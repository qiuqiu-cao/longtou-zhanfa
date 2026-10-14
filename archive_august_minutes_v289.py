"""Full August relay cohort, 1969 five-minute bars and independent day reference.

No outcomes select requests. Missing or conflicting rows remain in the cohort.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import public_minute_capture_v1 as minute
from archive_september_5min_v287 import sha,check_signal
from historical_ohlc_archive_v278 import URL as DAILY_URL
from next_session_forecast_lab_v271 import ROOT

COHORT=ROOT/'data/v279/daily_path_clues_20261002_v1/historical_relay_paths_LABEL_ONLY.csv'
START,END='2026-08-01','2026-09-01'
FIELDS=['code','minute','open','high','low','close','volume_shares','amount']


def daily_parse(payload,symbol):
    raw=(payload.get('data') or {}).get(symbol,{}).get('day')
    if not isinstance(raw,list) or not raw or any(not isinstance(r,list) or len(r)<6 for r in raw):
        raise ValueError('unadjusted day list required; adjusted data not substituted')
    frame=pd.DataFrame([r[:6] for r in raw],columns=['date','open','close','high','low','volume'])
    if frame.date.duplicated().any() or not frame.date.is_monotonic_increasing:
        raise ValueError('duplicate or unordered raw daily dates')
    frame=frame[frame.date.between(START,END)].copy()
    for key in ['open','close','high','low','volume']:frame[key]=pd.to_numeric(frame[key],errors='raise')
    valid=(np.isfinite(frame[['open','close','high','low','volume']]).all(axis=1)&frame.low.gt(0)
        &frame.high.ge(frame.low)&frame.open.between(frame.low,frame.high)
        &frame.close.between(frame.low,frame.high)&frame.volume.ge(0))
    if frame.empty or not valid.all():raise ValueError('invalid or unavailable reference daily bars')
    frame.insert(0,'symbol',symbol)
    return frame


def fetch(code,target):
    symbol=minute.live.sina_symbol(code)
    audits=[];frames={}
    for kind in ['minute','daily']:
        item=dict(code=code,symbol=symbol,kind=kind,requested_at=datetime.now().astimezone().isoformat())
        try:
            params=dict(symbol=symbol,scale='5',ma='no',datalen='1969') if kind=='minute' else \
                dict(param=f'{symbol},day,{START},{END},100,')
            url=minute.SINA_MINUTE_URL if kind=='minute' else DAILY_URL
            response=requests.get(url,params=params,headers=minute.SINA_HEADERS if kind=='minute' else None,timeout=8)
            path=target/(kind+'_raw')/(code+'.response');path.write_bytes(response.content)
            item.update(url=response.url,parameters=params,http_status=response.status_code,
                raw_path=str(path.resolve()),raw_sha256=sha(path),available_at=datetime.now().astimezone().isoformat())
            response.raise_for_status()
            if kind=='minute':
                raw=minute._extract_json_array(response.text)
                times=pd.Series([r.get('day') for r in raw],dtype='object')
                item.update(raw_rows=len(raw),duplicates_before_parser=int(times.duplicated().sum()),
                    invalid_raw_times=int(pd.to_datetime(times,errors='coerce').isna().sum()))
                if item['duplicates_before_parser'] or item['invalid_raw_times']:
                    raise ValueError('duplicate or invalid raw minute times')
                frame=minute.parse_minute_response(response.text,code)
                if frame.empty:raise ValueError('empty minute response is not complete')
                item.update(interval_minutes=5,requested_bar_count=1969,
                    first_bar=frame.minute.min(),last_bar=frame.minute.max())
            else:frame=daily_parse(response.json(),symbol)
            frames[kind]=frame;item.update(status='RAW_PARSED',parsed_rows=len(frame))
        except Exception as exc:item.update(status='UNAVAILABLE',error=str(exc)[:250])
        audits.append(item)
    return audits,frames


def run(output):
    target=Path(output)
    if target.exists():raise FileExistsError('immutable full-cohort archive exists')
    original=pd.read_csv(COHORT,dtype={'code':str},low_memory=False)
    cohort=original[original.date.between('2026-08-01','2026-08-31')].copy()
    if cohort.empty or cohort.duplicated(['date','code']).any() or not cohort.boards.ge(2).all() \
            or not cohort.code.str.fullmatch(r'(?:00|60)\d{4}').all():
        raise ValueError('original unique mainboard relay cohort required')
    codes=sorted(cohort.code.unique())
    target.mkdir(parents=True,exist_ok=False)
    for name in ['minute_raw','daily_raw']:(target/name).mkdir()
    cohort_path=target/'august_cohort_LABEL_ONLY.csv'
    cohort.to_csv(cohort_path,index=False,encoding='utf-8-sig')
    scope=dict(version='v289-full-august-five-minute-1',recorded_at=datetime.now().astimezone().isoformat(),
        candidate_rows=len(cohort),candidate_dates=int(cohort.date.nunique()),codes=codes,
        signal_range=['2026-08-01','2026-08-31'],daily_reference_range=[START,END],
        interval_minutes=5,requested_bar_count=1969,outcomes_used_to_select_codes=False,
        original_cohort_source=str(COHORT.resolve()),original_cohort_sha256=sha(COHORT))
    (target/'request_scope.json').write_text(json.dumps(scope,ensure_ascii=False,indent=2),encoding='utf-8')
    audits=[];minute_frames=[];daily_frames=[]
    with ThreadPoolExecutor(max_workers=4) as executor:
        jobs=[executor.submit(fetch,code,target) for code in codes]
        for i,job in enumerate(as_completed(jobs),1):
            audit,frames=job.result();audits.extend(audit)
            if 'minute' in frames:minute_frames.append(frames['minute'])
            if 'daily' in frames:daily_frames.append(frames['daily'])
            if i%20==0 or i==len(codes):print(json.dumps(dict(completed_codes=i,total_codes=len(codes))),flush=True)
    bars=pd.concat(minute_frames,ignore_index=True).sort_values(['code','minute']).reset_index(drop=True) \
        if minute_frames else pd.DataFrame(columns=FIELDS)
    reference=pd.concat(daily_frames,ignore_index=True).sort_values(['symbol','date']).reset_index(drop=True) \
        if daily_frames else pd.DataFrame(columns=['symbol','date','open','close','high','low','volume'])
    bars_path=target/'five_minute_bars_UNRECONCILED.csv';bars.to_csv(bars_path,index=False)
    daily_path=target/'unadjusted_daily_reference.csv';reference.to_csv(daily_path,index=False)
    groups={code:frame for code,frame in bars.groupby('code')}
    checks=[]
    for row in cohort.itertuples():
        check=check_signal(groups.get(row.code,bars.iloc[:0]),reference,row.code,row.date)
        matching=reference[reference.symbol.eq(minute.live.sina_symbol(row.code))&reference.date.eq(row.date)]
        check['reference_close_matches_original_pool']=bool(len(matching)==1 and abs(float(matching.close.iloc[0])-row.price)<=.005001)
        checks.append(check)
    checks_path=target/'signal_checks.json'
    checks_path.write_text(json.dumps(checks,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    report=dict(version=scope['version'],recorded_at=datetime.now().astimezone().isoformat(),
        status='RETROSPECTIVE_AUGUST_DATA_NOT_FORWARD_EVIDENCE',candidate_rows=len(cohort),candidate_dates=int(cohort.date.nunique()),
        requested_codes=len(codes),available_minute_codes=len(minute_frames),available_daily_codes=len(daily_frames),
        minute_rows=len(bars),daily_rows=len(reference),interval_minutes=5,
        price_reconciled_rows=sum(c['price_reconciled'] for c in checks),
        grid_complete_rows=sum(c['grid_complete'] for c in checks),
        volume_reconciled_rows=sum(c['volume_reconciled'] for c in checks),
        reference_close_matches_original_pool_rows=sum(c['reference_close_matches_original_pool'] for c in checks),
        complete_price_path_rows=sum(c['price_reconciled'] and c['grid_complete'] and c['reference_close_matches_original_pool'] for c in checks),
        outcomes_used_to_select_codes=False,missing_candidates_dropped=False,original_forecast_rewritten=False,
        minute_interval_upgraded=False,accuracy_claim_allowed=False,trade_authorized=False,audit=sorted(audits,key=lambda r:(r['code'],r['kind'])),
        source_hashes={str(Path(p).resolve()):sha(p) for p in [COHORT,Path(__file__),Path(minute.__file__),
            ROOT/'archive_september_5min_v287.py',ROOT/'historical_ohlc_archive_v278.py']},
        artifact_hashes={p.name:sha(p) for p in [cohort_path,target/'request_scope.json',bars_path,daily_path,checks_path]},
        limitations=['All receipts are October retrospective evidence; no original forecast backfill',
            'Five-minute OHLC lacks auction, tick order, continuous seal, queue fill and historical theme evidence',
            '1969 bars may start mid-session or omit traded intervals; no interpolation',
            'Original provider cohort is retained; no exchange-wide universe completeness claim',
            'Volume multiplier 100 is checked against independent raw day volume, not an official unit certification',
            'Historical labels already inspected; no pristine blind test and no stable sixty-percent claim'])
    (target/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return {k:report[k] for k in ['status','candidate_rows','candidate_dates','requested_codes','available_minute_codes',
        'available_daily_codes','minute_rows','price_reconciled_rows','grid_complete_rows','volume_reconciled_rows',
        'reference_close_matches_original_pool_rows','complete_price_path_rows']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    print(json.dumps(run(parser.parse_args().output),ensure_ascii=True))
