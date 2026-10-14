"""Full 2025Q4--2026Sep cohort, authentic 30-minute paths and independent days.

Streams normalized responses to keep memory bounded. Neither outcome nor
current security membership selects the original provider candidate cohort.
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
from archive_september_5min_v287 import sha
from historical_ohlc_archive_v278 import URL as DAILY_URL
from next_session_forecast_lab_v271 import ROOT

HISTORY=ROOT/'data/v279/daily_path_clues_20261002_v1/historical_relay_paths_LABEL_ONLY.csv'
SEPTEMBER=ROOT/'data/v279/daily_path_clues_20261002_v1/september_relay_paths_LABEL_ONLY.csv'
START,END='2025-09-01','2026-09-30'
FIELDS=['date','code','boards','price','daily_true_one_word']
BAR_FIELDS=['code','minute','open','high','low','close','volume_shares','amount']
DAY_FIELDS=['symbol','date','open','close','high','low','volume']
TIMES=['10:00','10:30','11:00','11:30','13:30','14:00','14:30','15:00']


def timestamp():return datetime.now().astimezone().isoformat()


def cohort():
    history=pd.read_csv(HISTORY,dtype={'code':str},usecols=FIELDS)
    history=history[history.date.between('2025-10-01','2026-08-31')]
    september=pd.read_csv(SEPTEMBER,dtype={'code':str},usecols=FIELDS)
    frame=pd.concat([history,september],ignore_index=True)
    if frame.empty or frame.duplicated(['date','code']).any() or not frame.boards.ge(2).all() \
            or not frame.code.str.fullmatch(r'(?:00|60)\d{4}').all():
        raise ValueError('unique original mainboard relay cohort required')
    return frame


def daily_parse(payload,symbol):
    raw=(payload.get('data') or {}).get(symbol,{}).get('day')
    if not isinstance(raw,list) or not raw or any(not isinstance(row,list) or len(row)<6 for row in raw):
        raise ValueError('unadjusted day list required; adjusted data not substituted')
    f=pd.DataFrame([row[:6] for row in raw],columns=DAY_FIELDS[1:])
    if f.date.duplicated().any() or not f.date.is_monotonic_increasing:
        raise ValueError('duplicate or unordered day dates')
    f=f[f.date.between(START,END)].copy()
    for field in DAY_FIELDS[2:]:f[field]=pd.to_numeric(f[field],errors='raise')
    valid=(np.isfinite(f[DAY_FIELDS[2:]]).all(axis=1)&f.low.gt(0)&f.high.ge(f.low)
        &f.open.between(f.low,f.high)&f.close.between(f.low,f.high)&f.volume.ge(0))
    if f.empty or not valid.all():raise ValueError('invalid or unavailable day prices')
    f.insert(0,'symbol',symbol)
    return f


def check_signal(bars,daily,code,day,price,one_word):
    rows=bars[bars.code.eq(code)&bars.minute.str[:10].eq(day)].sort_values('minute')
    reference=daily[daily.symbol.eq(minute.live.sina_symbol(code))&daily.date.eq(day)]
    result=dict(date=day,code=code,interval_minutes=30,observed_bars=len(rows),
        price_reconciled=False,grid_complete=False,missing_times=TIMES.copy(),
        volume_reconciled=False,original_pool_close_agrees=False,original_word_agrees=False,
        closing_price_is_daily_high=False,full_day_vwap=None,first_cap_high_bar_end=None,
        final_cap_price_bar_run=None,cap_price_bar_share=None,negative_max_pullback=None,
        eligible_for_frozen_forward=False,one_minute_evidence=False,auction_verified=False,
        continuous_seal_verified=False,real_reseal_count=None,actual_fill_verified=False)
    if len(reference)!=1:return result
    ref=reference.iloc[0]
    result.update(original_pool_close_agrees=bool(abs(ref.close-price)<=.005001),
        original_word_agrees=bool(np.isfinite(one_word) and float(np.isclose(ref.high,ref.low,atol=.001,rtol=0))==one_word),
        reference_ohlc={key:float(ref[key]) for key in ['open','high','low','close']})
    if rows.empty:return result
    times=set(rows.minute.str[11:16]);missing=sorted(set(TIMES)-times);extra=sorted(times-set(TIMES))
    duplicate=bool(rows.minute.duplicated().any())
    bounds=(np.isfinite(rows[['open','high','low','close']]).all(axis=1)&rows.low.gt(0)
        &rows.high.ge(rows[['open','low','close']].max(axis=1))&rows.low.le(rows[['open','high','close']].min(axis=1)))
    observed=dict(open=float(rows.open.iloc[0]),close=float(rows.close.iloc[-1]),high=float(rows.high.max()),low=float(rows.low.min()))
    agreement={key:abs(value-float(ref[key]))<=.005001 for key,value in observed.items()}
    price_ok=all(agreement.values()) and bool(bounds.all()) and not duplicate and not extra
    grid_ok=not missing and not extra and not duplicate
    volume_ok=bool(rows.volume_shares.notna().all() and rows.volume_shares.ge(0).all() and ref.volume>0
        and abs(float(rows.volume_shares.sum())-float(ref.volume)*100)<=100)
    traded=rows[rows.volume_shares.gt(0)]
    implied=traded.amount/traded.volume_shares
    unit_ok=bool(len(traded) and rows.amount.notna().all() and rows.amount.ge(0).all()
        and (implied.ge(traded.low-.020001)&implied.le(traded.high+.020001)).all())
    cap_ok=bool(result['original_pool_close_agrees'] and abs(ref.high-price)<=.005001)
    result.update(price_reconciled=price_ok,grid_complete=grid_ok,missing_times=missing,unexpected_times=extra,
        duplicate_times=duplicate,volume_reconciled=volume_ok,amount_volume_unit_consistent=unit_ok,
        closing_price_is_daily_high=cap_ok,observed_ohlc=observed,ohlc_agreement=agreement,
        observed_volume=float(rows.volume_shares.sum()),reference_raw_volume=float(ref.volume),
        volume_to_reference_scale_inferred=float(rows.volume_shares.sum()/ref.volume) if ref.volume>0 else None)
    if price_ok and grid_ok and volume_ok and unit_ok and rows.volume_shares.sum()>0:
        result['full_day_vwap']=float(rows.amount.sum()/rows.volume_shares.sum())
    if price_ok and grid_ok and cap_ok:
        at_cap=(rows[['open','high','low','close']]-price).abs().le(.005001).all(axis=1)
        run=0
        for value in at_cap.iloc[::-1]:
            if not value:break
            run+=1
        touched=rows.high.ge(price-.005001)
        result.update(final_cap_price_bar_run=run,cap_price_bar_share=float(at_cap.mean()),
            negative_max_pullback=float(rows.low.min()/price-1),
            first_cap_high_bar_end=rows.loc[touched,'minute'].iloc[0] if touched.any() else None)
    return result


def fetch(code,signals,target):
    symbol=minute.live.sina_symbol(code);audits=[];frames={}
    for kind in ['interval','daily']:
        item=dict(code=code,symbol=symbol,kind=kind,requested_at=timestamp())
        try:
            params=dict(symbol=symbol,scale='30',ma='no',datalen='1969') if kind=='interval' else \
                dict(param=f'{symbol},day,{START},{END},300,')
            response=requests.get(minute.SINA_MINUTE_URL if kind=='interval' else DAILY_URL,params=params,
                headers=minute.SINA_HEADERS if kind=='interval' else None,timeout=8)
            path=target/(kind+'_raw')/(code+'.response');path.write_bytes(response.content)
            item.update(url=response.url,parameters=params,http_status=response.status_code,raw_path=str(path.resolve()),
                raw_sha256=sha(path),available_at=timestamp())
            response.raise_for_status()
            if kind=='interval':
                raw=minute._extract_json_array(response.text)
                stamps=pd.Series([r.get('day') for r in raw],dtype='object')
                item.update(raw_rows=len(raw),duplicates_before_parser=int(stamps.duplicated().sum()),
                    invalid_raw_times=int(pd.to_datetime(stamps,errors='coerce').isna().sum()))
                if item['duplicates_before_parser'] or item['invalid_raw_times']:raise ValueError('invalid raw interval timestamps')
                frame=minute.parse_minute_response(response.text,code)
                if frame.empty:raise ValueError('empty response is not complete')
                item.update(interval_minutes=30,requested_bar_count=1969,first_bar=frame.minute.min(),last_bar=frame.minute.max())
            else:frame=daily_parse(response.json(),symbol)
            frames[kind]=frame;item.update(status='RAW_PARSED',parsed_rows=len(frame))
        except Exception as exc:item.update(status='UNAVAILABLE',error=str(exc)[:250])
        audits.append(item)
    checks=[check_signal(frames.get('interval',pd.DataFrame(columns=BAR_FIELDS)),
        frames.get('daily',pd.DataFrame(columns=DAY_FIELDS)),r.code,r.date,r.price,r.daily_true_one_word)
        for r in signals.itertuples()]
    return audits,frames,checks


def run(output):
    target=Path(output)
    if target.exists():raise FileExistsError('immutable full-cohort archive exists')
    candidates=cohort();codes=sorted(candidates.code.unique())
    source_hashes={str(Path(p).resolve()):sha(p) for p in [HISTORY,SEPTEMBER,Path(__file__),Path(minute.__file__)]}
    target.mkdir(parents=True,exist_ok=False)
    for kind in ['interval','daily']:(target/(kind+'_raw')).mkdir()
    cohort_path=target/'candidate_scope_NO_OUTCOMES.csv';candidates.to_csv(cohort_path,index=False,encoding='utf-8-sig')
    scope=dict(version='v290-full-long-interval-1',recorded_at=timestamp(),codes=codes,candidate_rows=len(candidates),
        candidate_dates=int(candidates.date.nunique()),signal_range=['2025-10-01','2026-09-30'],
        daily_range=[START,END],interval_minutes=30,bar_count=1969,workers=6,retries=0,
        outcome_selection_used=False,current_constituent_filter_used=False,source_hashes=source_hashes)
    scope_path=target/'request_scope.json';scope_path.write_text(json.dumps(scope,ensure_ascii=False,indent=2),encoding='utf-8')
    paths={'interval':target/'thirty_minute_bars_UNRECONCILED.csv','daily':target/'unadjusted_daily_reference.csv'}
    first={'interval':True,'daily':True};counts={'interval':0,'daily':0};available={'interval':0,'daily':0}
    audits=[];checks=[];groups={code:rows for code,rows in candidates.groupby('code')}
    with ThreadPoolExecutor(max_workers=6) as executor:
        pending={executor.submit(fetch,code,groups[code],target) for code in codes}
        for i,job in enumerate(as_completed(pending),1):
            pending.remove(job);items,frames,detail=job.result();audits.extend(items);checks.extend(detail)
            for kind,frame in frames.items():
                frame.to_csv(paths[kind],index=False,mode='w' if first[kind] else 'a',header=first[kind])
                first[kind]=False;counts[kind]+=len(frame);available[kind]+=1
            if i%50==0 or i==len(codes):
                print(json.dumps(dict(completed_codes=i,total_codes=len(codes),
                    signal_checks=len(checks),price_reconciled_signals=sum(c['price_reconciled'] for c in checks))),flush=True)
    for kind in paths:
        if first[kind]:pd.DataFrame(columns=BAR_FIELDS if kind=='interval' else DAY_FIELDS).to_csv(paths[kind],index=False)
    if any(sha(path)!=wanted for path,wanted in source_hashes.items()):raise ValueError('source changed during capture')
    checks.sort(key=lambda r:(r['date'],r['code']));check_path=target/'signal_checks.json'
    check_path.write_text(json.dumps(checks,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    report=dict(version=scope['version'],recorded_at=timestamp(),status='RETROSPECTIVE_THIRTY_MINUTE_RESEARCH_DATA',
        candidate_rows=len(candidates),candidate_dates=int(candidates.date.nunique()),requested_codes=len(codes),
        available_interval_codes=available['interval'],available_daily_codes=available['daily'],
        interval_rows=counts['interval'],daily_rows=counts['daily'],interval_minutes=30,
        price_reconciled_rows=sum(c['price_reconciled'] for c in checks),grid_complete_rows=sum(c['grid_complete'] for c in checks),
        pool_close_reconciled_rows=sum(c['original_pool_close_agrees'] for c in checks),
        one_word_agreement_rows=sum(c['original_word_agrees'] for c in checks),
        volume_reconciled_rows=sum(c['volume_reconciled'] for c in checks),
        complete_price_path_rows=sum(c['final_cap_price_bar_run'] is not None for c in checks),
        full_day_vwap_rows=sum(c['full_day_vwap'] is not None for c in checks),
        outcome_selection_used=False,missing_candidates_dropped=False,one_minute_evidence=False,
        original_forecast_rewritten=False,accuracy_claim_allowed=False,trade_authorized=False,
        source_hashes=source_hashes,artifact_hashes={p.name:sha(p) for p in [cohort_path,scope_path,check_path,*paths.values()]},
        audit=sorted(audits,key=lambda r:(r['code'],r['kind'])),
        limitations=['Thirty-minute OHLC cannot resolve one-/five-minute or tick order, auctions, continuous seals or actual fills',
            'Original provider cohort does not certify exchange-wide universe or historical security status',
            'All receipts obtained after the original forecast; no backfill or independent forward credit',
            'Prices, grid, volume scale, original pool close and word mask are checked separately; no imputation',
            'Previously viewed labels; data availability is not model improvement or stable sixty percent'])
    (target/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return {k:report[k] for k in ['status','candidate_rows','candidate_dates','requested_codes','available_interval_codes',
        'available_daily_codes','interval_rows','daily_rows','price_reconciled_rows','grid_complete_rows',
        'pool_close_reconciled_rows','one_word_agreement_rows','volume_reconciled_rows','complete_price_path_rows','full_day_vwap_rows']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    print(json.dumps(run(parser.parse_args().output),ensure_ascii=True))
