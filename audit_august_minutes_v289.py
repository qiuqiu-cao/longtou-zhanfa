"""Rebuild the full August cohort and both sources before path research."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import archive_august_minutes_v289 as engine
from next_session_expectation_contract_v271 import load_registered,now

LABEL_REFERENCE=engine.ROOT/'results/v279/daily_path_paired_20261002_v2/CONTROL_predictions_LABEL_ONLY.csv'


def run(directory,output):
    directory,target=Path(directory),Path(output)
    if target.exists():raise FileExistsError('immutable source audit exists')
    report=json.loads((directory/'report.json').read_text(encoding='utf-8'))
    scope=json.loads((directory/'request_scope.json').read_text(encoding='utf-8'))
    expected_sources={str(Path(p).resolve()) for p in [engine.COHORT,Path(engine.__file__),Path(engine.minute.__file__),
        engine.ROOT/'archive_september_5min_v287.py',engine.ROOT/'historical_ohlc_archive_v278.py']}
    if set(report['source_hashes'])!=expected_sources:raise ValueError('source input scope changed')
    for path,wanted in report['source_hashes'].items():
        if engine.sha(path)!=wanted:raise ValueError('source hash mismatch')
    for name,wanted in report['artifact_hashes'].items():
        path=(directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or engine.sha(path)!=wanted:
            raise ValueError('artifact hash mismatch')
    source=pd.read_csv(engine.COHORT,dtype={'code':str},low_memory=False)
    original=source[source.date.between('2026-08-01','2026-08-31')].reset_index(drop=True)
    cohort=pd.read_csv(directory/'august_cohort_LABEL_ONLY.csv',dtype={'code':str},low_memory=False)
    pd.testing.assert_frame_equal(cohort,original,check_dtype=False,rtol=1e-12,atol=1e-8)
    codes=set(cohort.code);expected_requests={(c,k) for c in codes for k in ['minute','daily']}
    requests=[(r['code'],r['kind']) for r in report['audit']]
    if len(requests)!=len(set(requests)) or set(requests)!=expected_requests or set(scope['codes'])!=codes:
        raise ValueError('full original code/source request denominator changed')
    frames={'minute':[],'daily':[]};unavailable=[]
    for i,item in enumerate(report['audit'],1):
        if 'raw_path' in item:
            path=Path(item['raw_path'])
            if not path.resolve().is_relative_to(directory.resolve()) or engine.sha(path)!=item['raw_sha256']:
                raise ValueError('raw response path or hash mismatch')
        if item['status']=='UNAVAILABLE':
            unavailable.append({k:item[k] for k in ['code','kind']});continue
        if item['status']!='RAW_PARSED' or item['http_status']!=200:
            raise ValueError('unexpected source status')
        text=Path(item['raw_path']).read_text(encoding='utf-8')
        if item['kind']=='minute':
            if item['parameters']!=dict(symbol=engine.minute.live.sina_symbol(item['code']),scale='5',ma='no',datalen='1969'):
                raise ValueError('wrong minute interval or depth')
            raw=engine.minute._extract_json_array(text)
            stamps=pd.Series([r.get('day') for r in raw],dtype='object')
            if len(raw)!=item['raw_rows'] or stamps.duplicated().any() or pd.to_datetime(stamps,errors='coerce').isna().any():
                raise ValueError('raw minute timestamp or count mismatch')
            frame=engine.minute.parse_minute_response(text,item['code'])
        else:frame=engine.daily_parse(json.loads(text),item['symbol'])
        if len(frame)!=item['parsed_rows'] or frame.empty:raise ValueError('parsed source count mismatch')
        frames[item['kind']].append(frame)
        if i%80==0:print(json.dumps(dict(verified_responses=i,total_responses=len(requests))),flush=True)
    bars=pd.concat(frames['minute'],ignore_index=True).sort_values(['code','minute']).reset_index(drop=True)
    daily=pd.concat(frames['daily'],ignore_index=True).sort_values(['symbol','date']).reset_index(drop=True)
    saved=pd.read_csv(directory/'five_minute_bars_UNRECONCILED.csv',dtype={'code':str})
    reference=pd.read_csv(directory/'unadjusted_daily_reference.csv')
    pd.testing.assert_frame_equal(saved,bars,check_dtype=False,rtol=1e-12,atol=1e-8)
    pd.testing.assert_frame_equal(reference,daily,check_dtype=False,rtol=1e-12,atol=1e-8)
    groups={code:frame for code,frame in bars.groupby('code')};checks=[];word_mismatches=[]
    for row in cohort.itertuples():
        check=engine.check_signal(groups.get(row.code,bars.iloc[:0]),daily,row.code,row.date)
        matching=daily[daily.symbol.eq(engine.minute.live.sina_symbol(row.code))&daily.date.eq(row.date)]
        check['reference_close_matches_original_pool']=bool(len(matching)==1 and abs(float(matching.close.iloc[0])-row.price)<=.005001)
        if len(matching)==1:
            truth=float(np.isclose(matching.high.iloc[0],matching.low.iloc[0],atol=.001,rtol=0))
            if not np.isfinite(row.daily_true_one_word) or truth!=row.daily_true_one_word:
                word_mismatches.append(dict(date=row.date,code=row.code,original=row.daily_true_one_word,reference=truth))
        checks.append(check)
    if checks!=json.loads((directory/'signal_checks.json').read_text(encoding='utf-8')):
        raise ValueError('candidate checks differ from raw reconstruction')
    actual=dict(candidate_rows=len(cohort),candidate_dates=int(cohort.date.nunique()),requested_codes=len(codes),
        available_minute_codes=len(frames['minute']),available_daily_codes=len(frames['daily']),
        minute_rows=len(bars),daily_rows=len(daily),interval_minutes=5,
        price_reconciled_rows=sum(c['price_reconciled'] for c in checks),
        grid_complete_rows=sum(c['grid_complete'] for c in checks),volume_reconciled_rows=sum(c['volume_reconciled'] for c in checks),
        reference_close_matches_original_pool_rows=sum(c['reference_close_matches_original_pool'] for c in checks),
        complete_price_path_rows=sum(c['price_reconciled'] and c['grid_complete'] and c['reference_close_matches_original_pool'] for c in checks))
    if any(report.get(key)!=value for key,value in actual.items()):raise ValueError('aggregate report mismatch')
    if any(report.get(key) is not False for key in ['outcomes_used_to_select_codes','missing_candidates_dropped',
            'original_forecast_rewritten','minute_interval_upgraded','accuracy_claim_allowed','trade_authorized']):
        raise ValueError('retrospective data cannot change forecast or scope')
    registry=engine.ROOT/'data/v271/expectation_registry/2026-09-30_2c74dc406eef7dcf.json'
    frozen=load_registered(registry)
    labels=pd.read_csv(LABEL_REFERENCE,dtype={'code':str},usecols=['date','code','target','label_valid'])
    labels=labels[labels.date.isin(cohort.date.unique())]
    if set(map(tuple,labels[['date','code']].to_numpy()))!=set(map(tuple,cohort[['date','code']].to_numpy())):
        raise ValueError('original label reference candidate scope mismatch')
    labeled=cohort[['date','code','promoted_next_day']].merge(labels,on=['date','code'],validate='one_to_one')
    valid=labeled.label_valid & labeled.target.notna()
    if not labeled.loc[valid,'target'].eq(labeled.loc[valid,'promoted_next_day']).all():
        raise ValueError('original promotion label disagrees with control')
    coverage=[]
    for day,group in cohort.groupby('date',sort=True):
        detail=[c for c in checks if c['date']==day]
        coverage.append(dict(date=day,candidates=len(detail),price_rows=sum(c['price_reconciled'] for c in detail),
            complete_path_rows=sum(c['price_reconciled'] and c['grid_complete'] and c['reference_close_matches_original_pool'] for c in detail),
            original_labels_complete=bool(labels.loc[labels.date.eq(day),'label_valid'].all()
                and labels.loc[labels.date.eq(day),'target'].notna().all())))
    result=dict(status='AUGUST_RAW_COHORT_AND_SIGNAL_CHECKS_RECONSTRUCTED',recorded_at=now().isoformat(),**actual,
        unavailable_requests=unavailable,original_one_word_reference_disagreements=word_mismatches,coverage_by_date=coverage,
        original_complete_label_dates=sum(r['original_labels_complete'] for r in coverage),
        frozen_registration_hash=frozen['sha256'],frozen_registry_byte_hash=engine.sha(registry),
        goal_verified=False,trade_authorized=False,
        source_hashes={str(Path(p).resolve()):engine.sha(p) for p in [directory/'report.json',engine.COHORT,
            directory/'august_cohort_LABEL_ONLY.csv',directory/'five_minute_bars_UNRECONCILED.csv',directory/'unadjusted_daily_reference.csv',
            directory/'signal_checks.json',LABEL_REFERENCE,Path(__file__),Path(engine.__file__),registry]})
    target.mkdir(parents=True,exist_ok=False)
    (target/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return {key:result[key] for key in ['status',*actual,'original_complete_label_dates','original_one_word_reference_disagreements']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--archive',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(run(args.archive,args.output),ensure_ascii=True))
