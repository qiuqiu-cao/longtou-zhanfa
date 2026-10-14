"""Reconstruct source responses and every original long-interval candidate."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import archive_long_interval_v290 as engine
from next_session_expectation_contract_v271 import now,load_registered

CONTROL=engine.ROOT/'results/v279/daily_path_paired_20261002_v2/CONTROL_predictions_LABEL_ONLY.csv'
PRIOR_FREEZE_AUDIT=engine.ROOT/'results/v289/august_minute_audit_20261002_v1/report.json'


def parse_source(item):
    text=Path(item['raw_path']).read_text(encoding='utf-8')
    if item['kind']=='interval':
        raw=engine.minute._extract_json_array(text)
        stamps=pd.Series([r.get('day') for r in raw],dtype='object')
        if stamps.duplicated().any() or pd.to_datetime(stamps,errors='coerce').isna().any():
            raise ValueError('duplicate or invalid raw interval stamps')
        frame=engine.minute.parse_minute_response(text,item['code'])
        return frame,len(raw)
    return engine.daily_parse(json.loads(text),item['symbol']),None


def run(directory,output):
    directory,target=Path(directory),Path(output)
    if target.exists():raise FileExistsError('immutable long-interval audit exists')
    report=json.loads((directory/'report.json').read_text(encoding='utf-8'))
    scope=json.loads((directory/'request_scope.json').read_text(encoding='utf-8'))
    expected_sources={str(Path(p).resolve()) for p in [engine.HISTORY,engine.SEPTEMBER,Path(engine.__file__),Path(engine.minute.__file__)]}
    if set(report['source_hashes'])!=expected_sources or scope['source_hashes']!=report['source_hashes']:
        raise ValueError('original source input scope changed')
    for path,wanted in report['source_hashes'].items():
        if engine.sha(path)!=wanted:raise ValueError('source hash mismatch')
    for name,wanted in report['artifact_hashes'].items():
        path=(directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or engine.sha(path)!=wanted:raise ValueError('artifact hash mismatch')
    original=engine.cohort()
    cohort=pd.read_csv(directory/'candidate_scope_NO_OUTCOMES.csv',dtype={'code':str})
    pd.testing.assert_frame_equal(cohort,original,check_dtype=False,rtol=1e-12,atol=1e-8)
    codes=set(cohort.code)
    keys=[(item['code'],item['kind']) for item in report['audit']]
    if len(keys)!=len(set(keys)) or set(keys)!={(code,kind) for code in codes for kind in ['interval','daily']} \
            or set(scope['codes'])!=codes or scope['interval_minutes']!=30:
        raise ValueError('full cohort or thirty-minute request denominator changed')
    bars=pd.read_csv(directory/'thirty_minute_bars_UNRECONCILED.csv',dtype={'code':str})
    daily=pd.read_csv(directory/'unadjusted_daily_reference.csv')
    if bars.duplicated(['code','minute']).any() or daily.duplicated(['symbol','date']).any():
        raise ValueError('normalized duplicate interval or day keys')
    bar_groups={code:group for code,group in bars.groupby('code',sort=False)}
    day_groups={symbol:group for symbol,group in daily.groupby('symbol',sort=False)}
    available={'interval':set(),'daily':set()};counts={'interval':0,'daily':0};unavailable=[]
    for i,item in enumerate(report['audit'],1):
        if 'raw_path' in item:
            path=Path(item['raw_path']).resolve()
            if not path.is_relative_to(directory.resolve()) or engine.sha(path)!=item['raw_sha256']:
                raise ValueError('raw response path or hash mismatch')
        expected=dict(symbol=engine.minute.live.sina_symbol(item['code']),scale='30',ma='no',datalen='1969') if item['kind']=='interval' else \
            dict(param=f"{item['symbol']},day,{engine.START},{engine.END},300,")
        if 'parameters' in item and item['parameters']!=expected:raise ValueError('request interval or day range changed')
        if item['status']=='UNAVAILABLE':
            # A successful response may not be hidden as missing to improve scope.
            if item.get('http_status')==200 and 'raw_path' in item:
                try:possible,_=parse_source(item)
                except (ValueError,KeyError,TypeError,json.JSONDecodeError):possible=pd.DataFrame()
                if not possible.empty:raise ValueError('available response labeled unavailable')
            unavailable.append({key:item.get(key) for key in ['code','kind','http_status','error']});continue
        if item['status']!='RAW_PARSED' or item['http_status']!=200:raise ValueError('unexpected source status')
        parsed,raw_count=parse_source(item)
        if len(parsed)!=item['parsed_rows'] or parsed.empty:raise ValueError('parsed row count mismatch')
        if raw_count is not None and (raw_count!=item['raw_rows'] or parsed.minute.min()!=item['first_bar'] or parsed.minute.max()!=item['last_bar']):
            raise ValueError('raw count or time window mismatch')
        if item['kind']=='interval':
            stored=bar_groups.get(item['code'],bars.iloc[:0]).sort_values('minute').reset_index(drop=True)
            parsed=parsed.sort_values('minute').reset_index(drop=True)
        else:
            stored=day_groups.get(item['symbol'],daily.iloc[:0]).sort_values('date').reset_index(drop=True)
            parsed=parsed.sort_values('date').reset_index(drop=True)
        pd.testing.assert_frame_equal(stored,parsed,check_dtype=False,rtol=1e-12,atol=1e-8)
        available[item['kind']].add(item['code']);counts[item['kind']]+=len(parsed)
        if i%200==0:print(json.dumps(dict(verified_responses=i,total_responses=len(keys))),flush=True)
    if set(bar_groups)!={code for code in available['interval']} or set(day_groups)!={engine.minute.live.sina_symbol(code) for code in available['daily']}:
        raise ValueError('normalized symbols differ from successful raw sources')
    checks=[]
    for row in cohort.itertuples():
        checks.append(engine.check_signal(bar_groups.get(row.code,bars.iloc[:0]),
            day_groups.get(engine.minute.live.sina_symbol(row.code),daily.iloc[:0]),row.code,row.date,row.price,row.daily_true_one_word))
    checks.sort(key=lambda r:(r['date'],r['code']))
    if checks!=json.loads((directory/'signal_checks.json').read_text(encoding='utf-8')):
        raise ValueError('candidate signal checks differ from reconstructed raw sources')
    actual=dict(candidate_rows=len(cohort),candidate_dates=int(cohort.date.nunique()),requested_codes=len(codes),
        available_interval_codes=len(available['interval']),available_daily_codes=len(available['daily']),
        interval_rows=counts['interval'],daily_rows=counts['daily'],interval_minutes=30,
        price_reconciled_rows=sum(r['price_reconciled'] for r in checks),grid_complete_rows=sum(r['grid_complete'] for r in checks),
        pool_close_reconciled_rows=sum(r['original_pool_close_agrees'] for r in checks),one_word_agreement_rows=sum(r['original_word_agrees'] for r in checks),
        volume_reconciled_rows=sum(r['volume_reconciled'] for r in checks),
        complete_price_path_rows=sum(r['final_cap_price_bar_run'] is not None for r in checks),
        full_day_vwap_rows=sum(r['full_day_vwap'] is not None for r in checks))
    if any(report.get(key)!=value for key,value in actual.items()):raise ValueError('aggregate report differs from source facts')
    if any(report.get(key) is not False for key in ['outcome_selection_used','missing_candidates_dropped','one_minute_evidence',
            'original_forecast_rewritten','accuracy_claim_allowed','trade_authorized']):raise ValueError('retrospective data cannot authorize score or forecast changes')
    labels=pd.read_csv(CONTROL,dtype={'code':str},usecols=['date','code','label_valid','target'])
    labels=labels[labels.date.isin(cohort.date.unique())]
    if set(map(tuple,labels[['date','code']].to_numpy()))!=set(map(tuple,cohort[['date','code']].to_numpy())):
        raise ValueError('original control label scope differs')
    complete=labels.groupby('date').apply(lambda g:bool(g.label_valid.all() and g.target.notna().all()),include_groups=False)
    coverage=[]
    for day,group in cohort.groupby('date',sort=True):
        detail=[r for r in checks if r['date']==day]
        coverage.append(dict(date=day,candidate_rows=len(group),price_rows=sum(r['price_reconciled'] for r in detail),
            complete_path_rows=sum(r['final_cap_price_bar_run'] is not None for r in detail),
            whole_original_price_path_complete=all(r['final_cap_price_bar_run'] is not None for r in detail),
            original_labels_complete=bool(complete.loc[day])))
    registry=engine.ROOT/'data/v271/expectation_registry/2026-09-30_2c74dc406eef7dcf.json'
    prior=json.loads(PRIOR_FREEZE_AUDIT.read_text(encoding='utf-8'))
    if engine.sha(registry)!=prior['frozen_registry_byte_hash']:raise ValueError('original frozen forecast changed')
    frozen=load_registered(registry)
    result=dict(status='LONG_INTERVAL_RAW_COHORT_AND_CHECKS_RECONSTRUCTED',recorded_at=now().isoformat(),**actual,
        unavailable_requests=unavailable,coverage_by_date=coverage,original_complete_label_dates=int(complete.sum()),
        whole_pool_complete_path_dates=sum(r['whole_original_price_path_complete'] for r in coverage),
        original_one_word_reference_disagreements=[{k:r[k] for k in ['date','code','reference_ohlc']} for r in checks
            if 'reference_ohlc' in r and not r['original_word_agrees']],
        original_pool_close_reference_disagreements=[{k:r[k] for k in ['date','code','reference_ohlc']} for r in checks
            if 'reference_ohlc' in r and not r['original_pool_close_agrees']],
        goal_verified=False,trade_authorized=False,frozen_registration_hash=frozen['sha256'],frozen_registry_byte_hash=engine.sha(registry),
        source_hashes={str(Path(p).resolve()):engine.sha(p) for p in [directory/'report.json',directory/'candidate_scope_NO_OUTCOMES.csv',
            directory/'thirty_minute_bars_UNRECONCILED.csv',directory/'unadjusted_daily_reference.csv',directory/'signal_checks.json',
            CONTROL,PRIOR_FREEZE_AUDIT,registry,Path(__file__),Path(engine.__file__)]})
    target.mkdir(parents=True,exist_ok=False)
    (target/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return {key:result[key] for key in ['status',*actual,'original_complete_label_dates','whole_pool_complete_path_dates']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--archive',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(run(args.archive,args.output),ensure_ascii=True))
