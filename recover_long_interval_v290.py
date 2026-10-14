"""One separate bounded recovery of failed requests; never overwrite the parent."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import pandas as pd
import archive_long_interval_v290 as engine
from audit_long_interval_v290 import parse_source


def run(parent,output):
    parent,target=Path(parent),Path(output)
    if target.exists():raise FileExistsError('immutable recovery exists')
    report=json.loads((parent/'report.json').read_text(encoding='utf-8'))
    for path,wanted in report['source_hashes'].items():
        if engine.sha(path)!=wanted:raise ValueError('parent source changed')
    for name,wanted in report['artifact_hashes'].items():
        if engine.sha(parent/name)!=wanted:raise ValueError('parent artifact changed')
    codes=sorted({r['code'] for r in report['audit'] if r['kind']=='interval' and r['status']=='UNAVAILABLE'})
    original=pd.read_csv(parent/'candidate_scope_NO_OUTCOMES.csv',dtype={'code':str})
    scope=original[original.code.isin(codes)].copy()
    target.mkdir(parents=True,exist_ok=False)
    for kind in ['interval','daily']:(target/(kind+'_raw')).mkdir()
    scope.to_csv(target/'recovery_scope_NO_OUTCOMES.csv',index=False)
    audits=[];checks=[];frames={'interval':[],'daily':[]}
    with ThreadPoolExecutor(max_workers=4) as executor:
        jobs=[executor.submit(engine.fetch,code,scope[scope.code.eq(code)],target) for code in codes]
        for job in jobs:
            items,observed,detail=job.result();audits.extend(items);checks.extend(detail)
            for kind,frame in observed.items():frames[kind].append(frame)
    paths=[]
    for kind in frames:
        frame=pd.concat(frames[kind],ignore_index=True) if frames[kind] else pd.DataFrame(columns=engine.BAR_FIELDS if kind=='interval' else engine.DAY_FIELDS)
        path=target/(kind+'_normalized.csv');frame.to_csv(path,index=False);paths.append(path)
    # Reparse each newly returned raw response rather than trust receipt metadata.
    for item in audits:
        if item['status']!='RAW_PARSED':continue
        if engine.sha(item['raw_path'])!=item['raw_sha256']:raise ValueError('recovery raw hash mismatch')
        raw,raw_count=parse_source(item)
        saved=pd.read_csv(target/(item['kind']+'_normalized.csv'),dtype={'code':str} if item['kind']=='interval' else None)
        selected=saved[saved.code.eq(item['code'])] if item['kind']=='interval' else saved[saved.symbol.eq(item['symbol'])]
        pd.testing.assert_frame_equal(selected.reset_index(drop=True),raw.reset_index(drop=True),check_dtype=False,rtol=1e-12,atol=1e-8)
        if raw_count is not None and raw_count!=item['raw_rows']:raise ValueError('recovery raw count differs')
    path=target/'signal_checks.json';path.write_text(json.dumps(checks,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');paths.append(path)
    result=dict(status='SEPARATE_BOUNDED_RECOVERY_RAW_RESPONSES_RECONSTRUCTED',recorded_at=engine.timestamp(),
        parent_original_candidate_rows=report['candidate_rows'],recovery_codes=codes,recovery_candidate_rows=len(scope),
        available_interval_codes=len(frames['interval']),available_daily_codes=len(frames['daily']),
        recovered_price_rows=sum(r['price_reconciled'] for r in checks),recovered_complete_price_paths=sum(r['final_cap_price_bar_run'] is not None for r in checks),
        outcome_selection_used=False,parent_archive_rewritten=False,original_forecast_rewritten=False,
        parent_label_or_accuracy_rewritten=False,accuracy_claim_allowed=False,trade_authorized=False,
        audit=audits,source_hashes={str(Path(p).resolve()):engine.sha(p) for p in [parent/'report.json',parent/'candidate_scope_NO_OUTCOMES.csv',
            Path(__file__),Path(engine.__file__)]},artifact_hashes={p.name:engine.sha(p) for p in [*paths,target/'recovery_scope_NO_OUTCOMES.csv']},
        limitation='Recovery is an additional later receipt, not replacement of the original snapshot or a new forward forecast')
    (target/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return {key:result[key] for key in ['status','recovery_codes','recovery_candidate_rows','available_interval_codes','available_daily_codes','recovered_price_rows','recovered_complete_price_paths']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(run(args.parent,args.output),ensure_ascii=True))
