"""Necessary-condition label diagnostics; never rewrite labels or select trades."""
import argparse
import json
from pathlib import Path
import pandas as pd
import archive_long_interval_v290 as source
from audit_long_interval_v290 import CONTROL
from closing_board_sanity_v290 import classify
from next_session_expectation_contract_v271 import now
from long_interval_layers_v290 import load as load_layers


def label_status(target,valid,signal_status,next_status):
    if not valid or pd.isna(target):return 'ORIGINAL_LABEL_UNRESOLVED'
    if target==1:
        if signal_status=='CLOSE_LIMIT_CONTRADICTION':return 'POSITIVE_CONTINUATION_SIGNAL_CONTRADICTION'
        if next_status=='CLOSE_LIMIT_CONTRADICTION':return 'POSITIVE_NEXT_CLOSE_CONTRADICTION'
        if signal_status=='NECESSARY_CONDITION_ONLY' and next_status=='NECESSARY_CONDITION_ONLY':
            return 'POSITIVE_NECESSARY_CONDITIONS_ONLY'
        return 'POSITIVE_NECESSARY_EVIDENCE_UNAVAILABLE'
    if target==0:
        return 'NEGATIVE_NEXT_CLOSE_NECESSARY_REJECTION_SUPPORTED' if next_status=='CLOSE_LIMIT_CONTRADICTION' \
            else 'NEGATIVE_REQUIRES_LIMIT_PRICE_VERIFICATION'
    raise ValueError('original target must be binary or missing')


def run(archive,audit,output,recovery=None):
    archive,audit,target=Path(archive),Path(audit),Path(output)
    if target.exists():raise FileExistsError('immutable label sanity report exists')
    scope,signal_checks,bars,daily,inputs=load_layers(archive,audit,recovery,with_bars=True)
    labels=pd.read_csv(CONTROL,dtype={'code':str},usecols=['date','code','target_date','target','label_valid'])
    labels=scope[['date','code']].merge(labels,on=['date','code'],how='left',validate='one_to_one')
    if labels.label_valid.isna().any():raise ValueError('original label scope changed')
    signals={(r['date'],r['code']):classify(r) for r in signal_checks}
    bar_groups={code:group for code,group in bars.groupby('code',sort=False)}
    day_groups={symbol:group for symbol,group in daily.groupby('symbol',sort=False)}
    records=[]
    for i,row in enumerate(labels.itertuples(),1):
        signal=signals[(row.date,row.code)];next_check=None;next_gate=None
        if pd.notna(row.target_date):
            reference=day_groups.get(source.minute.live.sina_symbol(row.code),daily.iloc[:0])
            match=reference[reference.date.eq(row.target_date)]
            if len(match)==1:
                cap=float(match.close.iloc[0]);word=float(abs(match.high.iloc[0]-match.low.iloc[0])<=.001)
                next_check=source.check_signal(bar_groups.get(row.code,bars.iloc[:0]),reference,row.code,row.target_date,cap,word)
                next_gate=classify(next_check)
        status=label_status(row.target,bool(row.label_valid),signal['status'],next_gate['status'] if next_gate else 'UNVERIFIED_SOURCE')
        records.append(dict(date=row.date,code=row.code,target_date=row.target_date if pd.notna(row.target_date) else None,
            original_target=float(row.target) if pd.notna(row.target) else None,original_label_valid=bool(row.label_valid),
            signal_status=signal['status'],next_status=next_gate['status'] if next_gate else 'UNVERIFIED_SOURCE',
            audit_status=status,next_day_check_LABEL_ONLY=next_check,
            label_rewritten=False,feature_use_allowed=False,trade_authorized=False))
        if i%600==0:print(json.dumps(dict(audited_labels=i,total_labels=len(labels))),flush=True)
    target.mkdir(parents=True,exist_ok=False)
    path=target/'closing_label_ledger_LABEL_ONLY.json'
    path.write_text(json.dumps(records,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    counts=pd.Series([r['audit_status'] for r in records]).value_counts().to_dict()
    result=dict(version='v290-close-label-necessary-condition-1',recorded_at=now().isoformat(),
        status='ORIGINAL_LABELS_PRESERVED_WITH_NECESSARY_CONTRADICTION_FLAGS',candidate_rows=len(records),
        status_counts={key:int(value) for key,value in counts.items()},
        positive_contradictions=[{k:r[k] for k in ['date','code','target_date','original_target','signal_status','next_status','audit_status']}
            for r in records if r['audit_status'] in ['POSITIVE_CONTINUATION_SIGNAL_CONTRADICTION','POSITIVE_NEXT_CLOSE_CONTRADICTION']],
        original_labels_rewritten=False,missing_candidates_dropped=False,original_forecast_rewritten=False,
        model_retrained=False,accuracy_claim_allowed=False,goal_verified=False,trade_authorized=False,
        limitations=['Necessary condition only; official daily limit price, ex-right reference and security-status history remain unverified',
            'High equal to close is not sufficient evidence of a limit-up or a contiguous multi-board episode',
            'Next-day checks are labels only; no next-day facts may enter signal features or ranking',
            'Original unresolved labels remain unresolved; no future dates are substituted',
            'Contradictions are retained and reported, not dropped to improve model performance'],
        source_hashes={str(Path(p).resolve()):source.sha(p) for p in [*inputs,CONTROL,Path(__file__),Path(source.__file__),
            Path(__file__).with_name('closing_board_sanity_v290.py')]},
        artifact_hashes={path.name:source.sha(path)})
    (target/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return {key:result[key] for key in ['status','candidate_rows','status_counts']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['archive','audit','output']:parser.add_argument('--'+key,required=True)
    parser.add_argument('--recovery')
    args=parser.parse_args();print(json.dumps(run(args.archive,args.audit,args.output,args.recovery),ensure_ascii=True))
