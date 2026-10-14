"""A necessary closing-board sanity check, never a sufficient limit-price test.

Contradictions preserve the entire source cohort and all original labels. The
ledger does not assume previous-day close is a valid ex-right reference price.
"""
import argparse
from decimal import Decimal,InvalidOperation
import json
from pathlib import Path
import pandas as pd
from archive_september_5min_v287 import sha
from next_session_expectation_contract_v271 import now
from long_interval_layers_v290 import load as load_layers


def classify(item):
    result={key:item[key] for key in ['date','code']}
    result.update(status='UNVERIFIED_SOURCE',necessary_close_high_passed=None,
        close_limit_confirmed=False,board_height_confirmed=False,trade_authorized=False,
        official_upper_limit_price=None,high_minus_close=None,
        reason='Independent intraday/day prices and original pool close must agree first')
    if not item.get('price_reconciled') or not item.get('original_pool_close_agrees'):
        return result
    values=item.get('observed_ohlc',{})
    try:
        high,close=Decimal(str(values['high'])),Decimal(str(values['close']))
    except (InvalidOperation,KeyError):return result
    if not high.is_finite() or not close.is_finite() or close<=0 or high<close:return result
    difference=high-close
    result['high_minus_close']=str(difference)
    if difference>=Decimal('0.01'):
        result.update(status='CLOSE_LIMIT_CONTRADICTION',necessary_close_high_passed=False,
            reason='Verified traded high exceeds verified close by at least one cent; closing at a fixed legal upper limit is contradicted')
    elif difference==0:
        result.update(status='NECESSARY_CONDITION_ONLY',necessary_close_high_passed=True,
            reason='High equals close; official limit price, applicable rules and contiguous episode are still unverified')
    else:
        result.update(status='UNRESOLVED_PRICE_PRECISION',reason='Nonzero sub-cent discrepancy is not silently rounded into a pass')
    return result


def run(archive,audit,output,recovery=None):
    archive,audit,target=Path(archive),Path(audit),Path(output)
    if target.exists():raise FileExistsError('immutable closing-board ledger exists')
    cohort,checks,_,_,inputs=load_layers(archive,audit,recovery)
    if len(checks)!=len(cohort) or set((r['date'],r['code']) for r in checks)!=set(map(tuple,cohort[['date','code']].to_numpy())):
        raise ValueError('original candidate denominator changed')
    ledger=[classify(item) for item in checks]
    target.mkdir(parents=True,exist_ok=False)
    path=target/'closing_board_ledger_NO_OUTCOMES.json'
    path.write_text(json.dumps(ledger,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    statuses=pd.Series([r['status'] for r in ledger]).value_counts().to_dict()
    report=dict(version='v290-necessary-closing-board-sanity-1',recorded_at=now().isoformat(),
        status='FULL_COHORT_NECESSARY_CONDITION_AUDIT',candidate_rows=len(cohort),
        status_counts={key:int(value) for key,value in statuses.items()},
        contradictions=[r for r in ledger if r['status']=='CLOSE_LIMIT_CONTRADICTION'],
        missing_candidates_dropped=False,original_labels_rewritten=False,original_forecast_rewritten=False,
        model_retrained=False,accuracy_claim_allowed=False,goal_verified=False,trade_authorized=False,
        inference='Under a fixed legal upper bound, a close at that bound cannot be below a verified traded high; price agreement alone does not establish a legal upper bound',
        limitations=['This is a necessary-condition diagnostic, not official limit-price or security-status certification',
            'High equal to close does not prove limit-up, uninterrupted boards, theme identity or actual fills',
            'Historical labels and predictions remain unchanged; conflicts cannot be removed to improve reported accuracy',
            'OHLC disagreement remains unknown; no assumption that raw previous close is an ex-right reference'],
        source_hashes={str(Path(p).resolve()):sha(p) for p in [*inputs,Path(__file__)]},
        artifact_hashes={path.name:sha(path)})
    (target/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return {key:report[key] for key in ['status','candidate_rows','status_counts']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['archive','audit','output']:parser.add_argument('--'+key,required=True)
    parser.add_argument('--recovery')
    args=parser.parse_args();print(json.dumps(run(args.archive,args.audit,args.output,args.recovery),ensure_ascii=True))
