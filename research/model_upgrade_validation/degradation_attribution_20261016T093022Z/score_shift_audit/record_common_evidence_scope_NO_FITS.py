"""Record shared information limits without selecting or fitting a new model."""
from pathlib import Path
import hashlib
import json

import joblib
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
REPO=ROOT.parents[2]
PRIOR=REPO/'research/model_upgrade_validation/positive_filter_20261016T091007Z'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run():
    data_path=PRIOR/'fixed_input_BEFORE_ANY_FIT.joblib';data=joblib.load(data_path)
    frame=data['full'];fields=data['fields']
    manifest=json.loads((PRIOR/'protocol_BEFORE_ANY_FIT.json').read_text(encoding='utf-8'))
    path=next(Path(p) for p in manifest['source_bindings'] if Path(p).name=='all41547_predictions_BEFORE_EVALUATION.csv')
    reference=pd.read_csv(path,dtype={'code':str},float_precision='round_trip',low_memory=False)
    outer=reference[reference.arm.eq(manifest['baseline'])][['date','code']]
    eval_frame=frame.merge(outer,on=['date','code'],validate='one_to_one')
    rows=[];shared_counts={}
    for scope,part in [('FULL9315',frame),('ORIGINAL_OUTER3777',eval_frame)]:
        multi=part[part.groupby('date').code.transform('size').gt(1)]
        shared=[]
        for c in fields:
            n=int(part[c].nunique(dropna=False));same=bool(multi.groupby('date')[c].nunique(dropna=False).le(1).all())
            if same:shared.append(c)
            rows.append(dict(scope=scope,field=c,rows=len(part),observed=int(part[c].notna().sum()),missing=int(part[c].isna().sum()),
                distinct_with_missing=n,globally_constant=n==1,shared_within_all_multi_candidate_dates=same,
                role_or_parent_rule_not_independent_ground_truth=c.startswith('causal_role_') or c.startswith('causal_parent_')))
        shared_counts[scope]=len(shared)
    pd.DataFrame(rows).to_csv(OUT/'all92_fields_information_scope_and_missingness.csv',index=False,encoding='utf-8-sig')
    lessons_path=PRIOR/'lessons/历轮经验总结_实施前.md'
    studies_path=PRIOR/'lessons/历轮研究总表_各自口径与原验收门.csv'
    sources=[data_path,path,lessons_path,studies_path,Path(__file__)]
    receipt=dict(new_fits=0,network_calls=0,old_files_modified=0,
        within_all_multi_candidate_day_shared_field_counts=shared_counts,
        full9315_all_empty_fields=[c for c in fields if frame[c].isna().all()],
        full9315_globally_constant_fields=[c for c in fields if frame[c].nunique(dropna=False)==1],
        historical_parent_rule_inputs_nonzero_not_missing_as_a_whole=True,
        actual_historical_theme_parent_identity_independent_truth_still_not_established=True,
        evidence_gaps_common_to_prior_algorithm_cost_proxy_changes=[
            'Shared market fields cannot themselves separate candidates with different outcomes on the same signal day',
            'Rule-derived roles/parent states and name/price proxies are not independent identity or theme truth',
            'Most algorithm and loss changes reuse the same historical price/volume evidence and target',
            'Late-acquired historical sources and current-universe survivor scope remain',
            'Repeated seen historical trials and overlapping monthly windows do not create new blind evidence',
            'Closing-promotion target does not certify execution, profit or multi-day continuation'],
        attribution='Descriptive evidence and competing mechanisms; not a unique causal proof',
        proposed_NO_META_REFIT_outer_predictions_generated_or_evaluated=False,
        source_bindings={str(p.resolve()):sha(p) for p in sources},
        output_bindings={str((OUT/'all92_fields_information_scope_and_missingness.csv').resolve()):sha(OUT/'all92_fields_information_scope_and_missingness.csv')})
    (OUT/'common_evidence_scope_NO_FITS_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['source_bindings','output_bindings']},ensure_ascii=False,indent=2))


if __name__=='__main__':run()
