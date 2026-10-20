"""Verify diagnostic boundaries, replay branch cases, and bound file hashes."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import json
import pandas as pd
import short_term_thinking_dataset_builder as b
from audit_current_shape import OUT,inspect_one


def main():
    sessions=['2026-09-04','2026-09-07'];indices={d:i for i,d in enumerate(sessions)}
    fields=['open','high','low','close','pre_close','volume']
    prior=(10,10,9,10,9,100)
    good=(10.8,11,10.7,11,10,200)
    key=('2026-09-07','000001')
    def call(current=good,previous=prior,pools=None):
        prices={}
        if current is not None:prices[key]=current
        if previous is not None:prices[('2026-09-04','000001')]=previous
        return inspect_one(*key,prices,pools or {},sessions,indices,fields)
    checks={}
    checks['official_adjacent_over_weekend']=call()['previous_official_date']=='2026-09-04'
    checks['valid_nonone_price']=call()['replay_current_price_shape']==0
    checks['nonclosing_never_assigned_nonword']=call(current=(10.8,11,10.7,10.9,10,200))['replay_current_price_shape'] is None
    checks['high_above_cap_rejected']=call(current=(10.8,11.01,10.7,11,10,200))['replay_daily_status']=='UNKNOWN_10PCT_SCOPE_CONFLICT'
    checks['explicit_preclose_conflict_rejected']=call(current=(10.8,11,10.7,11,9.9,200))['replay_daily_status']=='UNKNOWN_REFERENCE_CONFLICT'
    checks['zero_volume_rejected']=call(current=(10.8,11,10.7,11,10,0))['replay_daily_status']=='UNKNOWN_NONTRADING_OR_VOLUME'
    checks['missing_volume_rejected']=call(current=(10.8,11,10.7,11,10,None))['replay_daily_status']=='UNKNOWN_NONTRADING_OR_VOLUME'
    checks['missing_current_rejected']=call(current=None)['replay_daily_status']=='UNKNOWN_MISSING_DAILY'
    checks['preclose_alone_not_adjacent_substitute']=call(previous=None)['replay_daily_status']=='UNKNOWN_ADJACENT_REFERENCE_OR_SHAPE'
    checks['pool_fallback_exact_previous_only']=call(previous=None,pools={('2026-09-04','000001'):{'close':10}})['replay_current_price_shape']==0
    # Check original input/output hashes recorded by the isolated programs.
    verified={}
    for name in ['exact_daily_qualification_summary.json','raw_repair_candidate_summary.json','pipeline_trace_summary.json']:
        meta=b.read(OUT/name)
        for group in ['source_bindings','output_bindings']:
            for path,digest in meta.get(group,{}).items():
                if b.sha(path)!=digest:raise ValueError('bound file changed: '+path)
                verified[path]=digest
    all_rows=pd.read_csv(OUT/'all9315_exact_daily_qualification.csv',dtype={'code':str})
    sep=pd.read_csv(OUT/'september35_raw_repair_candidates_NO_WRITEBACK.csv',dtype={'code':str})
    evidence=b.read(OUT/'raw_same_and_adjacent_evidence.json')['rows']
    patches=b.read(OUT/'missing_field_candidate_patches_NO_WRITEBACK.json')['rows']
    checks['full9315_replay_equals_old_shape']=len(all_rows)==9315 and all_rows.replay_matches_stored_shape.all()
    checks['all35_kept_no_target_selection']=len(sep)==35 and sep.candidate_resolved.all()
    checks['all35_candidate_nonword']=sep.missing_fields_scenario_shape.eq(0).all()
    checks['no_candidate_proposal_fills_existing_nonmissing']=all(p['existing_value'] is None for p in patches)
    checks['all_evidence_dates_at_or_before_signal']=all(p['date']<=p['signal_date'] for p in evidence)
    checks['no_actual_receipt_time_backfill']=all(p['strict_original_signal_time_eligible'] is False for p in evidence)
    checks['6TP_5FP_withheld_preserved']=sep.withheld_raw_TP.sum()==6 and sep.withheld_raw_FP.sum()==5
    checks['no_current_zero_volume_cases']=not all_rows.raw_volume.eq(0).any()
    defects={d:int(all_rows.all_defects.map(lambda v:d in json.loads(v)).sum()) for d in ['MISSING_CURRENT_DAILY','MISSING_OR_NONPOSITIVE_ADJACENT_PREVIOUS_CLOSE','INVALID_DAILY_PRICE','EXPLICIT_PRE_CLOSE_CONFLICT','MISSING_OR_NONPOSITIVE_VOLUME','HIGH_ABOVE_EXACT_10PCT_CAP','NOT_EXACT_10PCT_CLOSING_LIMIT','EXPLICIT_SUSPENSION']}
    if not all(checks.values()):raise AssertionError({k:v for k,v in checks.items() if not v})
    outputs={str(p.resolve()):b.sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='verification_final.json'}
    b.write_new(OUT/'verification_final.json',dict(formed_at=b.now(),checks=checks,checks_passed=len(checks),verified_input_and_prior_output_bindings=len(verified),
        full9315_defect_flags_nonexclusive=defects,all_output_bindings=outputs,
        no_original_writeback=True,no_network=True,no_fits=True))
    print(json.dumps(dict(checks_passed=len(checks),verified_bindings=len(verified),defect_counts=defects),ensure_ascii=True,indent=2))


if __name__=='__main__':main()
