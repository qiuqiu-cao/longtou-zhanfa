"""Single fixed zero-fit PREFIX-for-FINAL counterfactual, no threshold search."""
from pathlib import Path
from datetime import datetime,timezone
import sys
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
sys.path.insert(0,str(REPO))
OLD=REPO/'research/model_upgrade_validation/positive_filter_20261016T091007Z'
sys.path.insert(0,str(OLD/'independent_verification'))
from verify_positive_filter_effects_without_fit import matrix,pick
from relay_local_recovery import verify_sources

INPUT=OLD/'execution/all49101_predictions_BEFORE_EVALUATION.csv'
BASE='SOURCE_PATCHED_FEATURE_CLASSIFIER'
ARMS={'POSITIVE_FILTER_SCORE_ONLY':'NO_META_REFIT_SCORE_ONLY','POSITIVE_FILTER_INDIVIDUAL_CONTEXT':'NO_META_REFIT_INDIVIDUAL_CONTEXT'}
PERIODS=[('2025H2','2025-07-01','2025-12-31'),('2026JanAug','2026-01-01','2026-08-31'),('2026Sep','2026-09-01','2026-09-30'),('2026Oct','2026-10-01','2026-10-15'),('HISTORICAL_POOLED','2025-07-01','2026-09-30')]
KEYS=['date','code']

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write_new(p,data):
    with Path(p).open('x',encoding='utf-8') as h:json.dump(data,h,ensure_ascii=False,indent=2)
def csv(p):return pd.read_csv(p,dtype={'code':str},float_precision='round_trip',low_memory=False)

def freeze():
    HERE.mkdir(parents=True,exist_ok=True)
    protocol_path=HERE/'protocol_BEFORE_ANY_INFERENCE.json'
    if protocol_path.exists():raise FileExistsError('one fixed zero-fit probe only')
    prior_path=OLD/'evaluation/fixed_study_result.json';prior=read(prior_path)
    recovered=verify_sources(prior['source_bindings'],REPO/'research/local_repair_20261016T051940Z/recovery_manifest.json')
    assert recovered['model_data_code_hash_exceptions']==0
    plans=[];sources={str(p.resolve()):sha(p) for p in [Path(__file__),INPUT,prior_path,
        OLD/'protocol_BEFORE_ANY_FIT.json',OLD/'independent_verification/verify_positive_filter_effects_without_fit.py',
        REPO/'relay_local_recovery.py',REPO/'research/local_repair_20261016T051940Z/recovery_manifest.json',
        HERE.parent/'meta_mechanism_audit/meta_mechanism_delivery_verified.json']}
    selections=csv(OLD/'execution/all32_training_only_selections.csv')
    assert len(selections)==32
    for record in selections.itertuples(index=False):
        completed_path=OLD/f'execution/{record.month}_{record.arm}_COMPLETED.json'
        completed=read(completed_path);prefix=completed['prefix_fit'];final=completed['final_fit']
        assert completed['selected_threshold']==record.selected_threshold
        original_selection=OLD/f'execution/{record.month}_SELECTION_BEFORE_FINAL_FIT_AND_OUTER.json'
        sources[str(completed_path.resolve())]=sha(completed_path);sources[str(original_selection.resolve())]=sha(original_selection)
        for saved in [prefix,final]:
            if saved['qualified']:
                assert sha(saved['model_path'])==saved['model_sha256']
                sources[str(Path(saved['model_path']).resolve())]=saved['model_sha256']
        plans.append(dict(month=record.month,original_arm=record.arm,new_arm=ARMS[record.arm],
            original_frozen_threshold=record.selected_threshold,prefix_qualified=prefix['qualified'],
            prefix_path=prefix.get('model_path'),prefix_sha256=prefix.get('model_sha256'),
            final_path=final.get('model_path'),final_sha256=final.get('model_sha256'),
            feature_schema=read(OLD/'protocol_BEFORE_ANY_FIT.json')['features'][record.arm]))
    assert len({p['month'] for p in plans})==16
    write_new(protocol_path,dict(formed_at_UTC=now(),user_authorization='Explicit parent authorization: fixed maxfit0 NO_META_REFIT probe before complete outer counterfactual effects',
        hypothesis='Replacing FINAL meta model with selection-time PREFIX at identical input and original frozen threshold may isolate refit/threshold transfer contribution',
        maximum_fits=0,network_calls_allowed=0,arms=ARMS,plans=plans,
        immutable_scope='All16 original outer months; old49101 original columns remain exact; new2x3777; original SOURCE raw scores/ranking/features/labels/qualification unchanged',
        sole_change='Use saved PREFIX preprocessor and Logistic in place of saved FINAL for the secondary score; identical original final effective threshold including explicit0 bypass',
        no_reselection=True,no_threshold_search=True,no_arm_selected_by_outer_effect=True,
        prediction_write_order='Save complete56655 predictions BEFORE evaluating effects',
        acceptance=dict(pooled_Precision_gain=.03,pooled_FP_fraction_max=.9,pooled_F1_not_lower=True,
            each_historical_fold_Recall_loss_max=.05,each_historical_fold_FP_not_higher=True,
            each_historical_fold_F1_not_lower=True,pooled_positive_Top3_full_slots_hits_not_lower=True),
        third_block_is_original_threshold_selection=True,previously_seen_history_not_blind=True,
        counterfactual_mechanism_probe_not_forward_forecast=True,event_probability_validated=False,auto_deployment=False,
        source_recovery=recovered,source_bindings=sources))

def run():
    protocol_path=HERE/'protocol_BEFORE_ANY_INFERENCE.json';protocol=read(protocol_path)
    for name,value in protocol['source_bindings'].items():assert sha(name)==value,('fixed source drift',name)
    write_new(HERE/'probe_STARTED.json',dict(formed_at_UTC=now(),protocol_sha256=sha(protocol_path),new_fits=0))
    old=csv(INPUT);assert len(old)==49101 and old.arm.nunique()==13
    parts=[];all_comparisons=[];five_rows=[]
    original_five=csv(OLD/'evaluation/all_filtered_candidates_GAINS_LOSSES_UNKNOWN.csv')
    for plan in protocol['plans']:
        source=old[old.arm.eq(plan['original_arm'])&old.date.str[:7].eq(plan['month'])].copy()
        mask=source.daily_one_word.eq(0)&source.verified_current_price_shape.eq(0)&source.base_call.eq('PROMOTE')
        score=np.full(len(source),np.nan)
        if plan['prefix_qualified']:
            assert sha(plan['prefix_path'])==plan['prefix_sha256']
            saved=joblib.load(plan['prefix_path']);assert saved['features']==plan['feature_schema']
            if mask.any():
                with threadpool_limits(limits=2):score[mask]=saved['model'].predict_proba(source.loc[mask,saved['features']])[:,list(saved['model'].named_steps['logistic'].classes_).index(1)]
        else:score[mask]=0.
        threshold=plan['original_frozen_threshold']
        assert source.filter_threshold.eq(threshold).all()
        reject=mask&(score<=threshold) if threshold!=0 else pd.Series(False,index=source.index)
        new=source.copy();new['arm']=plan['new_arm'];new['filter_score']=score;new['filter_applied']=reject
        new['filter_threshold']=threshold
        new['filter_reason']=np.where(reject,'SAVED_PREFIX_SCORE_NOT_ABOVE_ORIGINAL_THRESHOLD',np.where(mask,'EXPLICIT_ORIGINAL_NO_FILTER' if threshold==0 else 'RETAIN_ORIGINAL_POSITIVE','INHERIT_ORIGINAL_NONPOSITIVE_OR_INELIGIBLE'))
        new['call']=np.where(reject,'FAIL',new.base_call)
        for name in ['evaluation_call','research_call','aligned_evaluation_call','recovered_B_call']:new[name]=new.call
        new['second_stage_model_phase']='PREFIX_NO_META_REFIT_FIXED_PROBE'
        new['original_final_secondary_score']=source.filter_score.to_numpy()
        new['secondary_score_prefix_minus_final']=score-source.filter_score.to_numpy()
        new['research_validation_status']='FIXED_ZERO_FIT_COUNTERFACTUAL_NOT_ADOPTED'
        new['event_probability_validated']=False
        for field in ['score','b_score','ranking_score','target','label_valid','daily_one_word','verified_current_price_shape']:
            pd.testing.assert_series_equal(new[field],source[field],check_exact=True)
        new.to_csv(HERE/f"{plan['month']}_{plan['new_arm']}_predictions_BEFORE_EVALUATION.csv",index=False,encoding='utf-8-sig')
        parts.append(new)
        comparison=source[KEYS+['name','boards','target','label_valid','daily_one_word','verified_current_price_shape','base_call','ranking_score']].copy()
        comparison['original_arm']=plan['original_arm'];comparison['new_arm']=plan['new_arm']
        comparison['FINAL_secondary_score']=source.filter_score.to_numpy();comparison['PREFIX_secondary_score']=score
        comparison['original_fixed_threshold']=threshold
        comparison['FINAL_rejected']=source.filter_applied.astype(bool).to_numpy();comparison['PREFIX_rejected']=reject.to_numpy()
        comparison['FINAL_call']=source.evaluation_call.to_numpy();comparison['PREFIX_call']=new.evaluation_call.to_numpy()
        comparison['rejection_comparison']=np.select([comparison.FINAL_rejected&comparison.PREFIX_rejected,
            ~comparison.FINAL_rejected&comparison.PREFIX_rejected,comparison.FINAL_rejected&~comparison.PREFIX_rejected],
            ['BOTH_REJECTED','PREFIX_ONLY_REJECTED','FINAL_ONLY_REJECTED'],default='NEITHER_REJECTED')
        all_comparisons.append(comparison)
        if plan['original_arm']=='POSITIVE_FILTER_INDIVIDUAL_CONTEXT':
            subset=comparison.merge(original_five[KEYS],on=KEYS,how='inner',validate='one_to_one')
            if len(subset):five_rows.append(subset)
    added=pd.concat(parts,ignore_index=True);assert len(added)==7554 and added.groupby('arm').size().eq(3777).all()
    complete=pd.concat([old,added],ignore_index=True)
    assert len(complete)==56655 and complete.arm.nunique()==15
    pd.testing.assert_frame_equal(complete.iloc[:49101][old.columns],old,check_exact=True,check_dtype=False)
    assert not complete.duplicated(KEYS+['arm']).any()
    complete_path=HERE/'all56655_predictions_BEFORE_EVALUATION.csv'
    complete.to_csv(complete_path,index=False,encoding='utf-8-sig')
    comparison=pd.concat(all_comparisons,ignore_index=True)
    comparison.to_csv(HERE/'all7554_PREFIX_FINAL_full_cohort_comparisons.csv',index=False,encoding='utf-8-sig')
    five=pd.concat(five_rows,ignore_index=True);assert len(five)==5
    five.to_csv(HERE/'original_five_all_actual_PREFIX_FINAL_score_changes.csv',index=False,encoding='utf-8-sig')
    write_new(HERE/'all_predictions_COMPLETED_BEFORE_EFFECTS.json',dict(formed_at_UTC=now(),new_fits=0,old49101_rows_preserved=True,
        complete_predictions=56655,new_heads_each3777=True,predictions_sha256=sha(complete_path),protocol_sha256=sha(protocol_path)))
    # Effect computation begins only after complete predictions and comparisons
    # have been persisted. No labels influence a model/threshold decision.
    records=[];ranks=[];raw_picks=[];recs=[];rec_picks=[];differences=[]
    for period,first,last in PERIODS:
        for arm in [BASE]+list(ARMS)+list(ARMS.values()):
            sample=complete[complete.arm.eq(arm)&complete.date.between(first,last)]
            for scope,group in [('FULL_POOL',sample),('ORIGINAL_NONWORD',sample[sample.daily_one_word.eq(0)])]:records.append(dict(period=period,arm=arm,scope=scope,**matrix(group)))
            for k in [1,2,3]:
                stat,selected=pick(sample,k);ranks.append(dict(period=period,arm=arm,**stat))
                for row in selected.itertuples(index=False):raw_picks.append(dict(period=period,arm=arm,k=k,date=row.date,code=row.code,ranking_score=row.ranking_score,target=row.target,label_valid=row.label_valid))
            stat,selected=pick(sample,3,positive=True);recs.append(dict(period=period,arm=arm,**stat))
            for row in selected.itertuples(index=False):rec_picks.append(dict(period=period,arm=arm,date=row.date,code=row.code,ranking_score=row.ranking_score,target=row.target,label_valid=row.label_valid))
        part=comparison[comparison.date.between(first,last)]
        for (old_arm,new_arm,status),group in part.groupby(['original_arm','new_arm','rejection_comparison']):
            known=group.label_valid.astype(str).str.lower().eq('true')&group.target.isin([0,1])
            differences.append(dict(period=period,original_arm=old_arm,new_arm=new_arm,status=status,rows=len(group),
                known_positive=int((known&group.target.eq(1)).sum()),known_negative=int((known&group.target.eq(0)).sum()),unknown=int((~known).sum())))
    stats=pd.DataFrame(records);rank=pd.DataFrame(ranks);recs=pd.DataFrame(recs);picks=pd.DataFrame(raw_picks)
    for arm in ARMS.values():
        a=picks[picks.arm.eq(BASE)].drop(columns='arm').reset_index(drop=True)
        z=picks[picks.arm.eq(arm)].drop(columns='arm').reset_index(drop=True)
        pd.testing.assert_frame_equal(a,z,check_exact=True)
        current=added[added.arm.eq(arm)&added.date.eq('2026-10-15')]
        assert len(current)==6 and current.target.isna().all() and not current.label_valid.astype(str).str.lower().eq('true').any()
        historical=added[added.arm.eq(arm)&added.date.le('2026-09-30')&added.daily_one_word.eq(0)]
        assert len(historical)==2980
    table=stats[stats.scope.eq('ORIGINAL_NONWORD')].set_index(['period','arm']);gates={}
    for arm in ARMS.values():
        z=table.loc[('HISTORICAL_POOLED',arm)];a=table.loc[('HISTORICAL_POOLED',BASE)]
        guard=dict(pooled_Precision_gain3pp=bool(z.Precision>=a.Precision+.03),pooled_FP_drop10percent=bool(z.fp*10<=a.fp*9),pooled_F1_not_lower=bool(z.F1>=a.F1))
        for period,first,last in PERIODS[:3]:
            zz=table.loc[(period,arm)];aa=table.loc[(period,BASE)]
            guard[period+'_Recall_loss_within5pp']=bool(zz.Recall>=aa.Recall-.05)
            guard[period+'_FP_not_higher']=bool(zz.fp<=aa.fp)
            guard[period+'_F1_not_lower']=bool(zz.F1>=aa.F1)
        r=recs[recs.period.eq('HISTORICAL_POOLED')].set_index('arm')
        guard['pooled_positive_Top3_hits_not_lower']=bool(r.loc[arm,'hits']>=r.loc[BASE,'hits'])
        gates[arm]=dict(guards=guard,fixed_screen_pass=all(guard.values()))
    outputs={'all_same_pool_classification_metrics.csv':stats,'raw_observation_Top1Top2Top3_INVARIANT.csv':rank,
        'all_raw_Topk_rows_WITH_UNKNOWN.csv':picks,'positive_recommendation_Top3_FULL_SLOTS.csv':recs,
        'all_positive_Top3_rows_WITH_UNKNOWN.csv':pd.DataFrame(rec_picks),'all_rejection_comparison_counts_and_known_unknown.csv':pd.DataFrame(differences)}
    for name,frame in outputs.items():frame.to_csv(HERE/name,index=False,encoding='utf-8-sig')
    for name,value in protocol['source_bindings'].items():assert sha(name)==value
    files=list(HERE.glob('*.csv'))+[HERE/'all_predictions_COMPLETED_BEFORE_EFFECTS.json',HERE/'probe_STARTED.json']
    result=dict(formed_at_UTC=now(),new_fits=0,network_calls=0,old49101_all_original_columns_exact=True,complete_predictions=56655,
        arms=gates,raw_observation_rankings_identical=True,no_threshold_or_arm_reselection=True,current6_each_arm_unknown=True,
        historical_not_blind_not_forward=True,probability_validated=False,production_deployed=False,
        source_bindings=protocol['source_bindings']|{str(protocol_path.resolve()):sha(protocol_path)},
        output_bindings={str(p.resolve()):sha(p) for p in files})
    write_new(HERE/'fixed_probe_result.json',result)
    print(json.dumps({arm:z['fixed_screen_pass'] for arm,z in gates.items()},ensure_ascii=False))
    print(stats[stats.scope.eq('ORIGINAL_NONWORD')&stats.arm.isin([BASE]+list(ARMS.values()))][['period','arm','tp','fp','Recall','Precision','F1']].to_string(index=False))
    print(pd.DataFrame(differences)[lambda f:f.period.eq('HISTORICAL_POOLED')].to_string(index=False))

if __name__=='__main__':
    if len(sys.argv)!=2 or sys.argv[1] not in ['freeze','run']:raise SystemExit('choose freeze or run; no implicit retry')
    freeze() if sys.argv[1]=='freeze' else run()
