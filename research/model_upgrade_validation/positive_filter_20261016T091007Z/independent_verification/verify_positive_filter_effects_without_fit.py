"""Independent fixed-threshold choices and full-cohort effects, never fit/fetch."""
from pathlib import Path
from fractions import Fraction
from datetime import datetime, timezone
import sys
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
REPO=HERE.parents[3]
sys.path.insert(0,str(REPO))
from relay_local_recovery import verify_sources

EXEC=ROOT/'execution';EVAL=ROOT/'evaluation'
RECOVERY=REPO/'research/local_repair_20261016T051940Z/recovery_manifest.json'
REFERENCE=REPO/'research/model_upgrade_validation/iteration_research_20261016T043944Z/execution/all41547_predictions_BEFORE_EVALUATION.csv'
BASE='SOURCE_PATCHED_FEATURE_CLASSIFIER'
ARMS=['POSITIVE_FILTER_SCORE_ONLY','POSITIVE_FILTER_INDIVIDUAL_CONTEXT']
PERIODS=[('2025H2','2025-07-01','2025-12-31'),('2026JanAug','2026-01-01','2026-08-31'),('2026Sep','2026-09-01','2026-09-30'),('2026Oct','2026-10-01','2026-10-15'),('HISTORICAL_POOLED','2025-07-01','2026-09-30')]
THRESHOLDS=[0.]+[i/10 for i in range(1,10)]
KEYS=['date','code']

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def csv(path):return pd.read_csv(path,dtype={'code':str},float_precision='round_trip',low_memory=False)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def known(frame):return frame.label_valid.astype(str).str.lower().eq('true')&frame.target.isin([0,1])
def eligible(frame):return frame.daily_one_word.eq(0)&frame.verified_current_price_shape.eq(0)&frame.base_call.eq('PROMOTE')

def matrix(frame,column='evaluation_call'):
    k=frame[known(frame)];y=k.target.eq(1);p=k[column].eq('PROMOTE');n=k[column].eq('FAIL')
    tp=int((y&p).sum());fp=int((~y&p).sum());fn=int((y&n).sum());tn=int((~y&n).sum())
    ap=int((y&~p&~n).sum());an=int((~y&~p&~n).sum());pos=int(y.sum());neg=len(k)-pos
    rate=lambda a,z:a/z if z else np.nan
    return dict(candidates=len(frame),known=len(k),unknown_outcomes=len(frame)-len(k),positives=pos,negatives=neg,
        tp=tp,fp=fp,fn_called=fn,tn=tn,abstained_positives=ap,abstained_negatives=an,missed_positives=fn+ap,
        positive_calls=int(frame[column].eq('PROMOTE').sum()),negative_calls=int(frame[column].eq('FAIL').sum()),
        abstentions=int(frame[column].eq('UNKNOWN').sum()),positive_calls_unknown=int(frame.loc[~known(frame),column].eq('PROMOTE').sum()),
        Recall=rate(tp,pos),Precision=rate(tp,tp+fp),F1=rate(2*tp,pos+tp+fp),false_positive_rate=rate(fp,neg),
        decision_coverage=rate(int(p.sum()+n.sum()),len(k)),full_known_accuracy=rate(tp+tn,len(k)),called_accuracy=rate(tp+tn,int(p.sum()+n.sum())),
        utility_TP_minus_1p5FP=tp-1.5*fp)

def quality(stat,kind):
    tp=stat['tp'];fp=stat['fp'];pos=stat['positives']
    if kind=='Precision':return Fraction(tp,tp+fp) if tp+fp else None
    if kind=='Recall':return Fraction(tp,pos) if pos else None
    return Fraction(2*tp,pos+tp+fp) if pos+tp+fp else None

def choose(stats):
    baseline=stats[0.];allowed=[]
    for threshold in THRESHOLDS[1:]:
        stat=stats[threshold];p=quality(stat,'Precision');r=quality(stat,'Recall');f=quality(stat,'F1')
        bp=quality(baseline,'Precision');br=quality(baseline,'Recall');bf=quality(baseline,'F1')
        if None in [p,r,f,bp,br,bf]:continue
        if p>bp and r>=br-Fraction(1,20) and f>=bf and stat['fp']*10<=baseline['fp']*9:allowed.append(threshold)
    return sorted(allowed,key=lambda t:(-quality(stats[t],'Precision'),Fraction(str(t))))[0] if allowed else 0.

def apply(frame,score,threshold):
    view=frame.copy();mask=eligible(view)
    reject=mask&(np.asarray(score)<=threshold) if threshold!=0 else pd.Series(False,index=view.index)
    view['independent_rejected']=reject
    view['evaluation_call']=np.where(reject,'FAIL',view.base_call)
    return view

def pick(frame,k,positive=False):
    pool=frame[frame.daily_one_word.eq(0)].copy()
    if positive:pool=pool[pool.evaluation_call.eq('PROMOTE')]
    chosen=pool.sort_values(['date','ranking_score','code'],ascending=[True,False,True],na_position='last').groupby('date',sort=True).head(k)
    valid=known(chosen);slots=k*frame.date.nunique();hits=int((valid&chosen.target.eq(1)).sum())
    return dict(k=k,signal_days=frame.date.nunique(),full_slots=slots,selected=len(chosen),empty_slots=slots-len(chosen),
        known_selected=int(valid.sum()),unknown_selected=int((~valid).sum()),hits=hits,full_slot_hit_rate=hits/slots if slots else np.nan),chosen

def compare_stats(actual,expected):
    for key in ['candidates','known','unknown_outcomes','positives','positive_calls','negative_calls','abstentions','tp','fp','fn_called','tn','abstained_positives','abstained_negatives','missed_positives','Recall','Precision','F1','false_positive_rate','decision_coverage','full_known_accuracy','called_accuracy','utility_TP_minus_1p5FP']:
        z=expected[key]
        if z is None:z=np.nan
        assert np.isclose(actual[key],z,rtol=0,atol=1e-14,equal_nan=True),(key,actual[key],z)

def main():
    result_path=EVAL/'fixed_study_result.json';table_path=EXEC/'all49101_predictions_BEFORE_EVALUATION.csv'
    final_receipt=HERE/'positive_filter_effects_independent_receipt.json'
    if not result_path.exists():raise FileNotFoundError('wait for completed study evaluation; never fit or poll')
    if final_receipt.exists():raise FileExistsError('immutable independent receipt already exists')
    protocol=read(ROOT/'protocol_BEFORE_ANY_FIT.json');result=read(result_path)
    source_check=verify_sources(result['source_bindings'],RECOVERY)
    assert source_check['model_data_code_hash_exceptions']==0
    old=csv(REFERENCE);complete=csv(table_path)
    assert len(old)==41547 and len(complete)==49101 and complete.arm.nunique()==13
    assert complete.groupby('arm').size().eq(3777).all() and not complete.duplicated(KEYS+['arm']).any()
    pd.testing.assert_frame_equal(old,complete.iloc[:41547][old.columns],check_exact=True,check_dtype=False)
    assert complete.date.max()=='2026-10-15'
    base=complete[complete.arm.eq(BASE)].copy();base_index=base.set_index(KEYS)
    new={arm:complete[complete.arm.eq(arm)].copy() for arm in ARMS}
    route_checks=[];filtered=[]
    protected=['score','b_score','ranking_score','date','code','name','boards','target_date','target','label_valid','daily_one_word','frozen_original_nonword_identity','verified_current_price_shape']
    for arm,frame in new.items():
        indexed=frame.set_index(KEYS).loc[base_index.index].reset_index();reference=base_index.reset_index()
        pd.testing.assert_frame_equal(indexed[protected],reference[protected],check_exact=True,check_dtype=False)
        pd.testing.assert_series_equal(indexed.base_call,reference.evaluation_call,check_names=False,check_exact=True)
        mask=eligible(indexed)
        assert indexed.filter_threshold.isin(THRESHOLDS).all()
        reject=mask&indexed.filter_threshold.ne(0)&indexed.filter_score.le(indexed.filter_threshold)
        assert np.array_equal(indexed.filter_applied.astype(bool),reject)
        expected=np.where(reject,'FAIL',indexed.base_call)
        assert np.array_equal(indexed.call,expected)
        for column in ['evaluation_call','research_call','aligned_evaluation_call','recovered_B_call']:assert np.array_equal(indexed[column],expected)
        assert indexed.loc[~mask,'call'].equals(indexed.loc[~mask,'base_call'])
        assert indexed.loc[mask,'filter_score'].between(0,1).all()
        assert indexed.loc[~mask,'filter_score'].isna().all()
        assert indexed.event_probability_validated.astype(str).str.lower().eq('false').all()
        hist=indexed[indexed.date.le('2026-09-30')&indexed.daily_one_word.eq(0)]
        assert (len(hist),int(known(hist).sum()),int((~known(hist)).sum()),int(hist.target.eq(1).sum()))==(2980,2932,48,871)
        sep=indexed[indexed.date.between('2026-09-01','2026-09-30')&indexed.daily_one_word.eq(0)]
        assert (len(sep),int(known(sep).sum()),int(sep.target.eq(1).sum()),int((~known(sep)).sum()))==(143,107,40,36)
        current=indexed[indexed.date.eq('2026-10-15')]
        assert len(current)==6 and current.target.isna().all() and not known(current).any()
        route_checks.append(dict(arm=arm,rows=len(indexed),eligible_original_positives=int(mask.sum()),filtered_rows=int(reject.sum()),changed_noneligible=0,protected_scores_and_labels_exact=True,current6_unknown=True))
        one=indexed[reject].copy();one['filter_effect']=np.where(~known(one),'UNKNOWN_OUTCOME',np.where(one.target.eq(1),'NEW_MISSED_PROMOTION','REMOVED_FALSE_POSITIVE'))
        filtered.append(one)
    data=joblib.load(ROOT/'fixed_input_BEFORE_ANY_FIT.joblib');full=data['full']
    assert len(full)==9315 and len(data['plans'])==16
    threshold_rows=[];choice_rows=[];validation_scores=[]
    for plan in data['plans']:
        month=plan['month'];oof=csv(EXEC/f'{month}_complete_B_OOF_BEFORE_SELECTION.csv')
        assert not oof.duplicated(KEYS).any() and oof.daily_one_word.eq(0).all()
        assert sorted(oof.inner_block.unique())==[1,2,3]
        for block in [1,2,3]:assert oof[oof.inner_block.eq(block)].date.nunique()==63
        third=oof[oof.inner_block.eq(3)].copy();first=third.date.min()
        # Independently reconstruct the exact prefix/final membership, including
        # target maturity at the third-block boundary rather than outer cutoff.
        expected_prefix=oof[(oof.inner_block<3)&eligible(oof)&known(oof)&oof.date.lt(first)&oof.target_date.lt(first)]
        expected_final=oof[eligible(oof)&known(oof)&oof.date.lt(plan['cutoff'])&oof.target_date.lt(plan['cutoff'])]
        prefix=csv(EXEC/f'{month}_meta_prefix_training.csv');final=csv(EXEC/f'{month}_meta_final_training.csv')
        pd.testing.assert_frame_equal(prefix,expected_prefix.reset_index(drop=True),check_exact=True,check_dtype=False)
        pd.testing.assert_frame_equal(final,expected_final.reset_index(drop=True),check_exact=True,check_dtype=False)
        assert prefix.target_date.max()<first and final.target_date.max()<plan['cutoff']
        selection=read(EXEC/f'{month}_SELECTION_BEFORE_FINAL_FIT_AND_OUTER.json')
        assert selection['first_third_signal']==first and selection['outer_cutoff']==plan['cutoff']
        assert selection['outer_results_used_for_selection'] is False
        for arm in ARMS:
            selected_receipt=selection['choices'][arm]
            fit=selected_receipt['prefix_fit'];qualified=(int(prefix.target.eq(1).sum())>=20 and int(prefix.target.eq(0).sum())>=20 and prefix.date.nunique()>=20)
            assert fit['qualified']==qualified
            mask=eligible(third);scores=np.full(len(third),np.nan)
            if qualified:
                saved=joblib.load(fit['model_path'])
                assert saved['features']==protocol['features'][arm] and sha(fit['model_path'])==fit['model_sha256']
                if mask.any():
                    with threadpool_limits(limits=2):scores[mask]=saved['model'].predict_proba(third.loc[mask,saved['features']])[:,list(saved['model'].named_steps['logistic'].classes_).index(1)]
            else:scores[mask]=0.
            stats={threshold:matrix(apply(third,scores,threshold)) for threshold in THRESHOLDS}
            for threshold,stat in stats.items():
                compare_stats(stat,selected_receipt['statistics'][str(threshold)])
                threshold_rows.append(dict(month=month,arm=arm,threshold=threshold,**stat))
            selected=choose(stats) if qualified else 0.
            assert selected==selected_receipt['selected_threshold']
            completed=read(EXEC/f'{month}_{arm}_COMPLETED.json')
            expected_threshold=selected if completed['final_fit']['qualified'] else 0.
            assert completed['selected_threshold']==expected_threshold
            actual=new[arm][new[arm].date.str[:7].eq(month)]
            assert actual.filter_threshold.eq(expected_threshold).all()
            choice_rows.append(dict(month=month,arm=arm,prefix_qualified=qualified,prefix_rows=len(prefix),prefix_positive=int(prefix.target.eq(1).sum()),prefix_negative=int(prefix.target.eq(0).sum()),
                max_prefix_target=prefix.target_date.max(),third_first_signal=first,third_rows=len(third),third_known=int(known(third).sum()),third_unknown=int((~known(third)).sum()),
                selected_threshold=selected,final_effective_threshold=expected_threshold,max_final_target=final.target_date.max(),outer_cutoff=plan['cutoff'],
                third_block_is_threshold_selection_only=True,outer_results_not_used=True))
            small=third[KEYS+['target','label_valid','base_call','raw_score']].copy();small['arm']=arm;small['secondary_score_replayed']=scores
            validation_scores.append(small)
    records=[];raw_ranks=[];raw_picks=[];recommendations=[];recommended_rows=[]
    frames={arm:complete[complete.arm.eq(arm)].copy() for arm in ['BASE','ST_DUAL',BASE]+ARMS}
    for period,first,last in PERIODS:
        for arm,frame in frames.items():
            sample=frame[frame.date.between(first,last)]
            for scope,group in [('FULL_POOL',sample),('ORIGINAL_NONWORD',sample[sample.daily_one_word.eq(0)])]:records.append(dict(period=period,arm=arm,scope=scope,**matrix(group)))
            for k in [1,2,3]:
                stat,picks=pick(sample,k)
                raw_ranks.append(dict(period=period,arm=arm,**stat))
                for row in picks.itertuples(index=False):raw_picks.append(dict(period=period,arm=arm,k=k,date=row.date,code=row.code,ranking_score=row.ranking_score,target=row.target,label_valid=row.label_valid))
            stat,picks=pick(sample,3,positive=True)
            recommendations.append(dict(period=period,arm=arm,**stat))
            for row in picks.itertuples(index=False):recommended_rows.append(dict(period=period,arm=arm,date=row.date,code=row.code,ranking_score=row.ranking_score,target=row.target,label_valid=row.label_valid))
    matrices=pd.DataFrame(records);ranks=pd.DataFrame(raw_ranks);raw_picks=pd.DataFrame(raw_picks);recs=pd.DataFrame(recommendations);rec_picks=pd.DataFrame(recommended_rows)
    for arm in ARMS:
        original=raw_picks[raw_picks.arm.eq(BASE)].drop(columns='arm').reset_index(drop=True)
        filtered_raw=raw_picks[raw_picks.arm.eq(arm)].drop(columns='arm').reset_index(drop=True)
        pd.testing.assert_frame_equal(original,filtered_raw,check_exact=True)
    official=csv(EVAL/'complete_same_pool_classification.csv')
    for row in official.to_dict('records'):
        own=matrices[matrices.period.eq(row['period'])&matrices.arm.eq(row['arm'])&matrices.scope.eq(row['scope'])].iloc[0].to_dict()
        compare_stats(own,row)
    published_ranks=csv(EVAL/'raw_observation_Top1Top2Top3_INVARIANT.csv')
    for row in published_ranks.to_dict('records'):
        own=ranks[ranks.period.eq(row['period'])&ranks.arm.eq(row['arm'])&ranks.k.eq(row['k'])].iloc[0]
        for key in ['signal_days','full_slots','selected','known_selected','unknown_selected','empty_slots','hits','full_slot_hit_rate']:assert np.isclose(own[key],row[key],rtol=0,atol=1e-14,equal_nan=True)
    published_recs=csv(EVAL/'positive_recommendation_Top3_FULL_SLOTS.csv')
    for row in published_recs.to_dict('records'):
        own=recs[recs.period.eq(row['period'])&recs.arm.eq(row['arm'])].iloc[0]
        for key in ['signal_days','full_slots','selected','empty_slots','unknown_selected','hits']:assert own[key]==row[key],(row['period'],row['arm'],key)
    filtered=pd.concat(filtered,ignore_index=True);gains=[]
    for period,first,last in PERIODS:
        for arm in ARMS:
            group=filtered[filtered.arm.eq(arm)&filtered.date.between(first,last)&filtered.daily_one_word.eq(0)]
            gains.append(dict(period=period,arm=arm,filtered_total=len(group),removed_FP=int(group.filter_effect.eq('REMOVED_FALSE_POSITIVE').sum()),
                new_missed_promotions=int(group.filter_effect.eq('NEW_MISSED_PROMOTION').sum()),unknown_filtered=int(group.filter_effect.eq('UNKNOWN_OUTCOME').sum())))
    gains=pd.DataFrame(gains);index=matrices[matrices.scope.eq('ORIGINAL_NONWORD')].set_index(['period','arm']);guards={}
    for arm in ARMS:
        z=index.loc[('HISTORICAL_POOLED',arm)];old_base=index.loc[('HISTORICAL_POOLED',BASE)]
        guard=dict(pooled_Precision_gain3pp=bool(z.Precision>=old_base.Precision+.03),pooled_FP_drop10percent=bool(z.fp*10<=old_base.fp*9),pooled_F1_not_lower=bool(z.F1>=old_base.F1))
        for period,first,last in PERIODS[:3]:
            v=index.loc[(period,arm)];baseline=index.loc[(period,BASE)]
            guard[period+'_Recall_loss_within5pp']=bool(v.Recall>=baseline.Recall-.05)
            guard[period+'_FP_not_higher']=bool(v.fp<=baseline.fp)
            guard[period+'_F1_not_lower']=bool(v.F1>=baseline.F1)
        r=recs[recs.period.eq('HISTORICAL_POOLED')].set_index('arm')
        guard['pooled_positive_Top3_hits_not_lower']=bool(r.loc[arm,'hits']>=r.loc[BASE,'hits'])
        assert guard==result['arms'][arm]['guards'] and all(guard.values())==result['arms'][arm]['fixed_screen_pass']
        guards[arm]=guard
    outputs={'独立复算_全池与原B完整指标.csv':matrices,'独立复算_原raw观察Top1Top2Top3.csv':ranks,'独立复算_全部raw预选含未知.csv':raw_picks,
        '独立复算_正向推荐Top3完整槽位.csv':recs,'独立复算_正向推荐逐股含未知.csv':rec_picks,
        '独立复算_所有第三块十阈值完整B指标.csv':pd.DataFrame(threshold_rows),'独立审查_32月臂精确阈值选择.csv':pd.DataFrame(choice_rows),
        '独立重放_全部第三块前缀二级分数.csv':pd.concat(validation_scores,ignore_index=True),
        '独立复算_新增漏判移除误报与未知.csv':gains,'独立复算_全部过滤逐股得失.csv':filtered,'独立核验_路由及原分数守恒.csv':pd.DataFrame(route_checks)}
    HERE.mkdir(parents=True,exist_ok=True)
    for name,frame in outputs.items():frame.to_csv(HERE/name,index=False,encoding='utf-8-sig')
    receipt=dict(formed_at_UTC=datetime.now(timezone.utc).isoformat(),verification_fits=0,network_calls=0,no_oct16_results_read=True,
        original41547_all_columns_exact=True,all49101_complete13heads=True,new_heads_each3777=True,BASE_original_preserved=True,
        original_B_historical=[2980,2932,48,871],september_original_B=[143,107,40,36],current6_per_new_arm_unknown=True,
        only_eligible_original_positive_changed_to_FAIL=True,raw_scores_bscores_and_rankings_exact=True,
        threshold_choices32_independent_prefix_replay_and_Fraction=True,third_block_is_selection_not_unbiased_effect_validation=True,
        raw_observation_Top1Top2Top3_identical=True,positive_recommendation_Top3_complete_fixed_slots=True,
        selected_threshold_migration_after_final_refit_risk_remaining=True,old_failures_not_requalified=True,event_probability_validated=False,
        protected_source_verification=source_check,guards=guards,
        source_bindings={str(p.resolve()):sha(p) for p in [Path(__file__),ROOT/'protocol_BEFORE_ANY_FIT.json',ROOT/'fixed_input_BEFORE_ANY_FIT.joblib',REFERENCE,table_path,EXEC/'run_COMPLETED_BEFORE_EVALUATION.json',result_path,RECOVERY]},
        output_bindings={str((HERE/name).resolve()):sha(HERE/name) for name in outputs})
    final_receipt.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(verified=True,fit=0,source_verification=source_check,arms={arm:result['arms'][arm]['fixed_screen_pass'] for arm in ARMS}),ensure_ascii=False))
    print(matrices[matrices.arm.isin([BASE]+ARMS)&matrices.scope.eq('ORIGINAL_NONWORD')][['period','arm','tp','fp','Recall','Precision','F1']].to_string(index=False))

if __name__=='__main__':main()
