"""Independent warmup/selection/effects audit of saved monthly252 study."""
from pathlib import Path
from datetime import datetime,timezone
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
OLD=REPO/'research/model_upgrade_validation/positive_filter_20261016T091007Z'
sys.path.insert(0,str(OLD/'independent_verification'))
from verify_positive_filter_effects_without_fit import matrix,pick,choose,apply,compare_stats,eligible
from relay_local_recovery import verify_sources

ARMS=['MONTHLY252_OOF_FROZEN_SCORE_FILTER','MONTHLY252_OOF_FROZEN_INDIVIDUAL_FILTER']
KINDS=dict(zip(ARMS,['POSITIVE_FILTER_SCORE_ONLY','POSITIVE_FILTER_INDIVIDUAL_CONTEXT']))
BASE='SOURCE_PATCHED_FEATURE_CLASSIFIER'
PERIODS=[('2025H2','2025-07-01','2025-12-31'),('2026JanAug','2026-01-01','2026-08-31'),('2026Sep','2026-09-01','2026-09-30'),('2026Oct','2026-10-01','2026-10-15'),('HISTORICAL_POOLED','2025-07-01','2026-09-30')]
KEYS=['date','code'];THRESHOLDS=[0.]+[i/10 for i in range(1,10)]

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def csv(p):return pd.read_csv(p,dtype={'code':str},float_precision='round_trip',low_memory=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def known(frame):return frame.label_valid.astype(str).str.lower().eq('true')&frame.target.isin([0,1])

def main():
    EXEC=ROOT/'execution';EVAL=ROOT/'evaluation'
    receipt_path=HERE/'monthly252_independent_effect_receipt.json'
    if receipt_path.exists():raise FileExistsError('immutable audit already exists')
    result_path=EVAL/'fixed_study_result.json'
    if not result_path.exists():raise FileNotFoundError('wait for actual completed evaluation; never fit/poll')
    protocol=read(ROOT/'protocol_BEFORE_ANY_FIT.json');result=read(result_path)
    recovery=REPO/'research/local_repair_20261016T051940Z/recovery_manifest.json'
    protected=verify_sources(result['source_bindings'],recovery);assert protected['model_data_code_hash_exceptions']==0
    reference=OLD/'execution/all49101_predictions_BEFORE_EVALUATION.csv'
    old=csv(reference);all_rows=csv(EXEC/'all56655_predictions_BEFORE_EVALUATION.csv')
    assert len(old)==49101 and len(all_rows)==56655 and all_rows.arm.nunique()==15
    assert all_rows.groupby('arm').size().eq(3777).all() and not all_rows.duplicated(KEYS+['arm']).any()
    pd.testing.assert_frame_equal(old,all_rows.iloc[:49101][old.columns],check_exact=True,check_dtype=False)
    base=all_rows[all_rows.arm.eq(BASE)].copy();base_index=base.set_index(KEYS)
    heads={arm:all_rows[all_rows.arm.eq(arm)].copy() for arm in ARMS}
    complete_filtered=[];route_checks=[]
    for arm,frame in heads.items():
        aligned=frame.set_index(KEYS).loc[base_index.index].reset_index();original=base_index.reset_index()
        protected_columns=['date','code','name','boards','target_date','target','label_valid','daily_one_word','frozen_original_nonword_identity','verified_current_price_shape','score','b_score','ranking_score']
        pd.testing.assert_frame_equal(aligned[protected_columns],original[protected_columns],check_exact=True,check_dtype=False)
        pd.testing.assert_series_equal(aligned.base_call,original.evaluation_call,check_names=False,check_exact=True)
        mask=eligible(aligned);reject=mask&aligned.filter_threshold.ne(0)&aligned.filter_score.le(aligned.filter_threshold)
        assert aligned.filter_threshold.isin(THRESHOLDS).all() and np.array_equal(aligned.filter_applied.astype(bool),reject)
        expected=np.where(reject,'FAIL',aligned.base_call)
        for column in ['call','evaluation_call','research_call','aligned_evaluation_call','recovered_B_call']:assert np.array_equal(aligned[column],expected)
        assert aligned.loc[mask,'filter_score'].between(0,1).all() and aligned.loc[~mask,'filter_score'].isna().all()
        hist=aligned[aligned.daily_one_word.eq(0)&aligned.date.le('2026-09-30')]
        assert (len(hist),int(known(hist).sum()),int((~known(hist)).sum()),int(hist.target.eq(1).sum()))==(2980,2932,48,871)
        sep=aligned[aligned.daily_one_word.eq(0)&aligned.date.str.startswith('2026-09')]
        assert (len(sep),int(known(sep).sum()),int((~known(sep)).sum()),int(sep.target.eq(1).sum()))==(143,107,36,40)
        current=aligned[aligned.date.eq('2026-10-15')];assert len(current)==6 and current.target.isna().all() and not known(current).any()
        filtered=aligned[reject].copy();filtered['filter_effect']=np.where(~known(filtered),'UNKNOWN_OUTCOME',np.where(filtered.target.eq(1),'NEW_MISSED_PROMOTION','REMOVED_FALSE_POSITIVE'))
        complete_filtered.append(filtered)
        route_checks.append(dict(arm=arm,complete_rows=len(aligned),eligible_original_positives=int(mask.sum()),filtered_rows=int(reject.sum()),
            only_original_eligible_PROMOTE_to_FAIL=True,original_raw_and_ranking_preserved=True,current6_unknown=True))
    data=joblib.load(ROOT/'fixed_input_BEFORE_ANY_FIT.joblib');full=data['full'];shapes=data['shapes']
    assert len(full)==9315 and len(data['alignment_plans'])==16
    source_scores=csv(EXEC/'all_monthly252_source_scores_BEFORE_META.csv').set_index(KEYS)
    assert not source_scores.index.duplicated().any()
    choices=[];threshold_rows=[];third_scores=[];warmup=[];eligible_months=[]
    for plan in data['alignment_plans']:
        month=plan['month'];cutoff=plan['cutoff'];q={}
        for source_cutoff in plan['source_month_cutoffs']:
            available=full[known(full)&full.date.lt(source_cutoff)&full.target_date.lt(source_cutoff)]
            q[source_cutoff]=min(252,available.date.nunique())
        qualified=all(v==252 for v in q.values())
        assert q==plan['source_qualified_days'] and qualified==plan['eligible']
        choice_path=EXEC/f'{month}_SELECTION_BEFORE_OUTER_SCORE.json';selection=read(choice_path)
        assert selection['eligible']==qualified and selection['outer_results_used_for_selection'] is False and selection['final_meta_refit'] is False
        if not qualified:
            assert month in ['2025-07','2025-08','2025-09','2025-10']
            for arm in ARMS:
                recorded=selection['choices'][arm];assert recorded['threshold']==0 and recorded['reason']=='INSUFFICIENT_MONTHLY252_OOF_WARMUP'
                actual=heads[arm][heads[arm].date.str[:7].eq(month)]
                assert actual.filter_threshold.eq(0).all() and not actual.filter_applied.astype(bool).any()
                assert np.array_equal(actual.evaluation_call,actual.base_call)
                warmup.append(dict(month=month,arm=arm,complete_outer_rows=len(actual),threshold=0.,reason=recorded['reason'],
                    minimum_source_training_days=min(q.values()),denominator_retained=True,fit_count=0))
                choices.append(dict(month=month,arm=arm,warmup_eligible=False,threshold=0.,selection_statistics_present=False,prefix_fit_count=0))
            continue
        eligible_months.append(month)
        oof=csv(EXEC/f'{month}_complete_monthly252_OOF_BEFORE_META.csv')
        assert not oof.duplicated(KEYS).any() and oof.daily_one_word.eq(0).all()
        expected=[]
        for block,dates in enumerate(plan['validation_dates'],1):
            assert len(dates)==63
            complete=full[full.date.isin(dates)].copy()
            scores=source_scores.loc[pd.MultiIndex.from_frame(complete[KEYS])]
            complete['raw_score']=scores.raw_score.to_numpy();complete['first_stage_model_cutoff']=scores.first_stage_model_cutoff.to_numpy()
            assert complete.first_stage_model_cutoff.eq(complete.date.str[:7]+'-01').all()
            complete=complete.merge(shapes[KEYS+['verified_current_price_shape']],on=KEYS,validate='one_to_one')
            complete['daily_one_word']=complete.frozen_original_nonword_identity
            complete=complete[complete.daily_one_word.eq(0)].copy();assert not complete.verified_current_price_shape.eq(1).any()
            complete['base_call']=np.where(complete.verified_current_price_shape.eq(0),np.where(complete.raw_score.gt(.5),'PROMOTE','FAIL'),'UNKNOWN')
            complete['call']=complete.base_call;complete['ranking_score']=complete.raw_score
            raw=np.clip(complete.raw_score.to_numpy(float),1e-12,1-1e-12)
            complete['base_score_logit']=np.log(raw)-np.log1p(-raw);complete['inner_block']=block
            expected.append(complete)
        expected=pd.concat(expected,ignore_index=True)
        pd.testing.assert_frame_equal(oof,expected,check_exact=True,check_dtype=False)
        third=oof[oof.inner_block.eq(3)].copy();assert third.date.nunique()==63
        first=third.date.min();assert first==plan['first_third_signal']
        prefix=oof[(oof.inner_block<3)&eligible(oof)&known(oof)&oof.date.lt(first)&oof.target_date.lt(first)].copy()
        saved_prefix=csv(EXEC/f'{month}_frozen_meta_training.csv')
        pd.testing.assert_frame_equal(saved_prefix,prefix.reset_index(drop=True),check_exact=True,check_dtype=False)
        assert prefix.target_date.max()<first and oof[known(oof)].target_date.max()<cutoff
        for arm in ARMS:
            kind=KINDS[arm];recorded=selection['choices'][arm];fit=recorded['fit']
            qmeta=int(prefix.target.eq(1).sum())>=20 and int(prefix.target.eq(0).sum())>=20 and prefix.date.nunique()>=20
            assert qmeta==fit['qualified']
            mask=eligible(third);secondary=np.full(len(third),np.nan)
            if qmeta:
                assert sha(fit['model_path'])==fit['model_sha256'];model=joblib.load(fit['model_path'])
                assert model['features']==protocol['features'][kind]
                if mask.any():
                    with threadpool_limits(limits=2):secondary[mask]=model['model'].predict_proba(third.loc[mask,model['features']])[:,list(model['model'].named_steps['logistic'].classes_).index(1)]
            else:secondary[mask]=0.
            stats={t:matrix(apply(third,secondary,t)) for t in THRESHOLDS}
            for t,z in stats.items():
                compare_stats(z,recorded['statistics'][str(t)])
                threshold_rows.append(dict(month=month,arm=arm,threshold=t,**z))
            threshold=choose(stats) if qmeta else 0.
            assert threshold==recorded['threshold']
            actual=heads[arm][heads[arm].date.str[:7].eq(month)];assert actual.filter_threshold.eq(threshold).all()
            choices.append(dict(month=month,arm=arm,warmup_eligible=True,threshold=threshold,selection_statistics_present=True,
                prefix_fit_count=int(qmeta),prefix_rows=len(prefix),prefix_positive=int(prefix.target.eq(1).sum()),prefix_negative=int(prefix.target.eq(0).sum()),
                max_prefix_target=prefix.target_date.max(),third_first_signal=first,third_complete_B=len(third),third_known=int(known(third).sum()),
                third_unknown=int((~known(third)).sum()),outer_cutoff=cutoff,third_is_selection_not_validation=True,no_final_meta_refit=True))
            selected=third[KEYS+['target','label_valid','base_call','raw_score','first_stage_model_cutoff']].copy();selected['arm']=arm;selected['secondary_score_replayed']=secondary
            third_scores.append(selected)
    assert len(choices)==32 and len(warmup)==8 and len(eligible_months)==12 and len(threshold_rows)==240
    assert result['meta_fits']==24 and result['first_stage_backfill_fits']==6 and result['reused_source_models']==15 and result['first_stage_deployment_refits']==0
    assert not list(EXEC.glob('*FINAL*model.joblib'))
    metrics=[];rank_rows=[];raw_picks=[];recommendations=[];recommended=[]
    for period,first,last in PERIODS:
        for arm in ['BASE','ST_DUAL',BASE]+ARMS:
            frame=all_rows[all_rows.arm.eq(arm)&all_rows.date.between(first,last)]
            for scope,part in [('FULL_POOL',frame),('ORIGINAL_NONWORD',frame[frame.daily_one_word.eq(0)])]:metrics.append(dict(period=period,arm=arm,scope=scope,**matrix(part)))
            for k in [1,2,3]:
                stat,picks=pick(frame,k);rank_rows.append(dict(period=period,arm=arm,**stat))
                for row in picks.itertuples(index=False):raw_picks.append(dict(period=period,arm=arm,k=k,date=row.date,code=row.code,ranking_score=row.ranking_score,target=row.target,label_valid=row.label_valid))
            stat,picks=pick(frame,3,positive=True);recommendations.append(dict(period=period,arm=arm,**stat))
            for row in picks.itertuples(index=False):recommended.append(dict(period=period,arm=arm,date=row.date,code=row.code,ranking_score=row.ranking_score,target=row.target,label_valid=row.label_valid))
    metrics=pd.DataFrame(metrics);ranks=pd.DataFrame(rank_rows);raw_picks=pd.DataFrame(raw_picks);recs=pd.DataFrame(recommendations)
    for arm in ARMS:
        a=raw_picks[raw_picks.arm.eq(BASE)].drop(columns='arm').reset_index(drop=True);z=raw_picks[raw_picks.arm.eq(arm)].drop(columns='arm').reset_index(drop=True)
        pd.testing.assert_frame_equal(a,z,check_exact=True)
    official=csv(EVAL/'complete_same_pool_classification.csv')
    for row in official.to_dict('records'):
        own=metrics[metrics.period.eq(row['period'])&metrics.arm.eq(row['arm'])&metrics.scope.eq(row['scope'])].iloc[0].to_dict();compare_stats(own,row)
    official_rank=csv(EVAL/'raw_observation_Top1Top2Top3_INVARIANT.csv')
    for row in official_rank.to_dict('records'):
        own=ranks[ranks.period.eq(row['period'])&ranks.arm.eq(row['arm'])&ranks.k.eq(row['k'])].iloc[0]
        for key in ['signal_days','full_slots','selected','empty_slots','known_selected','unknown_selected','hits','full_slot_hit_rate']:assert np.isclose(own[key],row[key],rtol=0,atol=1e-14,equal_nan=True)
    official_rec=csv(EVAL/'positive_recommendation_Top3_FULL_SLOTS.csv')
    for row in official_rec.to_dict('records'):
        own=recs[recs.period.eq(row['period'])&recs.arm.eq(row['arm'])].iloc[0]
        for key in ['signal_days','full_slots','selected','empty_slots','unknown_selected','hits']:assert own[key]==row[key]
    filtered=pd.concat(complete_filtered,ignore_index=True);gains=[]
    for period,first,last in PERIODS:
        for arm in ARMS:
            part=filtered[filtered.arm.eq(arm)&filtered.date.between(first,last)]
            gains.append(dict(period=period,arm=arm,filtered_total=len(part),removed_FP=int(part.filter_effect.eq('REMOVED_FALSE_POSITIVE').sum()),
                new_missed_promotions=int(part.filter_effect.eq('NEW_MISSED_PROMOTION').sum()),unknown_filtered=int(part.filter_effect.eq('UNKNOWN_OUTCOME').sum())))
    index=metrics[metrics.scope.eq('ORIGINAL_NONWORD')].set_index(['period','arm']);guards={}
    for arm in ARMS:
        z=index.loc[('HISTORICAL_POOLED',arm)];baseline=index.loc[('HISTORICAL_POOLED',BASE)]
        g=dict(pooled_Precision_gain3pp=bool(z.Precision>=baseline.Precision+.03),pooled_FP_drop10percent=bool(z.fp*10<=baseline.fp*9),pooled_F1_not_lower=bool(z.F1>=baseline.F1))
        for period,first,last in PERIODS[:3]:
            z=index.loc[(period,arm)];baseline=index.loc[(period,BASE)]
            g[period+'_Recall_loss_within5pp']=bool(z.Recall>=baseline.Recall-.05)
            g[period+'_FP_not_higher']=bool(z.fp<=baseline.fp)
            g[period+'_F1_not_lower']=bool(z.F1>=baseline.F1)
        rec=recs[recs.period.eq('HISTORICAL_POOLED')].set_index('arm');g['pooled_positive_Top3_hits_not_lower']=bool(rec.loc[arm,'hits']>=rec.loc[BASE,'hits'])
        assert g==result['arms'][arm]['guards'] and all(g.values())==result['arms'][arm]['fixed_screen_pass'];guards[arm]=g
    outputs={'独立复算_全池与原B完整指标.csv':metrics,'独立复算_原raw观察Top1Top2Top3.csv':ranks,
        '独立复算_全部raw预选含未知.csv':raw_picks,'独立复算_正向推荐Top3完整槽位.csv':recs,
        '独立复算_正向推荐逐股含未知.csv':pd.DataFrame(recommended),'独立审查_32月臂暖启动与阈值选择.csv':pd.DataFrame(choices),
        '独立复算_24月臂第三块十阈值完整B统计.csv':pd.DataFrame(threshold_rows),'独立重放_第三块冻结元模型选择分数.csv':pd.concat(third_scores,ignore_index=True),
        '独立核验_前4月8槽无资格不滤完整分母.csv':pd.DataFrame(warmup),'独立复算_全部过滤逐股得失.csv':filtered,
        '独立复算_新增漏判移除误报与未知.csv':pd.DataFrame(gains),'独立核验_路由和原分数守恒.csv':pd.DataFrame(route_checks)}
    HERE.mkdir(parents=True,exist_ok=True)
    for name,frame in outputs.items():frame.to_csv(HERE/name,index=False,encoding='utf-8-sig')
    result=dict(formed_at_UTC=datetime.now(timezone.utc).isoformat(),verification_fits=0,network_calls=0,no_new_oct16_results_read=True,
        original49101_all_columns_exact=True,complete56655_fifteen_heads=True,new_heads_each3777=True,BASE_SOURCE_raw_values_unchanged=True,
        original_B=[2980,2932,48,871],september_B=[143,107,40,36],current6_each_head_unknown=True,
        monthly_selection_slots=32,first4months_bypass_slots=8,actual_third_block_threshold_selections=24,
        independent_third_block_threshold_statistics=240,exact_Fraction_choice_verified=True,prefix_target_maturity_verified=True,
        source_month_cutoff_natural_month_start_verified=True,complete_OOF_membership_and_unknowns_verified=True,
        raw_observation_Topk_invariant=True,positive_recommendation_Top3_full_slots_verified=True,only_original_eligible_positive_to_FAIL=True,
        no_final_meta_refit=True,guards=guards,protected_source_verification=protected,
        source_bindings={str(p.resolve()):sha(p) for p in [Path(__file__),OLD/'independent_verification/verify_positive_filter_effects_without_fit.py',ROOT/'protocol_BEFORE_ANY_FIT.json',ROOT/'fixed_input_BEFORE_ANY_FIT.joblib',reference,EXEC/'all56655_predictions_BEFORE_EVALUATION.csv',result_path]},
        output_bindings={str((HERE/name).resolve()):sha(HERE/name) for name in outputs})
    receipt_path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Independent effect audit passed; fits0, old source exceptions0, selection slots32/actual24/bypass8')
    print(metrics[metrics.scope.eq('ORIGINAL_NONWORD')&metrics.arm.isin([BASE]+ARMS)][['period','arm','tp','fp','Recall','Precision','F1']].to_string(index=False))

if __name__=='__main__':main()
