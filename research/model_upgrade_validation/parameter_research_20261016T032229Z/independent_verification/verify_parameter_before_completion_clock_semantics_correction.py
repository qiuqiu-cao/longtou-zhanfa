"""Independent replay of one fixed nested parameter study; zero fits/fetches."""
from pathlib import Path
from fractions import Fraction
from datetime import datetime, timezone
import sys
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import parameter_audit_primitives as a

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
REPO=HERE.parents[3]
sys.path.insert(0,str(REPO))
EXEC=ROOT/'execution'
OLD=REPO/'research/model_upgrade_validation/false_positive_20261016T030307Z'
REFERENCE=OLD/'execution/all30216_predictions_BEFORE_EVALUATION.csv'
BASE_ARM='SOURCE_PATCHED_FEATURE_CLASSIFIER'
ARMS=['PARAM_CAPACITY_OOF_FIXED_HALF','PARAM_CAPACITY_OOF_SELECTED_THRESHOLD']
PERIODS=[('2025H2','2025-07-01','2025-12-31'),('2026JanAug','2026-01-01','2026-08-31'),('2026Sep','2026-09-01','2026-09-30'),('2026Oct','2026-10-01','2026-10-15'),('HISTORICAL_POOLED','2025-07-01','2026-09-30')]

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def csv(path):return pd.read_csv(path,dtype={'code':str},float_precision='round_trip',low_memory=False)

def oof_statistics(rows,threshold):
    assert a.valid_labels(rows).all() and rows.target.nunique()==2
    assert rows.raw_score.notna().all() and np.isfinite(rows.raw_score).all()
    assert not rows.duplicated(['date','code']).any()
    view=rows.copy()
    eligible=view.verified_current_price_shape.eq(0)
    view['evaluation_call']='UNKNOWN'
    view.loc[eligible,'evaluation_call']=np.where(view.loc[eligible,'raw_score'].gt(threshold),'PROMOTE','FAIL')
    return a.matrix(view)

def f05(stat):return Fraction(5*stat['TP'],4*stat['TP']+4*stat['FP']+stat['positives'])
def recall(stat):return Fraction(stat['TP'],stat['positives'])
def f1(stat):return Fraction(2*stat['TP'],stat['positives']+stat['TP']+stat['FP'])
def allowed(stat,baseline):return recall(stat)>=recall(baseline)-Fraction(1,40) and f1(stat)>=f1(baseline)-Fraction(1,100)

def pick_capacity(stats):
    base=stats['BASE'];best='BASE'
    for name in ['COMPACT','EXPANDED']:
        if allowed(stats[name],base) and f05(stats[name])>f05(stats[best]):best=name
    return best

def pick_threshold(stats,baseline):
    candidates=[t for t,z in stats.items() if t!=.5 and allowed(z,baseline) and f05(z)>f05(stats[.5])]
    if not candidates:return .5
    return sorted(candidates,key=lambda t:(-f05(stats[t]),abs(Fraction(str(t))-Fraction(1,2)),-Fraction(str(t))))[0]

def assert_official_stat(own,official):
    for left,right in [('candidates','candidates'),('positives','positive'),('negatives','negative'),('TP','tp'),('FP','fp'),('Recall','Recall'),('F1','F1'),('F0p5','F05')]:
        assert np.isclose(own[left],official[right],rtol=0,atol=1e-14,equal_nan=True),(left,own[left],official[right])
    if np.isnan(own['Precision']):assert official['Precision'] is None
    else:assert abs(own['Precision']-official['Precision'])<1e-14
    assert Fraction(official['F05_numerator'],official['F05_denominator'])==f05(own)

def main():
    result_path=EXEC/'parameter_fixed_result.json'
    full_path=EXEC/'all37770_predictions_BEFORE_EVALUATION.csv'
    receipt_path=HERE/'parameter_study_independent_receipt.json'
    if not full_path.exists() or not result_path.exists():raise FileNotFoundError('wait for complete execution; do not fit or poll')
    if receipt_path.exists():raise FileExistsError('immutable independent verification already exists')
    protocol=read(ROOT/'protocol_BEFORE_ANY_FIT.json');result=read(result_path)
    hashes=protocol['source_bindings']|result['source_bindings']
    for path,expected in hashes.items():assert a.sha(path)==expected,('source SHA drift',path)
    expected_base=dict(learning_rate=.06,max_iter=120,max_leaf_nodes=7,max_depth=3,min_samples_leaf=25,l2_regularization=3.,early_stopping=False,random_state=20261015)
    assert protocol['baseline']==expected_base
    configs={'COMPACT':expected_base|dict(max_depth=2,max_leaf_nodes=3,min_samples_leaf=50,l2_regularization=6.),'EXPANDED':expected_base|dict(max_depth=4,max_leaf_nodes=15,min_samples_leaf=12,l2_regularization=1.5)}
    assert protocol['capacity_candidates']==configs and protocol['threshold_candidates']==[.4,.45,.5,.55,.6]
    assert protocol['policy_arms']==ARMS
    assert protocol['maximum_new_inner_fits']==96 and protocol['maximum_new_final_fits']==16 and protocol['maximum_total_fits']==112
    assert protocol['outer_data_used_for_selection'] is False and protocol['event_probability_validated'] is False
    old=csv(REFERENCE);complete=csv(full_path)
    a.assert_old_predictions_exact(old,complete)
    baseline=complete[complete.arm.eq(BASE_ARM)].set_index(['date','code'])
    new={arm:complete[complete.arm.eq(arm)].copy() for arm in ARMS}
    for arm,frame in new.items():
        a.assert_september_and_current_unknown(frame)
        indexed=frame.set_index(['date','code']).loc[baseline.index]
        for col in ['target_date','target','label_valid','daily_one_word','frozen_original_nonword_identity','verified_current_price_shape']:
            pd.testing.assert_series_equal(indexed[col],baseline[col],check_names=False,check_exact=True)
        assert indexed.event_probability_validated.astype(str).str.lower().eq('false').all()
        b_route=indexed.verified_current_price_shape.eq(0)
        pd.testing.assert_frame_equal(indexed.loc[~b_route,['score','call']],baseline.loc[~b_route,['score','call']],check_exact=True)
        expected_calls=np.where(indexed.loc[b_route,'b_score']>indexed.loc[b_route,'decision_threshold'],'PROMOTE','FAIL')
        assert np.array_equal(indexed.loc[b_route,'call'].to_numpy(),expected_calls)
        for col in ['evaluation_call','recovered_B_call','aligned_evaluation_call']:assert indexed[col].equals(indexed.call)
    left=new[ARMS[0]].set_index(['date','code']);right=new[ARMS[1]].set_index(['date','code']).loc[left.index]
    for col in ['b_score','ranking_score','selected_capacity_config']:
        pd.testing.assert_series_equal(left[col],right[col],check_names=False,check_exact=True)
    data=joblib.load(ROOT/'fixed_input_BEFORE_ANY_FIT.joblib');full=data['full'];fields=data['fields'];shape=data['shape']
    assert len(full)==9315 and len(fields)==92 and len(data['plans'])==16
    shape_lookup=shape.set_index(['date','code']).verified_current_price_shape
    blocks=[];choices=[];statistics=[];replays=[];new_inner=0;new_final=0
    for plan in data['plans']:
        month=plan['month'];prefix=EXEC/month;cutoff=plan['cutoff']
        window=a.outer_window(full,cutoff)
        parts={'BASE':[],'COMPACT':[],'EXPANDED':[]}
        selection=read(str(prefix)+'_SELECTION_BEFORE_FINAL_FIT_AND_OUTER_SCORE.json')
        completed=read(str(prefix)+'_COMPLETED.json')
        assert selection['outer_scores_or_labels_read_for_selection'] is False
        for block,past,validation in a.chronological_blocks(window):
            bvalid=validation[validation.frozen_original_nonword_identity.eq(0)].copy()
            ids=pd.MultiIndex.from_frame(bvalid[['date','code']])
            relevant_shape=shape_lookup.loc[ids].to_numpy()
            assert not np.any(relevant_shape==1),'non-B inherited route would need separate OOF audit'
            cached_model_path=OLD/'execution'/f'{month}_inner{block}_model.joblib'
            cached=joblib.load(cached_model_path)
            with threadpool_limits(limits=2):raw=cached['model'].predict_proba(validation[fields])[:,list(cached['model'].classes_).index(1)]
            raw_b=raw[validation.frozen_original_nonword_identity.eq(0)]
            cached_rows=csv(OLD/'execution'/f'{month}_inner{block}_all_validation_scores_BEFORE_EFFECTS.csv')
            assert a.scores_equal(cached_rows.raw_score,raw)<1e-14
            base_rows=bvalid[['date','code','target_date','target','label_valid','frozen_original_nonword_identity']].copy()
            base_rows['raw_score']=raw_b;base_rows['verified_current_price_shape']=relevant_shape
            parts['BASE'].append(base_rows)
            for config,params in configs.items():
                key=Path(str(prefix)+f'_inner{block}_{config}')
                model_file=Path(str(key)+'_model.joblib');saved=joblib.load(model_file)
                started=read(str(key)+'_STARTED.json');done=read(str(key)+'_COMPLETED.json')
                assert saved['features']==fields and saved['params']==params and saved['capacity']==config
                assert saved['cutoff']==validation.date.min() and saved['training_digest']==started['training_digest']==cached['training_digest']
                assert started['max_train_target']==past.target_date.max()<validation.date.min() and started['first_validation']==validation.date.min()
                assert done['model_sha256']==a.sha(model_file)
                actual=saved['model'].named_steps['hgb'].get_params()
                assert all(actual[k]==v for k,v in params.items())
                assert saved['model'].named_steps['hgb'].n_iter_==120
                rows=csv(str(key)+'_OOF_scores_BEFORE_SELECTION.csv')
                with threadpool_limits(limits=2):candidate_raw=saved['model'].predict_proba(bvalid[fields])[:,list(saved['model'].classes_).index(1)]
                error=a.scores_equal(rows.raw_score,candidate_raw)
                pd.testing.assert_frame_equal(rows.drop(columns='raw_score').reset_index(drop=True),base_rows.drop(columns='raw_score').reset_index(drop=True),check_exact=True,check_dtype=False)
                parts[config].append(rows);new_inner+=1
                blocks.append(dict(month=month,block=block,config=config,training_rows=len(past),training_days=past.date.nunique(),
                    max_training_target=past.target_date.max(),first_validation=validation.date.min(),validation_days=validation.date.nunique(),
                    outer_cutoff=cutoff,max_validation_target=validation.target_date.max(),original_B_validation_rows=len(rows),
                    strict_train_target_maturity=True,no_train_validation_overlap=True,max_saved_score_replay_error=error))
        oof={name:pd.concat(items,ignore_index=True) for name,items in parts.items()}
        stats={}
        for name,rows in oof.items():
            saved_rows=csv(str(prefix)+f'_{name}_complete_OOF_BEFORE_SELECTION.csv')
            assert rows.target_date.max()<cutoff and not rows.duplicated(['date','code']).any()
            # BASE replay may differ by machine precision from cached CSV; facts remain exact.
            pd.testing.assert_frame_equal(saved_rows.drop(columns='raw_score').reset_index(drop=True),rows.drop(columns='raw_score').reset_index(drop=True),check_exact=True,check_dtype=False)
            a.scores_equal(saved_rows.raw_score,rows.raw_score)
            # Selection operated on frozen cached scores; use exactly those bytes for rational decisions.
            oof[name]=saved_rows
            stats[name]=oof_statistics(saved_rows,.5)
            assert_official_stat(stats[name],selection['capacity_statistics'][name])
            statistics.append(dict(month=month,phase='CAPACITY',config=name,threshold=.5,**stats[name]))
        selected=pick_capacity(stats)
        assert selected==selection['selected_capacity']==completed['selected_capacity']
        threshold_stats={t:oof_statistics(oof[selected],t) for t in [.4,.45,.5,.55,.6]}
        threshold=pick_threshold(threshold_stats,stats['BASE'])
        assert threshold==selection['selected_threshold']==completed['selected_threshold']
        for t,z in threshold_stats.items():
            assert_official_stat(z,selection['threshold_statistics'][str(t)])
            statistics.append(dict(month=month,phase='THRESHOLD',config=selected,threshold=t,**z))
        assert selection['max_OOF_target']==oof[selected].target_date.max()<cutoff
        model_path=Path(completed['selected_final_model_path'])
        assert a.sha(model_path)==completed['selected_final_model_sha256']
        final=joblib.load(model_path)
        if selected=='BASE':
            assert not completed['selected_final_new_fit'] and model_path==Path(plan['final_model_path'])
            assert a.sha(model_path)==plan['final_model_sha256']
            assert not Path(str(prefix)+'_selected_FINAL_model.joblib').exists()
        else:
            assert completed['selected_final_new_fit'] and final['params']==configs[selected] and final['capacity']==selected
            assert final['features']==fields and final['training_digest']==plan['training_digest'] and final['cutoff']==cutoff
            started=read(str(prefix)+'_selected_FINAL_STARTED.json')
            assert started['capacity']==selected and started['training_digest']==plan['training_digest']
            assert selection['formed_at']<=started['formed_at']<=completed['formed_at']
            params=final['model'].named_steps['hgb'].get_params()
            assert all(params[k]==v for k,v in configs[selected].items())
            new_final+=1
        test=full[full.date.str[:7].eq(month)]
        with threadpool_limits(limits=2):raw=final['model'].predict_proba(test[fields])[:,list(final['model'].classes_).index(1)]
        keys=pd.MultiIndex.from_frame(test[['date','code']])
        for arm,t in zip(ARMS,[.5,threshold]):
            actual=new[arm].set_index(['date','code']).loc[keys]
            err=a.scores_equal(actual.b_score,raw)
            a.scores_equal(actual.ranking_score,raw)
            assert actual.decision_threshold.eq(t).all() and actual.selected_capacity_config.eq(selected).all()
            for row,score in zip(test.itertuples(index=False),raw):replays.append(dict(month=month,arm=arm,date=row.date,code=row.code,raw_score_replayed=float(score),max_error_in_month=err))
        choices.append(dict(month=month,cutoff=cutoff,selected_capacity=selected,selected_threshold=threshold,
            baseline_TP=stats['BASE']['TP'],baseline_FP=stats['BASE']['FP'],selected_TP=stats[selected]['TP'],selected_FP=stats[selected]['FP'],
            capacity_F05_strict_gain=selected!='BASE',selected_final_new_fit=selected!='BASE',
            max_OOF_target=oof[selected].target_date.max(),outer_results_used_for_selection=False))
    assert new_inner==96 and new_final<=16 and result['inner_new_fits']==96 and result['selected_final_new_fits']==new_final
    assert result['total_new_fits']==96+new_final and result['old_final_models_refitted']==0 and result['cached_baseline_inner_reused']==48
    assert len(list(EXEC.glob('*_inner*_model.joblib')))==96 and len(list(EXEC.glob('*_selected_FINAL_model.joblib')))==new_final
    stats_rows=[];rank_rows=[];picks=[]
    frames={BASE_ARM:baseline.reset_index()}|new
    for period,first,last in PERIODS:
        for arm,frame in frames.items():
            sample=frame[frame.date.between(first,last)]
            for scope,part in [('FULL_POOL',sample),('ORIGINAL_NONWORD',sample[sample.daily_one_word.eq(0)])]:
                stats_rows.append(dict(period=period,arm=arm,scope=scope,**a.matrix(part)))
            for k in [1,2,3]:
                stat,chosen=a.full_slot_topk(sample,k)
                rank_rows.append(dict(period=period,arm=arm,**stat))
                for row in chosen.itertuples(index=False):picks.append(dict(period=period,arm=arm,k=k,date=row.date,code=row.code,target=row.target,label_valid=row.label_valid,raw_score=row.ranking_score))
    matrices=pd.DataFrame(stats_rows);ranks=pd.DataFrame(rank_rows);selected_rows=pd.DataFrame(picks)
    official=csv(EXEC/'same_pool_full_classification_metrics.csv')
    for row in official.itertuples(index=False):
        own=matrices[matrices.period.eq(row.period)&matrices.arm.eq(row.arm)&matrices.scope.eq(row.scope)].iloc[0]
        for x,y in [('TP','tp'),('FP','fp'),('Recall','Recall'),('Precision','Precision'),('F1','F1')]:assert np.isclose(own[x],getattr(row,y),rtol=0,atol=1e-14,equal_nan=True)
    official_rank=csv(EXEC/'full_slot_Top1Top2Top3.csv')
    for row in official_rank.itertuples(index=False):
        own=ranks[ranks.period.eq(row.period)&ranks.arm.eq(row.arm)&ranks.k.eq(row.k)].iloc[0]
        for key in ['signal_days','full_slots','selected','empty_slots','known_selected','unknown_selected','hits','full_slot_hit_rate']:assert np.isclose(own[key],getattr(row,key),rtol=0,atol=1e-14,equal_nan=True)
    left=selected_rows[selected_rows.arm.eq(ARMS[0])].drop(columns='arm').reset_index(drop=True)
    right=selected_rows[selected_rows.arm.eq(ARMS[1])].drop(columns='arm').reset_index(drop=True)
    pd.testing.assert_frame_equal(left,right,check_exact=True)
    index=matrices[matrices.scope.eq('ORIGINAL_NONWORD')].set_index(['period','arm']);guards={}
    for arm in ARMS:
        z=index.loc[('HISTORICAL_POOLED',arm)];base=index.loc[('HISTORICAL_POOLED',BASE_ARM)]
        guard=dict(pooled_FP_drop_at_least10percent=bool(z.FP<=base.FP*.9),pooled_Precision_gain_at_least3pp=bool(z.Precision>=base.Precision+.03),pooled_F1_not_lower=bool(z.F1>=base.F1))
        for period,first,last in PERIODS[:3]:
            zz=index.loc[(period,arm)];bb=index.loc[(period,BASE_ARM)]
            guard[period+'_Recall_loss_within2p5pp']=bool(zz.Recall>=bb.Recall-.025)
            guard[period+'_FP_not_higher']=bool(zz.FP<=bb.FP)
            guard[period+'_F1_not_lower']=bool(zz.F1>=bb.F1)
        pooled_rank=ranks[ranks.period.eq('HISTORICAL_POOLED')&ranks.k.eq(3)].set_index('arm')
        guard['pooled_Top3_full_slot_hits_not_lower']=bool(pooled_rank.loc[arm,'hits']>=pooled_rank.loc[BASE_ARM,'hits'])
        assert guard==result['arms'][arm]['guards'] and all(guard.values())==result['arms'][arm]['fixed_screen_pass']
        guards[arm]=guard
    outputs={'独立核验_96新内层严格时间块与重放.csv':pd.DataFrame(blocks),'独立核验_16月训练期选容量与阈值.csv':pd.DataFrame(choices),
        '独立复算_所有容量与五阈值OOF指标.csv':pd.DataFrame(statistics),'独立重放_两策略完整7554原始分数.csv':pd.DataFrame(replays),
        '独立复算_完整池和原非一字指标.csv':matrices,'独立复算_完整槽位Top1Top2Top3.csv':ranks,'独立复算_全部预选保留未知.csv':selected_rows}
    for name,frame in outputs.items():frame.to_csv(HERE/name,index=False,encoding='utf-8-sig')
    receipt=dict(formed_at=datetime.now(timezone.utc).isoformat(),verification_new_fits=0,no_fetch=True,no_oct16_market_result_read=True,
        original30216_all_columns_exact=True,BASE_original_values_unchanged=True,all37770_complete10heads=True,original_september_B=[143,107,40,36],current6_each_policy_unknown=True,
        new_inner_models_replayed=96,reused_inner_models_replayed=48,monthly_selected_final_models_replayed=16,selected_new_final_models=new_final,
        capacity_threshold_selected_only_from_training_OOF=True,exact_fraction_choice_recomputed=True,policy_raw_scores_and_Topk_identical=True,
        event_probability_validated=False,source_hashes_verified=len(hashes),guards=guards,
        source_bindings={str(p.resolve()):a.sha(p) for p in [Path(__file__),HERE/'parameter_audit_primitives.py',ROOT/'protocol_BEFORE_ANY_FIT.json',ROOT/'fixed_input_BEFORE_ANY_FIT.joblib',REFERENCE,full_path,result_path]},
        output_bindings={str((HERE/name).resolve()):a.sha(HERE/name) for name in outputs})
    receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(verified=True,new_fits=0,source_hashes=len(hashes),selected_new_final_models=new_final),ensure_ascii=False))
    print(matrices[matrices.scope.eq('ORIGINAL_NONWORD')][['period','arm','TP','FP','Recall','Precision','F1','F0p5']].to_string(index=False))

if __name__=='__main__':main()
