"""One bounded nested chronological capacity/threshold parameter study.

Two fixed capacity candidates, one reused baseline, two declared policies.
No outer-month result participates in model or decision-threshold selection.
"""
from pathlib import Path
from fractions import Fraction
import argparse
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

import short_term_thinking_dataset_builder as b
import short_term_thinking_research as training
import model_upgrade_history as history
import expectation_345_research as original
import expectation_345_evaluation as evaluation
import relay_source_feature_research as source
import relay_temporal_score_calibration as temporal
from short_term_thinking_preprocessing import EmptyTrainingColumnGuard

ROOT=b.ROOT/'research/model_upgrade_validation/parameter_research_20261016T032229Z'
PREVIOUS=temporal.ROOT
REFERENCE=PREVIOUS/'execution/all30216_predictions_BEFORE_EVALUATION.csv'
BASE_ARM=source.ARM
ARMS=['PARAM_CAPACITY_OOF_FIXED_HALF','PARAM_CAPACITY_OOF_SELECTED_THRESHOLD']
CONFIGS={
    'COMPACT':training.PARAMS|dict(max_depth=2,max_leaf_nodes=3,min_samples_leaf=50,l2_regularization=6.),
    'EXPANDED':training.PARAMS|dict(max_depth=4,max_leaf_nodes=15,min_samples_leaf=12,l2_regularization=1.5),
}
THRESHOLDS=[.4,.45,.5,.55,.6]


def decision_statistics(oof,threshold):
    if oof.duplicated(['date','code']).any():raise ValueError('duplicate OOF identity')
    known=oof.target.isin([0,1])&oof.label_valid.fillna(False).astype(bool)
    if not known.all() or oof.target.nunique()!=2:raise ValueError('same two known OOF classes required')
    if not np.isfinite(oof.raw_score).all():raise ValueError('complete finite raw scores required')
    promote=oof.verified_current_price_shape.eq(0)&oof.raw_score.gt(threshold)
    tp=int((promote&oof.target.eq(1)).sum());fp=int((promote&oof.target.eq(0)).sum())
    positive=int(oof.target.eq(1).sum());fn=positive-tp
    denominator=5*tp+fn+4*fp
    return dict(candidates=len(oof),positive=positive,negative=len(oof)-positive,tp=tp,fp=fp,
        missed_positives=fn,Precision=tp/(tp+fp) if tp+fp else None,Recall=tp/positive,
        F1=2*tp/(2*tp+fn+fp),F05=5*tp/denominator if denominator else 0.,
        F05_numerator=5*tp,F05_denominator=denominator or 1,
        threshold=threshold,positive_calls=tp+fp,
        source_unknown=int(oof.verified_current_price_shape.ne(0).sum()))


def score_fraction(stat):return Fraction(stat['F05_numerator'],stat['F05_denominator'])


def guard(candidate,original_baseline):
    if candidate['candidates']!=original_baseline['candidates'] or candidate['positive']!=original_baseline['positive']:
        raise ValueError('selection denominator differs')
    recall=Fraction(candidate['tp'],candidate['positive'])
    baseline_recall=Fraction(original_baseline['tp'],original_baseline['positive'])
    f1=Fraction(2*candidate['tp'],2*candidate['tp']+candidate['missed_positives']+candidate['fp'])
    baseline_f1=Fraction(2*original_baseline['tp'],2*original_baseline['tp']+original_baseline['missed_positives']+original_baseline['fp'])
    return recall>=baseline_recall-Fraction(1,40) and f1>=baseline_f1-Fraction(1,100)


def select_capacity(statistics):
    baseline=statistics['BASE']
    eligible=[name for name in ['COMPACT','EXPANDED'] if guard(statistics[name],baseline)
        and score_fraction(statistics[name])>score_fraction(baseline)]
    # max returns the first equal item: fixed COMPACT precedence, never outer data.
    return max(eligible,key=lambda name:score_fraction(statistics[name])) if eligible else 'BASE'


def select_threshold(statistics,baseline):
    fixed=statistics[.5]
    eligible=[t for t in THRESHOLDS if t!=.5 and guard(statistics[t],baseline)
        and score_fraction(statistics[t])>score_fraction(fixed)]
    if not eligible:return .5
    return min(eligible,key=lambda t:(-score_fraction(statistics[t]),abs(Fraction(str(t))-Fraction(1,2)),-Fraction(str(t))))


def nonword_oof(validation,raw,shape):
    rows=validation[['date','code','target_date','target','label_valid','frozen_original_nonword_identity']].copy()
    rows['raw_score']=raw
    lookup=shape.set_index(['date','code']).verified_current_price_shape
    rows['verified_current_price_shape']=[lookup.loc[(r.date,r.code)] for r in rows.itertuples()]
    return rows[rows.frozen_original_nonword_identity.eq(0)].copy()


def apply_head(reference,raw,threshold,arm,config):
    if len(raw)!=len(reference) or not np.isfinite(raw).all():raise ValueError('same complete finite inference required')
    output=reference.copy();eligible=output.verified_current_price_shape.eq(0)
    output['arm']=arm;output['b_score']=raw;output['ranking_score']=raw
    output.loc[eligible,'score']=np.asarray(raw)[eligible]
    output.loc[eligible,'call']=np.where(np.asarray(raw)[eligible]>threshold,'PROMOTE','FAIL')
    for name in ['evaluation_call','recovered_B_call','aligned_evaluation_call']:output[name]=output.call
    output['same_source_aligned_score']=output.score
    output['selected_capacity_config']=config;output['decision_threshold']=threshold
    output['raw_score_not_event_probability']=True;output['event_probability_validated']=False
    output['research_validation_status']='FIXED_PARAMETER_RESEARCH_NOT_ADOPTED'
    output['classifier_head_kind']='HGB_CAPACITY_SELECTED_WITH_TEMPORAL_OOF'
    pd.testing.assert_frame_equal(output.loc[~eligible,['score','call']],reference.loc[~eligible,['score','call']],check_exact=True)
    return output


def prepare(tests_xml):
    if (ROOT/'protocol_BEFORE_ANY_FIT.json').exists():raise FileExistsError('one fixed parameter study only')
    import xml.etree.ElementTree as ET
    suites=list(ET.parse(tests_xml).getroot().iter('testsuite'))
    if not suites or any(int(x.get('failures',0))+int(x.get('errors',0)) for x in suites):raise ValueError('passing preflight required')
    last=b.read(PREVIOUS/'delivery/manual_request_verified_fixed_challenge_failed.json');b.verify(last['source_bindings'])
    data=joblib.load(PREVIOUS/'fixed_input_BEFORE_ANY_FIT.joblib')
    shape=joblib.load(source.ROOT/'prepared.joblib')['verified_shapes']
    checks=[];cached={}
    for plan in data['plans']:
        train=history.training_rows(data['full'],plan['cutoff'],'relay')
        if history._frame_digest(train,data['fields'])!=plan['training_digest']:raise ValueError('same252 training drift')
        for index,(past,valid) in enumerate(temporal.temporal_splits(train),1):
            model_path=PREVIOUS/'execution'/f"{plan['month']}_inner{index}_model.joblib"
            scores_path=PREVIOUS/'execution'/f"{plan['month']}_inner{index}_all_validation_scores_BEFORE_EFFECTS.csv"
            saved=joblib.load(model_path)
            expected=history._frame_digest(past,data['fields'])
            if saved['training_digest']!=expected or saved['features']!=data['fields']:raise ValueError('baseline inner cache drift')
            for path in [model_path,scores_path]:
                if str(path.resolve()) not in last['source_bindings'] or b.sha(path)!=last['source_bindings'][str(path.resolve())]:
                    raise ValueError('baseline raw model/OOF cache not originally bound')
                cached[str(path.resolve())]=b.sha(path)
            raw=pd.read_csv(scores_path,dtype={'code':str},float_precision='round_trip')
            pd.testing.assert_frame_equal(raw[['date','code','target_date','target','label_valid','frozen_original_nonword_identity']].reset_index(drop=True),
                valid[['date','code','target_date','target','label_valid','frozen_original_nonword_identity']].reset_index(drop=True),check_dtype=False,check_exact=True)
            checks.append(dict(month=plan['month'],block=index,training_days=past.date.nunique(),training_rows=len(past),
                max_train_target=past.target_date.max(),first_validation=valid.date.min(),validation_days=valid.date.nunique(),
                training_digest=expected,cached_model=str(model_path.resolve()),cached_raw_scores=str(scores_path.resolve())))
    ROOT.mkdir(parents=True,exist_ok=True)
    joblib.dump(data|dict(shape=shape),ROOT/'fixed_input_BEFORE_ANY_FIT.joblib',compress=3)
    pd.DataFrame(checks).to_csv(ROOT/'all48_cached_baseline_and_temporal_blocks.csv',index=False,encoding='utf-8-sig')
    sources=last['source_bindings']|cached|{str(path.resolve()):b.sha(path) for path in [Path(__file__),Path(tests_xml),
        b.ROOT/'tests/test_relay_parameter_research.py',REFERENCE,PREVIOUS/'fixed_input_BEFORE_ANY_FIT.joblib',
        source.ROOT/'prepared.joblib',ROOT/'fixed_input_BEFORE_ANY_FIT.joblib',ROOT/'all48_cached_baseline_and_temporal_blocks.csv']}
    b.write_new(ROOT/'protocol_BEFORE_ANY_FIT.json',dict(formed_at=b.now(),
        user_authorization='那就研究参数; new manual request, old platform goal usageLimited not changed',
        hypothesis='Existing depth3/leaf7/minleaf25/L2=3 may not suit corrected92 features; two fixed joint-capacity alternatives and strictly internal threshold selection may improve precision without excessive recall loss',
        baseline=training.PARAMS,capacity_candidates=CONFIGS,policy_arms=ARMS,threshold_candidates=THRESHOLDS,
        parameter_scope='Joint4-parameter capacity configurations, not attribution to any one parameter; lr/iterations/seed/weights/features/members/labels/source eligibility all fixed',
        temporal_selection='Three63-day inner blocks of same252 original mature signal-day window; inner train targets strictly before validation first signal; baseline48 inner scores/models reused exactly',
        capacity_selection='Original B OOF same denominator and source route, strict raw>.5; maximize F0.5=5TP/(5TP+FN+4FP); Recall>=original baseline-.025 and F1>=original baseline-.01; strict F0.5 gain required, otherwise BASE; new candidate tie COMPACT',
        threshold_selection='Same selected model OOF, fixed5-point raw threshold grid; guard against ORIGINAL BASE .5 (not selected model) to avoid stacked recall loss; strict F0.5 gain over selected config .5 or keep .5; tie nearest .5 then higher threshold, exact rational comparisons',
        labels='Only original known promotion labels, FN includes positive UNKNOWN; no role-label target; original daily identity complete retained',
        final_fit='After training-only monthly selection, fit only selected new capacity on original252 rows once; selected BASE reuses original final head. Never fit unselected final candidate',
        routing='Only verified_current_price_shape==0 uses own raw score/selected threshold; other routes exactly inherit frozen source head, including UNKNOWN',
        cached_inner_models=48,cached_final_models=16,maximum_new_inner_fits=96,maximum_new_final_fits=16,maximum_total_fits=112,
        fitting_budget_seconds=1800,repeat_failed_trials=False,outer_data_used_for_selection=False,
        acceptance=dict(pooled_FP_at_most_baseline_fraction=.9,pooled_Precision_gain_min=.03,pooled_F1_not_lower=True,
            each_historical_fold_Recall_loss_max=.025,each_historical_fold_FP_not_higher=True,each_historical_fold_F1_not_lower=True,
            pooled_Top3_full_slot_hits_not_lower=True),
        October_supplemental_only=True,current_target16_outcomes_never_read=True,
        historical_results_previously_seen_not_blind=True,late_source_original_PIT_not_certified=True,
        event_probability_validated=False,automatic_deployment=False,source_bindings=sources))
    return dict(frozen=True,new_fits=0,maximum_new_fits=112)


def evaluate(paired):
    records=[];rankings=[];selected=[]
    for period in temporal.PERIODS:
        frame=paired[paired.date.between(period['start'],period['end'])]
        for arm,part in frame.groupby('arm',sort=True):
            for scope,rows in [('FULL_POOL',part),('ORIGINAL_NONWORD',part[part.daily_one_word.eq(0)])]:
                view=rows.copy();view['call']=view.evaluation_call
                records.append(dict(period=period['id'],arm=arm,scope=scope,**evaluation.classification(view)))
            for k in [1,2,3]:
                stat,picks=source.full_slot_ranking(part,k);rankings.append(dict(period=period['id'],arm=arm,**stat))
                picks=picks.copy();picks['period']=period['id'];picks['k']=k;selected.append(picks)
    records=pd.DataFrame(records);ranks=pd.DataFrame(rankings);picks=pd.concat(selected,ignore_index=True)
    table=records[records.scope.eq('ORIGINAL_NONWORD')].set_index(['period','arm'])
    results={}
    for arm in ARMS:
        old=table.loc[('HISTORICAL_POOLED',BASE_ARM)];new=table.loc[('HISTORICAL_POOLED',arm)]
        guards=dict(pooled_FP_drop_at_least10percent=bool(new.fp<=old.fp*.9),
            pooled_Precision_gain_at_least3pp=bool(new.Precision>=old.Precision+.03),pooled_F1_not_lower=bool(new.F1>=old.F1))
        for fold in history.FOLDS:
            a=table.loc[(fold['id'],arm)];z=table.loc[(fold['id'],BASE_ARM)]
            guards[fold['id']+'_Recall_loss_within2p5pp']=bool(a.Recall>=z.Recall-.025)
            guards[fold['id']+'_FP_not_higher']=bool(a.fp<=z.fp)
            guards[fold['id']+'_F1_not_lower']=bool(a.F1>=z.F1)
        rank=ranks[ranks.period.eq('HISTORICAL_POOLED')&ranks.k.eq(3)].set_index('arm')
        guards['pooled_Top3_full_slot_hits_not_lower']=bool(rank.loc[arm,'hits']>=rank.loc[BASE_ARM,'hits'])
        results[arm]=dict(guards=guards,fixed_screen_pass=all(guards.values()))
    return dict(formed_at=b.now(),arms=results,production_deployed=False,event_probability_validated=False,not_blind=True),records,ranks,picks


def run():
    protocol=b.read(ROOT/'protocol_BEFORE_ANY_FIT.json');b.verify(protocol['source_bindings'])
    if protocol['capacity_candidates']!=CONFIGS or protocol['threshold_candidates']!=THRESHOLDS or protocol['baseline']!=training.PARAMS:
        raise ValueError('frozen parameter drift')
    b.write_new(ROOT/'experiment_STARTED.json',dict(formed_at=b.now(),protocol_sha256=b.sha(ROOT/'protocol_BEFORE_ANY_FIT.json')))
    output=ROOT/'execution';output.mkdir(exist_ok=False)
    data=joblib.load(ROOT/'fixed_input_BEFORE_ANY_FIT.joblib');full,fields,shape=data['full'],data['fields'],data['shape']
    refs=pd.read_csv(REFERENCE,dtype={'code':str},float_precision='round_trip',low_memory=False)
    baseline=refs[refs.arm.eq(BASE_ARM)].copy()
    start=time.monotonic();inner_fits=0;final_fits=0;parts=[];decisions=[];selection_table=[]
    def clock_guard():
        if time.monotonic()-start>1800:raise TimeoutError('fixed30-minute parameter budget reached')
    try:
        for plan in data['plans']:
            clock_guard();month=plan['month'];prefix=output/month
            window=history.training_rows(full,plan['cutoff'],'relay')
            if history._frame_digest(window,fields)!=plan['training_digest']:raise ValueError('same252 source window drift')
            oof_parts={'BASE':[],'COMPACT':[],'EXPANDED':[]}
            for index,(past,valid) in enumerate(temporal.temporal_splits(window),1):
                cached=PREVIOUS/'execution'/f'{month}_inner{index}_all_validation_scores_BEFORE_EFFECTS.csv'
                old=pd.read_csv(cached,dtype={'code':str},float_precision='round_trip')
                pd.testing.assert_frame_equal(old[['date','code','target','target_date']].reset_index(drop=True),
                    valid[['date','code','target','target_date']].reset_index(drop=True),check_dtype=False,check_exact=True)
                oof_parts['BASE'].append(nonword_oof(valid,old.raw_score.to_numpy(),shape))
                for name,params in CONFIGS.items():
                    clock_guard();key=Path(str(prefix)+f'_inner{index}_{name}')
                    b.write_new(str(key)+'_STARTED.json',dict(formed_at=b.now(),fit_number=inner_fits+1,params=params,
                        training_digest=history._frame_digest(past,fields),max_train_target=past.target_date.max(),first_validation=valid.date.min()))
                    model=Pipeline([('empty_training_columns',EmptyTrainingColumnGuard()),('hgb',HistGradientBoostingClassifier(**params))])
                    with threadpool_limits(limits=2):model.fit(past[fields],past.target.astype(int),hgb__sample_weight=original.weights(past,1.))
                    inner_fits+=1
                    joblib.dump(dict(model=model,features=fields,params=params,training_digest=history._frame_digest(past,fields),
                        cutoff=valid.date.min(),capacity=name),str(key)+'_model.joblib')
                    with threadpool_limits(limits=2):raw=model.predict_proba(valid[fields])[:,list(model.classes_).index(1)]
                    rows=nonword_oof(valid,raw,shape);oof_parts[name].append(rows)
                    rows.to_csv(str(key)+'_OOF_scores_BEFORE_SELECTION.csv',index=False,encoding='utf-8-sig')
                    b.write_new(str(key)+'_COMPLETED.json',dict(formed_at=b.now(),model_sha256=b.sha(str(key)+'_model.joblib'),validation_original_B=len(rows)))
            oof={name:pd.concat(value,ignore_index=True) for name,value in oof_parts.items()}
            for name,rows in oof.items():
                if rows.target_date.max()>=plan['cutoff']:raise ValueError('late inner target used for selection')
                pd.testing.assert_frame_equal(rows.drop(columns='raw_score'),oof['BASE'].drop(columns='raw_score'),check_exact=True)
                rows.to_csv(str(prefix)+f'_{name}_complete_OOF_BEFORE_SELECTION.csv',index=False,encoding='utf-8-sig')
            stats={name:decision_statistics(rows,.5) for name,rows in oof.items()}
            selected=select_capacity(stats)
            threshold_stats={t:decision_statistics(oof[selected],t) for t in THRESHOLDS}
            threshold=select_threshold(threshold_stats,stats['BASE'])
            for name,stat in stats.items():selection_table.append(dict(month=month,phase='CAPACITY',config=name,**stat))
            for t,stat in threshold_stats.items():selection_table.append(dict(month=month,phase='THRESHOLD',config=selected,**stat))
            selection=dict(formed_at=b.now(),month=month,cutoff=plan['cutoff'],selected_capacity=selected,selected_threshold=threshold,
                capacity_statistics=stats,threshold_statistics={str(k):v for k,v in threshold_stats.items()},
                max_OOF_target=oof[selected].target_date.max(),outer_scores_or_labels_read_for_selection=False)
            b.write_new(str(prefix)+'_SELECTION_BEFORE_FINAL_FIT_AND_OUTER_SCORE.json',selection)
            if selected=='BASE':
                model_path=Path(plan['final_model_path']);saved=joblib.load(model_path);model=saved['model']
                if b.sha(model_path)!=plan['final_model_sha256']:raise ValueError('old final source model drift')
                new_fit=False
            else:
                clock_guard();new_fit=True;model_path=Path(str(prefix)+'_selected_FINAL_model.joblib')
                params=CONFIGS[selected]
                b.write_new(str(prefix)+'_selected_FINAL_STARTED.json',dict(formed_at=b.now(),capacity=selected,
                    params=params,training_digest=plan['training_digest'],fit_number=final_fits+1))
                model=Pipeline([('empty_training_columns',EmptyTrainingColumnGuard()),('hgb',HistGradientBoostingClassifier(**params))])
                with threadpool_limits(limits=2):model.fit(window[fields],window.target.astype(int),hgb__sample_weight=original.weights(window,1.))
                final_fits+=1
                joblib.dump(dict(model=model,features=fields,cutoff=plan['cutoff'],capacity=selected,params=params,training_digest=plan['training_digest']),model_path)
            test=full[full.date.str[:7].eq(month)].copy()
            with threadpool_limits(limits=2):raw=model.predict_proba(test[fields])[:,list(model.classes_).index(1)]
            reference=baseline.set_index(['date','code']).loc[pd.MultiIndex.from_frame(test[['date','code']])].reset_index()
            if selected=='BASE':np.testing.assert_allclose(raw,reference.b_score,atol=1e-14,rtol=0)
            for arm,t in zip(ARMS,[.5,threshold]):
                prediction=apply_head(reference,raw,t,arm,selected)
                prediction['model_cutoff']=plan['cutoff']
                prediction.to_csv(str(prefix)+f'_{arm}_predictions_BEFORE_EVALUATION.csv',index=False,encoding='utf-8-sig')
                parts.append(prediction)
            selection.update(selected_final_new_fit=new_fit,selected_final_model_path=str(model_path.resolve()),
                selected_final_model_sha256=b.sha(model_path),inner_fits_this_month=6)
            b.write_new(str(prefix)+'_COMPLETED.json',selection);decisions.append(selection)
            print(month,selected,'threshold',threshold,'inner',inner_fits,'final',final_fits,flush=True)
        new=pd.concat(parts,ignore_index=True)
        if inner_fits!=96 or final_fits>16 or len(new)!=7554:raise ValueError('incomplete bounded study')
        all_heads=pd.concat([refs,new],ignore_index=True)
        pd.testing.assert_frame_equal(all_heads.iloc[:30216][refs.columns],refs,check_exact=True,check_dtype=False)
        all_heads.to_csv(output/'all37770_predictions_BEFORE_EVALUATION.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(selection_table).to_csv(output/'all_training_only_capacity_and_threshold_metrics.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame([{k:v for k,v in d.items() if k not in ['capacity_statistics','threshold_statistics']} for d in decisions]).to_csv(output/'all16_training_only_selections.csv',index=False,encoding='utf-8-sig')
        paired=pd.concat([baseline,new],ignore_index=True)
        result,stats,ranks,picks=evaluate(paired)
        stats.to_csv(output/'same_pool_full_classification_metrics.csv',index=False,encoding='utf-8-sig')
        ranks.to_csv(output/'full_slot_Top1Top2Top3.csv',index=False,encoding='utf-8-sig')
        picks.to_csv(output/'all_selected_rows_WITH_UNKNOWN.csv',index=False,encoding='utf-8-sig')
        current=new[new.date.eq('2026-10-15')]
        if len(current)!=12 or current.target.notna().any() or current.label_valid.fillna(False).any():raise ValueError('saved current6 per arm must stay unknown')
        current.to_csv(output/'current15_six_each_policy_target16_UNMATURED.csv',index=False,encoding='utf-8-sig')
        result.update(inner_new_fits=inner_fits,selected_final_new_fits=final_fits,total_new_fits=inner_fits+final_fits,
            old_final_models_refitted=0,cached_baseline_inner_reused=48,complete_predictions=len(all_heads),
            elapsed_seconds=time.monotonic()-start,source_bindings=protocol['source_bindings']|{str(p.resolve()):b.sha(p) for p in output.iterdir() if p.is_file()})
        b.verify(result['source_bindings']);b.write_new(output/'parameter_fixed_result.json',result)
        return dict(new_fits=inner_fits+final_fits,final_fits=final_fits,screens={k:v['fixed_screen_pass'] for k,v in result['arms'].items()},elapsed_seconds=result['elapsed_seconds'])
    except Exception as exc:
        b.write_new(ROOT/'FAILURE_STOP.json',dict(formed_at=b.now(),type=type(exc).__name__,error=str(exc),
            completed_inner_fits=inner_fits,completed_final_fits=final_fits,retry_authorized=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','run']);parser.add_argument('--tests-xml')
    args=parser.parse_args();print(prepare(args.tests_xml) if args.command=='prepare' else run())
