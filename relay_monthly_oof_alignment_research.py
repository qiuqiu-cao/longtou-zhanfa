"""Fixed study of monthly252-day prequential OOF and unchanged selected meta.

This is a distinct learning-procedure consistency hypothesis, not a retuning of
the failed filter. Immutable first-stage deployment heads are reused; only six
earlier monthly heads may need to be fitted to reconstruct honest OOF history.
"""
from pathlib import Path
import argparse
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

import relay_positive_filter_research as old
import short_term_thinking_dataset_builder as b
import short_term_thinking_research as training
import expectation_345_research as weights
import model_upgrade_history as history
import relay_source_feature_research as source
import relay_temporal_score_calibration as temporal
from short_term_thinking_preprocessing import EmptyTrainingColumnGuard

ROOT=b.ROOT/'research/model_upgrade_validation/rolling_oof_alignment_20261016T093022Z'
MECHANISM=b.ROOT/'research/model_upgrade_validation/degradation_attribution_20261016T093022Z'
PRIOR=old.ROOT/'delivery/study_delivery_verified_NO_PRODUCTION_CHANGE.json'
REFERENCE=old.ROOT/'execution/all49101_predictions_BEFORE_EVALUATION.csv'
ARMS=['MONTHLY252_OOF_FROZEN_SCORE_FILTER','MONTHLY252_OOF_FROZEN_INDIVIDUAL_FILTER']
KIND=dict(zip(ARMS,old.ARMS))


def month_cutoff(date):
    return date[:7]+'-01'


def plans(data):
    answer=[]
    for plan in data['plans']:
        window=history.training_rows(data['full'],plan['cutoff'],'relay')
        if history._frame_digest(window,data['fields'])!=plan['training_digest']:
            raise ValueError('unchanged outer252 window required')
        splits=temporal.temporal_splits(window)
        dates=[list(valid.date.unique()) for _,valid in splits]
        months=sorted({month_cutoff(d) for days in dates for d in days})
        qualification={cutoff:history.training_rows(data['full'],cutoff,'relay').date.nunique() for cutoff in months}
        answer.append(dict(month=plan['month'],cutoff=plan['cutoff'],validation_dates=dates,
            source_month_cutoffs=months,source_qualified_days=qualification,
            eligible=all(n==252 for n in qualification.values()),
            first_third_signal=min(dates[2]),original_training_digest=plan['training_digest']))
    return answer


def fixed_month_training(full,cutoff):
    rows=history.training_rows(full,cutoff,'relay')
    if rows.date.nunique()!=252 or rows.target_date.max()>=cutoff:
        raise ValueError('exact252 mature dates before natural month start required')
    return rows


def attach_inference(reference,full):
    out=reference.copy();out['base_call']=out.evaluation_call
    lookup=full.set_index(old.KEYS).loc[pd.MultiIndex.from_frame(out[old.KEYS])]
    np.testing.assert_array_equal(out.boards.to_numpy(),lookup.boards.to_numpy())
    for column in old.INDIVIDUAL:
        out[column]=lookup[column].to_numpy()
    out['base_score_logit']=old.logit(out.b_score)
    return out


def assemble_oof(full,shape,plan,source_scores):
    parts=[]
    lookup=source_scores.set_index(old.KEYS)
    for index,days in enumerate(plan['validation_dates'],1):
        rows=full[full.date.isin(days)].copy()
        if rows.duplicated(old.KEYS).any():raise ValueError('duplicate original full-pool input')
        got=lookup.loc[pd.MultiIndex.from_frame(rows[old.KEYS])]
        rows['raw_score']=got.raw_score.to_numpy()
        rows['first_stage_model_cutoff']=got.first_stage_model_cutoff.to_numpy()
        expected=rows.date.map(month_cutoff)
        if not rows.first_stage_model_cutoff.eq(expected).all():
            raise ValueError('OOF first-stage refresh must match deployment month-start procedure')
        rows=rows.merge(shape[old.KEYS+['verified_current_price_shape']],on=old.KEYS,validate='one_to_one')
        rows['daily_one_word']=rows.frozen_original_nonword_identity
        rows=rows[rows.daily_one_word.eq(0)].copy()
        if rows.verified_current_price_shape.eq(1).any():
            raise ValueError('unregistered A-route in original OOF B')
        rows['base_call']=np.where(rows.verified_current_price_shape.eq(0),
            np.where(rows.raw_score.gt(.5),'PROMOTE','FAIL'),'UNKNOWN')
        rows['call']=rows.base_call
        rows['ranking_score']=rows.raw_score
        rows['base_score_logit']=old.logit(rows.raw_score)
        rows['inner_block']=index
        parts.append(rows)
    answer=pd.concat(parts,ignore_index=True)
    if answer.duplicated(old.KEYS).any():raise ValueError('monthly OOF reused twice inside one outer month')
    return answer


def prepare(tests_xml):
    if (ROOT/'protocol_BEFORE_ANY_FIT.json').exists():raise FileExistsError('one fixed alignment study')
    import xml.etree.ElementTree as ET
    suites=list(ET.parse(tests_xml).getroot().iter('testsuite'))
    if not suites or any(int(s.get('errors',0))+int(s.get('failures',0)) for s in suites):
        raise ValueError('passing targeted preflight required')
    prior=b.read(PRIOR);old.verify(prior['source_bindings'])
    data=joblib.load(old.ROOT/'fixed_input_BEFORE_ANY_FIT.joblib')
    allocation=plans(data)
    needed=sorted({c for p in allocation if p['eligible'] for c in p['source_month_cutoffs']})
    existing={p['cutoff']:p for p in data['plans']}
    sources=prior['source_bindings'].copy();source_plans=[]
    for cutoff in needed:
        rows=fixed_month_training(data['full'],cutoff)
        dig=history._frame_digest(rows,data['fields'])
        item=dict(cutoff=cutoff,month=cutoff[:7],training_digest=dig,training_rows=len(rows),
            training_days=252,max_target=rows.target_date.max(),reuse=cutoff in existing)
        if item['reuse']:
            original=existing[cutoff]
            if original['training_digest']!=dig:raise ValueError('existing model same252 input mismatch')
            path=Path(original['final_model_path'])
            if b.sha(path)!=original['final_model_sha256']:raise ValueError('protected SOURCE head changed')
            item.update(model_path=str(path),model_sha256=b.sha(path))
        source_plans.append(item)
    ROOT.mkdir(parents=True,exist_ok=True)
    b.write_new(ROOT/'all16_outer_and_monthly_source_plans.json',dict(outer=allocation,source=source_plans))
    joblib.dump(data|dict(alignment_plans=allocation,monthly_source_plans=source_plans),ROOT/'fixed_input_BEFORE_ANY_FIT.joblib',compress=3)
    evidence=[MECHANISM/'meta_mechanism_audit/meta_mechanism_delivery_verified.json',
              MECHANISM/'score_shift_audit/score_shift_audit_BINDINGS.json']
    for path in evidence:
        receipt=b.read(path)
        binding=receipt.get('source_bindings',{})|receipt.get('output_bindings',{})
        if not binding:raise ValueError('source-bound mechanism evidence required before new protocol')
        old.verify(binding);sources.update(binding);sources[str(path.resolve())]=b.sha(path)
    for path in [Path(__file__),b.ROOT/'tests/test_relay_monthly_oof_alignment_research.py',
                 Path(tests_xml),PRIOR,REFERENCE,ROOT/'fixed_input_BEFORE_ANY_FIT.joblib',
                 ROOT/'all16_outer_and_monthly_source_plans.json']:
        sources[str(path.resolve())]=b.sha(path)
    b.write_new(ROOT/'protocol_BEFORE_ANY_FIT.json',dict(formed_at=b.now(),
        user_goal='找出退化的原因，从多轮的研究中找出问题的关键，对模型进行升级研究',
        hypothesis='Match first-stage OOF to natural-month deployment252-day procedure and preserve selected meta head so its threshold is not transferred to a refitted score scale',
        arms=ARMS,head_kind=KIND,features=old.FEATURES,meta_estimator=old.PARAMS,
        first_stage_estimator=training.PARAMS,first_stage_weights='Exact original.weights(train,1) balanced class mass; same full known training pool',
        meta_weights='Every row1; no class balancing; exact previous fixed Logistic',
        fixed_thresholds=old.THRESHOLDS,selection='Previous fixed exact Precision/Recall5pp/F1/FP guard, no new search',
        validation='Same outer252-date window split4x63; last3 blocks are genuine monthly252 pretrained OOF; prefix meta on first2 blocks with target strictly before third first day, third selects threshold; freeze meta and threshold for outer month, no final meta refit',
        warmup='Any required OOF natural-month start with fewer than252 mature signal dates makes entire outer month ineligible; inherit SOURCE/no-filter, retain all denominator',
        first_stage_new_fits_max=sum(not p['reuse'] for p in source_plans),
        meta_new_fits_max=2*sum(p['eligible'] for p in allocation),
        total_new_fits_max=sum(not p['reuse'] for p in source_plans)+2*sum(p['eligible'] for p in allocation),
        first_stage_deployment_refits=0,old_failed_models_retrained=False,fitting_budget_seconds=1800,
        primary_acceptance=dict(pooled_Precision_gain_min=.03,pooled_FP_fraction_max=.9,pooled_F1_not_lower=True,
            each_historical_Recall_loss_max=.05,each_historical_FP_not_higher=True,each_historical_F1_not_lower=True,
            pooled_positive_Top3_hits_not_lower=True),
        attribution='Joint training-procedure consistency change; does not uniquely identify training-size vs refresh-age or missing-information causes',
        history_previously_seen_not_blind=True,late_source_not_original_PIT=True,
        October_supplemental_only=True,current_target16_new_outcomes_read=False,
        no_post_effect_retry=True,event_probability_validated=False,deployed=False,source_bindings=sources))
    return dict(protocol_frozen=True,eligible_outer_months=sum(p['eligible'] for p in allocation),
        new_source_fits=sum(not p['reuse'] for p in source_plans),max_meta_fits=2*sum(p['eligible'] for p in allocation))


def run():
    protocol=b.read(ROOT/'protocol_BEFORE_ANY_FIT.json');old.verify(protocol['source_bindings'])
    if protocol['arms']!=ARMS or protocol['features']!=old.FEATURES or protocol['fixed_thresholds']!=old.THRESHOLDS:
        raise ValueError('fixed alignment schema drift')
    b.write_new(ROOT/'experiment_STARTED.json',dict(formed_at=b.now(),protocol_sha256=b.sha(ROOT/'protocol_BEFORE_ANY_FIT.json')))
    out=ROOT/'execution';out.mkdir(exist_ok=False)
    data=joblib.load(ROOT/'fixed_input_BEFORE_ANY_FIT.joblib');full,fields=data['full'],data['fields']
    refs=pd.read_csv(REFERENCE,dtype={'code':str},float_precision='round_trip',low_memory=False)
    baseline=refs[refs.arm.eq(source.ARM)].copy()
    source_frames=[];counts=dict(first_stage_backfill_fits=0,reused_source_models=0,meta_fits=0)
    start=time.monotonic()
    def budget():
        if time.monotonic()-start>1800:raise TimeoutError('fixed30minute budget')
    try:
        for plan in data['monthly_source_plans']:
            budget();month=plan['month'];train=fixed_month_training(full,plan['cutoff'])
            if history._frame_digest(train,fields)!=plan['training_digest']:raise ValueError('monthly source digest drift')
            if plan['reuse']:
                if b.sha(plan['model_path'])!=plan['model_sha256']:raise ValueError('immutable deployment head changed')
                saved=joblib.load(plan['model_path']);model=saved['model'];counts['reused_source_models']+=1
                model_path=Path(plan['model_path'])
            else:
                b.write_new(out/f'{month}_SOURCE_BACKFILL_FIT_STARTED.json',dict(formed_at=b.now(),plan=plan,estimator=training.PARAMS))
                model=Pipeline([('empty_training_columns',EmptyTrainingColumnGuard()),('hgb',HistGradientBoostingClassifier(**training.PARAMS))])
                with threadpool_limits(limits=2):
                    model.fit(train[fields],train.target.astype(int),hgb__sample_weight=weights.weights(train,1.))
                counts['first_stage_backfill_fits']+=1
                model_path=out/f'{month}_SOURCE_BACKFILL_model.joblib'
                joblib.dump(dict(model=model,features=fields,training_digest=plan['training_digest'],
                    cutoff=plan['cutoff'],estimator=training.PARAMS,negative_cost=1.,trained_at=b.now()),model_path,compress=3)
                b.write_new(out/f'{month}_SOURCE_BACKFILL_FIT_COMPLETED.json',dict(completed_at=b.now(),plan=plan,
                    model_path=str(model_path.resolve()),model_sha256=b.sha(model_path),n_iter=model.named_steps['hgb'].n_iter_))
            rows=full[full.date.str[:7].eq(month)].copy()
            with threadpool_limits(limits=2):raw=model.predict_proba(rows[fields])[:,list(model.classes_).index(1)]
            part=rows[old.KEYS].copy();part['raw_score']=raw;part['first_stage_model_cutoff']=plan['cutoff']
            part.to_csv(out/f'{month}_monthly252_OOF_raw_BEFORE_META.csv',index=False,encoding='utf-8-sig');source_frames.append(part)
            b.write_new(out/f'{month}_SOURCE_SCORES_COMPLETED.json',dict(completed_at=b.now(),plan=plan,
                model_path=str(model_path.resolve()),model_sha256=b.sha(model_path),rows=len(part)))
        all_source=pd.concat(source_frames,ignore_index=True)
        if all_source.duplicated(old.KEYS).any():raise ValueError('monthly SOURCE OOF duplicate')
        all_source.to_csv(out/'all_monthly252_source_scores_BEFORE_META.csv',index=False,encoding='utf-8-sig')
        outputs=[];choices=[]
        for plan in data['alignment_plans']:
            budget();month=plan['month']
            reference=attach_inference(baseline[baseline.date.str[:7].eq(month)],full)
            if plan['eligible']:
                oof=assemble_oof(full,data['shapes'],plan,all_source)
                oof.to_csv(out/f'{month}_complete_monthly252_OOF_BEFORE_META.csv',index=False,encoding='utf-8-sig')
                prefix=old.select_meta_rows(oof[oof.inner_block.lt(3)],plan['first_third_signal'])
                prefix.to_csv(out/f'{month}_frozen_meta_training.csv',index=False,encoding='utf-8-sig')
                validation=oof[oof.inner_block.eq(3)].copy()
            monthly_choices={};fitted={}
            for arm in ARMS:
                kind=KIND[arm]
                if plan['eligible']:
                    model,fit=old.fit_head(prefix,kind,out,month,'ALIGNED_FROZEN_PREFIX')
                    counts['meta_fits']+=fit['new_fits']
                    validation_score=old.scores(model,validation,kind)
                    stats={threshold:old.statistics(old.apply_filter(validation,validation_score,threshold)) for threshold in old.THRESHOLDS}
                    threshold=old.choose_threshold(stats) if model is not None else 0.
                    choice=dict(threshold=threshold,fit=fit,statistics={str(k):v for k,v in stats.items()},
                        max_training_target=prefix.target_date.max(),first_third_signal=plan['first_third_signal'])
                else:
                    model=None;choice=dict(threshold=0.,reason='INSUFFICIENT_MONTHLY252_OOF_WARMUP',source_qualified_days=plan['source_qualified_days'])
                monthly_choices[arm]=choice;fitted[arm]=model
            selection=dict(formed_at=b.now(),month=month,eligible=plan['eligible'],choices=monthly_choices,
                           outer_results_used_for_selection=False,final_meta_refit=False)
            b.write_new(out/f'{month}_SELECTION_BEFORE_OUTER_SCORE.json',selection)
            for arm in ARMS:
                kind=KIND[arm];threshold=monthly_choices[arm]['threshold']
                score=old.scores(fitted[arm],reference,kind)
                result=old.apply_filter(reference,score,threshold)
                result['arm']=arm;result['research_validation_status']='FIXED_ALIGNMENT_RESEARCH_NOT_ADOPTED'
                result['second_stage_head_kind']='MATCHED_MONTHLY252_OOF_SELECTED_META_FROZEN'
                for name in ['aligned_evaluation_call','recovered_B_call']:
                    if name in result:result[name]=result.research_call
                result.to_csv(out/f'{month}_{arm}_predictions_BEFORE_EVALUATION.csv',index=False,encoding='utf-8-sig')
                outputs.append(result)
                choices.append(dict(month=month,arm=arm,eligible=plan['eligible'],threshold=threshold,
                    meta_refitted_after_selection=False,outer_results_used_for_selection=False))
            print(month,'qualified',plan['eligible'],'thresholds',{k:v['threshold'] for k,v in monthly_choices.items()},flush=True)
        new=pd.concat(outputs,ignore_index=True)
        if any(len(f)!=3777 for _,f in new.groupby('arm')):raise ValueError('both complete3777 new heads required')
        all_heads=pd.concat([refs,new],ignore_index=True)
        pd.testing.assert_frame_equal(all_heads.iloc[:49101][refs.columns],refs,check_dtype=False,check_exact=True)
        all_heads.to_csv(out/'all56655_predictions_BEFORE_EVALUATION.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(choices).to_csv(out/'all32_training_only_selections.csv',index=False,encoding='utf-8-sig')
        current=new[new.date.eq('2026-10-15')]
        if len(current)!=12 or current.target.notna().any() or current.label_valid.fillna(False).any():raise ValueError('six unknown per new head must remain')
        current.to_csv(out/'current15_six_per_arm_UNMATURED.csv',index=False,encoding='utf-8-sig')
        if sum(counts[k] for k in ['first_stage_backfill_fits','meta_fits'])>protocol['total_new_fits_max']:raise ValueError('fit budget exceeded')
        old.verify(protocol['source_bindings'])
        b.write_new(out/'run_COMPLETED_BEFORE_EVALUATION.json',dict(completed_at=b.now(),**counts,
            first_stage_deployment_refits=0,complete_predictions=len(all_heads),elapsed_seconds=time.monotonic()-start,
            source_bindings=protocol['source_bindings']|{str(p.resolve()):b.sha(p) for p in out.iterdir() if p.is_file()}))
        return counts
    except Exception as exc:
        b.write_new(ROOT/'FAILURE_STOP.json',dict(formed_at=b.now(),error_type=type(exc).__name__,error=str(exc),counts=counts,retry_authorized=False));raise


def evaluate():
    done=b.read(ROOT/'execution/run_COMPLETED_BEFORE_EVALUATION.json');old.verify(done['source_bindings'])
    out=ROOT/'evaluation';out.mkdir(exist_ok=False)
    frame=pd.read_csv(ROOT/'execution/all56655_predictions_BEFORE_EVALUATION.csv',dtype={'code':str},float_precision='round_trip',low_memory=False)
    rows=frame[frame.arm.isin(['BASE','ST_DUAL',source.ARM]+ARMS)].copy()
    records=[];recs=[];ranks=[];picks=[]
    for period in temporal.PERIODS:
        for arm,part in rows[rows.date.between(period['start'],period['end'])].groupby('arm',sort=True):
            part=part.copy();part['call']=part.evaluation_call
            for scope,group in [('FULL_POOL',part),('ORIGINAL_NONWORD',part[part.daily_one_word.eq(0)])]:
                records.append(dict(period=period['id'],arm=arm,scope=scope,**old.statistics(group)))
            stat,selected=old.positive_top3(part);recs.append(dict(period=period['id'],arm=arm,**stat))
            picks.append(selected.assign(period=period['id'],selected_arm=arm))
            for k in [1,2,3]:
                stat,_=source.full_slot_ranking(part,k);ranks.append(dict(period=period['id'],arm=arm,**stat))
    stats=pd.DataFrame(records);recommendations=pd.DataFrame(recs)
    stats.to_csv(out/'complete_same_pool_classification.csv',index=False,encoding='utf-8-sig')
    recommendations.to_csv(out/'positive_recommendation_Top3_FULL_SLOTS.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(ranks).to_csv(out/'raw_observation_Top1Top2Top3_INVARIANT.csv',index=False,encoding='utf-8-sig')
    pd.concat(picks,ignore_index=True).to_csv(out/'all_positive_recommendations_WITH_UNKNOWN.csv',index=False,encoding='utf-8-sig')
    filtered=rows[rows.arm.isin(ARMS)&rows.filter_applied.eq(True)].copy()
    known=filtered.label_valid.fillna(False).astype(bool)&filtered.target.notna()
    filtered['filter_effect']=np.where(~known,'UNKNOWN_OUTCOME',np.where(filtered.target.eq(1),'NEW_MISSED_PROMOTION','REMOVED_FALSE_POSITIVE'))
    filtered.to_csv(out/'all_filtered_candidates_GAINS_LOSSES_UNKNOWN.csv',index=False,encoding='utf-8-sig')
    mapped=rows[rows.arm.isin([source.ARM]+ARMS)].copy();mapped['arm']=mapped.arm.replace(KIND)
    ci=old.bootstrap_intervals(mapped);ci['arm']=ci.arm.replace({v:k for k,v in KIND.items()})
    ci.to_csv(out/'exploratory_5day_block_intervals.csv',index=False,encoding='utf-8-sig')
    table=stats[stats.scope.eq('ORIGINAL_NONWORD')].set_index(['period','arm']);acceptance={}
    for arm in ARMS:
        z=table.loc[('HISTORICAL_POOLED',source.ARM)];a=table.loc[('HISTORICAL_POOLED',arm)]
        guards=dict(pooled_Precision_gain3pp=bool(a.Precision>=z.Precision+.03),
            pooled_FP_drop10percent=bool(a.fp*10<=z.fp*9),pooled_F1_not_lower=bool(a.F1>=z.F1))
        for fold in history.FOLDS:
            z=table.loc[(fold['id'],source.ARM)];a=table.loc[(fold['id'],arm)]
            guards[fold['id']+'_Recall_loss_within5pp']=bool(a.Recall>=z.Recall-.05)
            guards[fold['id']+'_FP_not_higher']=bool(a.fp<=z.fp)
            guards[fold['id']+'_F1_not_lower']=bool(a.F1>=z.F1)
        rec=recommendations[recommendations.period.eq('HISTORICAL_POOLED')].set_index('arm')
        guards['pooled_positive_Top3_hits_not_lower']=bool(rec.loc[arm,'hits']>=rec.loc[source.ARM,'hits'])
        acceptance[arm]=dict(guards=guards,fixed_screen_pass=all(guards.values()))
    result=dict(formed_at=b.now(),arms=acceptance,**{k:done[k] for k in ['first_stage_backfill_fits','meta_fits','reused_source_models','first_stage_deployment_refits']},
        history_previously_seen_not_blind=True,event_probability_validated=False,deployed=False,no_post_effect_retry=True,
        source_bindings=done['source_bindings']|{str(p.resolve()):b.sha(p) for p in out.iterdir() if p.is_file()})
    old.verify(result['source_bindings']);b.write_new(out/'fixed_study_result.json',result)
    return {k:v['fixed_screen_pass'] for k,v in acceptance.items()}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','run','evaluate']);parser.add_argument('--tests-xml')
    args=parser.parse_args();print(prepare(args.tests_xml) if args.command=='prepare' else run() if args.command=='run' else evaluate())
