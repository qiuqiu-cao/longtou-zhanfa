"""A fixed chronological study of saved boosting prefixes, never a refit.

Select30/60/120 solely on the original baseline's inner OOF rows. Final scoring
uses the corresponding prefix of the original252-day model, threshold0.5.
"""
from pathlib import Path
import argparse
import time
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import short_term_thinking_dataset_builder as b
import model_upgrade_history as history
import relay_parameter_research as parameter
import relay_temporal_score_calibration as temporal

ROOT=b.ROOT/'research/model_upgrade_validation/iteration_research_20261016T043944Z'
REFERENCE=parameter.ROOT/'execution/all37770_predictions_BEFORE_EVALUATION.csv'
ARM='PARAM_BASE_OOF_SELECTED_BOOSTING_PREFIX'
STAGES=[30,60,120]


def prefix_scores(pipeline,features,stages=STAGES):
    if not stages or stages!=sorted(set(stages)) or min(stages)<1:
        raise ValueError('ordered positive unique stages required')
    hgb=pipeline.named_steps['hgb']
    if max(stages)>hgb.n_iter_:raise ValueError('cannot extrapolate past saved boosting rounds')
    transformed=pipeline[:-1].transform(features)
    wanted=set(stages);result={}
    with threadpool_limits(limits=2):
        for index,values in enumerate(hgb.staged_predict_proba(transformed),1):
            if index in wanted:
                score=values[:,list(hgb.classes_).index(1)].copy()
                if not np.isfinite(score).all() or ((score<0)|(score>1)).any():raise ValueError('invalid saved prefix scores')
                result[index]=score
            if index==max(stages):break
    if set(result)!=wanted:raise ValueError('saved estimator yielded too few rounds')
    return result


def select_round(statistics):
    baseline=statistics[120]
    eligible=[stage for stage in [30,60] if parameter.guard(statistics[stage],baseline)
        and parameter.score_fraction(statistics[stage])>parameter.score_fraction(baseline)]
    # Equal improvements use the earlier stage; exact equality to120 retains120.
    return max(eligible,key=lambda stage:parameter.score_fraction(statistics[stage])) if eligible else 120


def prepare(tests_xml):
    if (ROOT/'protocol_BEFORE_ANY_EFFECT.json').exists():raise FileExistsError('one fixed prefix study only')
    import xml.etree.ElementTree as ET
    suites=list(ET.parse(tests_xml).getroot().iter('testsuite'))
    if not suites or any(int(x.get('errors',0))+int(x.get('failures',0)) for x in suites):raise ValueError('passing prefix tests required')
    previous_path=parameter.ROOT/'execution/parameter_fixed_result.json'
    previous=b.read(previous_path);b.verify(previous['source_bindings'])
    diagnostic_path=parameter.ROOT/'training_diagnostics/saved_training_diagnostic_receipt.json'
    diagnostic=b.read(diagnostic_path);b.verify(diagnostic['source_bindings'])
    data=joblib.load(parameter.ROOT/'fixed_input_BEFORE_ANY_FIT.joblib')
    checks=[]
    for plan in data['plans']:
        window=history.training_rows(data['full'],plan['cutoff'],'relay')
        if history._frame_digest(window,data['fields'])!=plan['training_digest']:raise ValueError('fixed252 source window drift')
        for index,(past,valid) in enumerate(temporal.temporal_splits(window),1):
            path=temporal.ROOT/'execution'/f"{plan['month']}_inner{index}_model.joblib"
            saved=joblib.load(path)
            if saved['training_digest']!=history._frame_digest(past,data['fields']):raise ValueError('cached original baseline mismatch')
            if saved['model'].named_steps['hgb'].n_iter_!=120:raise ValueError('exact saved120round baseline required')
            checks.append(dict(month=plan['month'],block=index,training_digest=saved['training_digest'],
                max_train_target=past.target_date.max(),first_validation=valid.date.min(),saved_model=str(path.resolve()),sha256=b.sha(path)))
    ROOT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(checks).to_csv(ROOT/'all48_original_inner_models_BEFORE_EFFECTS.csv',index=False,encoding='utf-8-sig')
    sources=previous['source_bindings']|{str(path.resolve()):b.sha(path) for path in [Path(__file__),Path(tests_xml),
        b.ROOT/'tests/test_relay_iteration_research.py',previous_path,diagnostic_path,
        parameter.ROOT/'fixed_input_BEFORE_ANY_FIT.joblib',REFERENCE,ROOT/'all48_original_inner_models_BEFORE_EFFECTS.csv']}
    b.write_new(ROOT/'protocol_BEFORE_ANY_EFFECT.json',dict(formed_at=b.now(),
        user_authorization='继续研究, continuing the explicitly authorized parameter study; old platform goal not modified',
        hypothesis='Fixed120 boosting rounds might retain excess training fit; selecting an earlier saved prefix using only chronological OOF may improve precision without unacceptable missed positives',
        evidence='Prior capacity study expanded model fit improves training loss but worsens39/48 OOF blocks; lr and120 rounds were unchanged, iteration not yet tested',
        scope='Only boosting prefix30/60/120 of saved original baseline; no capacity, class-weight, learning-rate, threshold, data, label or source changes',
        candidates=STAGES,threshold=.5,new_fits=0,model_mutation=False,
        inner='Same48 baseline inner models and252 windows, strict target maturity, original frozen B membership and verified0 route',
        selection='Strictly higher rational F0.5 than120, Recall>=120-.025, F1>=120-.01; otherwise120; tie30 over60. Original full denominator and positive unknown-source rows retained as missed, no outer outcomes',
        final='Same16 original saved final source heads, public staged_predict_proba prefix only; last120round output must reproduce frozen original score',
        ranking='Use selected raw prefix scores, event score uncalibrated; all nonverified0 score/call inherit old source unchanged',
        acceptance=dict(pooled_FP_max_baseline_fraction=.9,pooled_Precision_gain_min=.03,pooled_F1_not_lower=True,
                 each_historical_fold_Recall_loss_max=.025,each_historical_fold_FP_not_higher=True,
                 each_historical_fold_F1_not_lower=True,pooled_Top3_full_slot_hits_not_lower=True),
        read_only_phase_budget_seconds=1800,repeat_previous_trials=False,outer_results_not_used_for_selection=True,
        history_previously_seen_not_blind=True,late_source_original_PIT_not_certified=True,
        target16_results_not_read=True,automatic_deployment=False,event_probability_validated=False,
        source_bindings=sources))
    return dict(frozen=True,new_fits=0,models_reused=64,prefixes=STAGES)


def evaluate(paired):
    # Evaluate the one declared candidate and the same qualified baseline.
    records=[];rankings=[];selected=[]
    for period in temporal.PERIODS:
        sample=paired[paired.date.between(period['start'],period['end'])]
        for arm,part in sample.groupby('arm',sort=True):
            for scope,rows in [('FULL_POOL',part),('ORIGINAL_NONWORD',part[part.daily_one_word.eq(0)])]:
                view=rows.copy();view['call']=view.evaluation_call
                records.append(dict(period=period['id'],arm=arm,scope=scope,**parameter.evaluation.classification(view)))
            for k in [1,2,3]:
                stat,picks=parameter.source.full_slot_ranking(part,k)
                rankings.append(dict(period=period['id'],arm=arm,**stat))
                picks=picks.copy();picks['period']=period['id'];picks['k']=k;selected.append(picks)
    records=pd.DataFrame(records);ranks=pd.DataFrame(rankings)
    table=records[records.scope.eq('ORIGINAL_NONWORD')].set_index(['period','arm'])
    old=table.loc[('HISTORICAL_POOLED',parameter.BASE_ARM)];new=table.loc[('HISTORICAL_POOLED',ARM)]
    guards=dict(pooled_FP_drop_at_least10percent=bool(new.fp<=old.fp*.9),
        pooled_Precision_gain_at_least3pp=bool(new.Precision>=old.Precision+.03),pooled_F1_not_lower=bool(new.F1>=old.F1))
    for fold in history.FOLDS:
        a=table.loc[(fold['id'],ARM)];z=table.loc[(fold['id'],parameter.BASE_ARM)]
        guards[fold['id']+'_Recall_loss_within2p5pp']=bool(a.Recall>=z.Recall-.025)
        guards[fold['id']+'_FP_not_higher']=bool(a.fp<=z.fp)
        guards[fold['id']+'_F1_not_lower']=bool(a.F1>=z.F1)
    ranking=ranks[ranks.period.eq('HISTORICAL_POOLED')&ranks.k.eq(3)].set_index('arm')
    guards['pooled_Top3_full_slot_hits_not_lower']=bool(ranking.loc[ARM,'hits']>=ranking.loc[parameter.BASE_ARM,'hits'])
    return dict(formed_at=b.now(),guards=guards,fixed_screen_pass=all(guards.values()),
        new_fits=0,production_deployed=False,event_probability_validated=False,not_blind=True),records,ranks,pd.concat(selected,ignore_index=True)


def run():
    protocol=b.read(ROOT/'protocol_BEFORE_ANY_EFFECT.json');b.verify(protocol['source_bindings'])
    if protocol['candidates']!=STAGES:raise ValueError('fixed stage drift')
    b.write_new(ROOT/'experiment_STARTED.json',dict(formed_at=b.now(),new_fits=0,protocol_sha256=b.sha(ROOT/'protocol_BEFORE_ANY_EFFECT.json')))
    output=ROOT/'execution';output.mkdir(exist_ok=False)
    data=joblib.load(parameter.ROOT/'fixed_input_BEFORE_ANY_FIT.joblib')
    full,fields,shape=data['full'],data['fields'],data['shape']
    refs=pd.read_csv(REFERENCE,dtype={'code':str},float_precision='round_trip',low_memory=False)
    baseline=refs[refs.arm.eq(parameter.BASE_ARM)].copy()
    parts=[];choices=[];selection_metrics=[];start=time.monotonic()
    try:
        for plan in data['plans']:
            if time.monotonic()-start>1800:raise TimeoutError('read-only30min budget')
            month=plan['month'];window=history.training_rows(full,plan['cutoff'],'relay');oof={stage:[] for stage in STAGES}
            for block,(past,valid) in enumerate(temporal.temporal_splits(window),1):
                path=temporal.ROOT/'execution'/f'{month}_inner{block}_model.joblib';saved=joblib.load(path)
                if saved['training_digest']!=history._frame_digest(past,fields):raise ValueError('strict original inner training drift')
                values=prefix_scores(saved['model'],valid[fields])
                cached=pd.read_csv(temporal.ROOT/'execution'/f'{month}_inner{block}_all_validation_scores_BEFORE_EFFECTS.csv',dtype={'code':str},float_precision='round_trip')
                np.testing.assert_allclose(values[120],cached.raw_score,atol=1e-14,rtol=0)
                all_rows=[]
                for stage in STAGES:
                    rows=parameter.nonword_oof(valid,values[stage],shape);rows['boosting_rounds']=stage;rows['block']=block
                    oof[stage].append(rows);all_rows.append(rows)
                pd.concat(all_rows,ignore_index=True).to_csv(output/f'{month}_inner{block}_all3prefix_OOF_BEFORE_SELECTION.csv',index=False,encoding='utf-8-sig')
            oof={stage:pd.concat(rows,ignore_index=True) for stage,rows in oof.items()}
            statistics={stage:parameter.decision_statistics(rows,.5) for stage,rows in oof.items()}
            selected=select_round(statistics)
            if any(rows.target_date.max()>=plan['cutoff'] for rows in oof.values()):raise ValueError('outer target in inner selection')
            for stage,stats in statistics.items():selection_metrics.append(dict(month=month,rounds=stage,**stats))
            choice=dict(formed_at=b.now(),month=month,cutoff=plan['cutoff'],selected_rounds=selected,
                statistics={str(k):v for k,v in statistics.items()},max_inner_target=oof[120].target_date.max(),
                outer_scores_or_labels_used_for_selection=False,new_fits=0)
            b.write_new(output/f'{month}_SELECTION_BEFORE_OUTER_SCORE.json',choice)
            path=Path(plan['final_model_path'])
            if b.sha(path)!=plan['final_model_sha256']:raise ValueError('protected final model drift')
            saved=joblib.load(path);test=full[full.date.str[:7].eq(month)].copy()
            values=prefix_scores(saved['model'],test[fields],sorted(set([selected,120])))
            reference=baseline.set_index(['date','code']).loc[pd.MultiIndex.from_frame(test[['date','code']])].reset_index()
            np.testing.assert_allclose(values[120],reference.b_score,atol=1e-14,rtol=0)
            result=parameter.apply_head(reference,values[selected],.5,ARM,'BASE_SAVED_PREFIX_'+str(selected))
            result['selected_boosting_rounds']=selected
            result['classifier_head_kind']='SAVED_BASE_HGB_PREFIX_SELECTED_WITH_TEMPORAL_OOF'
            result.to_csv(output/f'{month}_predictions_BEFORE_EVALUATION.csv',index=False,encoding='utf-8-sig');parts.append(result)
            choices.append(dict(month=month,selected_rounds=selected,max_inner_target=oof[120].target_date.max(),new_fits=0))
            b.write_new(output/f'{month}_COMPLETED.json',dict(completed_at=b.now(),selection_formed_at=choice['formed_at'],
                selected_rounds=selected,new_fits=0,prediction_sha256=b.sha(output/f'{month}_predictions_BEFORE_EVALUATION.csv'),final_model_sha256=b.sha(path)))
            print(month,'prefix',selected,'fits0',flush=True)
        new=pd.concat(parts,ignore_index=True)
        if len(new)!=3777:raise ValueError('same complete3777 observations required')
        all_heads=pd.concat([refs,new],ignore_index=True)
        pd.testing.assert_frame_equal(all_heads.iloc[:37770][refs.columns],refs,check_exact=True,check_dtype=False)
        all_heads.to_csv(output/'all41547_predictions_BEFORE_EVALUATION.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(choices).to_csv(output/'all16_training_only_prefix_selections.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(selection_metrics).to_csv(output/'all48_training_only_prefix_metrics.csv',index=False,encoding='utf-8-sig')
        result,stats,ranks,picks=evaluate(pd.concat([baseline,new],ignore_index=True))
        stats.to_csv(output/'same_pool_classification_metrics.csv',index=False,encoding='utf-8-sig')
        ranks.to_csv(output/'full_slot_Top1Top2Top3.csv',index=False,encoding='utf-8-sig')
        picks.to_csv(output/'all_selected_rows_WITH_UNKNOWN.csv',index=False,encoding='utf-8-sig')
        current=new[new.date.eq('2026-10-15')]
        if len(current)!=6 or current.target.notna().any() or current.label_valid.fillna(False).any():raise ValueError('current saved six must remain unknown')
        current.to_csv(output/'current15_six_target16_UNMATURED.csv',index=False,encoding='utf-8-sig')
        result.update(complete_predictions=len(all_heads),protected_models_changed=False,elapsed_seconds=time.monotonic()-start,
            source_bindings=protocol['source_bindings']|{str(p.resolve()):b.sha(p) for p in output.iterdir() if p.is_file()})
        b.verify(result['source_bindings']);b.write_new(output/'iteration_fixed_result.json',result)
        return dict(new_fits=0,screen_pass=result['fixed_screen_pass'],elapsed_seconds=result['elapsed_seconds'])
    except Exception as exc:
        b.write_new(ROOT/'FAILURE_STOP.json',dict(formed_at=b.now(),new_fits=0,type=type(exc).__name__,error=str(exc),retry_authorized=False));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','run']);parser.add_argument('--tests-xml')
    args=parser.parse_args();print(prepare(args.tests_xml) if args.command=='prepare' else run())
