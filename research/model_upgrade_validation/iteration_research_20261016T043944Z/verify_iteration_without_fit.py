"""Separate read-only verifier; uses staged decision logits, not study helpers.

This is a second implementation run by the primary agent, not a new independent
sub-agent review. No sklearn fit or protected-file mutation is performed.
"""
from pathlib import Path
from fractions import Fraction
import importlib.util
import sys
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
sys.path.insert(0,str(REPO))
import short_term_thinking_dataset_builder as b
import model_upgrade_history as history

PARAM=REPO/'research/model_upgrade_validation/parameter_research_20261016T032229Z'
OLD=REPO/'research/model_upgrade_validation/false_positive_20261016T030307Z'
ARM='PARAM_BASE_OOF_SELECTED_BOOSTING_PREFIX'
BASE='SOURCE_PATCHED_FEATURE_CLASSIFIER'
spec=importlib.util.spec_from_file_location('separate_parameter_audit_primitives',PARAM/'independent_verification/parameter_audit_primitives.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def raw_logits(model,X):
    # Alternate public scoring path: logistic transform of staged decisions.
    hgb=model.named_steps['hgb'];X=model[:-1].transform(X);out={}
    with threadpool_limits(limits=2):
        for stage,logits in enumerate(hgb.staged_decision_function(X),1):
            if stage in [30,60,120]:out[stage]=expit(np.asarray(logits).reshape(-1))
    assert hgb.n_iter_==120 and sorted(out)==[30,60,120]
    return out


def statistic(rows):
    valid=audit.valid_labels(rows)
    assert valid.all() and rows.target.nunique()==2
    call=rows.verified_current_price_shape.eq(0)&rows.raw_score.gt(.5)
    tp=int((call&rows.target.eq(1)).sum());fp=int((call&rows.target.eq(0)).sum());positive=int(rows.target.eq(1).sum())
    return dict(tp=tp,fp=fp,positive=positive,
        f=Fraction(5*tp,5*tp+positive-tp+4*fp),
        recall=Fraction(tp,positive),f1=Fraction(2*tp,positive+tp+fp))


def choose(stats):
    old=stats[120];choice=120;best=old['f']
    for stage in [30,60]:
        value=stats[stage]
        good=value['recall']>=old['recall']-Fraction(1,40) and value['f1']>=old['f1']-Fraction(1,100)
        if good and value['f']>best:choice=stage;best=value['f']
    return choice


def run():
    out=ROOT/'verification'
    if out.exists():raise FileExistsError('new immutable verification directory required')
    protocol=b.read(ROOT/'protocol_BEFORE_ANY_EFFECT.json');b.verify(protocol['source_bindings'])
    effect=b.read(ROOT/'execution/iteration_fixed_result.json');b.verify(effect['source_bindings'])
    data=joblib.load(PARAM/'fixed_input_BEFORE_ANY_FIT.joblib')
    full,fields=data['full'],data['fields'];shape=data['shape'].set_index(['date','code']).verified_current_price_shape
    previous_path=PARAM/'execution/all37770_predictions_BEFORE_EVALUATION.csv'
    final_path=ROOT/'execution/all41547_predictions_BEFORE_EVALUATION.csv'
    previous=pd.read_csv(previous_path,dtype={'code':str},float_precision='round_trip',low_memory=False)
    final=pd.read_csv(final_path,dtype={'code':str},float_precision='round_trip',low_memory=False)
    assert len(previous)==37770 and len(final)==41547 and final.arm.nunique()==11
    pd.testing.assert_frame_equal(previous,final.iloc[:37770][previous.columns],check_exact=True,check_dtype=False)
    assert final.groupby('arm').size().eq(3777).all() and not final.duplicated(['date','code','arm']).any()
    candidate=final[final.arm.eq(ARM)].copy();audit.assert_september_and_current_unknown(candidate)
    decisions=[];replays=[];maximum=0.;model_count=0
    for plan in data['plans']:
        month=plan['month'];window=audit.outer_window(full,plan['cutoff'])
        assert history._frame_digest(window,fields)==plan['training_digest']
        blocks={stage:[] for stage in [30,60,120]}
        for index,past,valid in audit.chronological_blocks(window):
            path=OLD/'execution'/f'{month}_inner{index}_model.joblib';saved=joblib.load(path)
            assert saved['training_digest']==history._frame_digest(past,fields)
            assert past.target_date.max()<valid.date.min()
            values=raw_logits(saved['model'],valid[fields]);model_count+=1
            emitted=pd.read_csv(ROOT/'execution'/f'{month}_inner{index}_all3prefix_OOF_BEFORE_SELECTION.csv',dtype={'code':str},float_precision='round_trip')
            eligible=valid[valid.frozen_original_nonword_identity.eq(0)]
            mask=valid.frozen_original_nonword_identity.eq(0).to_numpy()
            for stage in [30,60,120]:
                expected=eligible[['date','code','target_date','target','label_valid','frozen_original_nonword_identity']].copy()
                expected['verified_current_price_shape']=[shape.loc[(row.date,row.code)] for row in expected.itertuples()]
                expected['raw_score']=values[stage][mask]
                saved_rows=emitted[emitted.boosting_rounds.eq(stage)]
                pd.testing.assert_frame_equal(expected.drop(columns='raw_score').reset_index(drop=True),saved_rows[expected.columns.drop('raw_score')].reset_index(drop=True),check_exact=True,check_dtype=False)
                maximum=max(maximum,audit.scores_equal(expected.raw_score,saved_rows.raw_score))
                blocks[stage].append(expected)
            replays.append(dict(month=month,block=index,train_days=past.date.nunique(),validation_days=valid.date.nunique(),
                train_target_max=past.target_date.max(),validation_first=valid.date.min(),all3prefixes_match=True))
        stats={stage:statistic(pd.concat(rows,ignore_index=True)) for stage,rows in blocks.items()}
        stage=choose(stats)
        choice=b.read(ROOT/'execution'/f'{month}_SELECTION_BEFORE_OUTER_SCORE.json')
        complete=b.read(ROOT/'execution'/f'{month}_COMPLETED.json')
        assert stage==choice['selected_rounds']==complete['selected_rounds']
        assert choice['max_inner_target']<plan['cutoff'] and choice['formed_at']<=complete['completed_at']
        path=Path(plan['final_model_path']);assert b.sha(path)==plan['final_model_sha256']
        model=joblib.load(path)['model'];test=full[full.date.str[:7].eq(month)].copy();values=raw_logits(model,test[fields]);model_count+=1
        expected_index=pd.MultiIndex.from_frame(test[['date','code']])
        baseline=previous[previous.arm.eq(BASE)].set_index(['date','code']).loc[expected_index]
        emitted=candidate.set_index(['date','code']).loc[expected_index]
        maximum=max(maximum,audit.scores_equal(values[120],baseline.b_score),audit.scores_equal(values[stage],emitted.b_score),audit.scores_equal(values[stage],emitted.ranking_score))
        is_B=baseline.verified_current_price_shape.eq(0).to_numpy()
        assert (emitted.call.to_numpy()[is_B]==np.where(values[stage][is_B]>.5,'PROMOTE','FAIL')).all()
        pd.testing.assert_frame_equal(emitted.loc[~is_B,['score','call']],baseline.loc[~is_B,['score','call']],check_exact=True,check_dtype=False)
        decisions.append(dict(month=month,selected_rounds=stage,only_training_OOF=True,max_inner_target=choice['max_inner_target'],new_fits=0))
    recorded=pd.read_csv(ROOT/'execution/same_pool_classification_metrics.csv')
    recorded_ranks=pd.read_csv(ROOT/'execution/full_slot_Top1Top2Top3.csv')
    summaries=[];rankings=[]
    periods=history.FOLDS+[dict(id='2026Oct',start='2026-10-01',end='2026-10-15'),dict(id='HISTORICAL_POOLED',start='2025-07-01',end='2026-09-30')]
    for period in periods:
        for name in [BASE,ARM]:
            sample=final[final.arm.eq(name)&final.date.between(period['start'],period['end'])]
            for scope,rows in [('FULL_POOL',sample),('ORIGINAL_NONWORD',sample[sample.daily_one_word.eq(0)])]:
                stat=audit.matrix(rows);expected=recorded[recorded.period.eq(period['id'])&recorded.arm.eq(name)&recorded.scope.eq(scope)].iloc[0]
                for actual_field,saved_field in [('TP','tp'),('FP','fp'),('Recall','Recall'),('Precision','Precision'),('F1','F1')]:
                    np.testing.assert_allclose(stat[actual_field],expected[saved_field],atol=1e-14,rtol=0,equal_nan=True)
                summaries.append(dict(period=period['id'],arm=name,scope=scope,**stat))
            for k in [1,2,3]:
                stat,picks=audit.full_slot_topk(sample,k)
                expected=recorded_ranks[recorded_ranks.period.eq(period['id'])&recorded_ranks.arm.eq(name)&recorded_ranks.k.eq(k)].iloc[0]
                for column in stat:np.testing.assert_allclose(stat[column],expected[column],atol=1e-14,rtol=0,equal_nan=True)
                rankings.append(dict(period=period['id'],arm=name,**stat))
    # Recheck all protected hashes after the alternate-scoring path.
    b.verify(protocol['source_bindings']);out.mkdir()
    pd.DataFrame(decisions).to_csv(out/'all16_prefix_choices_recomputed.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(replays).to_csv(out/'all48_strict_inner_replays.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(summaries).to_csv(out/'classification_metrics_second_implementation.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(rankings).to_csv(out/'full_slot_Topk_second_implementation.csv',index=False,encoding='utf-8-sig')
    paths=[Path(__file__),ROOT/'protocol_BEFORE_ANY_EFFECT.json',ROOT/'execution/iteration_fixed_result.json',final_path,previous_path,*out.glob('*.csv')]
    b.write_new(out/'primary_agent_second_implementation_verification.json',dict(verified_at=b.now(),passed=True,
        implementation='Separate read-only code, public staged_decision_function + expit',new_fits=0,
        subagent_independent_review=False,subagent_unavailable_due_usage_limit=True,
        models_replayed=model_count,all48_inner_and16_original_final_models_replayed=True,
        maximum_score_difference=maximum,original37770_all_columns_unchanged=True,complete_outputs=41547,
        source_rows_per_arm=3777,arms=11,original_September_B=[143,107,40,36],current6_unknown=True,
        original_model_hashes_verified_before_and_after=True,training_only_exact_prefix_selection=True,
        event_probability_validated=False,fixed_screen_pass_as_recorded=effect['fixed_screen_pass'],
        source_bindings={str(path.resolve()):b.sha(path) for path in paths}))
    print(dict(models_replayed=model_count,max_difference=maximum,old37770_unchanged=True,new_fits=0))


if __name__=='__main__':run()
