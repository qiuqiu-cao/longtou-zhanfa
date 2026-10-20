"""Align the new feature head to the exact same frozen source routes.

This is an append-only routing correction; all saved fits and raw predictions
remain immutable. Non-B routes inherit the equally qualified reference.
"""
from pathlib import Path
import argparse
import json
import shutil
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import relay_source_feature_research as research
import short_term_thinking_dataset_builder as b

ROOT=research.ROOT


def predict_with_same_source_routes(test,raw_score,shape,reference):
    """Actual inference entry: verified B uses own score; every other route
    inherits the exact equally-qualified frozen reference, including unknown.

    Reference fact/source metadata are copied by exact signal/code identity.
    """
    source=reference.set_index(['date','code'])
    if source.index.duplicated().any():raise ValueError('duplicate fixed reference identity')
    shapes=shape.set_index(['date','code']).verified_current_price_shape
    if len(test)!=len(raw_score):raise ValueError('prediction count mismatch')
    output=[]
    for raw,row in zip(raw_score,test.itertuples(index=False)):
        if not np.isfinite(raw) or not 0<=raw<=1:raise ValueError('finite absolute B score required')
        saved=source.loc[(row.date,row.code)].to_dict()
        effective=shapes.loc[(row.date,row.code)]
        inherited_call=saved.get('evaluation_call',saved.get('recovered_B_call',saved['call']))
        call='PROMOTE' if raw>.5 else 'FAIL' if effective==0 else inherited_call
        if effective!=0:call=inherited_call
        value=float(raw) if effective==0 else saved['score']
        saved.update(date=row.date,code=row.code,arm=research.ARM,ranking_score=float(raw),b_score=float(raw),
            score=value,call=call,evaluation_call=call,recovered_B_call=call,
            effective_verified_shape=effective,verified_current_price_shape=effective,
            aligned_evaluation_call=call,same_source_aligned_score=value,
            route_alignment_reason='VERIFIED_B_OWN_ABSOLUTE_SCORE' if effective==0 else 'NON_B_ROUTE_EXACT_FROZEN_SOURCE_QUALIFIED_REFERENCE_INHERITED',
            classifier_head_kind='ABSOLUTE_HGB_EVENT_HEAD',uncalibrated_event_score=True)
        output.append(saved)
    return pd.DataFrame(output)


def align_routes(predictions):
    out=predictions.copy()
    if 'aligned_evaluation_call' in out:raise ValueError('already aligned')
    baseline=out[out.arm.eq('CORRECTED_CAUSAL')].set_index(['date','code'])
    if baseline.index.duplicated().any():raise ValueError('duplicate frozen reference')
    out['effective_verified_shape']=out.verified_current_price_shape
    out['aligned_evaluation_call']=out.evaluation_call
    out['same_source_aligned_score']=out.score
    out['route_alignment_reason']='ORIGINAL_SAME_SOURCE_REFERENCE_OR_VERIFIED_B_ROUTE'
    for i,row in out[out.arm.eq(research.ARM)].iterrows():
        if row.effective_verified_shape==0:
            # A real absolute score remains separate from a relative rank.
            expected='PROMOTE' if row.b_score>.5 else 'FAIL'
            if row.call!=expected:raise ValueError('verified B route differs from fixed absolute score')
            continue
        saved=baseline.loc[(row.date,row.code)]
        out.loc[i,'aligned_evaluation_call']=saved.evaluation_call
        out.loc[i,'same_source_aligned_score']=saved.score
        out.loc[i,'route_alignment_reason']='NON_B_ROUTE_EXACT_FROZEN_SOURCE_QUALIFIED_REFERENCE_INHERITED'
    pd.testing.assert_frame_equal(out[predictions.columns],predictions,check_exact=True)
    return out


def run(tests_xml):
    output=ROOT/'same_source_route_alignment'
    if output.exists():raise FileExistsError('new immutable alignment directory required')
    protocol=b.read(ROOT/'protocol_BEFORE_ANY_FIT.json');b.verify(protocol['source_bindings'])
    fixed=b.read(ROOT/'execution/fixed_source_feature_result.json');b.verify(fixed['source_bindings'])
    source_path=ROOT/'execution/complete18885_predictions_BEFORE_EVALUATION.csv'
    baseline_path=research.REFERENCE/'all_predictions_with_isolated_qualification.csv'
    output.mkdir()
    b.write_new(output/'policy_BEFORE_ROUTING_RECOMPUTATION.json',dict(formed_at=b.now(),
        reason='User requires identical source routes: only verified shape0 is a new B direction; new shape1 or unresolved uses exact frozen qualified baseline route',
        no_new_fits=True,old_fit_failure_and_predictions_retained=True,threshold_unchanged=True,
        inherited_routes='CORRECTED_CAUSAL evaluation_call and original score for every non-verified0 new-feature row',
        source_bindings={str(p.resolve()):b.sha(p) for p in [Path(__file__),Path(research.__file__),ROOT/'protocol_BEFORE_ANY_FIT.json',source_path,baseline_path,research.REFERENCE/'source_qualification_repair_receipt.json',Path(tests_xml)]}))
    original_prediction=pd.read_csv(source_path,dtype={'code':str},low_memory=False)
    # Preserve the entire six-arm final reference, not a reduced four-arm view.
    all_baseline=pd.read_csv(baseline_path,dtype={'code':str},low_memory=False)
    baseline=all_baseline[all_baseline.arm.eq('CORRECTED_CAUSAL')].copy()
    baseline['evaluation_call']=baseline.recovered_B_call
    old_new=original_prediction[original_prediction.arm.eq(research.ARM)].copy()
    old_shape=old_new[['date','code','verified_current_price_shape']]
    new=predict_with_same_source_routes(old_new,old_new.b_score.to_numpy(),old_shape,baseline)
    new['pre_alignment_executed_score']=old_new.score.to_numpy()
    new['pre_alignment_executed_call']=old_new.call.to_numpy()
    # All original22662 cells remain exact. New columns express only the added
    # wrapper semantics; no old scores/calls are repurposed.
    all_baseline['evaluation_call']=all_baseline.recovered_B_call
    all_baseline['effective_verified_shape']=all_baseline.verified_current_price_shape
    all_baseline['aligned_evaluation_call']=all_baseline.evaluation_call
    all_baseline['same_source_aligned_score']=all_baseline.score
    all_baseline['route_alignment_reason']='ORIGINAL_FINAL_SOURCE_QUALIFIED_REFERENCE'
    aligned=pd.concat([all_baseline,new],ignore_index=True)
    if len(aligned)!=26439:raise ValueError('complete7x3777 expected')
    # Appending a dict-built new head can promote a pandas StringDtype column
    # to object; old cells, including every score/call, still match exactly.
    pd.testing.assert_frame_equal(pd.read_csv(baseline_path,dtype={'code':str},low_memory=False),aligned.iloc[:22662][pd.read_csv(baseline_path,nrows=0).columns],check_exact=True,check_dtype=False)
    aligned.to_csv(output/'complete26439_predictions_WITH_SAME_SOURCE_ROUTES.csv',index=False,encoding='utf-8-sig')
    evaluation_view=aligned.copy();evaluation_view['evaluation_call']=evaluation_view.aligned_evaluation_call
    result,stats,ranks,picks=research.evaluate(evaluation_view)
    stats['metric_applicability']='ABSOLUTE_EVENT_HEAD'
    pair=stats.arm.str.startswith('PAIRWISE')
    stats.loc[pair,'metric_applicability']='NO_ABSOLUTE_EVENT_HEAD'
    for column in ['tp','fp','Recall','Precision','F1','called_accuracy','full_known_accuracy']:
        stats.loc[pair,column]=np.nan
    stats.to_csv(output/'same_pool_classification_metrics.csv',index=False,encoding='utf-8-sig')
    ranks.to_csv(output/'complete_full_slot_Top1_Top2_Top3.csv',index=False,encoding='utf-8-sig')
    picks.to_csv(output/'all_selected_rows_WITH_SAME_SOURCE_ROUTES.csv',index=False,encoding='utf-8-sig')
    # The same original function that produced the training patch is reused for
    # a genuine current-six inference replay. It changes zero current cells.
    original_data=joblib.load(research.OLD/'trainable_corrected_dataset_QUALIFIED.joblib')
    verified=pd.read_csv(research.REFERENCE/'all244_verified_qualification_rows.csv',dtype={'code':str})
    training_copy,training_changes=research.patched_feature_copy(original_data['full'],original_data['fields'],verified,47)
    prepared=joblib.load(ROOT/'prepared.joblib')
    pd.testing.assert_frame_equal(training_copy,prepared['full'],check_exact=True)
    current=original_data['full'][original_data['full'].date.eq('2026-10-15')].copy()
    current_qualifications=verified[verified.date.eq('2026-10-15')].copy()
    current_copy,current_changes=research.patched_feature_copy(current,original_data['fields'],current_qualifications,0)
    pd.testing.assert_frame_equal(current_copy,current,check_exact=True)
    current_shapes=research.verified_shapes(current,current_qualifications)
    october=b.read(ROOT/'execution/2026-10_COMPLETED.json')
    if b.sha(october['model_path'])!=october['model_sha256']:raise ValueError('October saved model hash changed')
    model=joblib.load(october['model_path'])['model']
    with threadpool_limits(limits=2):score=model.predict_proba(current_copy[original_data['fields']])[:,list(model.classes_).index(1)]
    current_ref=baseline[baseline.date.eq('2026-10-15')].copy()
    current_inference=predict_with_same_source_routes(current_copy,score,current_shapes,current_ref)
    current_saved=aligned[aligned.arm.eq(research.ARM)&aligned.date.eq('2026-10-15')].copy()
    if not np.allclose(current_saved.b_score,score,atol=1e-15,rtol=0):raise ValueError('current shared-feature inference differs')
    if current_saved.target.notna().any() or current_saved.label_valid.fillna(False).any():raise ValueError('future6 incorrectly matured')
    current_compare=current_saved.set_index(['date','code']).loc[pd.MultiIndex.from_frame(current_inference[['date','code']])]
    if not np.allclose(current_inference.b_score,current_compare.b_score,atol=1e-15,rtol=0) or list(current_inference.call)!=list(current_compare.call):
        raise ValueError('shared wrapper/current inference mismatch')
    current_saved.to_csv(output/'current15_six_SHARED_FEATURE_API_target16_UNMATURED.csv',index=False,encoding='utf-8-sig')
    b.write_new(output/'current15_six_SHARED_FEATURE_API_receipt.json',dict(formed_at=b.now(),signal_date='2026-10-15',official_target_date='2026-10-16',rows=6,
        shared_apply_function='relay_source_feature_research.patched_feature_copy',training_changes=len(training_changes),current_changes=len(current_changes),
        inference_wrapper='relay_source_feature_alignment.predict_with_same_source_routes',wrapper_actually_called=True,
        effective_shape_column='effective_verified_shape',original_shape_metadata_unchanged=True,parent_path_role_metadata_unchanged=True,
        new_fits=0,existing_october_model_path=october['model_path'],existing_october_model_sha256=october['model_sha256'],october_model_original_fit_reused_for_this_replay=True,
        known_current_outcomes=0,formal_forecast_append=False,model_scores_uncalibrated=True))
    snapshots=output/'executed_source_snapshots';snapshots.mkdir()
    for path in [Path(research.__file__),Path(__file__),b.ROOT/'tests/test_relay_source_feature_research.py',b.ROOT/'tests/test_relay_source_feature_alignment.py']:
        shutil.copyfile(path,snapshots/path.name)
    result.update(original_experiment_new_fits=fixed['new_fits'],original_experiment_reused_windows=fixed['windows_reused'],routing_recompute_new_fits=0,
        non_B_routes_inherited=int(aligned.route_alignment_reason.eq('NON_B_ROUTE_EXACT_FROZEN_SOURCE_QUALIFIED_REFERENCE_INHERITED').sum()),
        original_prediction_rows_and_columns_unchanged=True,known_future_six_outcomes=0,
        source_bindings={str(p.resolve()):b.sha(p) for p in [*output.rglob('*')] if p.is_file()})
    b.write_new(output/'same_source_aligned_fixed_result.json',result)
    print(json.dumps(dict(new_fits=fixed['new_fits'],reused_windows=fixed['windows_reused'],routing_refits=0,screen_pass=result['screen_pass'],current6_feature_changes=0),ensure_ascii=True))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--tests-xml',required=True)
    args=parser.parse_args();run(args.tests_xml)
