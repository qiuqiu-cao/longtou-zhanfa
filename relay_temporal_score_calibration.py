"""One fixed training-only temporal OOF score-scale challenge, no threshold search.

Protected final classifiers are reused. Only inner chronological models and one
Platt map per month are fitted. Calibration is not certified event probability.
"""
from pathlib import Path
import argparse
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

import short_term_thinking_dataset_builder as b
import short_term_thinking_research as training
import model_upgrade_history as history
import expectation_345_research as original
import expectation_345_evaluation as metrics
import relay_source_feature_research as source
from short_term_thinking_preprocessing import EmptyTrainingColumnGuard

ROOT = b.ROOT / 'research/model_upgrade_validation/false_positive_20261016T030307Z'
PREVIOUS = source.ROOT.parent
ARM = 'TEMPORAL_OOF_CALIBRATED_SOURCE_HEAD'
REFERENCE = PREVIOUS / 'source_feature_training/same_source_route_alignment/complete26439_predictions_WITH_SAME_SOURCE_ROUTES.csv'
BASE_ARM = 'SOURCE_PATCHED_FEATURE_CLASSIFIER'
CALIBRATOR_PARAMS = dict(C=1., solver='lbfgs', max_iter=2000, random_state=20261016)
PERIODS = history.FOLDS + [dict(id='2026Oct', start='2026-10-01', end='2026-10-15'), dict(id='HISTORICAL_POOLED', start='2025-07-01', end='2026-09-30')]


def temporal_splits(window):
    days = sorted(window.date.unique())
    if len(days) != 252 or window.duplicated(['date', 'code']).any():
        raise ValueError('exact original252-day complete training window required')
    splits = []
    for block in range(1, 4):
        validation_days = days[63*block:63*(block+1)]
        cutoff = validation_days[0]
        before = window[window.date.lt(cutoff) & window.target_date.lt(cutoff)].copy()
        valid = window[window.date.isin(validation_days)].copy()
        if before.empty or before.target.nunique() != 2 or before.target_date.max() >= cutoff:
            raise ValueError('both classes and strictly matured inner training required')
        if set(before.date) & set(valid.date):
            raise ValueError('inner train-validation overlap')
        splits.append((before, valid))
    return splits


def calibration_weights(frame):
    """Each observed original-B signal day has mass1, no class balancing."""
    if frame.empty or frame.target.nunique() != 2 or not frame.label_valid.fillna(False).all():
        raise ValueError('two known classes required for calibration')
    return 1. / frame.groupby('date').date.transform('size')


def logit_column(raw):
    values = np.asarray(raw, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('finite original0to1 scores required')
    clipped = np.clip(values, 1e-12, 1-1e-12)
    return (np.log(clipped) - np.log1p(-clipped)).reshape(-1, 1)


def calibrated_scores(calibrator, raw):
    if calibrator.coef_.shape != (1, 1) or calibrator.coef_[0, 0] <= 0:
        raise ValueError('nonpositive calibration slope cannot reverse a ranking')
    return calibrator.predict_proba(logit_column(raw))[:, list(calibrator.classes_).index(1)]


def apply_head(reference, raw, adjusted):
    if len(reference) != len(raw) or len(raw) != len(adjusted):
        raise ValueError('same complete ordered cohort required')
    np.testing.assert_allclose(reference.b_score.to_numpy(float), raw, atol=1e-14, rtol=0)
    output = reference.copy()
    eligible = output.verified_current_price_shape.eq(0)
    output['arm'] = ARM
    output['raw_b_score_before_calibration'] = raw
    output['b_score'] = adjusted
    # Preserve the original ranking rather than pretending scale calibration
    # adds within-day discrimination or changing tie order by rounding.
    output.loc[eligible, 'score'] = np.asarray(adjusted)[eligible]
    output.loc[eligible, 'call'] = np.where(np.asarray(adjusted)[eligible] > .5, 'PROMOTE', 'FAIL')
    for field in ['evaluation_call', 'recovered_B_call', 'aligned_evaluation_call']:
        output[field] = output.call
    output['same_source_aligned_score'] = output.score
    output['calibrator_training_only_temporal_OOF'] = True
    output['calibration_map_trained'] = True
    output['event_probability_validated'] = False
    output['research_validation_status'] = 'FIXED_RESEARCH_NOT_ADOPTED'
    output['classifier_head_kind'] = 'TEMPORAL_OOF_SCORE_SCALE_RESEARCH_HEAD'
    pd.testing.assert_series_equal(output.ranking_score, reference.ranking_score, check_exact=True)
    pd.testing.assert_frame_equal(output.loc[~eligible, ['score', 'call']], reference.loc[~eligible, ['score', 'call']], check_exact=True)
    return output


def prepare(tests_xml):
    lock = ROOT / 'protocol_BEFORE_ANY_FIT.json'
    if lock.exists():
        raise FileExistsError('fixed challenge already registered')
    import xml.etree.ElementTree as ET
    suites = list(ET.parse(tests_xml).getroot().iter('testsuite'))
    if not suites or any(int(x.get('failures', 0))+int(x.get('errors', 0)) for x in suites):
        raise ValueError('passing targeted preflight tests required')
    old_receipt = b.read(PREVIOUS / 'delivery/batch_verified_NOT_MODEL_ACCEPTANCE.json')
    b.verify(old_receipt['source_bindings'])
    data = joblib.load(source.ROOT / 'prepared.joblib')
    full, fields = data['full'].copy(), data['fields']
    if len(full) != 9315 or len(fields) != 92:
        raise ValueError('same complete9315 source-feature input required')
    identity_path = b.ROOT / 'research/model_upgrade_validation/short_term_thinking_20261015T071000Z/dataset.joblib'
    identity = joblib.load(identity_path)['features'].set_index(['date', 'code']).daily_one_word
    refs = pd.read_csv(REFERENCE, dtype={'code': str}, low_memory=False)
    outer_identity = refs[refs.arm.eq('BASE')].set_index(['date', 'code']).daily_one_word
    identity = identity[~identity.index.isin(outer_identity.index)]
    identity = pd.concat([identity, outer_identity])
    full['frozen_original_nonword_identity'] = [identity.loc[(r.date, r.code)] for r in full.itertuples()]
    plans, blocks, models = [], [], {}
    for check in data['training_checks']:
        window = history.training_rows(full, check['cutoff'], 'relay')
        if history._frame_digest(window, fields) != check['new_digest']:
            raise ValueError('fixed source training digest drift')
        for index, (past, valid) in enumerate(temporal_splits(window), 1):
            subset = valid[valid.frozen_original_nonword_identity.eq(0)]
            if subset.target.nunique() != 2:
                raise ValueError('two calibration classes required in each temporal block')
            blocks.append(dict(month=check['month'], block=index, cutoff=valid.date.min(),
                train_days=past.date.nunique(), train_rows=len(past), max_train_target=past.target_date.max(),
                validation_days=valid.date.nunique(), validation_rows=len(valid),
                nonword_calibration_rows=len(subset), nonword_positives=int(subset.target.eq(1).sum()),
                training_digest=history._frame_digest(past, fields)))
        record_path = source.ROOT / f"execution/{check['month']}_COMPLETED.json"
        saved = b.read(record_path)
        if b.sha(saved['model_path']) != saved['model_sha256']:
            raise ValueError('saved source head changed')
        models[saved['model_path']] = saved['model_sha256']
        models[str(record_path.resolve())] = b.sha(record_path)
        plans.append(dict(month=check['month'], cutoff=check['cutoff'], training_digest=check['new_digest'],
            final_model_path=saved['model_path'], final_model_sha256=saved['model_sha256']))
    ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(blocks).to_csv(ROOT / 'all48_temporal_blocks_BEFORE_ANY_FIT.csv', index=False, encoding='utf-8-sig')
    joblib.dump(dict(full=full, fields=fields, plans=plans), ROOT / 'fixed_input_BEFORE_ANY_FIT.joblib', compress=3)
    sources = old_receipt['source_bindings'] | models | {str(p.resolve()): b.sha(p) for p in [
        Path(__file__), Path(tests_xml), b.ROOT/'tests/test_relay_temporal_score_calibration.py',
        source.ROOT/'prepared.joblib', REFERENCE, identity_path,
        ROOT/'fixed_input_BEFORE_ANY_FIT.joblib', ROOT/'all48_temporal_blocks_BEFORE_ANY_FIT.csv']}
    b.write_new(lock, dict(formed_at=b.now(), arm=ARM,
        user_authorization='New manual request: 为什么误报这么多能不能再优化; original platform goal remains usageLimited and is not resumed here',
        evidence='Uncalibrated class-balanced scores; overlapping TP/FP score ranges; prior cost1.5 loses many positives; no prior failed challenge rerun',
        hypothesis='A training-only temporal OOF score map may reduce false positives while preserving enough recall; it cannot add within-day ranking information',
        train_window='Same252 mature signal dates and source-patched92 inputs; inner validation blocks2/3/4 of4x63; train strictly earlier signals and target_date before first validation signal',
        membership='Original9245 historical nonword identity, with exact outer frozen BASE3777 identity override; current engine identity never replaces original denominator',
        inner_estimator=training.PARAMS, inner_weights='Exact original.weights(train,1), no new cost',
        calibration=dict(input='logit(raw) clipped only at1e-12 endpoints', estimator=CALIBRATOR_PARAMS,
            selection='Original frozen nonword known-label OOF rows', weights='1/known original-B candidates per date; day mass1, no class balancing, no row-count renormalization',
            nonpositive_slope='Stop as ineligible; no reversed scores or new trial'),
        final_estimator='Reuse exact16 frozen final source heads; no final-head refits',
        decision='Only verified_current_price_shape==0 uses own calibrated score strictly>.5; all non-B/unknown routes inherit frozen source head unchanged',
        ranking='Preserve original source raw ranking_score exactly; no claimed Top-k improvement from monotonic calibration',
        plans=plans, max_inner_HGB_fits=48, max_calibrator_fits=16, max_total_new_fits=64,
        fitting_budget_seconds=1800, search=False, threshold_search=False, old_failed_cost_not_repeated=True,
        acceptance=dict(pooled_Precision_gain_min=.03, pooled_FP_fraction_max=.9, pooled_F1_not_lower=True,
            each_historical_fold_FP_not_higher=True, each_historical_fold_Recall_loss_max=.025,
            complete_daily_Topk_picks_identical=True),
        previously_seen_history_not_blind=True, late_source_original_PIT_not_certified=True,
        event_probability_validated=False, auto_deployment=False, current_target16_never_read=True,
        source_bindings=sources))
    return dict(frozen=True, new_fits=0, planned_inner=48, planned_calibrators=16)


def evaluate(paired):
    records, rankings, selections = [], [], []
    for period in PERIODS:
        sample = paired[paired.date.between(period['start'], period['end'])]
        for arm, frame in sample.groupby('arm', sort=True):
            for scope, part in [('FULL_POOL', frame), ('ORIGINAL_NONWORD', frame[frame.daily_one_word.eq(0)])]:
                view = part.copy(); view['call'] = view.evaluation_call
                records.append(dict(period=period['id'], arm=arm, scope=scope, **metrics.classification(view)))
            for k in [1, 2, 3]:
                stat, selected = source.full_slot_ranking(frame, k)
                rankings.append(dict(period=period['id'], arm=arm, **stat))
                selected = selected.copy(); selected['period'] = period['id']; selected['k'] = k
                selections.append(selected)
    record = pd.DataFrame(records)
    table = record[record.scope.eq('ORIGINAL_NONWORD')].set_index(['period', 'arm'])
    old, new = table.loc[('HISTORICAL_POOLED', BASE_ARM)], table.loc[('HISTORICAL_POOLED', ARM)]
    guards = dict(pooled_Precision_gain_at_least3pp=bool(new.Precision >= old.Precision+.03),
        pooled_FP_drop_at_least10percent=bool(new.fp <= old.fp*.9), pooled_F1_not_lower=bool(new.F1 >= old.F1))
    for fold in history.FOLDS:
        a, z = table.loc[(fold['id'], ARM)], table.loc[(fold['id'], BASE_ARM)]
        guards[fold['id']+'_FP_not_higher'] = bool(a.fp <= z.fp)
        guards[fold['id']+'_Recall_loss_within2p5pp'] = bool(a.Recall >= z.Recall-.025)
    saved = pd.concat(selections, ignore_index=True)
    old_picks = saved[saved.arm.eq(BASE_ARM)][['period','k','date','code']].reset_index(drop=True)
    new_picks = saved[saved.arm.eq(ARM)][['period','k','date','code']].reset_index(drop=True)
    pd.testing.assert_frame_equal(old_picks, new_picks, check_exact=True)
    guards['all_complete_Topk_picks_identical'] = True
    return dict(formed_at=b.now(), guards=guards, fixed_screen_pass=all(guards.values()),
        deployed=False, calibrated_map_trained=True, event_probability_validated=False,
        new_ranking_information=False, not_blind=True), record, pd.DataFrame(rankings), saved


def run():
    protocol = b.read(ROOT/'protocol_BEFORE_ANY_FIT.json'); b.verify(protocol['source_bindings'])
    if protocol['inner_estimator'] != training.PARAMS or protocol['calibration']['estimator'] != CALIBRATOR_PARAMS:
        raise ValueError('frozen parameters drift')
    b.write_new(ROOT/'experiment_STARTED.json', dict(formed_at=b.now(), protocol_sha256=b.sha(ROOT/'protocol_BEFORE_ANY_FIT.json')))
    output = ROOT/'execution'; output.mkdir(exist_ok=False)
    data = joblib.load(ROOT/'fixed_input_BEFORE_ANY_FIT.joblib')
    full, fields = data['full'], data['fields']
    refs = pd.read_csv(REFERENCE, dtype={'code': str}, low_memory=False)
    fixed = refs[refs.arm.eq(BASE_ARM)].copy()
    start = time.monotonic(); hgb_fits = 0; map_fits = 0; predictions = []
    try:
        for plan in data['plans']:
            if time.monotonic()-start > 1800:
                raise TimeoutError('fixed30min training budget reached')
            window = history.training_rows(full, plan['cutoff'], 'relay')
            if history._frame_digest(window, fields) != plan['training_digest']:
                raise ValueError('outer source training digest changed')
            parts = []
            prefix = output/plan['month']
            for index, (past, valid) in enumerate(temporal_splits(window), 1):
                key = Path(str(prefix)+f'_inner{index}')
                b.write_new(str(key)+'_STARTED.json', dict(formed_at=b.now(), training_digest=history._frame_digest(past, fields),
                    max_train_target=past.target_date.max(), validation_first=valid.date.min(), inner_fit_number=hgb_fits+1))
                model = Pipeline([('empty_training_columns', EmptyTrainingColumnGuard()), ('hgb', HistGradientBoostingClassifier(**training.PARAMS))])
                with threadpool_limits(limits=2):
                    model.fit(past[fields], past.target.astype(int), hgb__sample_weight=original.weights(past, 1.))
                hgb_fits += 1
                joblib.dump(dict(model=model, features=fields, cutoff=valid.date.min(), training_digest=history._frame_digest(past, fields)), str(key)+'_model.joblib')
                with threadpool_limits(limits=2):
                    raw = model.predict_proba(valid[fields])[:, list(model.classes_).index(1)]
                scored = valid[['date','code','target_date','target','label_valid','frozen_original_nonword_identity']].copy()
                scored['raw_score'] = raw; scored['inner_block'] = index
                scored.to_csv(str(key)+'_all_validation_scores_BEFORE_EFFECTS.csv', index=False, encoding='utf-8-sig')
                parts.append(scored[scored.frozen_original_nonword_identity.eq(0)])
                b.write_new(str(key)+'_COMPLETED.json', dict(formed_at=b.now(), model_sha256=b.sha(str(key)+'_model.joblib'), rows=len(valid)))
            oof = pd.concat(parts, ignore_index=True)
            if oof.duplicated(['date','code']).any() or oof.target_date.max() >= plan['cutoff']:
                raise ValueError('OOF overlap or late target')
            oof['calibration_weight'] = calibration_weights(oof)
            oof.to_csv(str(prefix)+'_OOF_calibration_input_BEFORE_FIT.csv', index=False, encoding='utf-8-sig')
            b.write_new(str(prefix)+'_calibrator_STARTED.json', dict(formed_at=b.now(), rows=len(oof), input_sha256=b.sha(str(prefix)+'_OOF_calibration_input_BEFORE_FIT.csv'), fit_number=map_fits+1))
            calibrator = LogisticRegression(**CALIBRATOR_PARAMS)
            with threadpool_limits(limits=2):
                calibrator.fit(logit_column(oof.raw_score), oof.target.astype(int), sample_weight=oof.calibration_weight)
            map_fits += 1
            calibrated_scores(calibrator, [.1,.5,.9])  # Reject a reversing slope.
            joblib.dump(dict(calibrator=calibrator, features=fields, plan=plan, params=CALIBRATOR_PARAMS), str(prefix)+'_calibrator.joblib')
            saved = joblib.load(plan['final_model_path'])
            if b.sha(plan['final_model_path']) != plan['final_model_sha256']:
                raise ValueError('frozen final classifier changed')
            test = full[full.date.str[:7].eq(plan['month'])].copy()
            with threadpool_limits(limits=2):
                raw = saved['model'].predict_proba(test[fields])[:, list(saved['model'].classes_).index(1)]
            baseline = fixed.set_index(['date','code']).loc[pd.MultiIndex.from_frame(test[['date','code']])].reset_index()
            np.testing.assert_allclose(raw, baseline.b_score, atol=1e-14, rtol=0)
            prediction = apply_head(baseline, raw, calibrated_scores(calibrator, raw))
            prediction['calibrator_month'] = plan['month']
            prediction.to_csv(str(prefix)+'_predictions_BEFORE_EVALUATION.csv', index=False, encoding='utf-8-sig')
            predictions.append(prediction)
            b.write_new(str(prefix)+'_COMPLETED.json', dict(formed_at=b.now(),
                inner_HGB_fits=3, calibrator_fits=1, final_classifier_refits=0,
                calibrator_coef=float(calibrator.coef_[0,0]), calibrator_intercept=float(calibrator.intercept_[0]),
                calibration_rows=len(oof), calibration_days=oof.date.nunique(),
                calibrator_sha256=b.sha(str(prefix)+'_calibrator.joblib'), prediction_sha256=b.sha(str(prefix)+'_predictions_BEFORE_EVALUATION.csv')))
            print(plan['month'], 'saved', len(prediction), 'inner', hgb_fits, 'maps', map_fits, flush=True)
        new = pd.concat(predictions, ignore_index=True)
        if len(new) != 3777 or hgb_fits != 48 or map_fits != 16:
            raise ValueError('incomplete fixed calibration comparison')
        all_heads = pd.concat([refs, new], ignore_index=True)
        pd.testing.assert_frame_equal(all_heads.iloc[:len(refs)][refs.columns], refs, check_dtype=False, check_exact=True)
        all_heads.to_csv(output/'all30216_predictions_BEFORE_EVALUATION.csv', index=False, encoding='utf-8-sig')
        paired = pd.concat([fixed, new], ignore_index=True)
        result, stats, ranks, picks = evaluate(paired)
        stats.to_csv(output/'full_same_scope_classification_metrics.csv', index=False, encoding='utf-8-sig')
        ranks.to_csv(output/'same_full_slot_Top1Top2Top3.csv', index=False, encoding='utf-8-sig')
        picks.to_csv(output/'all_selected_rows_WITH_UNKNOWN.csv', index=False, encoding='utf-8-sig')
        current = new[new.date.eq('2026-10-15')]
        if len(current) != 6 or current.target.notna().any() or current.label_valid.fillna(False).any():
            raise ValueError('saved current6 target16 must stay unknown')
        current.to_csv(output/'current15_six_target16_UNMATURED.csv', index=False, encoding='utf-8-sig')
        result.update(inner_HGB_fits=hgb_fits, calibrator_fits=map_fits, total_new_fits=hgb_fits+map_fits,
            old_final_classifier_refits=0, paired_predictions=len(all_heads), elapsed_seconds=time.monotonic()-start,
            source_bindings=protocol['source_bindings'] | {str(p.resolve()):b.sha(p) for p in output.iterdir() if p.is_file()})
        b.verify(result['source_bindings']); b.write_new(output/'calibration_fixed_result.json', result)
        return dict(new_fits=64, screen_pass=result['fixed_screen_pass'], elapsed_seconds=result['elapsed_seconds'])
    except Exception as exc:
        b.write_new(ROOT/'FAILURE_STOP.json', dict(formed_at=b.now(), type=type(exc).__name__, error=str(exc),
            inner_HGB_fits=hgb_fits, calibrator_fits=map_fits, retry_authorized=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','run'])
    parser.add_argument('--tests-xml')
    args = parser.parse_args()
    print(prepare(args.tests_xml) if args.command == 'prepare' else run())
