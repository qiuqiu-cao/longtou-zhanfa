"""One fixed chronological positive-candidate filtering study.

The first-stage SOURCE models are immutable. Two small, unweighted Logistic
heads use genuinely out-of-training first-stage predictions. No network/API,
production changes, or post-result parameter retries are involved.
"""
from pathlib import Path
from fractions import Fraction
import argparse
import hashlib
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import short_term_thinking_dataset_builder as b
import model_upgrade_history as history
import expectation_345_evaluation as metrics
import relay_source_feature_research as source
import relay_temporal_score_calibration as temporal
import relay_iteration_research as iteration
import relay_local_recovery as recovery

ROOT = b.ROOT/'research/model_upgrade_validation/positive_filter_20261016T091007Z'
RECOVERY = b.ROOT/'research/local_repair_20261016T051940Z/recovery_manifest.json'
PRIOR = iteration.ROOT/'delivery/combined_parameter_delivery_verified_NO_UPGRADE.json'
REFERENCE = iteration.ROOT/'execution/all41547_predictions_BEFORE_EVALUATION.csv'
INPUT = temporal.ROOT/'fixed_input_BEFORE_ANY_FIT.joblib'
SHAPES = source.ROOT/'prepared.joblib'
BASE_ARM = source.ARM
ARMS = ['POSITIVE_FILTER_SCORE_ONLY', 'POSITIVE_FILTER_INDIVIDUAL_CONTEXT']
INDIVIDUAL = ['boards', 'height_gap', 'same_height_rivals', 'turnover',
    'amplitude', 'log_signal_amount', 'price_same_height_percentile',
    'signal_amount_same_height_percentile', 'turnover_same_height_percentile']
FEATURES = {ARMS[0]: ['base_score_logit'], ARMS[1]: ['base_score_logit']+INDIVIDUAL}
PARAMS = dict(C=1., solver='lbfgs', max_iter=2000, class_weight=None,
              random_state=20261016)
THRESHOLDS = [0.]+[i/10 for i in range(1, 10)]
KEYS = ['date', 'code']


def verify(bindings):
    return recovery.verify_sources(bindings, RECOVERY)


def digest(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()


def logit(raw):
    values = np.asarray(raw, float)
    if not np.isfinite(values).all() or ((values < 0)|(values > 1)).any():
        raise ValueError('finite first-stage scores in0..1 required')
    values = np.clip(values, 1e-12, 1-1e-12)
    return np.log(values)-np.log1p(-values)


def eligible(frame):
    return (frame.daily_one_word.eq(0) & frame.verified_current_price_shape.eq(0)
            & frame.base_call.eq('PROMOTE'))


def select_meta_rows(oof, before):
    if oof.duplicated(KEYS).any():
        raise ValueError('duplicate OOF candidate; never mix monthly caches')
    known = oof.label_valid.fillna(False).astype(bool) & oof.target.isin([0, 1])
    return oof[eligible(oof) & known & oof.date.lt(before)
               & oof.target_date.notna() & oof.target_date.lt(before)].copy()


def qualification(frame):
    positive = int(frame.target.eq(1).sum())
    negative = int(frame.target.eq(0).sum())
    days = frame.date.nunique()
    return dict(rows=len(frame), positives=positive, negatives=negative,
                signal_days=days, qualified=positive >= 20 and negative >= 20 and days >= 20)


def apply_filter(frame, secondary_score, threshold):
    """The explicit zero sentinel bypasses filtering even for zero scores."""
    if threshold not in THRESHOLDS:
        raise ValueError('unregistered filter threshold')
    score = np.asarray(secondary_score, float)
    if len(frame) != len(score):
        raise ValueError('complete ordered secondary score required')
    mask = eligible(frame).to_numpy()
    if mask.any() and (not np.isfinite(score[mask]).all()
                       or ((score[mask] < 0)|(score[mask] > 1)).any()):
        raise ValueError('eligible secondary score missing or invalid')
    result = frame.copy()
    reject = mask & (score <= threshold) if threshold else np.zeros(len(frame), bool)
    result['filter_score'] = score
    result['filter_threshold'] = threshold
    result['filter_applied'] = reject
    result['filter_reason'] = np.where(reject, 'SECONDARY_SCORE_NOT_ABOVE_FROZEN_THRESHOLD',
        np.where(mask, 'EXPLICIT_NO_FILTER' if threshold == 0 else 'RETAIN_BASE_POSITIVE',
                 'INHERIT_ORIGINAL_NONPOSITIVE_OR_INELIGIBLE'))
    result['research_call'] = np.where(reject, 'FAIL', result.base_call)
    result['call'] = result.research_call
    result['evaluation_call'] = result.research_call
    result['event_probability_validated'] = False
    return result


def statistics(frame):
    return metrics.classification(frame)


def fraction(stat, kind):
    tp, fp, pos = stat['tp'], stat['fp'], stat['positives']
    if kind == 'Precision':
        return Fraction(tp, tp+fp) if tp+fp else None
    if kind == 'Recall':
        return Fraction(tp, pos) if pos else None
    if kind == 'F1':
        return Fraction(2*tp, tp+fp+pos) if tp+fp+pos else None
    raise ValueError(kind)


def choose_threshold(stats):
    base = stats[0.]
    candidates = []
    for threshold in THRESHOLDS[1:]:
        value = stats[threshold]
        p, r, f = [fraction(value, k) for k in ['Precision', 'Recall', 'F1']]
        bp, br, bf = [fraction(base, k) for k in ['Precision', 'Recall', 'F1']]
        if (None not in [p, r, f, bp, br, bf] and p > bp
            and r >= br-Fraction(1, 20) and f >= bf
            and value['fp']*10 <= base['fp']*9):
            candidates.append(threshold)
    return min(candidates, key=lambda t: (-fraction(stats[t], 'Precision'), t)) if candidates else 0.


def positive_top3(frame):
    """Select before reading labels. Unknown outcomes and empty slots remain."""
    days = sorted(frame.date.unique())
    pool = frame[frame.daily_one_word.eq(0) & frame.call.eq('PROMOTE')]
    picks = pool.sort_values(['date', 'ranking_score', 'code'], ascending=[True, False, True],
                            na_position='last').groupby('date', sort=True).head(3).copy()
    known = picks.label_valid.fillna(False).astype(bool) & picks.target.notna()
    count = dict(signal_days=len(days), full_slots=3*len(days), selected=len(picks),
        empty_slots=3*len(days)-len(picks), unknown_selected=int((~known).sum()),
        hits=int((known & picks.target.eq(1)).sum()))
    return count, picks


def fit_head(rows, arm, output, month, phase):
    columns = FEATURES[arm]
    info = qualification(rows)
    if not info['qualified']:
        return None, dict(**info, new_fits=0, reason='INSUFFICIENT_META_LABELS_OR_DAYS')
    name = f'{month}_{arm}_{phase}'
    receipt = dict(formed_at=b.now(), month=month, arm=arm, phase=phase,
        features=columns, estimator=PARAMS, weights='Every row weight1; no class balancing',
        training_digest=digest(rows[KEYS+['target_date', 'target']+columns]),
        training_dtypes={c:str(rows[c].dtype) for c in KEYS+['target_date','target']+columns},
        max_training_target=rows.target_date.max(), **info)
    b.write_new(output/f'{name}_FIT_STARTED.json', receipt)
    model = Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True,
                                               keep_empty_features=True)),
                      ('scale', StandardScaler()), ('logistic', LogisticRegression(**PARAMS))])
    with warnings.catch_warnings(), threadpool_limits(limits=2):
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(rows[columns], rows.target.astype(int))
    if int(model.named_steps['logistic'].n_iter_.max()) >= PARAMS['max_iter']:
        raise ValueError('fixed second-stage estimator did not converge; no retry')
    saved = receipt | dict(model=model, second_stage_score_not_certified_probability=True)
    path = output/f'{name}_model.joblib'
    joblib.dump(saved, path, compress=3)
    b.write_new(output/f'{name}_FIT_COMPLETED.json', receipt | dict(completed_at=b.now(),
        model_path=str(path.resolve()), model_sha256=b.sha(path),
        n_iter=int(model.named_steps['logistic'].n_iter_.max()), new_fits=1))
    return model, receipt | dict(model_path=str(path.resolve()), model_sha256=b.sha(path), new_fits=1)


def scores(model, frame, arm):
    values = np.full(len(frame), np.nan)
    mask = eligible(frame).to_numpy()
    if model is None:
        values[mask] = 0.
    elif mask.any():
        with threadpool_limits(limits=2):
            values[mask] = model.predict_proba(frame.loc[mask, FEATURES[arm]])[:,
                list(model.named_steps['logistic'].classes_).index(1)]
    return values


def complete_oof(data, shapes, plan, output=None):
    """Complete cached validation dates, including rows with unknown labels."""
    full, fields = data['full'], data['fields']
    window = history.training_rows(full, plan['cutoff'], 'relay')
    if history._frame_digest(window, fields) != plan['training_digest']:
        raise ValueError('fixed252 member/label/field digest drift')
    frames, checks = [], []
    for block, (past, valid) in enumerate(temporal.temporal_splits(window), 1):
        model_path = temporal.ROOT/'execution'/f"{plan['month']}_inner{block}_model.joblib"
        cache_path = temporal.ROOT/'execution'/f"{plan['month']}_inner{block}_all_validation_scores_BEFORE_EFFECTS.csv"
        saved = joblib.load(model_path)
        if saved['training_digest'] != history._frame_digest(past, fields) or saved['features'] != fields:
            raise ValueError('cached first-stage training identity drift')
        complete = full[full.date.isin(valid.date.unique())].copy()
        if complete.duplicated(KEYS).any():
            raise ValueError('duplicate complete validation identity')
        with threadpool_limits(limits=2):
            raw = saved['model'].predict_proba(complete[fields])[:, 1]
        complete['raw_score'] = raw
        cache = pd.read_csv(cache_path, dtype={'code': str}, float_precision='round_trip')
        ordered = complete.set_index(KEYS).loc[pd.MultiIndex.from_frame(cache[KEYS])]
        np.testing.assert_allclose(ordered.raw_score, cache.raw_score, atol=1e-14, rtol=0)
        pd.testing.assert_frame_equal(cache[['date','code','target_date','target','label_valid',
            'frozen_original_nonword_identity']].reset_index(drop=True),
            valid[['date','code','target_date','target','label_valid',
            'frozen_original_nonword_identity']].reset_index(drop=True), check_dtype=False, check_exact=True)
        complete = complete.merge(shapes[KEYS+['verified_current_price_shape']], on=KEYS, validate='one_to_one')
        complete['daily_one_word'] = complete.frozen_original_nonword_identity
        complete = complete[complete.daily_one_word.eq(0)].copy()
        if complete.verified_current_price_shape.eq(1).any():
            raise ValueError('OOF original-B contains A route; no unregistered A-model substitution')
        complete['base_call'] = np.where(complete.verified_current_price_shape.eq(0),
            np.where(complete.raw_score.gt(.5), 'PROMOTE', 'FAIL'), 'UNKNOWN')
        complete['base_score_logit'] = logit(complete.raw_score)
        complete['ranking_score'] = complete.raw_score
        complete['call'] = complete.base_call
        complete['inner_block'] = block
        complete['was_in_original_cached_validation'] = pd.MultiIndex.from_frame(complete[KEYS]).isin(
            pd.MultiIndex.from_frame(cache[KEYS]))
        checks.append(dict(month=plan['month'], block=block, original_cache_rows=len(cache),
            full_validation_rows=len(full[full.date.isin(valid.date.unique())]),
            original_B_rows=len(complete), original_B_unknown_outcomes=int((~(
                complete.label_valid.fillna(False).astype(bool) & complete.target.notna())).sum()),
            cache_completion_B_rows=int((~complete.was_in_original_cached_validation).sum()),
            first_validation=str(valid.date.min()), last_validation=str(valid.date.max()),
            max_first_stage_training_target=past.target_date.max(), first_stage_new_fits=0))
        frames.append(complete)
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(KEYS).any():
        raise ValueError('OOF temporal block overlap')
    if output is not None:
        result.to_csv(output/f"{plan['month']}_complete_B_OOF_BEFORE_SELECTION.csv", index=False, encoding='utf-8-sig')
    return result, checks


def prepare(tests_xml, lessons_receipt):
    if (ROOT/'protocol_BEFORE_ANY_FIT.json').exists():
        raise FileExistsError('one fixed positive-filter study only')
    import xml.etree.ElementTree as ET
    suites = list(ET.parse(tests_xml).getroot().iter('testsuite'))
    if not suites or any(int(s.get('errors', 0))+int(s.get('failures', 0)) for s in suites):
        raise ValueError('passing targeted preflight required')
    prior = b.read(PRIOR)
    checks = verify(prior['source_bindings'])
    lessons = b.read(lessons_receipt)
    lesson_sources = lessons.get('source_bindings', {}) | lessons.get('output_bindings', {})
    if not lesson_sources:
        raise ValueError('source-bound history summary must be saved before protocol')
    verify(lesson_sources)
    data = joblib.load(INPUT)
    shapes = joblib.load(SHAPES)['verified_shapes']
    if len(data['full']) != 9315 or len(data['fields']) != 92 or len(data['plans']) != 16:
        raise ValueError('fixed source input9315/92/16 required')
    checks_rows = []
    bindings = prior['source_bindings'] | lesson_sources
    for plan in data['plans']:
        window = history.training_rows(data['full'], plan['cutoff'], 'relay')
        if history._frame_digest(window, data['fields']) != plan['training_digest']:
            raise ValueError('same252 training digest drift')
        for block, (past, valid) in enumerate(temporal.temporal_splits(window), 1):
            p = temporal.ROOT/'execution'/f"{plan['month']}_inner{block}_model.joblib"
            c = temporal.ROOT/'execution'/f"{plan['month']}_inner{block}_all_validation_scores_BEFORE_EFFECTS.csv"
            for path in [p, c]:
                if bindings.get(str(path.resolve())) != b.sha(path):
                    raise ValueError('first-stage cache not protected by existing receipt')
            checks_rows.append(dict(month=plan['month'], block=block,
                first_validation=valid.date.min(), max_train_target=past.target_date.max(),
                train_days=past.date.nunique(), validation_days=valid.date.nunique()))
    ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks_rows).to_csv(ROOT/'all48_cached_training_boundaries.csv', index=False, encoding='utf-8-sig')
    joblib.dump(data | dict(shapes=shapes), ROOT/'fixed_input_BEFORE_ANY_FIT.joblib', compress=3)
    for path in [Path(__file__), b.ROOT/'tests/test_relay_positive_filter_research.py',
                 Path(tests_xml), Path(lessons_receipt), RECOVERY, PRIOR, REFERENCE, INPUT, SHAPES,
                 ROOT/'fixed_input_BEFORE_ANY_FIT.joblib', ROOT/'all48_cached_training_boundaries.csv']:
        bindings[str(path.resolve())] = b.sha(path)
    b.write_new(ROOT/'protocol_BEFORE_ANY_FIT.json', dict(formed_at=b.now(),
        user_authorization='PLEASE IMPLEMENT THIS PLAN; Precision primary; maximum Recall loss5pp for this new study only',
        hypothesis='Individual price/volume/cohort facts may distinguish TP and FP inside immutable SOURCE positives',
        arms=ARMS, features=FEATURES, estimator=PARAMS, sample_weights='Every row1; class_weight=None',
        thresholds=THRESHOLDS, explicit_zero_sentinel=True,
        baseline=BASE_ARM, baseline_historical=dict(tp=225, fp=303, known=2932, unknown=48, positives=871),
        first_stage_refits=0, max_second_stage_fits=64, fitting_budget_seconds=1800,
        split='Each month separate: first two cached chronological OOF blocks train meta; target_date strictly before block3 first signal; block3 selects threshold only; final refit all3 known eligible OOF with target_date strictly before month cutoff',
        oof_completion='Use same saved first-stage model to score complete validation dates including unknown-label rows; reproduce all original cached scores; completion is inference only and not original-time receipt',
        meta_eligibility='Original frozen B, verified shape0, original raw>.5, known mature target, minimum20 positive20 negative20 signal dates',
        inner_selection='Complete original-B counts; strict Precision improvement; Recall loss<=.05, F1>=baseline, FP<=.9 baseline; maximize exact Precision then lower threshold; otherwise explicit no-filter0',
        routing='Only original-B verified0 base PROMOTE may become research FAIL; all other directions unchanged; original scores/rankings remain exact',
        acceptance=dict(pooled_precision_gain=.03, pooled_FP_fraction_max=.9, pooled_F1_not_lower=True,
            each_historical_fold_Recall_loss_max=.05, each_historical_fold_FP_not_higher=True,
            each_historical_fold_F1_not_lower=True, pooled_positive_recommendation_Top3_hits_not_lower=True),
        bootstrap=dict(signal_day_block=5, replicates=2000, seed=20261016,
            two_arm_pointwise_interval_quantiles=[.0125,.9875], exploratory_not_confirmatory=True),
        known_risks=['First-stage OOF training lengths62/125/188 vs outer252',
            'Final meta refit changes preprocessing, coefficients and score scale; frozen threshold may not transfer',
            'Previously seen history, late-source and survivor-scope limits remain'],
        current_target16_new_outcomes_read=False, October_supplemental_only=True,
        old_failures_requalified=False, event_probability_validated=False, production_deployed=False,
        source_recovery=checks, source_bindings=bindings))
    return dict(protocol_frozen=True, first_stage_new_fits=0, maximum_new_fits=64)


def run():
    protocol = b.read(ROOT/'protocol_BEFORE_ANY_FIT.json')
    verify(protocol['source_bindings'])
    if protocol['features'] != FEATURES or protocol['thresholds'] != THRESHOLDS or protocol['estimator'] != PARAMS:
        raise ValueError('fixed protocol/config drift')
    b.write_new(ROOT/'experiment_STARTED.json', dict(formed_at=b.now(), protocol_sha256=b.sha(ROOT/'protocol_BEFORE_ANY_FIT.json')))
    output = ROOT/'execution'
    output.mkdir(exist_ok=False)
    data = joblib.load(ROOT/'fixed_input_BEFORE_ANY_FIT.joblib')
    refs = pd.read_csv(REFERENCE, dtype={'code': str}, float_precision='round_trip', low_memory=False)
    baseline = refs[refs.arm.eq(BASE_ARM)].copy()
    if len(refs) != 41547 or len(baseline) != 3777:
        raise ValueError('all11 immutable old heads and complete SOURCE rows required')
    full, shapes = data['full'], data['shapes']
    results, selections, boundaries = [], [], []
    start = time.monotonic()
    try:
        for plan in data['plans']:
            if time.monotonic()-start > 1800:
                raise TimeoutError('fixed30min fitting budget exhausted')
            month = plan['month']
            oof, check = complete_oof(data, shapes, plan, output)
            boundaries.extend(check)
            first_third = oof[oof.inner_block.eq(3)].date.min()
            prefix = select_meta_rows(oof[oof.inner_block.lt(3)], first_third)
            final = select_meta_rows(oof, plan['cutoff'])
            validation = oof[oof.inner_block.eq(3)].copy()
            prefix.to_csv(output/f'{month}_meta_prefix_training.csv', index=False, encoding='utf-8-sig')
            final.to_csv(output/f'{month}_meta_final_training.csv', index=False, encoding='utf-8-sig')
            choices = {}
            for arm in ARMS:
                fitted, fit_receipt = fit_head(prefix, arm, output, month, 'PREFIX')
                raw = scores(fitted, validation, arm)
                table = {t: statistics(apply_filter(validation, raw, t)) for t in THRESHOLDS}
                selected = choose_threshold(table) if fitted is not None else 0.
                choices[arm] = dict(selected_threshold=selected, prefix_fit=fit_receipt,
                    statistics={str(t): s for t,s in table.items()},
                    choice_only_no_unbiased_validation_claim=True)
            selection = dict(formed_at=b.now(), month=month, outer_cutoff=plan['cutoff'],
                first_third_signal=first_third, max_prefix_target=prefix.target_date.max(),
                max_final_target=final.target_date.max(), choices=choices,
                outer_results_used_for_selection=False)
            b.write_new(output/f'{month}_SELECTION_BEFORE_FINAL_FIT_AND_OUTER.json', selection)
            month_ref = baseline[baseline.date.str[:7].eq(month)].copy()
            for arm in ARMS:
                fitted, final_receipt = fit_head(final, arm, output, month, 'FINAL')
                threshold = choices[arm]['selected_threshold'] if fitted is not None else 0.
                inference = month_ref.copy()
                inference['base_call'] = inference.evaluation_call
                # Preserve the exact immutable source columns; add only missing model features.
                lookup = full.set_index(KEYS)
                index = pd.MultiIndex.from_frame(inference[KEYS])
                for field in INDIVIDUAL:
                    values = lookup.loc[index, field].to_numpy()
                    if field in inference:
                        np.testing.assert_allclose(inference[field].to_numpy(float), values.astype(float), equal_nan=True, atol=0, rtol=0)
                    else:
                        inference[field] = values
                inference['base_score_logit'] = logit(inference.b_score)
                new = apply_filter(inference, scores(fitted, inference, arm), threshold)
                new['arm'] = arm
                new['second_stage_head_kind'] = 'CONDITIONAL_POSITIVE_CANDIDATE_LOGISTIC'
                new['research_validation_status'] = 'FIXED_RESEARCH_NOT_ADOPTED'
                # Public aligned call fields refer to this research head, originals remain explicit.
                for name in ['aligned_evaluation_call', 'recovered_B_call']:
                    if name in new:
                        new[name] = new.research_call
                new.to_csv(output/f'{month}_{arm}_predictions_BEFORE_EVALUATION.csv', index=False, encoding='utf-8-sig')
                results.append(new)
                selections.append(dict(month=month, arm=arm, selected_threshold=threshold,
                    prefix_qualified=choices[arm]['prefix_fit']['qualified'], final_qualified=final_receipt['qualified'],
                    max_prefix_target=prefix.target_date.max(), first_third_signal=first_third,
                    max_final_target=final.target_date.max(), cutoff=plan['cutoff']))
                b.write_new(output/f'{month}_{arm}_COMPLETED.json', dict(completed_at=b.now(),
                    selection_formed_at=selection['formed_at'], selected_threshold=threshold,
                    prefix_fit=choices[arm]['prefix_fit'], final_fit=final_receipt,
                    first_stage_new_fits=0, outer_results_used_for_selection=False))
            print(month, 'thresholds', {arm: choices[arm]['selected_threshold'] for arm in ARMS}, flush=True)
        new = pd.concat(results, ignore_index=True)
        if any(len(part) != 3777 for _,part in new.groupby('arm')):
            raise ValueError('both new heads require complete3777 candidates')
        all_heads = pd.concat([refs,new], ignore_index=True)
        pd.testing.assert_frame_equal(all_heads.iloc[:len(refs)][refs.columns], refs, check_exact=True, check_dtype=False)
        all_heads.to_csv(output/'all49101_predictions_BEFORE_EVALUATION.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(selections).to_csv(output/'all32_training_only_selections.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(boundaries).to_csv(output/'all48_first_stage_cache_replays_and_completion.csv', index=False, encoding='utf-8-sig')
        current = new[new.date.eq('2026-10-15')]
        if len(current) != 12 or current.target.notna().any() or current.label_valid.fillna(False).any():
            raise ValueError('current six per arm must remain unknown')
        current.to_csv(output/'current15_six_per_arm_UNKNOWN.csv', index=False, encoding='utf-8-sig')
        started = list(output.glob('*_FIT_STARTED.json'))
        completed = list(output.glob('*_FIT_COMPLETED.json'))
        if len(started) != len(completed) or len(started) > 64:
            raise ValueError('research fit budget or completion mismatch')
        verify(protocol['source_bindings'])
        bindings = protocol['source_bindings'] | {str(p.resolve()): b.sha(p) for p in output.iterdir() if p.is_file()}
        b.write_new(output/'run_COMPLETED_BEFORE_EVALUATION.json', dict(completed_at=b.now(),
            new_second_stage_fits=len(started), first_stage_new_fits=0,
            complete_predictions=len(all_heads), old_rows_exact=len(refs),
            elapsed_seconds=time.monotonic()-start, source_bindings=bindings))
        return dict(second_stage_fits=len(started), complete_predictions=len(all_heads), effects_not_yet_evaluated=True)
    except Exception as exc:
        b.write_new(ROOT/'FAILURE_STOP.json', dict(formed_at=b.now(), error_type=type(exc).__name__,
            error=str(exc), new_fit_attempts=len(list(output.glob('*_FIT_STARTED.json'))),
            completed_fits=len(list(output.glob('*_FIT_COMPLETED.json'))), retry_authorized=False))
        raise


def evaluate():
    completed = b.read(ROOT/'execution/run_COMPLETED_BEFORE_EVALUATION.json')
    verify(completed['source_bindings'])
    output = ROOT/'evaluation'
    output.mkdir(exist_ok=False)
    all_heads = pd.read_csv(ROOT/'execution/all49101_predictions_BEFORE_EVALUATION.csv',
                            dtype={'code':str}, float_precision='round_trip', low_memory=False)
    chosen = all_heads[all_heads.arm.isin(['BASE','ST_DUAL',BASE_ARM]+ARMS)].copy()
    records, ranks, recommendations, picks = [], [], [], []
    for period in temporal.PERIODS:
        part = chosen[chosen.date.between(period['start'],period['end'])]
        for arm, frame in part.groupby('arm',sort=True):
            frame = frame.copy(); frame['call'] = frame.evaluation_call
            for scope, group in [('FULL_POOL',frame),('ORIGINAL_NONWORD',frame[frame.daily_one_word.eq(0)])]:
                records.append(dict(period=period['id'],arm=arm,scope=scope,**statistics(group)))
            for k in [1,2,3]:
                stat, selection = source.full_slot_ranking(frame,k)
                ranks.append(dict(period=period['id'],arm=arm,**stat))
            stat, selection = positive_top3(frame)
            recommendations.append(dict(period=period['id'],arm=arm,**stat))
            picks.append(selection.assign(period=period['id'],selected_arm=arm))
    records, ranks, recs = pd.DataFrame(records), pd.DataFrame(ranks), pd.DataFrame(recommendations)
    table = records[records.scope.eq('ORIGINAL_NONWORD')].set_index(['period','arm'])
    acceptance = {}
    for arm in ARMS:
        a, z = table.loc[('HISTORICAL_POOLED',arm)], table.loc[('HISTORICAL_POOLED',BASE_ARM)]
        guards = dict(pooled_Precision_gain3pp=bool(a.Precision >= z.Precision+.03),
            pooled_FP_drop10percent=bool(a.fp*10 <= z.fp*9), pooled_F1_not_lower=bool(a.F1 >= z.F1))
        for fold in history.FOLDS:
            a,z = table.loc[(fold['id'],arm)], table.loc[(fold['id'],BASE_ARM)]
            guards[fold['id']+'_Recall_loss_within5pp'] = bool(a.Recall >= z.Recall-.05)
            guards[fold['id']+'_FP_not_higher'] = bool(a.fp <= z.fp)
            guards[fold['id']+'_F1_not_lower'] = bool(a.F1 >= z.F1)
        r = recs[recs.period.eq('HISTORICAL_POOLED')].set_index('arm')
        guards['pooled_positive_Top3_hits_not_lower'] = bool(r.loc[arm,'hits'] >= r.loc[BASE_ARM,'hits'])
        acceptance[arm] = dict(guards=guards,fixed_screen_pass=all(guards.values()))
    records.to_csv(output/'complete_same_pool_classification.csv', index=False, encoding='utf-8-sig')
    ranks.to_csv(output/'raw_observation_Top1Top2Top3_INVARIANT.csv', index=False, encoding='utf-8-sig')
    recs.to_csv(output/'positive_recommendation_Top3_FULL_SLOTS.csv', index=False, encoding='utf-8-sig')
    pd.concat(picks,ignore_index=True).to_csv(output/'all_positive_recommendations_WITH_UNKNOWN.csv', index=False, encoding='utf-8-sig')
    filtered = chosen[chosen.arm.isin(ARMS) & chosen.filter_applied.fillna(False)].copy()
    filtered['known_outcome'] = filtered.label_valid.fillna(False).astype(bool) & filtered.target.notna()
    filtered['filter_effect'] = np.where(~filtered.known_outcome,'UNKNOWN_OUTCOME',
        np.where(filtered.target.eq(1),'NEW_MISSED_PROMOTION','REMOVED_FALSE_POSITIVE'))
    filtered.to_csv(output/'all_filtered_candidates_GAINS_LOSSES_UNKNOWN.csv', index=False, encoding='utf-8-sig')
    bootstrap = bootstrap_intervals(chosen)
    bootstrap.to_csv(output/'exploratory_5day_block_intervals.csv', index=False, encoding='utf-8-sig')
    result = dict(formed_at=b.now(),arms=acceptance, new_second_stage_fits=completed['new_second_stage_fits'],
        first_stage_new_fits=0, history_previously_seen_not_blind=True,
        threshold_transfer_risk_not_eliminated=True, event_probability_validated=False,
        production_deployed=False, no_post_effect_retry=True,
        source_bindings=completed['source_bindings'] | {str(p.resolve()):b.sha(p) for p in output.iterdir() if p.is_file()})
    verify(result['source_bindings'])
    b.write_new(output/'fixed_study_result.json',result)
    return dict(fits=result['new_second_stage_fits'], arms={k:v['fixed_screen_pass'] for k,v in acceptance.items()})


def bootstrap_intervals(chosen):
    rows = []
    for period in temporal.PERIODS:
        sample = chosen[chosen.date.between(period['start'],period['end']) & chosen.daily_one_word.eq(0)]
        dates = sorted(sample.date.unique())
        if not dates:
            continue
        day_stats = {}
        for arm in [BASE_ARM]+ARMS:
            one = sample[sample.arm.eq(arm)].copy();one['call'] = one.evaluation_call
            day_stats[arm] = np.asarray([[s['tp'],s['fp'],s['positives'],s['known']-s['positives']]
                for d in dates for s in [statistics(one[one.date.eq(d)])]],float)
        rng = np.random.default_rng(20261016)
        starts = rng.integers(0,len(dates),size=(2000,int(np.ceil(len(dates)/5))))
        indices = ((starts[:,:,None]+np.arange(5))%len(dates)).reshape(2000,-1)[:,:len(dates)]
        def measures(values):
            tp,fp,pos,neg = values.T
            return dict(Precision=np.divide(tp,tp+fp,out=np.full(2000,np.nan),where=(tp+fp)>0),
                Recall=np.divide(tp,pos,out=np.full(2000,np.nan),where=pos>0),
                F1=np.divide(2*tp,tp+fp+pos,out=np.full(2000,np.nan),where=(tp+fp+pos)>0),FP=fp)
        old = measures(day_stats[BASE_ARM][indices].sum(axis=1))
        for arm in ARMS:
            new = measures(day_stats[arm][indices].sum(axis=1))
            for name in old:
                delta = new[name]-old[name];finite = delta[np.isfinite(delta)]
                rows.append(dict(period=period['id'],arm=arm,metric=name,
                    lower=float(np.quantile(finite,.0125)) if len(finite) else None,
                    upper=float(np.quantile(finite,.9875)) if len(finite) else None,
                    finite_replicates=len(finite),total_replicates=2000,block_signal_days=5,
                    two_arms_adjustment_only=True,previous_sequential_trials_not_adjusted=True,
                    exploratory_not_confirmatory=True))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','evaluate'])
    parser.add_argument('--tests-xml');parser.add_argument('--lessons-receipt')
    args=parser.parse_args()
    result = prepare(args.tests_xml,args.lessons_receipt) if args.command=='prepare' else run() if args.command=='run' else evaluate()
    print(result)
