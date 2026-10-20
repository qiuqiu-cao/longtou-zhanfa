"""Append-only review of saved predictions; never fit or alter the experiment."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RUN = OUT.parent
PROJECT = RUN.parents[2]
ARMS = {'ROLL252': 'roll_score', 'PULLBACK_PERSONALITY': 'new_score', 'FROZEN_BASE': 'base_score_saved'}
FOLDS = [('2025H2', '2025-07-01', '2025-12-31'),
         ('2026JanAug', '2026-01-01', '2026-08-31'),
         ('2026Sep', '2026-09-01', '2026-09-30')]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.bool_, np.integer)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write(name, value):
    with (OUT / name).open('x', encoding='utf-8') as stream:
        json.dump(clean(value), stream, ensure_ascii=False, indent=2, allow_nan=False)


def check_originals(scope):
    for path, digest in {**scope['source_artifacts_before_changes'], **scope['protected_inputs']}.items():
        assert sha(path) == digest, path


def valid_label(frame):
    return frame.label_valid.fillna(False).astype(bool) & frame.target.notna()


def metric(frame, score):
    valid = valid_label(frame) & frame[score].notna()
    sub = frame[valid]
    positive = sub.target.eq(1)
    chosen = sub[score].gt(.5)
    tp = int((positive & chosen).sum())
    fp = int((~positive & chosen).sum())
    fn = int((positive & ~chosen).sum())
    tn = int((~positive & ~chosen).sum())
    return dict(total=len(frame), known=len(sub), unknown=len(frame) - len(sub),
                TP=tp, FP=fp, FN=fn, TN=tn,
                Accuracy=(tp+tn)/len(sub) if len(sub) else None,
                Precision=tp/(tp+fp) if tp+fp else None,
                Recall=tp/(tp+fn) if tp+fn else None,
                F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)


def top(frame, score, k, complete_dates):
    chosen = frame.sort_values(['date', score, 'code'], ascending=[True, False, True], kind='stable').groupby('date', sort=False).head(k)
    valid = valid_label(chosen)
    return dict(selected=len(chosen), known=int(valid.sum()), unknown=int((~valid).sum()),
                hits=int(chosen.loc[valid, 'target'].sum()), days=chosen.date.nunique(),
                complete_pool_signal_dates=complete_dates,
                no_subgroup_candidate_dates=complete_dates-chosen.date.nunique(),
                known_pick_hit_rate=float(chosen.loc[valid, 'target'].mean()) if valid.any() else None,
                outcome_coverage=float(valid.mean()) if len(chosen) else None)


def bootstrap_official(frame, sessions, start, end):
    """Zero metric contribution preserves time spacing; it does not invent outcomes."""
    dates = [d for d in sessions if start <= d <= end]
    assert dates and dates == sorted(set(dates))
    known = frame[valid_label(frame)].copy()
    assert set(frame.date).issubset(dates)
    counts = np.zeros((len(dates), 6), dtype=np.int64)
    date_index = {d: i for i, d in enumerate(dates)}
    for date, group in known.groupby('date', sort=True):
        y = group.target.eq(1)
        for arm, score in enumerate(['new_score', 'roll_score']):
            pred = group[score].gt(.5)
            counts[date_index[date], arm*3:arm*3+3] = [int((y & pred).sum()), int((~y & pred).sum()), int((y & ~pred).sum())]
    n = len(dates)
    rng = np.random.default_rng(20261020)
    starts = rng.integers(0, n, size=(2000, (n+4)//5))
    indices = ((starts[:, :, None] + np.arange(5)) % n).reshape(2000, -1)[:, :n]
    total = counts[indices].sum(axis=1)
    newden = 2*total[:, 0] + total[:, 1] + total[:, 2]
    oldden = 2*total[:, 3] + total[:, 4] + total[:, 5]
    defined = (newden > 0) & (oldden > 0)
    delta = 2*total[defined, 0]/newden[defined] - 2*total[defined, 3]/oldden[defined]
    return dict(lower=float(np.quantile(delta, .025)) if len(delta) else None,
                upper=float(np.quantile(delta, .975)) if len(delta) else None,
                official_sessions=n, candidate_signal_dates=frame.date.nunique(),
                known_contribution_dates=known.date.nunique(),
                dates_without_known_contribution=n-known.date.nunique(),
                draws=2000, undefined_draws=int((~defined).sum()), block=5, seed=20261020,
                axis_start=dates[0], axis_end=dates[-1],
                method='paired circular blocks on complete inherited official calendar; zero counts retain absent/unknown sessions without relabeling outcomes',
                interpretation='Descriptive repeated historical research; no selection-adjusted confidence guarantee.')


def direction_changes(frame):
    changed = frame.new_score.gt(.5).ne(frame.roll_score.gt(.5))
    f = frame[changed].copy()
    known = valid_label(f)
    f['roll_direction'] = f.roll_score.gt(.5)
    f['new_direction'] = f.new_score.gt(.5)
    f['delta_TP'] = np.where(known, f.target.eq(1).astype(int)*(f.new_direction.astype(int)-f.roll_direction.astype(int)), np.nan)
    f['delta_FP'] = np.where(known, f.target.eq(0).astype(int)*(f.new_direction.astype(int)-f.roll_direction.astype(int)), np.nan)
    return f


def corrected_case(predictions, bars, code, date, sessions):
    row = predictions[predictions.code.eq(code) & predictions.date.eq(date)].iloc[0]
    anchor, launch = row.anchor_date, row.launch_date
    pull_days = [d for d in sessions if anchor < d < launch]
    data = bars[bars.code.eq(code) & bars.date.between(anchor, date)].sort_values('date').copy()
    assert not data.date.duplicated().any()
    anchor_bar = data[data.date.eq(anchor)].iloc[0]
    launch_bar = data[data.date.eq(launch)].iloc[0]
    pull = data[data.date.isin(pull_days)]
    assert len(pull) == len(pull_days) == row.pullback_days
    calculated = dict(pullback_mean_volume_ratio=pull.volume.mean()/anchor_bar.volume,
                      pullback_min_volume_ratio=pull.volume.min()/anchor_bar.volume,
                      pullback_last_volume_ratio=pull.iloc[-1].volume/anchor_bar.volume,
                      relaunch_volume_ratio=launch_bar.volume/pull.volume.mean(),
                      pullback_low_vs_anchor_low=pull.low.min()/anchor_bar.low-1)
    for field, value in calculated.items():
        assert np.isclose(value, row[field], atol=1e-12, rtol=1e-12)
    data[['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'volume_unit']].to_csv(OUT / (code+'_'+date+'_BARS_THROUGH_SIGNAL_ONLY.csv'), index=False, encoding='utf-8-sig')
    return dict(date=date, code=code, name=row['name'], signal_boards=int(row.boards),
                prediction_transition=f'{int(row.boards)}进{int(row.boards)+1}',
                target_date=row.target_date, actual_promoted=bool(row.target),
                anchor_date=anchor, launch_date=launch, pullback_days=len(pull),
                roll_score=row.roll_score, new_score=row.new_score,
                raw_bar_recalculated=calculated,
                bar_volume_unit_missing=int(data.volume_unit.isna().sum()),
                number_of_bars=len(data),
                illustrative_selection='POST_HOC_AFTER_RESULTS; input table ends at signal, outcome stored separately in this case record',
                attribution='Model output changed after jointly adding21features; these two cases do not isolate individual-feature causation or washout intent.')


def main():
    scope = read(OUT / 'review_scope_BEFORE_CHANGES.json')
    check_originals(scope)
    implementation = read(RUN / 'implementation_BEFORE_FITS.json')
    prepared = joblib.load(RUN / 'prepared_FEATURES_BEFORE_FITS.joblib')
    sessions = prepared['sessions']
    frame = pd.read_csv(RUN / 'full_paired_predictions.csv', dtype={'code': str}, float_precision='round_trip', low_memory=False)
    original_results = read(RUN / 'results.json')
    assert len(frame) == 3707 and not frame.duplicated(['date', 'code']).any()
    assert valid_label(frame).sum() == 3640 and frame.boards.ge(2).all()
    assert frame[list(ARMS.values())].notna().all().all()
    # Original monthly scores/models remain byte-identical. Saved replay checks
    # from the earlier Codex review can be reused; this audit adds no inference.
    completed = []
    for plan in implementation['training_plans']:
        month = plan['month']
        receipt = read(RUN / (month+'_fit_COMPLETED.json'))
        assert sha(RUN / (month+'_PULLBACK_PERSONALITY.joblib')) == receipt['model_sha']
        assert sha(RUN / (month+'_predictions_BEFORE_VALIDATION.csv')) == receipt['prediction_sha']
        assert plan['train_max_target'] < plan['cutoff']
        assert plan['train_signal_days'] == 252
        completed.append(dict(month=month, original_model_and_prediction_hashes_match=True,
                              training_targets_strictly_before_month=True))

    bindings = implementation['bindings']
    engine_path = next(k for k in bindings if k.endswith('engine_context_WITH_CURRENT15.joblib'))
    engine = joblib.load(engine_path)
    identity = engine['features'][['date', 'code', 'daily_identity_status']]
    frame = frame.merge(identity, on=['date', 'code'], validate='one_to_one')
    strict = frame.daily_one_word.eq(0)
    original = frame.actual_one_word_daily.eq(0)
    scopes = {
        'all': pd.Series(True, index=frame.index),
        'strict_daily_nonword': strict,
        'original_actual_nonword': original,
        'strict_daily_oneword': frame.daily_one_word.eq(1),
        'strict_identity_unknown': frame.daily_one_word.isna(),
        'cap_pullback': frame.pullback_has_anchor.eq(1),
        'no_pullback': frame.pullback_has_anchor.eq(0),
        'pullback_unknown': frame.pullback_has_anchor.isna(),
    }
    summary = {}
    for fold, start, end in FOLDS:
        in_fold = frame.date.between(start, end)
        dates = frame.loc[in_fold, 'date'].nunique()
        result = {}
        for name, mask in scopes.items():
            part = frame[in_fold & mask]
            result[name] = {arm: dict(metrics=metric(part, score), top={str(k): top(part, score, k, dates) for k in (1, 2, 3)}) for arm, score in ARMS.items()}
        # Compare the original frozen report, using explicit renamed scope only.
        for new_name, old_name in [('all', 'all'), ('strict_daily_nonword', 'actual_nonword'), ('cap_pullback', 'cap_pullback')]:
            for arm in ARMS:
                assert result[new_name][arm]['metrics'] == original_results['folds'][fold][old_name][arm]['metrics']
                for k in ('1', '2', '3'):
                    for key, value in original_results['folds'][fold][old_name][arm]['top'][k].items():
                        assert result[new_name][arm]['top'][k][key] == value
        result['official_calendar_F1_delta95_strict'] = bootstrap_official(frame[in_fold & strict], sessions, start, end)
        result['official_calendar_F1_delta95_original'] = bootstrap_official(frame[in_fold & original], sessions, start, end)
        result['original_compressed_date_interval_retained'] = original_results['folds'][fold]['nonword_F1_delta95']
        summary[fold] = result

    september = frame[frame.date.between('2026-09-01', '2026-09-30')].copy()
    changes = direction_changes(september)
    change_cols = ['date', 'code', 'name', 'boards', 'target_date', 'target', 'label_valid',
                   'actual_one_word_daily', 'daily_one_word', 'daily_identity_status', 'pullback_status',
                   'roll_score', 'new_score', 'roll_direction', 'new_direction', 'delta_TP', 'delta_FP']
    changes[change_cols].to_csv(OUT / 'SEPTEMBER_ALL_DIRECTION_CHANGES.csv', index=False, encoding='utf-8-sig')
    discrepancy = september[september.actual_one_word_daily.notna() & september.daily_one_word.isna()].copy()
    discrepancy['roll_direction'] = discrepancy.roll_score.gt(.5)
    discrepancy['new_direction'] = discrepancy.new_score.gt(.5)
    discrepancy[change_cols[:-2]].to_csv(OUT / 'SEPTEMBER_IDENTITY_SCOPE_DIFFERENCES.csv', index=False, encoding='utf-8-sig')
    assert len(discrepancy) == 13

    # These are illustrative cases already selected in Gemini's report, never
    # newly selected validation cases or evidence of washout causation.
    cases = [corrected_case(frame, engine['bars'], code, date, sessions)
             for code, date in [('000607', '2025-07-10'), ('600744', '2025-07-09')]]
    write('CORRECTED_CASE_FACTS.json', cases)
    volume_unknown = engine['bars'].volume_unit.isna()
    volume_audit = dict(total_archived_bars=len(engine['bars']),
                        volume_unit_missing_bars=int(volume_unknown.sum()),
                        declared_volume_units=engine['bars'].volume_unit.fillna('UNDECLARED').value_counts().to_dict(),
                        prepared_rows=len(prepared['full']),
                        rows_with_volume_ratio=int(prepared['full'].pullback_mean_volume_ratio.notna().sum()),
                        limitation='A <=100x adjacent-volume-change check cannot prove consistent units. The original builder did not consume volume_unit; normalized unit provenance is not independently certified.',
                        fixed_scores_preserved=True, retrain_after_known_results=False)
    write('VOLUME_UNIT_QUALIFICATION_LIMITATION.json', volume_audit)
    strict_metrics = summary['2026Sep']['strict_daily_nonword']
    original_metrics = summary['2026Sep']['original_actual_nonword']
    strict_tp_delta = strict_metrics['PULLBACK_PERSONALITY']['metrics']['TP'] - strict_metrics['ROLL252']['metrics']['TP']
    original_tp_delta = original_metrics['PULLBACK_PERSONALITY']['metrics']['TP'] - original_metrics['ROLL252']['metrics']['TP']
    known_changes = changes[valid_label(changes)]
    write('CORRECTED_RESULTS.json', dict(
        formed_at=now(), full_rows=len(frame), known=int(valid_label(frame).sum()), unknown=int((~valid_label(frame)).sum()),
        all_arms={arm: metric(frame, score) for arm, score in ARMS.items()}, folds=summary,
        stage_scope='All3707signals are >=2boards; not a first-board1to2 test.',
        original_fixed_gate_unchanged=original_results['fixed_screen'],
        original_fixed_screen_passed=original_results['screen_passed'],
        comparison_scope='strict_daily_nonword versus original_actual_nonword reported in parallel; neither silently replaces the other',
        september_changes=dict(total=len(changes), known=len(known_changes), unknown=len(changes)-len(known_changes),
                               strict_tp_delta=strict_tp_delta, original_tp_delta=original_tp_delta),
        uncertainty='Official session bootstrap repairs time spacing, not source missingness or repeated-testing bias.',
        new_fits=0, new_live_predictions=0, orders=0, promotion_allowed=False))
    write('MODEL_REVIEW_STATUS.json', dict(
        formed_at=now(), experiment='PULLBACK_PERSONALITY',
        status='REJECTED_FIXED_COMPARISON_RETAIN_DIAGNOSTIC_ONLY',
        inference_use='Historical diagnostics only; do not replace production BASE or use as first-board acceptance evidence.',
        review_adjustments='Evaluation scope, uncertainty axis, case facts and volume-provenance wording corrected. No fitted weights or scores changed.',
        primary_reasons=['Fair monthly baseline full-pool accuracy69.0934% exceeds challenger68.5440%.',
                         'Original September nonword cohort TP6/FP4/FN34 remains unchanged; strict subset improvement does not transfer to full cohort.',
                         'Original fixed Recall+10pp and Top2 gates failed.',
                         'This relay pool has no first-board signals.',
                         'Volume normalization provenance remains unverified; source survivorship and late availability limits remain.'],
        production_changed=False, schedule_changed=False, original_goal_status='paused',
        new_fits=0, new_matured_results=0, actual_trades=0))
    check_originals(scope)
    write('verification.json', dict(
        formed_at=now(), status='PASS', immutable_original_files=len(scope['source_artifacts_before_changes']),
        protected_input_bindings=len(scope['protected_inputs']), original_months=completed,
        full_cohort_metrics_reproduced=True, original_fold_and_top_metrics_reproduced=True,
        official_calendar_unknown_sessions_preserved=True,
        all13identity_discrepancies_and_all_direction_changes_saved=True,
        case_ratios_recomputed_from_original_bars=True,
        earlier_model_reload_checks_reused='CODEX_saved_artifact_review.json, byte-identical15savedmodels/scores; no duplicate fitting or broad rerun',
        new_fits=0, new_forward_results=0, orders=0,
        artifact_sha256={p.name: sha(p) for p in OUT.iterdir() if p.is_file()}))
    print(json.dumps(clean(dict(status='PASS', all_arms={arm: metric(frame, score) for arm, score in ARMS.items()},
                               strict_september_interval=summary['2026Sep']['official_calendar_F1_delta95_strict'],
                               original_september_interval=summary['2026Sep']['official_calendar_F1_delta95_original'],
                               changes=changes[change_cols].to_dict('records'))), ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
