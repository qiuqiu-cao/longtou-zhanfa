"""Fixed rolling numerical market comparison; inspected historical diagnosis only."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from market_interval_research_v300 import METRICS

CHANNELS = list(METRICS)
ARMS = ('LEVEL', 'RIDGE_LEVEL', 'RIDGE_LEVEL_DELTA')
KNOWN = 'KNOWN_DIAGNOSTIC_POINT'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def features(obs):
    """The caller must retain the complete official grid, including missing days."""
    level = obs[CHANNELS].astype(float).copy()
    delta = level.diff()
    delta.columns = [name + '_delta' for name in CHANNELS]
    return {'LEVEL': level, 'LEVEL_DELTA': pd.concat([level, delta], axis=1)}


def _bounded(point, metric):
    low, high = METRICS[metric][2:4]
    if low is not None:
        point = max(point, low)
    if high is not None:
        point = min(point, high)
    return float(point)


def forecast(obs, sessions, positions):
    if sessions != sorted(set(sessions)) or obs.index.tolist() != sessions:
        raise ValueError('complete ordered official date grid required')
    if list(obs.columns) != CHANNELS:
        raise ValueError('only the eight frozen factual channels are allowed')
    if np.isinf(obs.to_numpy(float)).any():
        raise ValueError('infinite observations are invalid')
    x = features(obs)
    rows = []
    for pos in positions:
        if not 0 <= pos < len(sessions):
            raise ValueError('signal position outside official grid')
        signal = sessions[pos]
        target = sessions[pos + 1] if pos + 1 < len(sessions) else None
        level_start = max(0, pos + 1 - 120)
        ends = np.arange(max(1, level_start), pos + 1)
        source_positions = ends - 1
        current_available = obs.iloc[pos].notna().any()
        for metric in CHANNELS:
            levels = obs[metric].iloc[level_start:pos + 1].dropna()
            row = {'signal_date': signal, 'target_date': target, 'metric': metric,
                   'arm': 'LEVEL', 'status': KNOWN if len(levels) >= 60 else 'INSUFFICIENT_HISTORY',
                   'prediction': _bounded(levels.median(), metric) if len(levels) >= 60 else np.nan,
                   'train_n': len(levels),
                   'max_training_target_date': levels.index.max() if len(levels) else None,
                   'window_start': sessions[level_start],
                   'training_digest': hashlib.sha256(levels.to_csv().encode()).hexdigest(),
                   'dropped_all_missing_features': '[]', 'forward_evidence': False}
            rows.append(row)
            labels = obs[metric].iloc[ends].to_numpy(float)
            available_x = obs.iloc[source_positions].notna().any(axis=1).to_numpy()
            keep = np.isfinite(labels) & available_x
            for arm, name in [('RIDGE_LEVEL', 'LEVEL'), ('RIDGE_LEVEL_DELTA', 'LEVEL_DELTA')]:
                training = x[name].iloc[source_positions].iloc[np.flatnonzero(keep)].copy()
                y = labels[keep]
                active = training.columns[training.notna().any(axis=0)].tolist()
                dropped = sorted(set(training.columns) - set(active))
                status = ('UNKNOWN_CURRENT_LEVELS' if not current_available else
                          'INSUFFICIENT_HISTORY' if len(y) < 60 or not active else KNOWN)
                prediction = np.nan
                if status == KNOWN:
                    model = make_pipeline(SimpleImputer(strategy='median', add_indicator=True),
                                          StandardScaler(), Ridge(alpha=10.0, fit_intercept=True))
                    model.fit(training[active], y)
                    prediction = _bounded(model.predict(x[name].iloc[[pos]][active])[0], metric)
                end_dates = [sessions[i] for i in ends[keep]]
                if end_dates and max(end_dates) > signal:
                    raise AssertionError('training label reaches beyond signal close')
                binding = training.copy()
                binding['LABEL_FOR_TRAINING_ONLY'] = y
                rows.append({'signal_date': signal, 'target_date': target, 'metric': metric,
                             'arm': arm, 'status': status, 'prediction': prediction,
                             'train_n': len(y), 'max_training_target_date': max(end_dates) if end_dates else None,
                             'window_start': sessions[level_start],
                             'training_digest': hashlib.sha256(binding.to_csv().encode()).hexdigest(),
                             'dropped_all_missing_features': json.dumps(dropped), 'forward_evidence': False})
    return pd.DataFrame(rows)


def score(predictions, obs):
    result = predictions.copy()
    result['actual'] = [obs.at[d, k] if d in obs.index else np.nan
                        for d, k in zip(result.target_date, result.metric)]
    result['evaluated'] = result.status.eq(KNOWN) & result.actual.notna()
    result['absolute_error'] = abs(result.prediction - result.actual).where(result.evaluated)
    result['squared_error'] = ((result.prediction - result.actual) ** 2).where(result.evaluated)
    return result


def direction_metrics(actual, point):
    actual, point = np.asarray(actual), np.asarray(point)
    if not len(actual):
        return None
    y, p = actual > 0, point > 0
    tp = int((y & p).sum()); tn = int((~y & ~p).sum())
    fp = int((~y & p).sum()); fn = int((y & ~p).sum())
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {'n': len(y), 'positive_count': int(y.sum()), 'called_positive': int(p.sum()),
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'accuracy': (tp + tn) / len(y),
            'balanced_accuracy': (recall + specificity) / 2 if recall is not None and specificity is not None else None,
            'precision': tp / (tp + fp) if tp + fp else None,
            'complete_positive_recall': recall,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            'all_positive_accuracy': float(y.mean()), 'all_nonpositive_accuracy': float((~y).mean()),
            'Brier': None, 'probability_calibration': 'UNAVAILABLE_POINT_FORECAST'}


def summarize(scores, protocol):
    summary, pairs = [], []
    for period, (start, end) in protocol['periods'].items():
        part = scores[scores.signal_date.between(start, end)]
        grid = sorted(part.signal_date.unique())
        for metric in CHANNELS:
            metric_rows = part[part.metric.eq(metric)]
            for arm in ARMS:
                group = metric_rows[metric_rows.arm.eq(arm)]
                known = group[group.evaluated]
                summary.append({'period': period, 'metric': metric, 'arm': arm,
                                'registered_dates': len(group), 'predicted_dates': int(group.status.eq(KNOWN).sum()),
                                'known_target_dates': int(group.actual.notna().sum()), 'scored_dates': len(known),
                                'scoring_coverage': len(known) / len(group) if len(group) else None,
                                'MAE': float(known.absolute_error.mean()) if len(known) else None,
                                'RMSE': float(np.sqrt(known.squared_error.mean())) if len(known) else None,
                                'direction': direction_metrics(known.actual, known.prediction)
                                if metric in ('sh_a_return_pct', 'sz_a_return_pct', 'yesterday_premium') else None})
            baseline = metric_rows[metric_rows.arm.eq('LEVEL')].set_index('signal_date')
            for arm in ARMS[1:]:
                challenger = metric_rows[metric_rows.arm.eq(arm)].set_index('signal_date')
                paired = baseline[['evaluated', 'actual', 'prediction', 'absolute_error']].join(
                    challenger[['evaluated', 'prediction', 'absolute_error']], lsuffix='_base', rsuffix='_challenger')
                known = paired[paired.evaluated_base & paired.evaluated_challenger]
                difference = (known.absolute_error_challenger - known.absolute_error_base).reindex(grid)
                interval = None
                if metric == 'yesterday_premium' and len(known):
                    rng = np.random.default_rng(protocol['uncertainty']['seed'])
                    values = difference.to_numpy(float); estimates = []
                    for _ in range(protocol['uncertainty']['draws']):
                        starts = rng.integers(0, len(grid), size=int(np.ceil(len(grid) / 5)))
                        indexes = np.concatenate([(s + np.arange(5)) % len(grid) for s in starts])[:len(grid)]
                        sample = values[indexes]
                        if np.isfinite(sample).any(): estimates.append(float(np.nanmean(sample)))
                    interval = np.quantile(estimates, protocol['uncertainty']['tails']).tolist()
                pairs.append({'period': period, 'metric': metric, 'arm': arm, 'paired_dates': len(known),
                              'registered_dates': len(grid),
                              'baseline_MAE': float(known.absolute_error_base.mean()) if len(known) else None,
                              'challenger_MAE': float(known.absolute_error_challenger.mean()) if len(known) else None,
                              'MAE_difference': float(difference.mean()) if len(known) else None,
                              'adjusted_primary_difference_interval': interval,
                              'baseline_direction': direction_metrics(known.actual, known.prediction_base)
                              if metric in ('sh_a_return_pct', 'sz_a_return_pct', 'yesterday_premium') else None,
                              'challenger_direction': direction_metrics(known.actual, known.prediction_challenger)
                              if metric in ('sh_a_return_pct', 'sz_a_return_pct', 'yesterday_premium') else None})
    return summary, pairs


def run(protocol_path):
    path = Path(protocol_path); out = path.parent
    if (out / 'predictions_NO_LABELS_NOT_FORWARD.csv').exists():
        raise FileExistsError('fixed run already exists; no repeated fitting or overwrite')
    protocol = json.loads(path.read_text(encoding='utf-8'))
    if (protocol['input_channels'] != CHANNELS
            or list(protocol['challengers']) != list(ARMS[1:])
            or protocol['estimator']['alpha'] != 10.0
            or protocol['primary_target'] != 'yesterday_premium'):
        raise ValueError('protocol does not match the frozen implemented comparison')
    for filename, expected in protocol['input_hashes'].items():
        if digest(filename) != expected:
            raise ValueError('input hash mismatch: ' + filename)
    source = next(Path(p) for p in protocol['input_hashes'] if Path(p).name == 'observations_DIAGNOSTIC.csv')
    calendar = next(Path(p) for p in protocol['input_hashes'] if Path(p).name == 'calendar.json')
    sessions = json.loads(calendar.read_text(encoding='utf-8'))['sessions']
    obs = pd.read_csv(source, index_col=0)
    obs.index = obs.index.astype(str)
    if obs.index.tolist() != sessions:
        raise ValueError('cached observations no longer match original full official grid')
    positions = [i for i, day in enumerate(sessions) if protocol['signal_dates'][0] <= day <= protocol['signal_dates'][1]]
    predictions = forecast(obs, sessions, positions)
    predictions.to_csv(out / 'predictions_NO_LABELS_NOT_FORWARD.csv', index=False)
    write_json(out / 'prediction_receipt.json', {'saved_at': datetime.now(timezone.utc).isoformat(),
               'protocol_sha256': digest(path), 'implementation_sha256': digest(__file__),
               'prediction_sha256': digest(out / 'predictions_NO_LABELS_NOT_FORWARD.csv'),
               'new_forward_samples': 0, 'status': 'SAVED_BEFORE_EVALUATION_NOT_FORWARD'})
    scores = score(predictions, obs)
    scores.to_csv(out / 'scores_LABEL_ONLY.csv', index=False)
    summary, pairs = summarize(scores, protocol)
    pd.DataFrame([{k: v for k, v in row.items() if k != 'direction'} for row in summary]).to_csv(
        out / 'metric_summary.csv', index=False)
    gates = {}
    for arm in ARMS[1:]:
        ev = next(r for r in pairs if r['period'] == 'EVALUATION_DIAGNOSTIC' and r['metric'] == 'yesterday_premium' and r['arm'] == arm)
        dev = next(r for r in pairs if r['period'] == 'DEVELOPMENT_DIAGNOSTIC' and r['metric'] == 'yesterday_premium' and r['arm'] == arm)
        base = next(r for r in summary if r['period'] == 'EVALUATION_DIAGNOSTIC' and r['metric'] == 'yesterday_premium' and r['arm'] == 'LEVEL')
        ch = next(r for r in summary if r['period'] == 'EVALUATION_DIAGNOSTIC' and r['metric'] == 'yesterday_premium' and r['arm'] == arm)
        checks = {'sample_at_least60': ev['paired_dates'] >= 60,
                  'primary_MAE_lower': ev['MAE_difference'] is not None and ev['MAE_difference'] < 0,
                  'adjusted_interval_below0': ev['adjusted_primary_difference_interval'] is not None and ev['adjusted_primary_difference_interval'][1] < 0,
                  'coverage_not_lower': ch['scored_dates'] >= base['scored_dates'],
                  'direction_accuracy_not_lower': ev['challenger_direction'] is not None and ev['challenger_direction']['accuracy'] >= ev['baseline_direction']['accuracy'],
                  'development_MAE_not_worse': dev['MAE_difference'] is not None and dev['MAE_difference'] <= 0}
        gates[arm] = {'checks': checks, 'historical_candidate_gate_passed': all(checks.values()),
                      'production_upgrade_allowed': False}
    report = {'recorded_at': datetime.now(timezone.utc).isoformat(),
              'status': 'FIXED_ROLLING_RIDGE_DIAGNOSTIC_NOT_MODEL_UPGRADE',
              'protocol_sha256': digest(path), 'implementation_sha256': digest(__file__),
              'primary_metric': 'next-session yesterday-limit-pool median-feedback MAE',
              'signal_dates': len(positions), 'predictions': len(predictions),
              'summary': summary, 'paired_comparisons': pairs, 'candidate_gates': gates,
              'Brier': None, 'new_forward_samples': 0, 'pure_blind': False,
              'stable60_proved': False, 'production_model_changed': False,
              'comparison_is_not_direct_production_V272_improvement': True,
              'limitations': ['previously inspected historical archive, not genuine PIT',
                             'provider scope not exchange full-market certification',
                             'direction signs not the existing three-class ecology contract',
                             'no theme-state truth, probability calibration, execution or return claim'],
              'artifact_sha256': {name: digest(out / name) for name in
                                 ['predictions_NO_LABELS_NOT_FORWARD.csv', 'scores_LABEL_ONLY.csv', 'metric_summary.csv']}}
    write_json(out / 'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    args = parser.parse_args()
    report = run(args.protocol)
    print(json.dumps({'status': report['status'], 'signal_dates': report['signal_dates'],
                      'candidate_gates': report['candidate_gates'], 'new_forward_samples': 0}, ensure_ascii=False))
