"""One event-cohort challenge. Default/precheck are strictly zero-fit.

The actual peer must accept an independently reviewed frozen SHA before execute.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import sys
import traceback
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import short_term_thinking_research as old
import model_upgrade_history as history
from short_term_thinking_preprocessing import EmptyTrainingColumnGuard

PREVIOUS = ROOT / 'research/relay_joint_research_20261021T1142Z/final_prevalidation'
SOURCE = ROOT / 'research/model_upgrade_validation/short_term_thinking_20261015T071000Z/historical_final/dataset.joblib'
SAVED = SOURCE.parent.parent / 'execution_after_binning_repair'
spec = importlib.util.spec_from_file_location('fixed3_metrics_only', PREVIOUS / 'fixed_path_validation.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)
ADDED = ['event_cohort_matched', 'event_cohort_size', 'event_cohort_primary_member',
         'event_cohort_damage_ratio', 'event_cohort_repair_ratio',
         'event_anchor_status', 'event_cohort_amount_share_delta']
ARM = 'ST_A_EVENT_COHORT_FIXED3'

def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_new(p, value):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(comparison.clean(value), f, ensure_ascii=False, indent=2, allow_nan=False)
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def prepare(features):
    data = joblib.load(SOURCE)
    full = old.merged(data)
    f = pd.read_csv(features, dtype={'code': str})
    assert len(f) == 9245 and not f.duplicated(['date', 'code']).any()
    assert not set(ADDED) & set(old.FEATURES) and len(old.FEATURES) == 82
    assert not {'target', 'next_open_gap', 'large_loss', 'net_return'} & set(ADDED)
    for c in ADDED:
        values = pd.to_numeric(f[c], errors='raise')
        assert not np.isinf(values.to_numpy(dtype=float)).any(), c
    merged = full.merge(f[['date', 'code'] + ADDED], on=['date', 'code'], validate='one_to_one')
    assert len(merged) == 9245
    return merged

def precheck(out, features, engine):
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'EXPERIMENT_STARTED.json').exists()
    full = prepare(features)
    control_path = PREVIOUS / 'CONTROL_FIXED3_3707.csv'
    assert sha(control_path) == '7bc02b36d7bdf6117b3b773c36d8629898bef6793eb095a555bb5066eebcbfe3'
    control = pd.read_csv(control_path, dtype={'code': str}, low_memory=False)
    expected = full.loc[full.date.between('2025-07-01', '2026-09-30')].sort_values(['date', 'code']).reset_index(drop=True)
    ctl = control.sort_values(['date', 'code']).reset_index(drop=True)
    for c in ['date', 'code', 'target_date', 'target', 'boards', 'daily_one_word', 'label_valid']:
        pd.testing.assert_series_equal(expected[c], ctl[c], check_dtype=False, check_names=False)
    assert len(control) == 3707 and control.date.nunique() == 302 and control.target.isna().sum() == 67
    checks = []
    for fold in history.FOLDS:
        train = history.training_rows(full, fold['start'], 'relay')
        saved = joblib.load(SAVED / (fold['start'][:7] + '_ST_DUAL_model.joblib'))
        assert saved['features'] == old.FEATURES and saved['estimator'] == old.PARAMS
        assert saved['cutoff'] == fold['start'] and saved['training_rows'] == len(train)
        assert train.target.notna().all() and train.date.nunique() == 252
        assert train.target_date.max() < fold['start']
        test = full.loc[full.date.between(fold['start'], fold['end'])]
        checks.append(dict(fold=fold, train_rows=len(train), days=252,
            max_target=train.target_date.max(), test_rows=len(test), test_days=int(test.date.nunique()),
            training_digest=history._frame_digest(train, old.FEATURES + ADDED),
            weight_digest=hashlib.sha256(old.training_weights(train).to_numpy(float).tobytes()).hexdigest()))
    paths = [SOURCE, control_path, engine, features, Path(__file__),
             PREVIOUS / 'fixed_path_validation.py', PREVIOUS / 'FROZEN_PROTOCOL.json',
             ROOT / 'short_term_thinking_research.py', ROOT / 'short_term_thinking_preprocessing.py',
             ROOT / 'short_term_thinking_dataset_builder.py', ROOT / 'model_upgrade_history.py',
             ROOT / 'historical_symbolic_context.py', ROOT / 'symbolic_name_theme_v217.py',
             ROOT / 'security_master_v4.py', ROOT / 'data/v4/security_master_intervals.csv']
    paths += [SAVED / (f['start'][:7] + '_ST_DUAL_model.joblib') for f in history.FOLDS]
    receipt = dict(at_utc=now(), fits=0, arm=ARM, fields=old.FEATURES + ADDED,
        added_fields=ADDED, estimator=old.PARAMS, checks=checks, control_metrics=comparison.metrics(control),
        feature_file=str(features.resolve()), engine_file=str(engine.resolve()),
        control_file=str(control_path.resolve()), gates=read(PREVIOUS / 'FROZEN_PROTOCOL.json')['gates'],
        sources={str(p.resolve()): sha(p) for p in paths},
        causal_qualification_requires_separate_codex_review=True,
        maximum_new_fits=3, maximum_fits_per_cutoff=1, first_board='PAUSED',
        history_seen=True, new_blind_test=False, monthly_fits_authorized=False,
        production_deployed=False)
    write_new(out / 'PRECHECK.json', receipt)
    print(json.dumps(dict(status='PRECHECK_ZERO_FITS', checks=checks, control=receipt['control_metrics']['nonword_top1'])))

def execute(out):
    protocol_path = out / 'FROZEN_PROTOCOL.json'
    p = read(protocol_path)
    assert p['status'] == 'CODEX_SIGNED_AWAITING_REAL_GEMINI_ACK_BEFORE_FIT'
    ack = read(out / 'GEMINI_ACK_BEFORE_FIT.json')
    assert ack['accepted'] is True and ack['protocol_sha256'] == sha(protocol_path)
    assert ack['actual_executor'] == 'Antigravity Gemini'
    assert p['fields'] == old.FEATURES + ADDED and p['estimator'] == old.PARAMS
    for path, digest in p['sources'].items(): assert sha(path) == digest, path
    write_new(out / 'EXPERIMENT_STARTED.json', dict(at_utc=now(), protocol_sha256=sha(protocol_path), maximum_fits=3))
    fits = 0
    try:
        full = prepare(Path(p['feature_file']))
        parts = []
        for check in p['checks']:
            fold = check['fold']
            train = history.training_rows(full, fold['start'], 'relay')
            assert history._frame_digest(train, p['fields']) == check['training_digest']
            weights = old.training_weights(train)
            assert hashlib.sha256(weights.to_numpy(float).tobytes()).hexdigest() == check['weight_digest']
            write_new(out / (fold['id'] + '_FIT_STARTED.json'), dict(at_utc=now(), cutoff=fold['start'], attempt=1))
            model = Pipeline([('empty_training_columns', EmptyTrainingColumnGuard()),
                              ('hgb', HistGradientBoostingClassifier(**old.PARAMS))])
            fits += 1
            with threadpool_limits(limits=2): model.fit(train[p['fields']], train.target.astype(int), hgb__sample_weight=weights)
            model_path = out / (fold['id'] + '_MODEL.joblib')
            joblib.dump(dict(model=model, features=p['fields'], cutoff=fold['start'], params=old.PARAMS,
                             training_digest=check['training_digest']), model_path)
            test = full.loc[full.date.between(fold['start'], fold['end'])].copy()
            with threadpool_limits(limits=2): test['score'] = model.predict_proba(test[p['fields']])[:, 1]
            test['arm'] = ARM
            test[['date', 'code', 'boards', 'target_date', 'target', 'label_valid', 'daily_one_word', 'score', 'arm']].to_csv(
                out / (fold['id'] + '_PREDICTIONS_BEFORE_EVALUATION.csv'), index=False, encoding='utf-8-sig')
            parts.append(test)
            write_new(out / (fold['id'] + '_FIT_COMPLETED.json'), dict(at_utc=now(), fits_attempted=fits, model_sha256=sha(model_path)))
        new = pd.concat(parts, ignore_index=True)
        new.to_csv(out / 'ALL_PREDICTIONS_BEFORE_EVALUATION.csv', index=False, encoding='utf-8-sig')
        c = pd.read_csv(p['control_file'], dtype={'code': str}, low_memory=False)
        a, b, delta = comparison.metrics(new), comparison.metrics(c), comparison.paired(c, new)
        g = p['gates']
        gates = dict(nonword_top1_min=a['nonword_top1']['hits'] >= g['nonword_top1_min_hits'],
            same_algorithm_gain=delta['gain_hits'] >= g['min_gain_hits_vs_control'], paired_lower_positive=delta['lower'] > 0,
            full_top1_protected=a['full_top1']['hits'] >= g['full_top1_min_hits'],
            nonword_top2_slots_protected=a['nonword_top2']['hits'] >= g['base_nonword_top2_slot_hits'],
            nonword_top3_slots_protected=a['nonword_top3']['hits'] >= g['base_nonword_top3_slot_hits'],
            nonword_top2_dates_protected=a['nonword_top2']['any_hit_days'] >= g['nonword_top2_any_min_days'],
            nonword_top3_dates_protected=a['nonword_top3']['any_hit_days'] >= g['nonword_top3_any_min_days'])
        folds = {}
        for f in history.FOLDS:
            x = comparison.metrics(new.loc[new.date.between(f['start'], f['end'])])
            y = comparison.metrics(c.loc[c.date.between(f['start'], f['end'])])
            folds[f['id']] = dict(new=x, control=y)
            gates[f['id'] + '_top1_no_loss'] = x['nonword_top1']['hits'] >= y['nonword_top1']['hits']
            for metric in ('Recall', 'Precision', 'F1'):
                av, bv = x['nonword_classification'][metric], y['nonword_classification'][metric]
                gates[f['id'] + '_' + metric + '_protected'] = av is not None and bv is not None and av >= bv - g['each_fold_recall_precision_f1_max_decline']
        for path, digest in p['sources'].items(): assert sha(path) == digest, path
        result = dict(at_utc=now(), status='TRAINED_AND_EVALUATED', actual_fits_attempted=fits,
            rows=len(new), dates=int(new.date.nunique()), unknown=int(new.target.isna().sum()),
            new_metrics=a, control_metrics=b, paired_top1=delta, folds=folds, gates=gates,
            all_gates_passed=all(gates.values()), monthly_fits_authorized=False, production_deployed=False,
            historical_seen=True, new_blind_test=False, realized_trading_win_rate=None)
        write_new(out / 'RESULT.json', result)
        print(json.dumps(dict(new=a['nonword_top1'], control=b['nonword_top1'], paired=delta, all_pass=all(gates.values()))))
    except BaseException as error:
        write_new(out / 'FAILURE.json', dict(at_utc=now(), fits_attempted=fits, error=str(error), traceback=traceback.format_exc(), rerun_authorized=False))
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['precheck', 'execute'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--features', type=Path)
    parser.add_argument('--engine', type=Path)
    args = parser.parse_args()
    if args.mode == 'precheck': precheck(args.out.resolve(), args.features.resolve(), args.engine.resolve())
    else: execute(args.out.resolve())
