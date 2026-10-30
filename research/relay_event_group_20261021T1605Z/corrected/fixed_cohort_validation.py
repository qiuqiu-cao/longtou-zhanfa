"""
One preregistered relay feature challenge: ST_A_EVENT_COHORT_FIXED3.
Precheck never fits a model (0 fits). Execution requires signed Gemini ACK and frozen protocol.
"""
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import importlib.util
import json
import os
import sys
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))

import short_term_thinking_research as old
import short_term_thinking_dataset_builder as builder
import model_upgrade_history as history
from short_term_thinking_preprocessing import EmptyTrainingColumnGuard

SOURCE = ROOT / 'research/model_upgrade_validation/short_term_thinking_20261015T071000Z/historical_final/dataset.joblib'
SAVED = ROOT / 'research/model_upgrade_validation/short_term_thinking_20261015T071000Z/execution_after_binning_repair'
FEATURES_CSV = OUT / 'event_group_features_9245.csv'

ADDED = [
    'event_cohort_matched',
    'event_cohort_size',
    'event_cohort_primary_member',
    'event_cohort_damage_ratio',
    'event_cohort_repair_ratio',
    'event_anchor_status',
    'event_cohort_amount_share_delta'
]
ARM = 'ST_A_EVENT_COHORT_FIXED3'
FOLDS = history.FOLDS

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value

def write_new(path, value):
    with Path(path).open('w', encoding='utf-8') as handle:
        json.dump(clean(value), handle, ensure_ascii=False, indent=2, allow_nan=False)

def select(frame, k=1, nonword=True):
    x = frame[frame.daily_one_word.eq(0)] if nonword else frame
    return x.sort_values(['date', 'score', 'code'], ascending=[True, False, True]).groupby('date', sort=True).head(k)

def metrics(frame):
    total = int(frame.date.nunique())
    answer = {}
    for scope, nonword in [('nonword', True), ('full', False)]:
        for k in (1, 2, 3):
            picks = select(frame, k, nonword)
            answer[f'{scope}_top{k}'] = dict(
                hits=int(picks.target.eq(1).sum()),
                slots=len(picks),
                known=int(picks.target.notna().sum()),
                unknown=int(picks.target.isna().sum()),
                dates=total,
                any_hit_days=int(picks[picks.target.eq(1)].date.nunique())
            )
    for scope, nonword in [('nonword', True), ('full', False)]:
        x = frame[frame.daily_one_word.eq(0)] if nonword else frame
        y = x[x.target.notna() & x.label_valid.fillna(False).astype(bool)]
        pred = y.score.gt(.5)
        truth = y.target.eq(1)
        tp = int((pred & truth).sum())
        fp = int((pred & ~truth).sum())
        fn = int((~pred & truth).sum())
        tn = int((~pred & ~truth).sum())
        answer[scope + '_classification'] = dict(
            TP=tp, FP=fp, FN=fn, TN=tn,
            Recall=tp / (tp + fn) if tp + fn else None,
            Precision=tp / (tp + fp) if tp + fp else None,
            F1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
        )
    answer['by_board'] = {
        str(b): dict(
            rows=len(g),
            known=int(g.target.notna().sum()),
            unknown=int(g.target.isna().sum()),
            top1_hits=int(select(g).target.eq(1).sum()),
            top1_slots=len(select(g))
        )
        for b, g in frame.groupby('boards')
    }
    return answer

def paired(left, right):
    days = sorted(left.date.unique())
    a = select(left).set_index('date').target.reindex(days)
    z = select(right).set_index('date').target.reindex(days)
    delta = z.eq(1).astype(float).to_numpy() - a.eq(1).astype(float).to_numpy()
    rng = np.random.default_rng(20261021)
    n = len(days)
    length = min(5, n)
    starts = rng.integers(0, n, size=(4000, (n + length - 1) // length))
    indices = ((starts[:, :, None] + np.arange(length)) % n).reshape(4000, -1)[:, :n]
    values = delta[indices].mean(axis=1)
    return dict(
        gain_hits=int(delta.sum()),
        dates=n,
        lower=float(np.quantile(values, .025)),
        upper=float(np.quantile(values, .975)),
        draws=4000,
        block_signal_dates=5,
        exploratory_seen_history=True,
        unknown_left=int(a.isna().sum()),
        unknown_right=int(z.isna().sum())
    )

def prepare():
    data = joblib.load(SOURCE)
    full = old.merged(data)
    feat_df = pd.read_csv(FEATURES_CSV, dtype={'code': str})
    feat_df['code'] = feat_df.code.astype(str).str.zfill(6)
    
    # Merge the 7 added features
    merge_cols = ['date', 'code'] + ADDED
    full = full.merge(feat_df[merge_cols], on=['date', 'code'], validate='one_to_one')
    assert len(full) == 9245 and not full.duplicated(['date', 'code']).any()
    assert len(old.FEATURES) == 82 and not set(ADDED) & set(old.FEATURES)
    assert not {'next_open_gap', 'target', 'large_loss', 'net_return'} & set(old.FEATURES + ADDED)
    return data, full

def precheck():
    data, full = prepare()
    checks = []
    controls = []
    for fold in FOLDS:
        train = history.training_rows(full, fold['start'], 'relay')
        test = full[full.date.between(fold['start'], fold['end'])].copy()
        month = fold['start'][:7]
        saved = joblib.load(SAVED / (month + '_ST_DUAL_model.joblib'))
        assert saved['features'] == old.FEATURES and saved['estimator'] == old.PARAMS
        assert saved['cutoff'] == fold['start'] and saved['training_rows'] == len(train)
        assert train.date.nunique() == 252 and train.target_date.max() < fold['start']
        assert train.target.notna().all()
        with threadpool_limits(limits=2):
            test['score'] = saved['model'].predict_proba(test[old.FEATURES])[:, 1]
        test['arm'] = 'ST_DUAL_FIXED3_CONTROL'
        controls.append(test)
        checks.append(dict(
            fold=fold,
            train_rows=len(train),
            days=252,
            max_target=train.target_date.max(),
            training_digest=history._frame_digest(train, old.FEATURES + ADDED),
            weight_digest=hashlib.sha256(old.training_weights(train).to_numpy(float).tobytes()).hexdigest(),
            test_rows=len(test),
            test_days=test.date.nunique()
        ))
    control = pd.concat(controls, ignore_index=True)
    assert len(control) == 3707 and control.date.nunique() == 302 and int(control.target.isna().sum()) == 67
    control.to_csv(OUT / 'CONTROL_FIXED3_3707.csv', index=False, encoding='utf-8-sig')
    
    paths = [
        SOURCE,
        FEATURES_CSV,
        Path(__file__),
        ROOT / 'short_term_thinking_research.py',
        ROOT / 'short_term_thinking_preprocessing.py',
        ROOT / 'model_upgrade_history.py'
    ]
    paths += [SAVED / (fold['start'][:7] + '_ST_DUAL_model.joblib') for fold in FOLDS]
    receipt = dict(
        at_utc=now(),
        fits=0,
        rows=9245,
        evaluation_rows=3707,
        evaluation_dates=302,
        evaluation_unknown=67,
        fields=old.FEATURES + ADDED,
        params=old.PARAMS,
        checks=checks,
        control_metrics=metrics(control),
        sources={str(p.resolve()): sha(p) for p in paths}
    )
    write_new(OUT / 'PRECHECK.json', receipt)
    print(json.dumps(clean(dict(fits=0, checks=checks, control_metrics=receipt['control_metrics'])), ensure_ascii=True))

def execute():
    protocol_path = OUT / 'FROZEN_PROTOCOL.json'
    ack_path = OUT / 'GEMINI_ACK_BEFORE_FIT.json'
    assert protocol_path.exists(), "FROZEN_PROTOCOL.json does not exist. Awaiting Codex freeze."
    assert ack_path.exists(), "GEMINI_ACK_BEFORE_FIT.json does not exist."

    protocol = json.loads(protocol_path.read_text('utf-8'))
    ack = json.loads(ack_path.read_text('utf-8'))
    assert ack['protocol_sha256'] == sha(protocol_path) and ack['accepted'] is True, "ACK mismatch or not accepted."
    for path, digest in protocol['sources'].items():
        assert sha(path) == digest, f"Digest mismatch on source: {path}"

    write_new(OUT / 'EXPERIMENT_STARTED.json', dict(
        at_utc=now(),
        protocol_sha256=sha(protocol_path),
        max_new_fits=3
    ))
    fits = 0
    try:
        data, full = prepare()
        parts = []
        for check in protocol['checks']:
            fold = check['fold']
            train = history.training_rows(full, fold['start'], 'relay')
            assert history._frame_digest(train, old.FEATURES + ADDED) == check['training_digest']
            weights = old.training_weights(train)
            assert hashlib.sha256(weights.to_numpy(float).tobytes()).hexdigest() == check['weight_digest']

            write_new(OUT / f"{fold['id']}_FIT_STARTED.json", dict(
                at_utc=now(),
                cutoff=fold['start'],
                attempt=1,
                train_rows=len(train)
            ))
            model = Pipeline([
                ('empty_training_columns', EmptyTrainingColumnGuard()),
                ('hgb', HistGradientBoostingClassifier(**old.PARAMS))
            ])
            fits += 1
            with threadpool_limits(limits=2):
                model.fit(train[old.FEATURES + ADDED], train.target.astype(int), hgb__sample_weight=weights)

            joblib.dump(dict(
                model=model,
                features=old.FEATURES + ADDED,
                cutoff=fold['start'],
                params=old.PARAMS,
                training_digest=check['training_digest']
            ), OUT / f"{fold['id']}_MODEL.joblib")

            test = full[full.date.between(fold['start'], fold['end'])].copy()
            with threadpool_limits(limits=2):
                test['score'] = model.predict_proba(test[old.FEATURES + ADDED])[:, 1]
            test['arm'] = ARM
            test[['date', 'code', 'boards', 'target_date', 'target', 'label_valid', 'daily_one_word', 'score', 'arm']].to_csv(
                OUT / f"{fold['id']}_PREDICTIONS_BEFORE_EVALUATION.csv", index=False, encoding='utf-8-sig'
            )
            parts.append(test)
            write_new(OUT / f"{fold['id']}_FIT_COMPLETED.json", dict(
                at_utc=now(),
                fits_attempted=fits,
                model_sha256=sha(OUT / f"{fold['id']}_MODEL.joblib")
            ))

        new = pd.concat(parts, ignore_index=True)
        control = pd.read_csv(OUT / 'CONTROL_FIXED3_3707.csv', dtype={'code': str}, low_memory=False)
        assert len(new) == 3707 and new.date.nunique() == 302
        a = metrics(new)
        c = metrics(control)
        delta = paired(control, new)
        g = protocol['gates']

        gates = dict(
            nonword_top1_min=a['nonword_top1']['hits'] >= g['nonword_top1_min_hits'],
            same_algorithm_gain=delta['gain_hits'] >= g['min_gain_hits_vs_control'],
            paired_lower_positive=delta['lower'] > 0,
            full_top1_protected=a['full_top1']['hits'] >= 209,
            nonword_top2_slots_protected=a['nonword_top2']['hits'] >= g['base_nonword_top2_slot_hits'],
            nonword_top3_slots_protected=a['nonword_top3']['hits'] >= g['base_nonword_top3_slot_hits'],
            nonword_top2_dates_protected=a['nonword_top2']['any_hit_days'] >= 184,
            nonword_top3_dates_protected=a['nonword_top3']['any_hit_days'] >= 233
        )

        folds = {}
        for fold in FOLDS:
            x = new[new.date.between(fold['start'], fold['end'])]
            y = control[control.date.between(fold['start'], fold['end'])]
            am, cm = metrics(x), metrics(y)
            folds[fold['id']] = {'new': am, 'control': cm}
            gates[f"{fold['id']}_top1_no_loss"] = am['nonword_top1']['hits'] >= cm['nonword_top1']['hits']
            for metric in ('Recall', 'Precision', 'F1'):
                av, cv = am['nonword_classification'][metric], cm['nonword_classification'][metric]
                gates[f"{fold['id']}_{metric}_protected"] = (av is not None and cv is not None and av >= cv - .025)

        for path, digest in protocol['sources'].items():
            assert sha(path) == digest, path

        result = dict(
            at_utc=now(),
            status='TRAINED_AND_EVALUATED',
            actual_fits_attempted=fits,
            rows=3707,
            dates=302,
            unknown=int(new.target.isna().sum()),
            new_metrics=a,
            control_metrics=c,
            paired_top1=delta,
            folds=folds,
            gates=gates,
            all_gates_passed=all(gates.values()),
            monthly_fits_authorized=False,
            production_deployed=False,
            historical_seen=True,
            new_blind_test=False,
            realized_trading_win_rate=None,
            limitations=[
                'Current-universe survivorship and original source qualifications persist.',
                'Historical screen is not a live execution or causal capital identity result.'
            ]
        )
        write_new(OUT / 'RESULT.json', result)
        print(json.dumps(clean(dict(
            status=result['status'],
            fits=fits,
            new_nonword_top1=a['nonword_top1'],
            control_nonword_top1=c['nonword_top1'],
            full_top1=a['full_top1'],
            gates=gates,
            all_pass=result['all_gates_passed']
        )), ensure_ascii=True))
    except BaseException as error:
        write_new(OUT / 'FAILURE.json', dict(
            at_utc=now(),
            fits_attempted=fits,
            error=str(error),
            traceback=traceback.format_exc(),
            rerun_authorized=False
        ))
        raise

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'execute':
        execute()
    else:
        precheck()
