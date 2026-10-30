"""Recompute saved fixed-three predictions without importing the peer runner.

No training, score transformation, threshold search or source mutation.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parents[3]
CONTROL = WORK / 'research/relay_joint_research_20261021T1142Z/final_prevalidation/CONTROL_FIXED3_3707.csv'
FOLDS = [('2025H2', '2025-07-01', '2025-12-31'),
         ('2026JanAug', '2026-01-01', '2026-08-31'),
         ('2026Sep', '2026-09-01', '2026-09-30')]

def picks(d, k=1, nonword=True):
    eligible = d.loc[d.daily_one_word.eq(0)] if nonword else d
    return eligible.sort_values(['date', 'score', 'code'], ascending=[True, False, True]).groupby('date', sort=True).head(k)

def calculate(d):
    result = {}
    for scope, nonword in [('nonword', True), ('full', False)]:
        for k in (1, 2, 3):
            p = picks(d, k, nonword)
            result[f'{scope}_top{k}'] = dict(hits=int(p.target.eq(1).sum()), slots=len(p),
                unknown=int(p.target.isna().sum()), dates=int(d.date.nunique()),
                any_hit_days=int(p.loc[p.target.eq(1), 'date'].nunique()))
        e = d.loc[d.daily_one_word.eq(0)] if nonword else d
        known = e.loc[e.target.notna() & e.label_valid.fillna(False).astype(bool)]
        yes = known.score.gt(.5)
        truth = known.target.eq(1)
        tp, fp, fn, tn = [int(v.sum()) for v in [yes & truth, yes & ~truth, ~yes & truth, ~yes & ~truth]]
        result[scope + '_classification'] = dict(TP=tp, FP=fp, FN=fn, TN=tn,
            Recall=tp/(tp+fn) if tp+fn else None,
            Precision=tp/(tp+fp) if tp+fp else None,
            F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
    return result

def compare(control, new):
    days = sorted(control.date.unique())
    left = picks(control).set_index('date').target.reindex(days)
    right = picks(new).set_index('date').target.reindex(days)
    delta = right.eq(1).to_numpy(dtype=float) - left.eq(1).to_numpy(dtype=float)
    rng = np.random.default_rng(20261021)
    starts = rng.integers(0, len(days), size=(4000, (len(days)+4)//5))
    idx = ((starts[:, :, None] + np.arange(5)) % len(days)).reshape(4000, -1)[:, :len(days)]
    values = delta[idx].mean(axis=1)
    return dict(gain_hits=int(delta.sum()), lower=float(np.quantile(values, .025)),
                upper=float(np.quantile(values, .975)), unknown_left=int(left.isna().sum()),
                unknown_right=int(right.isna().sum()), dates=len(days))

def main():
    folder = Path(sys.argv[1]).resolve()
    files = sorted(p for p in folder.glob('*_PREDICTIONS_BEFORE_EVALUATION.csv') if not p.name.startswith('ALL_'))
    assert len(files) == 3, files
    d = pd.concat([pd.read_csv(p, dtype={'code': str}) for p in files], ignore_index=True)
    c = pd.read_csv(CONTROL, dtype={'code': str}, low_memory=False)
    keys = ['date', 'code']
    d = d.sort_values(keys).reset_index(drop=True)
    c = c.sort_values(keys).reset_index(drop=True)
    assert len(d) == 3707 and d.date.nunique() == 302 and d.target.isna().sum() == 67
    assert not d.duplicated(keys).any() and d.score.notna().all()
    assert np.isfinite(d.score).all() and d.score.between(0, 1).all()
    for column in keys + ['boards', 'target_date', 'target', 'label_valid', 'daily_one_word']:
        pd.testing.assert_series_equal(d[column], c[column], check_dtype=False, check_names=False)
    new, control, paired = calculate(d), calculate(c), compare(c, d)
    gates = dict(nonword_top1_min=new['nonword_top1']['hits'] >= 140,
        same_algorithm_gain=paired['gain_hits'] >= 6,
        paired_lower_positive=paired['lower'] > 0,
        full_top1_protected=new['full_top1']['hits'] >= 209,
        nonword_top2_slots_protected=new['nonword_top2']['hits'] >= 246,
        nonword_top3_slots_protected=new['nonword_top3']['hits'] >= 356,
        nonword_top2_dates_protected=new['nonword_top2']['any_hit_days'] >= 184,
        nonword_top3_dates_protected=new['nonword_top3']['any_hit_days'] >= 233)
    folds = {}
    for name, start, end in FOLDS:
        a = calculate(d.loc[d.date.between(start, end)])
        b = calculate(c.loc[c.date.between(start, end)])
        folds[name] = dict(new=a, control=b)
        gates[name+'_top1_no_loss'] = a['nonword_top1']['hits'] >= b['nonword_top1']['hits']
        for metric in ('Recall', 'Precision', 'F1'):
            av, bv = a['nonword_classification'][metric], b['nonword_classification'][metric]
            gates[name+'_'+metric+'_protected'] = av is not None and bv is not None and av >= bv-.025
    receipt = dict(at_utc=datetime.now(timezone.utc).isoformat(), fits_by_auditor=0,
        rows=len(d), dates=int(d.date.nunique()), unknown=int(d.target.isna().sum()),
        new_metrics=new, control_metrics=control, paired_top1=paired, folds=folds,
        gates=gates, all_gates_passed=all(gates.values()), historical_seen=True,
        new_blind_test=False, realized_trading_win_rate=None,
        prediction_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    result_path = folder / 'CODEX_INDEPENDENT_PREDICTION_AUDIT.json'
    with result_path.open('x', encoding='utf-8') as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2)
    print(json.dumps(dict(new=new['nonword_top1'], control=control['nonword_top1'],
                         paired=paired, all_pass=all(gates.values())), ensure_ascii=True))

if __name__ == '__main__':
    main()
