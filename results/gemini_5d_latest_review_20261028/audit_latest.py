from pathlib import Path
from datetime import datetime, timezone
import contextlib
import hashlib
import importlib.util
import io
import json
import shutil
import sys

import joblib
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'research/model_upgrade_validation/unified_dragon_breakthrough_20261028'
OUT = Path(__file__).resolve().parent
SNAP = OUT / 'source_snapshot'
SNAP.mkdir(exist_ok=True)

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

names = [
    'build_and_train_5d_dragon_model.py', 'backtest_and_evaluate_5d_dragon.py',
    'evaluate_20261028_case.py', 'UNIFIED_5D_DRAGON_BREAKTHROUGH_REPORT.md',
    'model_5d_dragon_breakthrough.joblib', 'feature_list_5d.json',
    'oof_5d_predictions.csv', 'unified_5d_metrics.json',
    'comprehensive_backtest_metrics.json', 'case_study_20261028_ranking.csv',
    'feature_importances_5d.csv',
]
before = {n: sha(SRC / n) for n in names}
for n in names:
    target = SNAP / n
    if target.exists():
        raise RuntimeError('Refusing to overwrite an earlier source snapshot')
    shutil.copy2(SRC / n, target)

oof = pd.read_csv(SNAP / 'oof_5d_predictions.csv', dtype={'code_str': str})
top = oof[oof.daily_rank == 1]
case = pd.read_csv(SNAP / 'case_study_20261028_ranking.csv', dtype={'code_str': str})
case['actual_saved_rank'] = np.arange(1, len(case) + 1)
model = joblib.load(SNAP / 'model_5d_dragon_breakthrough.joblib')
features = json.loads((SNAP / 'feature_list_5d.json').read_text(encoding='utf-8'))
replayed = model.predict_proba(case[features].fillna(0))[:, 1]

# Causal mutation: all previous-day inputs and today's open remain unchanged;
# only today's final close/low and corresponding final float cap are changed.
# This synthetic fixture tests feature causality; it is not a market result.
spec = importlib.util.spec_from_file_location('gemini_5d_under_review', SNAP / 'build_and_train_5d_dragon_model.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
dates = pd.bdate_range('2025-01-01', periods=35).strftime('%Y-%m-%d').tolist()
rows = []
for code in ['000001', '000002']:
    prev = 10.0
    for j, day in enumerate(dates):
        price = {27: 11.0, 28: 12.1, 29: 13.31, 30: 14.64}.get(j, 10.0) if code == '000001' else 10.0
        op = round(prev * 1.05, 2) if j == 30 and code == '000001' else price
        rows.append(dict(date=day, code=code, name=code, open=op, high=max(op, price),
                         low=min(op, price), close=price, pre_close=prev, volume=1e7,
                         amount=price * 1e7, turnover=15.0, float_cap=price * 1e8,
                         st=False, new_listing=False))
        prev = price
fixture = pd.DataFrame(rows)
changed = fixture.copy()
m = changed.code.eq('000001') & changed.date.eq(dates[30])
changed.loc[m, ['close', 'low', 'float_cap']] = [11.98, 11.98, 11.98e8]
original_reader = pd.read_csv

def engineer(data):
    def mocked_read(path, **kwargs):
        if Path(path).name == 'stock_basics_industry_original.csv':
            return pd.DataFrame({'symbol': ['000001', '000002'], 'industry': ['fixture', 'fixture']})
        return data.copy()
    pd.read_csv = mocked_read
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return module.load_data_and_engineer_5d()
    finally:
        pd.read_csv = original_reader

a, b = engineer(fixture), engineer(changed)
am = a.code_str.eq('000001') & a.date.eq(dates[30])
bm = b.code_str.eq('000001') & b.date.eq(dates[30])
av, bv = a.loc[am].iloc[0], b.loc[bm].iloc[0]
mutation = {f: {'original': float(av[f]), 'changed_today_close': float(bv[f])}
            for f in ['mkt_promote_rate', 'mkt_total_dt', 'float_cap_yi', 'prev_bad_crash_count', 'has_bad_crash_gene']}
assert av.open == bv.open
assert av.prev_close == bv.prev_close
assert av.mkt_promote_rate != bv.mkt_promote_rate
assert av.prev_bad_crash_count != bv.prev_bad_crash_count

# Field-unit evidence from the actual archived Sina rows.
bars = pd.read_csv(ROOT / 'data/v3/imported/daily-bars.csv', usecols=['source', 'volume', 'amount', 'high', 'low'])
z = bars[bars.source.str.startswith('Sina') & (bars.volume > 0)].copy()
right = z.amount / z.volume
used = right / 100.0
vwap_check = dict(rows=len(z),
    amount_div_volume_inside_daily_range_pct=round(((right >= z.low-0.02) & (right <= z.high+0.02)).mean()*100, 4),
    used_vwap_inside_daily_range_pct=round(((used >= z.low-0.02) & (used <= z.high+0.02)).mean()*100, 4))

after = {n: sha(SRC / n) for n in names}
result = dict(
    formed_at_utc=datetime.now(timezone.utc).isoformat(), new_fits=0, orders=0,
    source_hashes=before, reviewed_sources_unchanged=before == after,
    oof_rows=len(oof), oof_dates=oof.date.nunique(), oof_date_range=[oof.date.min(), oof.date.max()],
    duplicate_date_code_rows=int(oof.duplicated(['date', 'code_str']).sum()),
    top1_reported_label_hits=int(top.target_limit_up.sum()), top1_count=len(top),
    top1_reported_label_hit_pct=round(top.target_limit_up.mean()*100, 2),
    top1_day_loss_at_least_4pct=int(top.target_big_loss.sum()),
    top1_mean_open_close_pct=round(top.target_day_return.mean()*100, 4),
    saved_case_prediction_max_abs_error=float(np.max(np.abs(replayed-case.pred_5d_score))),
    saved_case_head=case.head(4)[['actual_saved_rank','code_str','name','pred_5d_score','execution_status']].to_dict('records'),
    causal_mutation_synthetic_only=mutation, actual_sina_vwap_unit_check=vwap_check,
    accepted_for_production=False,
    primary_reasons=['same-day outcome leaked via market promotion and down-limit counts',
                     'current-day close-based float cap leaked into features',
                     'historical crash event backdated before the crash was observable',
                     'Sina volume shares divided by extra 100',
                     'daily amplitude used as a proxy for tail order changes',
                     'report case ranking contradicts saved model output',
                     'open-to-close compounding cannot represent A-share T+1 portfolio return'],
)
(OUT / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
