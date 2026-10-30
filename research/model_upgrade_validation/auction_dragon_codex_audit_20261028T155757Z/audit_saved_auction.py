"""Read-only audit of Gemini's saved auction research; never fit or overwrite it."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import shutil
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=FutureWarning)
ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
SRC = ROOT / 'research/model_upgrade_validation/auction_dragon_model_20261028'
SNAP = OUT / 'source_snapshot'
assert not SNAP.exists(), 'Use a new audit folder for a new snapshot.'
SNAP.mkdir()

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()

paths = list(SRC.glob('*')) + [ROOT / 'AGENTS.md', ROOT / 'leader_system_v3.py']
paths += [ROOT / 'templates/同花顺公式_龙头竞价非一字选股_v2026.txt',
          ROOT / 'templates/同花顺问财_竞价大概率涨停选股_v2026.txt']
before = {}
for path in paths:
    if path.is_file():
        rel = path.relative_to(ROOT)
        target = SNAP / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        before[str(rel)] = sha(path)
        shutil.copy2(path, target)
        assert sha(target) == before[str(rel)]
snap_src = SNAP / SRC.relative_to(ROOT)
spec = importlib.util.spec_from_file_location('auction_audit_snapshot_builder', snap_src / 'build_and_train_auction_models.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print('Only build_auction_dataset is called; no fitting or source output mutation.', flush=True)
df = module.build_auction_dataset()

is20 = df['code_str'].str.startswith(('300', '301', '688'))
# Diagnostic regular-price comparator; not a qualified historical price-limit registry.
exact_price = np.floor(df.pre_close * np.where(is20, 1.20, 1.10) * 100 + 0.5 + 1e-8) / 100
df['regular_limit_diagnostic'] = exact_price
df['regular_zt_diagnostic'] = (df.close >= exact_price - .001) & (df.close == df.high)
df['regular_touch_diagnostic'] = df.high >= exact_price - .001
df['label_false_positive_diagnostic'] = df.is_zt & ~df.regular_zt_diagnostic

features = ['open_gap', 'open_vs_prev_high', 'open_vs_prev_vwap', 'open_vs_prev_open',
    'open_vs_ma5', 'open_vs_ma20', 'prev_board_streak', 'prev_turnover',
    'prev_amount_yi', 'prev_amplitude_pct', 'float_cap_yi', 'pre_close',
    'is_capacity_dragon', 'is_small_cap_trap', 'is_low_price', 'is_ma_bull']
relay = df[df.prev_is_zt].copy()
non_one = relay[relay.is_non_one_word].copy()
train = non_one.dropna(subset=features + ['target_limit_up']).copy()
dates = sorted(train.date.unique())
folds = [(dates[0], dates[int(len(dates) * a)], dates[int(len(dates) * a) + 1], dates[int(len(dates) * b)] if b < 1 else dates[-1])
         for a, b in [(.40, .55), (.55, .70), (.70, .85), (.85, 1)]]

observed_volume = (df.volume > 0) & (df.amount > 0) & (df.low > 0)
vwap_share = df.amount / df.volume
vwap_in_lot_code = df.amount / (df.volume * 100)
unit_stats = {
    'observed_rows': int(observed_volume.sum()),
    'amount_div_volume_inside_ohlc': int((observed_volume & vwap_share.between(df.low - .011, df.high + .011)).sum()),
    'amount_div_volume100_inside_ohlc': int((observed_volume & vwap_in_lot_code.between(df.low - .011, df.high + .011)).sum()),
    'training_open_vs_prev_vwap_quantiles': train.open_vs_prev_vwap.quantile([0, .5, .99, 1]).to_dict(),
}

signal = (df.prev_is_zt & df.is_non_one_word & df.open_gap.between(.02, .07)
          & (df.open >= df.prev_high) & (df.prev_amount_yi >= 3)
          & df.prev_turnover.between(8, 32))
candidate = df[signal].copy()
model = joblib.load(snap_src / 'model_auction_relay_dragon.joblib')
candidate['ml_score_saved_full_fit'] = model.predict_proba(candidate[features].values)[:, 1]
candidate = candidate.sort_values(['date', 'ml_score_saved_full_fit'], ascending=[True, False])
candidate['daily_rank'] = candidate.groupby('date').cumcount() + 1
top1 = candidate[candidate.daily_rank == 1].copy()
train_keys = set(zip(train.date, train.code_str))
candidate['in_final_fit_dataset'] = [(d, c) in train_keys for d, c in zip(candidate.date, candidate.code_str)]
top1['in_final_fit_dataset'] = [(d, c) in train_keys for d, c in zip(top1.date, top1.code_str)]

market = df.groupby('date').agg(total_zt=('is_zt', 'sum'), total_touch=('is_touch_zt', 'sum'))
market['failed_rate'] = (market.total_touch - market.total_zt) / market.total_touch.replace(0, 1)
market['same_day_gate'] = (market.total_zt >= 15) & (market.failed_rate <= .45)
market['previous_observed_day_gate'] = market.same_day_gate.shift(1)
gated = top1[top1.date.isin(market.index[market.same_day_gate])]

saved_trades = pd.read_csv(snap_src / 'strategy_trade_log.csv', dtype={'code_str': str})
saved_trades['code_str'] = saved_trades.code_str.str.zfill(6)
candidate_keys = set(zip(candidate.date, candidate.code_str))
saved_keys = set(zip(saved_trades.date, saved_trades.code_str))
assert candidate_keys == saved_keys
candidate = candidate.merge(saved_trades[['date', 'code_str', 'return_trailing', 'holding_days', 'return_t1_open']], on=['date', 'code_str'], how='left')

calendar = sorted(df.date.unique())
day_idx = {d: i for i, d in enumerate(calendar)}
gap = df.date.map(day_idx) - df.prev_date.map(day_idx)
df['previous_observed_bar_gap'] = gap

raw_status = pd.read_csv(ROOT / 'data/v3/imported/daily-bars.csv',
    usecols=['source', 'status_verified', 'universe_complete', 'st', 'new_listing'])
bad_labels = df[df.label_false_positive_diagnostic]
bad_labels[['date', 'code_str', 'name', 'pre_close', 'open', 'high', 'close', 'limit_price', 'regular_limit_diagnostic']].to_csv(OUT / 'limit_label_mismatch_diagnostic.csv', index=False, encoding='utf-8-sig')
candidate[['date', 'code_str', 'name', 'open', 'close', 'target_limit_up', 'regular_zt_diagnostic', 'ml_score_saved_full_fit', 'daily_rank', 'in_final_fit_dataset', 'return_trailing', 'holding_days', 'return_t1_open']].to_csv(OUT / 'saved_model_candidate_replay.csv', index=False, encoding='utf-8-sig')

def candidate_stats(frame):
    return {'samples': len(frame), 'dates': frame.date.nunique(), 'reported_label_hits': int(frame.target_limit_up.sum()),
            'reported_label_rate': float(frame.target_limit_up.mean()),
            'regular_price_diagnostic_hits': int(frame.regular_zt_diagnostic.sum()),
            'regular_price_diagnostic_rate': float(frame.regular_zt_diagnostic.mean())}

result = {
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'fits_performed_by_this_audit': 0,
    'source_snapshot_sha256': before,
    'raw_daily_bars_sha256': sha(ROOT / 'data/v3/imported/daily-bars.csv'),
    'dataset': {'rows': len(df), 'first_date': df.date.min(), 'last_date': df.date.max(),
        'codes': df.code_str.nunique(), 'relay_rows': len(relay), 'non_one_relay_rows': len(non_one),
        'final_training_rows': len(train), 'oof_rows_from_first_test_start': int((train.date >= folds[0][2]).sum()),
        'folds_reconstructed_without_fitting': folds,
        'non_adjacent_previous_bar_rows': int((gap > 1).sum()),
        'non_adjacent_previous_bar_training_rows': int((gap.loc[train.index] > 1).sum()),
        '20pct_board_training_rows': int(train.code_str.str.startswith(('300', '301', '688')).sum()),
        'raw_sources': raw_status.source.value_counts(dropna=False).to_dict(),
        'raw_status_verified': raw_status.status_verified.value_counts(dropna=False).to_dict(),
        'raw_universe_complete': raw_status.universe_complete.value_counts(dropna=False).to_dict()},
    'volume_unit_check': unit_stats,
    'regular_limit_diagnostics': {'label_false_positives_all_rows': len(bad_labels),
        'training_positive_count': int(train.target_limit_up.sum()),
        'training_positive_below_regular_price': int(train.label_false_positive_diagnostic.sum()),
        'note': 'Diagnostic for standard 10%/20% prices only; excludes no special-event errors. No formal relabel/retraining performed.'},
    'replay': {'all_candidate_stats': candidate_stats(candidate), 'daily_top1_stats': candidate_stats(top1),
        'same_day_gated_top1_stats': candidate_stats(gated),
        'candidate_in_final_fit_rows': int(candidate.in_final_fit_dataset.sum()),
        'top1_in_final_fit_rows': int(top1.in_final_fit_dataset.sum()),
        'same_day_gate_vs_previous_observed_day_different_days': int((market.same_day_gate != market.previous_observed_day_gate).sum()),
        'saved_strict_t1_positive_returns': int((saved_trades.return_t1_open > 0).sum()),
        'saved_strict_t1_win_rate': float((saved_trades.return_t1_open > 0).mean()),
        'saved_strict_t1_mean_return': float(saved_trades.return_t1_open.mean()),
        'multi_day_trades': int((saved_trades.holding_days > 1).sum()),
        'trade_keys_equal_saved_log': candidate_keys == saved_keys},
    'artifact_limitations': {'walkforward_models_saved': False, 'row_level_oof_saved': False,
        'verified_auction_amount_in_training': False, 'verified_auction_orderbook_in_training': False,
        'ths_client_compile_or_field_parity_executed': False},
}
result['source_changed_during_audit'] = [str(p.relative_to(ROOT)) for p in paths if p.is_file() and sha(p) != before[str(p.relative_to(ROOT))]]
(OUT / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k not in ['source_snapshot_sha256']}, ensure_ascii=False, indent=2, default=str))
