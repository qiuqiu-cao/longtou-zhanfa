"""Comprehensive benchmark of the 4 levers on 7,254 qualified candidates across 286 dates.
No retraining, no data snooping, exact reproduction of transaction costs and constraints.
"""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from legacy_industry_snapshot import attach_legacy_groups

ROOT = Path(__file__).resolve().parent
cal = json.loads((ROOT / 'data/v279/exchange_calendar_audit_20261002_v1/calendar.json').read_text(encoding='utf-8'))['sessions']
pos = {d: i for i, d in enumerate(cal)}

# Load bar data
SOURCE = ROOT / 'data/v16/point_in_time_rebuild_cutoff_20260831_v2'
cols = ['date','code','name','open','high','low','close','pre_close','volume','amount','turnover','float_cap','st','new_listing','status_verified']
bars1 = pd.read_csv(SOURCE / 'point_in_time_current_daily_bars.csv', usecols=cols, dtype={'code':str}, low_memory=False)
bars2 = pd.read_csv(SOURCE / 'verified_recovered_daily_bars.csv', usecols=cols, dtype={'code':str}, low_memory=False)
bars = pd.concat([bars1, bars2], ignore_index=True)
bars['code'] = bars['code'].str.zfill(6)
bars['date'] = bars['date'].astype(str).str[:10]
bars = bars.drop_duplicates(['date', 'code'], keep='last')
numeric = ['open','high','low','close','pre_close','volume','amount','turnover','float_cap']
bars[numeric] = bars[numeric].apply(pd.to_numeric, errors='coerce')
bars['scope_known'] = bars['status_verified'].astype(str).str.lower().isin(['true', '1', '1.0'])
bars['eligible'] = bars['scope_known'] & ~bars['st'].astype(str).str.lower().isin(['true', '1', '1.0']) & ~bars['new_listing'].astype(str).str.lower().isin(['true', '1', '1.0']) & bars['code'].str.startswith(('000','001','002','003','600','601','603','605'))
bars['p'] = bars['date'].map(pos)
bars = bars[bars['p'].notna()].sort_values(['code', 'p']).reset_index(drop=True)
bars['p'] = bars['p'].astype(int)
bars['valid_price'] = bars[['open','high','low','close','pre_close']].gt(0).all(axis=1) & bars['high'].ge(bars[['open','close','low']].max(axis=1)) & bars['low'].le(bars[['open','close']].min(axis=1))
bars['cap'] = np.floor(bars['pre_close'].values * 110 + 0.5 + 1e-8)
bars['floor'] = np.floor(bars['pre_close'].values * 90 + 0.5 + 1e-8)

# Load predictions and screen
s1_preds = pd.read_csv(ROOT / 'research/first_second_stage_20261020_run1/S1_ALL_PREDICTIONS.csv', dtype={'code':str})
height_preds = pd.read_csv(ROOT / 'research/first_board_leader_20261020T114600Z/evaluation_ALL_PREDICTIONS.csv', dtype={'code':str})
screen = pd.read_csv(ROOT / 'research/first_board_leader_emotion_20261021T1706Z/ALL28503_PATHS_AND_ORIGINAL_SCREEN.csv', dtype={'code':str})

test = screen[screen['signal_date'].between('2025-07-01', '2026-08-31')].copy()
test_qualified = test[test['allowed'].astype(str).str.lower().isin(['true', '1', '1.0'])].copy()

# Merge predictions
df = test_qualified.merge(height_preds[['signal_date', 'code', 'model_score']].rename(columns={'model_score': 'height_score'}), on=['signal_date', 'code'], validate='one_to_one')
df = df.merge(s1_preds[['signal_date', 'code', 'model_score']].rename(columns={'model_score': 's1_score'}), on=['signal_date', 'code'], validate='one_to_one')

# Attach industry
df = attach_legacy_groups(df)

# Fast bar lookup builder for a set of codes
def build_lookup_for_codes(codes):
    sub = bars[bars['code'].isin(codes)]
    return sub.set_index(['date', 'code']).to_dict('index')

def simulate_strategy(picks, look=None):
    if look is None:
        look = build_lookup_for_codes(set(picks['code']))
    by_entry = {cal[int(r.p)+1]: r for r in picks.itertuples() if int(r.p)+1 < len(cal)}
    start = min(by_entry) if by_entry else cal[-1]
    end = bars['date'].max()
    cash = 1_000_000.0
    positions = []
    events = []
    equity = []
    last = {}
    
    for p, day in enumerate(cal):
        if not start <= day <= end: continue
        mark = cash + sum(t['shares'] * last.get(t['code'], t['entry_price']) for t in positions)
        candidate = by_entry.get(day)
        if candidate is not None:
            bar = look.get((day, candidate.code))
            reason = None
            if bar is None or not bar['valid_price'] or not bar['eligible']: reason = 'ENTRY_UNKNOWN_OR_INELIGIBLE'
            elif round(bar['open']*100) >= bar['cap']: reason = 'OPEN_AT_UPPER_LIMIT_NO_QUEUE_ASSUMPTION'
            elif round(bar['open']*100) <= bar['floor']: reason = 'OPEN_AT_LOWER_LIMIT_CANCEL'
            if reason:
                events.append(dict(signal_date=candidate.signal_date, code=candidate.code, entry_date=day, status=reason))
            else:
                price = math.ceil(bar['open'] * 1.001 * 100 - 1e-8) / 100
                budget = min(cash / 1.001, mark * 0.20, max(0, float(candidate.amount)) * 0.01 if pd.notna(candidate.amount) else 0)
                shares = int(budget / price / 100) * 100
                if shares and round(price * 100) < bar['cap']:
                    value = shares * price
                    fee = max(5, value * 0.0003) + value * 0.00001
                    cash -= value + fee
                    positions.append(dict(signal_date=candidate.signal_date, code=candidate.code, entry_date=day, entry_p=p, planned_exit_p=p+2, shares=shares, entry_price=price, entry_cost=value+fee, status='OPEN', blocked_days=0))
                else:
                    events.append(dict(signal_date=candidate.signal_date, code=candidate.code, entry_date=day, status='ENTRY_LOT_OR_SLIPPAGE_LIMIT_CANCEL'))
        for t in list(positions):
            bar = look.get((day, t['code']))
            if bar and bar['valid_price']: last[t['code']] = bar['close']
            if p < t['planned_exit_p']: continue
            if bar is None or not bar['valid_price'] or not bar['eligible'] or round(bar['close']*100) <= bar['floor'] or not (bar['volume'] > 0):
                t['blocked_days'] += 1; continue
            price = math.floor(bar['close'] * 0.999 * 100 + 1e-8) / 100
            value = t['shares'] * price
            fee = max(5, value * 0.0003) + value * (0.00001 + 0.0005)
            cash += value - fee
            t.update(status='SCENARIO_CLOSED', exit_date=day, exit_price=price, net_pnl=value-fee-t['entry_cost'], net_return=(value-fee)/t['entry_cost']-1)
            events.append(t.copy())
            positions.remove(t)
        equity.append(dict(date=day, equity=cash+sum(t['shares']*last.get(t['code'], t['entry_price']) for t in positions), cash=cash, open_positions=len(positions)))
    
    events.extend([{**t, 'status': 'OPEN_UNRESOLVED', 'mark': last.get(t['code'], t['entry_price'])} for t in positions])
    trades = pd.DataFrame(events)
    curve = pd.DataFrame(equity)
    closed = trades[trades['status'].eq('SCENARIO_CLOSED')] if len(trades) else trades
    returns = closed['net_return'] if len(closed) else pd.Series(dtype=float)
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    total_ret = curve['equity'].iloc[-1] / 1e6 - 1 if len(curve) else None
    max_dd = float((curve['equity'] / curve['equity'].cummax() - 1).min()) if len(curve) else None
    win_rate = float((returns > 0).mean()) if len(returns) else None
    p_factor = closed.loc[closed['net_pnl'] > 0, 'net_pnl'].sum() / -closed.loc[closed['net_pnl'] < 0, 'net_pnl'].sum() if len(closed) and closed['net_pnl'].lt(0).any() else None
    
    return dict(selected=len(picks), closed=len(closed), total_return=total_ret, max_dd=max_dd, win_rate=win_rate, profit_factor=p_factor, pnl_sum=closed['net_pnl'].sum() if len(closed) else 0)

# Check defense condition
def is_defense_day(r):
    flags = []
    if r.get('market_down_limits', 0) >= 8: flags.append('down>=8')
    if r.get('market_failed_rate', 0) >= 0.40: flags.append('fail>=40%')
    if r.get('previous_first_promotion', 1.0) < 0.10: flags.append('p1<10%')
    if r.get('market_positive_fraction', 1.0) < 0.25: flags.append('pos<25%')
    return len(flags) >= 2

defense_series = df.groupby('signal_date').first().apply(is_defense_day, axis=1)
defense_dates = set(defense_series[defense_series].index)
print(f'Total dates: {df["signal_date"].nunique()}, Defense dates: {len(defense_dates)}')

# Sector peer calculation
known_ind = df[df['legacy_industry_group_PROXY'].notna()].copy()
peer_counts = known_ind.groupby(['signal_date', 'legacy_industry_group_PROXY'])['code'].transform('count')
df.loc[known_ind.index, 'peer_count'] = peer_counts
df['peer_count'] = df['peer_count'].fillna(0)

# Benchmarking
configs = [
    ('1. Legacy Unscreened Height (Reference)', height_preds.sort_values(['signal_date', 'model_score', 'code'], ascending=[True, False, True]).groupby('signal_date').head(1)),
    ('2. Screened Pure Height (w=0.0)', df.sort_values(['signal_date', 'height_score', 'code'], ascending=[True, False, True]).groupby('signal_date').head(1)),
    ('3. Screened Pure S1 (w=1.0)', df.sort_values(['signal_date', 's1_score', 'code'], ascending=[True, False, True]).groupby('signal_date').head(1)),
    ('4. Screened Fusion w=0.5', df.assign(score=0.5*df['s1_score']+0.5*df['height_score']).sort_values(['signal_date', 'score', 'code'], ascending=[True, False, True]).groupby('signal_date').head(1)),
    ('5. Screened Fusion w=0.7', df.assign(score=0.7*df['s1_score']+0.3*df['height_score']).sort_values(['signal_date', 'score', 'code'], ascending=[True, False, True]).groupby('signal_date').head(1)),
    ('6. Screened Fusion w=0.5 + Defense', df[~df['signal_date'].isin(defense_dates)].assign(score=0.5*df['s1_score']+0.5*df['height_score']).sort_values(['signal_date', 'score', 'code'], ascending=[True, False, True]).groupby('signal_date').head(1)),
    ('7. Screened Fusion w=0.7 + Defense', df[~df['signal_date'].isin(defense_dates)].assign(score=0.7*df['s1_score']+0.3*df['height_score']).sort_values(['signal_date', 'score', 'code'], ascending=[True, False, True]).groupby('signal_date').head(1)),
]

# Run all simulations
for name, picks in configs:
    res = simulate_strategy(picks)
    p_next = picks['promotion_next_LABEL_ONLY'].mean() if 'promotion_next_LABEL_ONLY' in picks else np.nan
    r4 = picks['reach4_LABEL_ONLY'].mean() if 'reach4_LABEL_ONLY' in picks else np.nan
    print(f'=== {name} ===')
    print(f'  Picks: {res["selected"]}, Closed: {res["closed"]}, Win Rate: {res["win_rate"]*100:.2f}%, Total Ret: {res["total_return"]*100:.2f}%, MaxDD: {res["max_dd"]*100:.2f}%, PnL: {res["pnl_sum"]:.2f}, 1-to-2: {p_next*100:.2f}%, R4: {r4*100:.2f}%')
