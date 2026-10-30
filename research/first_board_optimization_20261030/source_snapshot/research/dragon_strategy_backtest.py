"""Dragon Strategy Backtest: Testing the 5 core short-term trading dimensions.
1. Pattern: Breakout & MA Bull Alignment vs Oversold Bear Trap.
2. Small-Cap Flexibility: Preference for <= 35亿 float cap (Chuanzhi Education DNA).
3. Sector Resonance: 2~7 peers in sector vs solo stocks.
4. Market Emotion Defense: Broken board nuclear dump (down_limits >= 8, fail_rate >= 40%, etc.).
5. Comprehensive Dragon Ranker.
"""
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import json
import math
import numpy as np
import pandas as pd
from legacy_industry_snapshot import attach_legacy_groups
cal = json.loads((ROOT / 'data/v279/exchange_calendar_audit_20261002_v1/calendar.json').read_text(encoding='utf-8'))['sessions']
pos = {d: i for i, d in enumerate(cal)}

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

s1_preds = pd.read_csv(ROOT / 'research/first_second_stage_20261020_run1/S1_ALL_PREDICTIONS.csv', dtype={'code':str})
height_preds = pd.read_csv(ROOT / 'research/first_board_leader_20261020T114600Z/evaluation_ALL_PREDICTIONS.csv', dtype={'code':str})
screen = pd.read_csv(ROOT / 'research/first_board_leader_emotion_20261021T1706Z/ALL28503_PATHS_AND_ORIGINAL_SCREEN.csv', dtype={'code':str})

test = screen[screen['signal_date'].between('2025-07-01', '2026-08-31')].copy()
test_qualified = test[test['allowed'].astype(str).str.lower().isin(['true', '1', '1.0'])].copy()

df = test_qualified.merge(height_preds[['signal_date', 'code', 'model_score']].rename(columns={'model_score': 'height_score'}), on=['signal_date', 'code'], validate='one_to_one')
df = df.merge(s1_preds[['signal_date', 'code', 'model_score']].rename(columns={'model_score': 's1_score'}), on=['signal_date', 'code'], validate='one_to_one')

df = attach_legacy_groups(df)

# Compute peer count
known_ind = df[df['legacy_industry_group_PROXY'].notna()].copy()
peer_counts = known_ind.groupby(['signal_date', 'legacy_industry_group_PROXY'])['code'].transform('count')
df.loc[known_ind.index, 'peer_count'] = peer_counts
df['peer_count'] = df['peer_count'].fillna(0)

# Build pattern features
# 1. Breakout & MA Bull Alignment
df['is_breakout'] = df['breakout_20'].fillna(-1) >= 0
df['is_ma_bull'] = (df['close_ma5'] > df['close_ma20']) & (df['close_ma20'] > df['close_ma60'])
df['is_high_position'] = df['position_60'].fillna(0) >= 0.65

# 2. Oversold Bear Trap (Down at bottom 30% and far below 60MA)
df['is_oversold_trap'] = (df['position_60'].fillna(1.0) < 0.30) & (df['close_ma60'].fillna(0.0) < -0.08)

# 3. Small-cap advantage
df['is_small_cap'] = df['float_cap'] <= 35e8

# 4. Sector Resonance (2~7 peers) vs Solo stock (1 peer)
df['is_sector_resonance'] = df['peer_count'].between(2, 7)
df['is_solo_stock'] = df['peer_count'].le(1)

# Market Emotion Defense (nuclear dump)
def is_defense_day(r):
    flags = []
    if r.get('market_down_limits', 0) >= 8: flags.append('down>=8')
    if r.get('market_failed_rate', 0) >= 0.40: flags.append('fail>=40%')
    if r.get('previous_first_promotion', 1.0) < 0.10: flags.append('p1<10%')
    if r.get('market_positive_fraction', 1.0) < 0.25: flags.append('pos<25%')
    return len(flags) >= 2

defense_series = df.groupby('signal_date').first().apply(is_defense_day, axis=1)
defense_dates = set(defense_series[defense_series].index)

def build_lookup_for_codes(codes):
    sub = bars[bars['code'].isin(codes)]
    return sub.set_index(['date', 'code']).to_dict('index')

def simulate_strategy(picks, look=None, veto_flat_down_open=False):
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
            # 09:25 竞价一票否决：平开或低开取消买单
            elif veto_flat_down_open and bar['open'] <= bar['pre_close']: reason = 'AUCTION_FLAT_OR_DOWN_CANCEL'
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

# Construct Composite Enhanced Dragon Score
# Base: 0.6 * s1_score + 0.4 * height_score
df['base_fusion'] = 0.6 * df['s1_score'] + 0.4 * df['height_score']

# Pattern adjustments:
# +0.02 for Breakout & MA bull
# -0.05 for Oversold Bear Trap
pattern_adj = np.where(df['is_breakout'] & df['is_ma_bull'], 0.02, 0.0)
pattern_adj = np.where(df['is_oversold_trap'], -0.05, pattern_adj)

# Small-cap adjustment (+0.025 for <= 35亿 float cap)
cap_adj = np.where(df['is_small_cap'], 0.025, 0.0)

# Sector resonance adjustment (+0.015 for 2~7 peers, -0.015 for solo)
sector_adj = np.where(df['is_sector_resonance'], 0.015, np.where(df['is_solo_stock'], -0.015, 0.0))

df['dragon_score'] = df['base_fusion'] + pattern_adj + cap_adj + sector_adj

# Prepare comparison configs
configs = [
    ('1. 原基准: 5:5机械融合 (无竞价过滤)', df.assign(score=df['base_fusion']).sort_values(['signal_date', 'score', 'code'], ascending=[True, False, True]).groupby('signal_date').head(1), False),
    ('2. 加入图形过滤 (淘汰超跌熊坑/奖励突破多头)', df.assign(score=df['base_fusion'] + pattern_adj).sort_values(['signal_date', 'score', 'code'], ascending=[True, False, True]).groupby('signal_date').head(1), False),
    ('3. 加入传智式小市值偏好 (<=35亿)', df.assign(score=df['base_fusion'] + cap_adj).sort_values(['signal_date', 'score', 'code'], ascending=[True, False, True]).groupby('signal_date').head(1), False),
    ('4. 加入题材板块共振 (2~7只协同)', df.assign(score=df['base_fusion'] + sector_adj).sort_values(['signal_date', 'score', 'code'], ascending=[True, False, True]).groupby('signal_date').head(1), False),
    ('5. 图形+小盘+题材 全维度融合 (无防守)', df.sort_values(['signal_date', 'dragon_score', 'code'], ascending=[True, False, True]).groupby('signal_date').head(1), False),
    ('6. 全维度融合 + 恶性退潮断板防守', df[~df['signal_date'].isin(defense_dates)].sort_values(['signal_date', 'dragon_score', 'code'], ascending=[True, False, True]).groupby('signal_date').head(1), False),
    ('7. 最终实战闭环: 全维度融合+防守+09:25竞价平低开否决', df[~df['signal_date'].isin(defense_dates)].sort_values(['signal_date', 'dragon_score', 'code'], ascending=[True, False, True]).groupby('signal_date').head(1), True),
]

results = []
for name, picks, veto in configs:
    res = simulate_strategy(picks, veto_flat_down_open=veto)
    p_next = picks['promotion_next_LABEL_ONLY'].mean() if 'promotion_next_LABEL_ONLY' in picks else np.nan
    r4 = picks['reach4_LABEL_ONLY'].mean() if 'reach4_LABEL_ONLY' in picks else np.nan
    
    date_picks = picks['signal_date'].unique()
    top3_all = df[df['signal_date'].isin(date_picks)].sort_values(['signal_date', 'dragon_score' if 'dragon' in name or '融合' in name else 'base_fusion', 'code'], ascending=[True, False, True]).groupby('signal_date').head(3)
    p_next_top3 = top3_all['promotion_next_LABEL_ONLY'].mean()
    r4_top3 = top3_all['reach4_LABEL_ONLY'].mean()
    
    results.append({
        'name': name,
        'picks': res['selected'],
        'closed': res['closed'],
        'top1_p_next': p_next,
        'top1_r4': r4,
        'top3_p_next': p_next_top3,
        'top3_r4': r4_top3,
        'win_rate': res['win_rate'],
        'total_ret': res['total_return'],
        'max_dd': res['max_dd'],
        'pnl_sum': res['pnl_sum']
    })
    print(f'Done: {name}')

res_df = pd.DataFrame(results)
res_df.to_csv(ROOT / 'research/dragon_strategy_backtest_results.csv', index=False, encoding='utf-8-sig')

# Print formatted summary table
print('\n' + '='*90)
print(f'{"方案名称":<36} | {"Top1一进二":<8} | {"实盘胜率":<8} | {"累计回报":<10} | {"最大回撤":<10}')
print('-'*90)
for r in results:
    p_str = f"{r['top1_p_next']*100:.2f}%" if pd.notna(r['top1_p_next']) else "N/A"
    w_str = f"{r['win_rate']*100:.2f}%" if pd.notna(r['win_rate']) else "N/A"
    ret_str = f"{r['total_ret']*100:.2f}%" if pd.notna(r['total_ret']) else "N/A"
    dd_str = f"{r['max_dd']*100:.2f}%" if pd.notna(r['max_dd']) else "N/A"
    print(f"{r['name']:<34} | {p_str:<8} | {w_str:<8} | {ret_str:<10} | {dd_str:<10}")
print('='*90)
