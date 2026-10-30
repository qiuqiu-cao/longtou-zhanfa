"""
calculate_exact_premiums.py:
Computes the exact return premiums across all Tiers:
"""

import sys
import json
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

daily = pd.read_csv('research/model_upgrade_validation/dragon_auction_and_second_wave_20261027T091600Z/daily_competition_round6.csv')
oof = pd.read_csv('research/model_upgrade_validation/dragon_auction_and_second_wave_20261027T091600Z/oof_round6_predictions.csv')
daily_bars = pd.read_csv('data/v3/imported/daily-bars.csv')

daily['date_str'] = pd.to_datetime(daily['date']).dt.strftime('%Y-%m-%d')
daily['code_str'] = daily['top1_code'].astype(str).str.zfill(6)
oof['date_str'] = pd.to_datetime(oof['date']).dt.strftime('%Y-%m-%d')
oof['code_str'] = oof['code'].astype(str).str.zfill(6)

daily_bars['date_str'] = pd.to_datetime(daily_bars['date']).dt.strftime('%Y-%m-%d')
daily_bars['code_str'] = daily_bars['code'].astype(str).str.zfill(6)
daily_bars = daily_bars.sort_values(['code_str', 'date_str']).reset_index(drop=True)

bars_by_code = {c: g[['date_str', 'open', 'high', 'low', 'close']].values for c, g in daily_bars.groupby('code_str')}

merged = daily.merge(oof, on=['date_str', 'code_str'], suffixes=('', '_oof'))

t2_open_premium = []
t2_high_premium = []
t2_close_return = []

for idx, r in merged.iterrows():
    c = r['code_str']
    d = r['date_str']
    b = bars_by_code.get(c)
    if b is None:
        t2_open_premium.append(np.nan)
        t2_high_premium.append(np.nan)
        t2_close_return.append(np.nan)
        continue
    
    dates = b[:, 0]
    future = b[dates > d]
    if len(future) < 2:
        t2_open_premium.append(np.nan)
        t2_high_premium.append(np.nan)
        t2_close_return.append(np.nan)
        continue
        
    bar_t1 = future[0] # Day t+1 (3进4当天)
    bar_t2 = future[1] # Day t+2 (次日卖出日)
    
    t1_close = bar_t1[4]
    
    t2_open = bar_t2[1]
    t2_high = bar_t2[2]
    t2_close = bar_t2[4]
    
    open_prem = (t2_open - t1_close) / t1_close
    high_prem = (t2_high - t1_close) / t1_close
    close_ret = (t2_close - t1_close) / t1_close
    
    t2_open_premium.append(open_prem)
    t2_high_premium.append(high_prem)
    t2_close_return.append(close_ret)

merged['t2_open_premium'] = t2_open_premium
merged['t2_high_premium'] = t2_high_premium
merged['t2_close_return'] = t2_close_return

tiers = [
    ("Tier 0: 裸模型基准 (无过滤)", merged),
    ("Tier 1: 原竞价红开 (>0.0%)", merged[merged['top1_open_gap'] > 0.0]),
    ("Tier 2: 增强竞价 (>=+2.0%)", merged[merged['top1_open_gap'] >= 0.02]),
    ("Tier 3: 竞价 + 潮汐闸门", merged[(merged['top1_open_gap'] >= 0.02) & (merged['prob_regime_dragon'] >= 0.55)]),
    ("Tier 4: 黄金强化级 (>=+2% + 潮汐 + 分>=0.35)", merged[(merged['top1_open_gap'] >= 0.02) & (merged['prob_regime_dragon'] >= 0.55) & (merged['top1_score'] >= 0.35)]),
    ("Tier 5: 极品巅峰级 (>=+2% + 潮汐 + 分>=0.40)", merged[(merged['top1_open_gap'] >= 0.02) & (merged['prob_regime_dragon'] >= 0.55) & (merged['top1_score'] >= 0.40)]),
    ("核按钮低开 (<=0.0%)", merged[merged['top1_open_gap'] <= 0.0]),
    ("0%~+2% 假弱转强陷阱", merged[(merged['top1_open_gap'] > 0.0) & (merged['top1_open_gap'] < 0.02)])
]

output_rows = []
print("="*105)
print(f"{'战法层级':<32} | {'样本':<5} | {'次日开盘溢价':<12} | {'次日冲高最高':<12} | {'次日收盘盈亏':<12} | {'10日持股最大':<12} | {'吃大面率(<-5%)':<12}")
print("="*105)

for name, sub in tiers:
    n = len(sub)
    mean_open_p = sub['t2_open_premium'].dropna().mean()
    mean_high_p = sub['t2_high_premium'].dropna().mean()
    mean_close_r = sub['t2_close_return'].dropna().mean()
    mean_10d_max = sub['forward_10d_max_return'].dropna().mean()
    loss_rate_5 = (sub['t2_close_return'].dropna() <= -0.05).mean()
    
    row_data = {
        "tier": name,
        "n": n,
        "t2_open_premium": float(mean_open_p),
        "t2_high_premium": float(mean_high_p),
        "t2_close_return": float(mean_close_r),
        "forward_10d_max_return": float(mean_10d_max),
        "big_loss_rate_5": float(loss_rate_5)
    }
    output_rows.append(row_data)
    
    print(f"{name:<30} | {n:<5} | {mean_open_p*100:+7.2f}%     | {mean_high_p*100:+7.2f}%     | {mean_close_r*100:+7.2f}%     | {mean_10d_max*100:+7.2f}%     | {loss_rate_5*100:6.2f}%")

with open('research/model_upgrade_validation/dragon_auction_and_second_wave_20261027T091600Z/exact_premiums_summary.json', 'w', encoding='utf-8') as f:
    json.dump(output_rows, f, indent=2, ensure_ascii=False)
print("Saved to exact_premiums_summary.json")
