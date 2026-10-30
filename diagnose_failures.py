"""
diagnose_failures.py:
Analyzes why Top-1 candidates fail even when they open positive (Open > 0.0%).
Identifies the core failure patterns and microstructural traps.
"""

import numpy as np
import pandas as pd

oof = pd.read_csv('research/model_upgrade_validation/dragon_auction_and_second_wave_20261027T091600Z/oof_round6_predictions.csv')
daily = pd.read_csv('research/model_upgrade_validation/dragon_auction_and_second_wave_20261027T091600Z/daily_competition_round6.csv')

# Link each daily Top-1 to its full row in OOF
oof['code_str'] = oof['code'].astype(str).str.zfill(6)
oof['date_str'] = pd.to_datetime(oof['date']).dt.strftime('%Y-%m-%d')
daily['code_str'] = daily['top1_code'].astype(str).str.zfill(6)
daily['date_str'] = pd.to_datetime(daily['date']).dt.strftime('%Y-%m-%d')

merged = daily.merge(oof, left_on=['date_str', 'code_str'], right_on=['date_str', 'code_str'], suffixes=('', '_oof'))

# Positive open pool
pos_pool = merged[merged['is_auction_pos'] == 1].copy()
failures = pos_pool[pos_pool['top1_is_3to4'] == 0]
successes = pos_pool[pos_pool['top1_is_3to4'] == 1]

print(f"Total Positive Open Top-1 Days: {len(pos_pool)}")
print(f"Successes: {len(successes)} ({len(successes)/len(pos_pool):.1%})")
print(f"Failures: {len(failures)} ({len(failures)/len(pos_pool):.1%})")

# Factor 1: Open Gap distribution
print("\n--- Factor 1: Open Gap Bands ---")
bins = [-0.01, 0.02, 0.05, 0.07, 0.095, 0.11]
labels = ['0%~2% (Weak Positive)', '2%~5% (Moderate Sweet)', '5%~7% (Strong Sweet)', '7%~9.5% (High Danger Zone)', '>=9.5% (One-word Ceiling)']
pos_pool['open_band'] = pd.cut(pos_pool['top1_open_gap'], bins=bins, labels=labels)
for band, g in pos_pool.groupby('open_band', observed=False):
    rate_3to4 = g['top1_is_3to4'].mean()
    rate_space = g['top1_is_space_leader'].mean()
    print(f"{band} (N={len(g)}): 3进4胜率={rate_3to4:.1%}, 空间高标率={rate_space:.1%}")

# Factor 2: One-word Streak before 3-board (Unexchanged Profit Trap)
print("\n--- Factor 2: One-Word Streak (一字断层出货陷阱) ---")
for streak in [0, 1, 2, 3]:
    sub = pos_pool[pos_pool['one_word_streak'] == streak]
    if len(sub) > 0:
        print(f"One-word Streak = {streak} (N={len(sub)}): 3进4胜率={sub['top1_is_3to4'].mean():.1%}, 空间高标率={sub['top1_is_space_leader'].mean():.1%}")

# Factor 3: Turnover level on 3-board day
print("\n--- Factor 3: Day t Turnover Bands ---")
pos_pool['turnover_band'] = pd.cut(pos_pool['turnover'], bins=[-0.01, 0.03, 0.10, 0.20, 0.35, 1.0], labels=['<3% (Ultra Low)', '3%~10% (Low-Mid)', '10%~20% (Healthy Active)', '20%~35% (High Churn)', '>35% (Exhaustion)'])
for band, g in pos_pool.groupby('turnover_band', observed=False):
    print(f"{band} (N={len(g)}): 3进4胜率={g['top1_is_3to4'].mean():.1%}, 空间高标率={g['top1_is_space_leader'].mean():.1%}")

# Factor 4: Same-height rivals in market
print("\n--- Factor 4: Same Height Rivals (同板位竞争数量) ---")
for rivals in [0, 1, 2, 3, 4, 5]:
    sub = pos_pool[pos_pool['same_height_rivals'] == rivals]
    if len(sub) > 0:
        print(f"Rivals = {rivals} (N={len(sub)}): 3进4胜率={sub['top1_is_3to4'].mean():.1%}, 空间高标率={sub['top1_is_space_leader'].mean():.1%}")

# Factor 5: Theme Diffusion (Theme first board count)
print("\n--- Factor 5: Theme First Board Support (题材首板助攻扩散) ---")
pos_pool['theme_assist'] = pd.cut(pos_pool['theme_first_board_count'], bins=[-0.01, 0.5, 2.5, 5.5, 100], labels=['0 (孤勇者)', '1~2 (微弱扩散)', '3~5 (梯队爆发)', '>=6 (超级主线)'])
for band, g in pos_pool.groupby('theme_assist', observed=False):
    print(f"{band} (N={len(g)}): 3进4胜率={g['top1_is_3to4'].mean():.1%}, 空间高标率={g['top1_is_space_leader'].mean():.1%}")
