import pandas as pd
import numpy as np

df = pd.read_csv('data/v16/point_in_time_rebuild_cutoff_20260831_v2/leader_candidates.csv')
b3 = df[df['boards'] == 3].copy()
b3['target_5'] = (b3['terminal_board'] >= 5).astype(int)

print("=== MARGINAL SUCCESS RATE (Reach >= 5 Boards) ===")
print("Overall Base Rate:", f"{b3['target_5'].mean():.3%}", f"({b3['target_5'].sum()}/{len(b3)})")

# 1. Price
b3['price_bin'] = pd.qcut(b3['price'], 4, labels=['Q1(Low)', 'Q2(Mid-Low)', 'Q3(Mid-High)', 'Q4(High)'])
print("\nBy Price Quartile:")
print(b3.groupby('price_bin', observed=False)['target_5'].agg(['count', 'mean']))

# 2. Market Cap (float_cap_log)
b3['cap_bin'] = pd.qcut(b3['float_cap_log'], 4, labels=['Q1(Small)', 'Q2(Mid-Small)', 'Q3(Mid-Large)', 'Q4(Large)'])
print("\nBy Market Cap Quartile:")
print(b3.groupby('cap_bin', observed=False)['target_5'].agg(['count', 'mean']))

# 3. Turnover
b3['to_bin'] = pd.cut(b3['turnover'], bins=[-1, 0.03, 0.10, 0.20, 1.0], labels=['<3%(One-word)', '3-10%(Low-Med)', '10-20%(Healthy)', '>20%(High)'])
print("\nBy Turnover Bin:")
print(b3.groupby('to_bin', observed=False)['target_5'].agg(['count', 'mean']))

# 4. Amplitude
b3['amp_bin'] = pd.cut(b3['amplitude'], bins=[-1, 2.0, 6.0, 10.0, 100.0], labels=['<=2%(Tight/One-word)', '2-6%(Solid)', '6-10%(Moderate)', '>10%(Chaotic/Bomb)'])
print("\nBy Amplitude Bin:")
print(b3.groupby('amp_bin', observed=False)['target_5'].agg(['count', 'mean']))

# 5. Cohort Rivals (same_height_rivals)
print("\nBy Same Height Rivals:")
print(b3.groupby('same_height_rivals', observed=False)['target_5'].agg(['count', 'mean']))

# 6. Is Space Leader
print("\nBy Is Space Leader:")
print(b3.groupby('is_space_leader', observed=False)['target_5'].agg(['count', 'mean']))
