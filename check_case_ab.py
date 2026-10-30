import sys
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('data/v16/point_in_time_rebuild_cutoff_20260831_v2/leader_candidates.csv')
daily_max = df.groupby('date')['boards'].max().rename('market_max_board')
df = df.merge(daily_max, on='date', how='left')

b3 = df[df['boards'] == 3].copy()
b3['is_already_space_leader'] = (b3['boards'] == b3['market_max_board']).astype(int)

total = len(b3)
case_a = int(b3['is_already_space_leader'].sum())
case_b = total - case_a

print(f"Total 3-board events: {total}")
print(f"Case A: Already Space Leader on Day t (market max == 3): {case_a} ({case_a/total:.2%})")
print(f"Case B: Not Space Leader on Day t (market max >= 4): {case_b} ({case_b/total:.2%})")

# Check terminal board for Case A vs Case B
print("\nCase A (Already High Board):")
print(f"  Reached >= 4 boards: {(b3[b3['is_already_space_leader']==1]['terminal_board']>=4).mean():.2%}")
print(f"  Reached >= 5 boards: {(b3[b3['is_already_space_leader']==1]['terminal_board']>=5).mean():.2%}")
print(f"  Reached >= 6 boards: {(b3[b3['is_already_space_leader']==1]['terminal_board']>=6).mean():.2%}")

print("\nCase B (Not High Board at Day t):")
print(f"  Reached >= 4 boards: {(b3[b3['is_already_space_leader']==0]['terminal_board']>=4).mean():.2%}")
print(f"  Reached >= 5 boards: {(b3[b3['is_already_space_leader']==0]['terminal_board']>=5).mean():.2%}")
print(f"  Reached >= 6 boards: {(b3[b3['is_already_space_leader']==0]['terminal_board']>=6).mean():.2%}")
