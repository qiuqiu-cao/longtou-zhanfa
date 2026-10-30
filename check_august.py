import pandas as pd

df = pd.read_csv('data/v16/point_in_time_rebuild_cutoff_20260831_v2/leader_candidates.csv')
b3 = df[df['boards'] == 3].copy()
b3_aug = b3[b3['date'].str.startswith('2026-08')].copy()

print("August 2026 3-board candidates:")
print(b3_aug[['date', 'code', 'name', 'terminal_board', 'price', 'turnover', 'amplitude']])
