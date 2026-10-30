"""audit_continuous_streaks.py

Verifies Fix 1 & Fix 4:
1. Every 3-board candidate event must strictly bind to its UNIQUE continuous streak along adjacent official trading sessions.
2. Checks whether terminal_board in leader_candidates.csv matches the adjacent official trading day sequence.
3. Identifies any event where a gap exists or where an event is unsettled.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data/v16/point_in_time_rebuild_cutoff_20260831_v2/leader_candidates.csv"

df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
sessions = sorted(df['date'].unique())
session_map = {s: i for i, s in enumerate(sessions)}

b3 = df[df['boards'] == 3].copy().sort_values(['date', 'code']).reset_index(drop=True)
print(f"Total 3-board candidate events: {len(b3)}")

# Audit each 3-board event
discrepancies = []
mature_settled = 0
unsettled = 0

for idx, row in b3.iterrows():
    c = row['code']
    d = row['date']
    orig_tb = row['terminal_board']
    orig_td = row['terminal_date']
    s_idx = session_map[d]
    
    # Trace forward along official sessions
    streak = [3]
    curr_s_idx = s_idx
    expected_b = 4
    
    while True:
        curr_s_idx += 1
        if curr_s_idx >= len(sessions):
            # Hit end of available calendar
            break
        next_date = sessions[curr_s_idx]
        # Check if code c appears on next_date with boards == expected_b
        match = df[(df['date'] == next_date) & (df['code'] == c)]
        if len(match) == 1 and match.iloc[0]['boards'] == expected_b:
            streak.append(expected_b)
            expected_b += 1
        else:
            # Streak broken on next_date
            break
            
    verified_max_board = max(streak)
    last_streak_date = sessions[s_idx + len(streak) - 1]
    
    # If streak hit calendar boundary (e.g. at 2026-08-31)
    is_at_boundary = (s_idx + len(streak) == len(sessions))
    
    if is_at_boundary:
        unsettled += 1
    else:
        mature_settled += 1
        
    if pd.notna(orig_tb) and verified_max_board != orig_tb:
        discrepancies.append({
            'date': d.strftime('%Y-%m-%d'),
            'code': c,
            'name': row['name'],
            'orig_terminal_board': orig_tb,
            'verified_streak_max': verified_max_board,
            'streak_length': len(streak),
            'last_streak_date': last_streak_date.strftime('%Y-%m-%d'),
            'is_at_boundary': is_at_boundary
        })

print(f"Mature Settled Events (within dataset): {mature_settled}")
print(f"Unsettled Events at Cutoff: {unsettled}")
print(f"Discrepancies found with original terminal_board: {len(discrepancies)}")
if discrepancies:
    print(pd.DataFrame(discrepancies).head(20).to_string())
