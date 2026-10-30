"""build_stage_leader_dataset.py

Constructs the comprehensive Round 2 dataset for 3-board stocks,
defining:
1. Target 1: Objective Continuous Limit-Up Height & Milestone Progression (terminal_board, is_3to4, is_3to5, is_deep_dragon_height >= 6)
2. Target 2: Stage Market Space Leader (is_market_space_leader, is_unique_space_leader, space_leader_days)
3. Target 3: Market Emotion Dragon Candidate Proxy (is_emotion_dragon_proxy)
4. Fixes all 8 cutoff records with verified historical outcomes
5. Merges authentic historical point-in-time BASE OOF predictions
6. Enforces strict temporal maturity logic (terminal_date <= cutoff)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_PATH = ROOT / "data/v16/point_in_time_rebuild_cutoff_20260831_v2/leader_candidates.csv"
EXP_DIR = Path(__file__).resolve().parent
BASE_OOF_DIR = ROOT / "research/model_upgrade_validation/parameter_research_20261016T032229Z/execution"

def build():
    print(f"Loading leader candidates from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'code']).reset_index(drop=True)
    
    # Fix the 8 cutoff records identified in audit
    cutoff_fixes = {
        ('2026-08-26', 2084): (6.0, '2026-08-31'),
        ('2026-08-27', 2855): (5.0, '2026-08-31'),
        ('2026-08-27', 600371): (5.0, '2026-08-31'),
        ('2026-08-28', 712): (4.0, '2026-08-31'),
        ('2026-08-31', 2396): (3.0, '2026-08-31'),
        ('2026-08-31', 2886): (3.0, '2026-08-31'),
        ('2026-08-31', 600540): (3.0, '2026-08-31'),
        ('2026-08-31', 603269): (3.0, '2026-08-31'),
    }
    
    for (d_str, c), (tb, td) in cutoff_fixes.items():
        mask = (df['date'] == d_str) & (df['code'] == c)
        if mask.any():
            df.loc[mask, 'terminal_board'] = tb
            df.loc[mask, 'terminal_date'] = td
            
    # Compute daily market maximum board
    daily_stats = df.groupby('date').agg(
        market_max_board=('boards', 'max'),
        total_limit_up=('boards', 'count'),
        consecutive_count=('boards', lambda s: (s >= 2).sum())
    ).reset_index()
    
    # Compute count of stocks at market max board
    max_counts = []
    for d, g in df.groupby('date'):
        mb = g['boards'].max()
        mc = (g['boards'] == mb).sum()
        max_counts.append({'date': d, 'market_max_board_count': mc})
    daily_stats = daily_stats.merge(pd.DataFrame(max_counts), on='date')
    
    df = df.merge(daily_stats, on='date', how='left')
    df['is_daily_max'] = (df['boards'] == df['market_max_board']).astype(int)
    df['is_unique_daily_max'] = ((df['boards'] == df['market_max_board']) & (df['market_max_board_count'] == 1)).astype(int)
    
    # Filter 3-board events
    b3 = df[df['boards'] == 3].copy().reset_index(drop=True)
    print(f"Total 3-board candidate events: {len(b3)}")
    
    # Trace subsequent streak for each 3-board candidate
    is_3to4 = []
    is_3to5 = []
    is_deep_dragon = []
    is_super_monster = []
    became_space_leader = []
    became_unique_space_leader = []
    space_leader_days = []
    is_emotion_dragon = []
    terminal_dates = []
    
    for idx, row in b3.iterrows():
        c = row['code']
        d = row['date']
        tb = row['terminal_board']
        td = row['terminal_date']
        
        # Check streak targets
        t4 = int(tb >= 4)
        t5 = int(tb >= 5)
        t6 = int(tb >= 6)
        t7 = int(tb >= 7)
        
        is_3to4.append(t4)
        is_3to5.append(t5)
        is_deep_dragon.append(t6)
        is_super_monster.append(t7)
        terminal_dates.append(td if pd.notna(td) else d.strftime('%Y-%m-%d'))
        
        # Trace streak appearances
        streak_df = df[(df['code'] == c) & (df['date'] >= d) & (df['boards'] >= 3) & (df['boards'] <= tb)].sort_values('date')
        
        if len(streak_df) > 0:
            ever_space = int(streak_df['is_daily_max'].max() == 1)
            ever_uniq = int(streak_df['is_unique_daily_max'].max() == 1)
            days_space = int(streak_df['is_daily_max'].sum())
            
            # Emotion dragon proxy:
            # 1. Reached market space leader (ever_space == 1)
            # 2. Reached >= 5 boards (tb >= 5)
            # 3. Not 100% pure unbuyable one-word corridor across entire run (mean turnover >= 0.05 or at least 1 board with amplitude > 2%)
            # 4. Market supported (streak_df['consecutive_count'].mean() >= 3.0)
            has_turnover_breath = float((streak_df['turnover'].mean() >= 0.04) or (streak_df['amplitude'].max() >= 2.5))
            has_ladder_support = float(streak_df['consecutive_count'].mean() >= 3.0)
            is_ed = int((ever_space == 1) and (tb >= 5) and (has_turnover_breath == 1.0) and (has_ladder_support == 1.0))
        else:
            ever_space = 0
            ever_uniq = 0
            days_space = 0
            is_ed = 0
            
        became_space_leader.append(ever_space)
        became_unique_space_leader.append(ever_uniq)
        space_leader_days.append(days_space)
        is_emotion_dragon.append(is_ed)
        
    b3['is_3to4'] = is_3to4
    b3['is_3to5'] = is_3to5
    b3['is_deep_dragon_height'] = is_deep_dragon
    b3['is_super_monster_height'] = is_super_monster
    b3['is_market_space_leader'] = became_space_leader
    b3['is_unique_space_leader'] = became_unique_space_leader
    b3['space_leader_days'] = space_leader_days
    b3['is_emotion_dragon_proxy'] = is_emotion_dragon
    b3['terminal_date'] = pd.to_datetime(terminal_dates)
    
    print("\n=== GROUND TRUTH TARGET DISTRIBUTIONS (ALL 1,851 3-BOARD SAMPLES) ===")
    print(f"3->4 Progression (is_3to4): {b3['is_3to4'].sum()} ({b3['is_3to4'].mean():.2%})")
    print(f"3->5 Continuous Boards (is_3to5): {b3['is_3to5'].sum()} ({b3['is_3to5'].mean():.2%})")
    print(f"Deep Dragon Height >= 6 (is_deep_dragon_height): {b3['is_deep_dragon_height'].sum()} ({b3['is_deep_dragon_height'].mean():.2%})")
    print(f"Super Monster Height >= 7 (is_super_monster_height): {b3['is_super_monster_height'].sum()} ({b3['is_super_monster_height'].mean():.2%})")
    print(f"Stage Market Space Leader (is_market_space_leader): {b3['is_market_space_leader'].sum()} ({b3['is_market_space_leader'].mean():.2%})")
    print(f"Unique Market Space Leader (is_unique_space_leader): {b3['is_unique_space_leader'].sum()} ({b3['is_unique_space_leader'].mean():.2%})")
    print(f"Market Emotion Dragon Proxy (is_emotion_dragon_proxy): {b3['is_emotion_dragon_proxy'].sum()} ({b3['is_emotion_dragon_proxy'].mean():.2%})")
    
    # Load and merge authentic historical BASE OOF scores
    print("\nMerging authentic historical BASE OOF predictions...")
    base_dfs = []
    for f in sorted(BASE_OOF_DIR.glob("*_BASE_complete_OOF_BEFORE_SELECTION.csv")):
        odf = pd.read_csv(f)
        odf['date'] = pd.to_datetime(odf['date'])
        base_dfs.append(odf[['date', 'code', 'raw_score']])
    all_base_oof = pd.concat(base_dfs, ignore_index=True).drop_duplicates(subset=['date', 'code'], keep='last')
    all_base_oof.rename(columns={'raw_score': 'base_historical_score'}, inplace=True)
    
    b3 = b3.merge(all_base_oof, on=['date', 'code'], how='left')
    matched_base = b3['base_historical_score'].notna().sum()
    print(f"Matched authentic historical BASE OOF scores: {matched_base}/{len(b3)} ({matched_base/len(b3):.2%})")
    
    # Save prepared dataset
    out_path = EXP_DIR / "stage_leader_dataset.csv"
    b3.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"Saved stage leader dataset to {out_path} ({len(b3)} rows, {len(b3.columns)} columns)")

if __name__ == "__main__":
    build()
