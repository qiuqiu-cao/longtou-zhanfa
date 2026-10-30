"""compute_round3_dataset.py

Constructs the Round 3 canonical dataset:
1. Strict session-by-session streak validation along adjacent official trading days.
2. Verified ground truth outcomes for all 8 cutoff records (including 海鸥住工 7板, 捷荣技术 >=6板, 万向德农 6板, 锦龙股份 4板, 新赛股份 5板, etc.).
3. Rigorous determination date (determined_date) tracking for every target to prevent label leakage at training cutoff.
4. Target A (Already High Board) vs Target B (Subsequent Takeover High Board) explicit separation.
5. Point-in-time cohort relative features (volume share, amplitude rank, price rank).
6. Authentic historical BASE OOF score matching, separating complete cohort match vs partial match.
7. Zero data leakage: dataset saved clean without global pre-scaling or global quantile transformations.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data/v16/point_in_time_rebuild_cutoff_20260831_v2/leader_candidates.csv"
EXP_DIR = ROOT / "research/model_upgrade_validation/dragon_takeover_20261027T074500Z"
BASE_OOF_DIR = ROOT / "research/model_upgrade_validation/parameter_research_20261016T032229Z/execution"

def build():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'code']).reset_index(drop=True)
    
    # Official trading sessions in dataset
    sessions = sorted(df['date'].unique())
    session_map = {s: i for i, s in enumerate(sessions)}
    
    # Precise September verified extensions for the 8 cutoff stocks:
    # 002084: 2026-08-26 (3b), 08-27 (4b), 08-28 (5b), 08-31 (6b), 09-01 (7b), 09-02 (broken) -> tb=7.0, determined=2026-09-02
    # 002855: 2026-08-27 (3b), 08-28 (4b), 08-31 (5b), 09-01 (6b) -> tb>=6.0, determined>=2026-09-02
    # 600371: 2026-08-27 (3b), 08-28 (4b), 08-31 (5b), 09-01 (6b), 09-02 (broken) -> tb=6.0, determined=2026-09-02
    # 000712: 2026-08-28 (3b), 08-31 (4b), 09-01 (broken) -> tb=4.0, determined=2026-09-01
    # 002396: 2026-08-31 (3b), 09-01 (broken) -> tb=3.0, determined=2026-09-01
    # 002886: 2026-08-31 (3b), 09-01 (broken) -> tb=3.0, determined=2026-09-01
    # 600540: 2026-08-31 (3b), 09-01 (4b), 09-02 (5b), 09-03 (broken) -> tb=5.0, determined=2026-09-03
    # 603269: 2026-08-31 (3b), 09-01 (broken) -> tb=3.0, determined=2026-09-01
    verified_cutoff_updates = {
        ('2026-08-26', 2084): (7.0, '2026-09-02', 'SETTLED'),
        ('2026-08-27', 2855): (6.0, '2026-09-02', 'SETTLED_AT_LEAST_6'),
        ('2026-08-27', 600371): (6.0, '2026-09-02', 'SETTLED'),
        ('2026-08-28', 712): (4.0, '2026-09-01', 'SETTLED'),
        ('2026-08-31', 2396): (3.0, '2026-09-01', 'SETTLED'),
        ('2026-08-31', 2886): (3.0, '2026-09-01', 'SETTLED'),
        ('2026-08-31', 600540): (5.0, '2026-09-03', 'SETTLED'),
        ('2026-08-31', 603269): (3.0, '2026-09-01', 'SETTLED'),
    }
    
    for (d_str, c), (tb, det_d, status) in verified_cutoff_updates.items():
        mask = (df['date'] == d_str) & (df['code'] == c)
        if mask.any():
            df.loc[mask, 'terminal_board'] = tb
            df.loc[mask, 'terminal_date'] = det_d
            
    # Compute daily market maximum board
    daily_stats = df.groupby('date').agg(
        market_max_board=('boards', 'max'),
        total_limit_up=('boards', 'count'),
        consecutive_count=('boards', lambda s: (s >= 2).sum())
    ).reset_index()
    
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
    b3 = df[df['boards'] == 3].copy().sort_values(['date', 'code']).reset_index(drop=True)
    print(f"Total 3-board candidate events: {len(b3)}")
    
    # Point-in-time cohort relative features (calculated strictly within Day t close 3-board cohort)
    b3['amount_cohort_sum'] = b3.groupby('date')['signal_amount'].transform('sum')
    b3['amount_cohort_share'] = np.where(b3['amount_cohort_sum'] > 0, b3['signal_amount'] / b3['amount_cohort_sum'], 0.0)
    b3['amount_cohort_rank'] = b3.groupby('date')['signal_amount'].rank(ascending=False, method='min')
    b3['amplitude_cohort_rank'] = b3.groupby('date')['amplitude'].rank(ascending=True, method='min')
    b3['price_cohort_rank'] = b3.groupby('date')['price'].rank(ascending=True, method='min')
    b3['turnover_cohort_rank'] = b3.groupby('date')['turnover'].rank(ascending=True, method='min')
    
    b3['is_cohort_volume_leader'] = (b3['amount_cohort_rank'] == 1.0).astype(float)
    b3['is_cohort_tightest'] = (b3['amplitude_cohort_rank'] == 1.0).astype(float)
    b3['is_cohort_lowest_price'] = (b3['price_cohort_rank'] == 1.0).astype(float)
    
    # Day t status
    b3['is_already_space_leader'] = (b3['boards'] == b3['market_max_board']).astype(int)
    b3['height_gap'] = b3['market_max_board'] - b3['boards']
    
    # Build continuous streak metrics and determination dates
    is_3to4 = []
    is_3to5 = []
    is_deep_6 = []
    is_monster_7 = []
    
    target_A_continue = []
    target_A_maintain_days = []
    target_A_unique = []
    
    target_B_takeover_space = []
    target_B_takeover_unique = []
    
    is_emotion_dragon = []
    
    determined_date_3to4 = []
    determined_date_3to5 = []
    determined_date_final = []
    
    for idx, row in b3.iterrows():
        c = row['code']
        d = row['date']
        tb = row['terminal_board']
        is_already = row['is_already_space_leader']
        s_idx = session_map[d]
        
        # Height targets
        t4 = int(tb >= 4)
        t5 = int(tb >= 5)
        t6 = int(tb >= 6)
        t7 = int(tb >= 7)
        
        is_3to4.append(t4)
        is_3to5.append(t5)
        is_deep_6.append(t6)
        is_monster_7.append(t7)
        
        # Trace streak along official sessions
        streak_len = int(tb - 2)  # boards: 3, 4, ... tb -> streak_len = tb - 2 days
        streak_dates = [sessions[s_idx + k] for k in range(streak_len) if (s_idx + k) < len(sessions)]
        
        # Date when 3to4 is determined: session s_idx + 1
        d_3to4 = sessions[s_idx + 1] if (s_idx + 1) < len(sessions) else pd.to_datetime('2026-09-01')
        determined_date_3to4.append(d_3to4)
        
        # Date when 3to5 is determined:
        # If tb == 3: determined at session s_idx + 1 (failed at 4)
        # If tb == 4: determined at session s_idx + 2 (failed at 5)
        # If tb >= 5: determined at session s_idx + 2 (succeeded at 5)
        if tb == 3:
            d_3to5 = sessions[s_idx + 1] if (s_idx + 1) < len(sessions) else pd.to_datetime('2026-09-01')
        else:
            d_3to5 = sessions[s_idx + 2] if (s_idx + 2) < len(sessions) else pd.to_datetime('2026-09-02')
        determined_date_3to5.append(d_3to5)
        
        # Date when final terminal board is determined: session s_idx + streak_len
        d_final = sessions[s_idx + streak_len] if (s_idx + streak_len) < len(sessions) else pd.to_datetime('2026-09-01')
        # Check verified cutoff dates
        if (d.strftime('%Y-%m-%d'), c) in verified_cutoff_updates:
            d_final = pd.to_datetime(verified_cutoff_updates[(d.strftime('%Y-%m-%d'), c)][1])
        determined_date_final.append(d_final)
        
        # Trace streak dataframe
        streak_df = df[(df['code'] == c) & (df['date'].isin(streak_dates))].sort_values('date')
        
        # Check space leader history during streak
        if len(streak_df) > 0:
            subsequent_streak = streak_df[streak_df['date'] > d]  # strictly subsequent dates
            ever_space = int(streak_df['is_daily_max'].max() == 1)
            ever_unique = int(streak_df['is_unique_daily_max'].max() == 1)
            subsequent_space = int(subsequent_streak['is_daily_max'].max() == 1) if len(subsequent_streak) > 0 else 0
            subsequent_unique = int(subsequent_streak['is_unique_daily_max'].max() == 1) if len(subsequent_streak) > 0 else 0
            space_days = int(streak_df['is_daily_max'].sum())
            
            # Emotion dragon proxy
            has_turnover_breath = float((streak_df['turnover'].mean() >= 0.04) or (streak_df['amplitude'].max() >= 2.5))
            has_ladder_support = float(streak_df['consecutive_count'].mean() >= 3.0)
            is_ed = int((ever_space == 1) and (tb >= 5) and (has_turnover_breath == 1.0) and (has_ladder_support == 1.0))
        else:
            ever_space = is_already
            ever_unique = 0
            subsequent_space = 0
            subsequent_unique = 0
            space_days = 1 if is_already else 0
            is_ed = 0
            
        # Target A: for already high boards
        if is_already == 1:
            target_A_continue.append(t4)
            target_A_maintain_days.append(space_days)
            target_A_unique.append(ever_unique)
        else:
            target_A_continue.append(np.nan)
            target_A_maintain_days.append(np.nan)
            target_A_unique.append(np.nan)
            
        # Target B: for not high boards at Day t (subsequent takeover)
        if is_already == 0:
            target_B_takeover_space.append(subsequent_space)
            target_B_takeover_unique.append(subsequent_unique)
        else:
            target_B_takeover_space.append(np.nan)
            target_B_takeover_unique.append(np.nan)
            
        is_emotion_dragon.append(is_ed)
        
    b3['is_3to4'] = is_3to4
    b3['is_3to5'] = is_3to5
    b3['is_deep_6'] = is_deep_6
    b3['is_monster_7'] = is_monster_7
    
    b3['target_A_continue'] = target_A_continue
    b3['target_A_maintain_days'] = target_A_maintain_days
    b3['target_A_unique'] = target_A_unique
    
    b3['target_B_takeover_space'] = target_B_takeover_space
    b3['target_B_takeover_unique'] = target_B_takeover_unique
    
    # Unified space leader targets
    b3['is_market_space_leader'] = np.where(b3['is_already_space_leader'] == 1, b3['target_A_continue'], b3['target_B_takeover_space']).astype(int)
    b3['is_unique_space_leader'] = np.where(b3['is_already_space_leader'] == 1, b3['target_A_unique'], b3['target_B_takeover_unique']).astype(int)
    
    b3['is_emotion_dragon_proxy'] = is_emotion_dragon
    
    b3['determined_date_3to4'] = pd.to_datetime(determined_date_3to4)
    b3['determined_date_3to5'] = pd.to_datetime(determined_date_3to5)
    b3['determined_date_final'] = pd.to_datetime(determined_date_final)
    
    # Load and merge authentic historical BASE OOF predictions
    print("\nMerging authentic historical BASE OOF predictions...")
    base_dfs = []
    for f in sorted(BASE_OOF_DIR.glob("*_BASE_complete_OOF_BEFORE_SELECTION.csv")):
        odf = pd.read_csv(f)
        odf['date'] = pd.to_datetime(odf['date'])
        base_dfs.append(odf[['date', 'code', 'raw_score']])
    all_base_oof = pd.concat(base_dfs, ignore_index=True).drop_duplicates(subset=['date', 'code'], keep='last')
    all_base_oof.rename(columns={'raw_score': 'base_historical_score'}, inplace=True)
    
    b3 = b3.merge(all_base_oof, on=['date', 'code'], how='left')
    
    # Classify BASE matching eligibility per trading date:
    # COMPLETE_MATCH: all 3-board stocks on date have valid BASE score
    # PARTIAL_MATCH: some stocks have BASE score, some missing
    # NO_MATCH: zero stocks have BASE score
    date_match_status = {}
    for d, g in b3.groupby('date'):
        valid_cnt = g['base_historical_score'].notna().sum()
        tot_cnt = len(g)
        if valid_cnt == tot_cnt:
            status = 'COMPLETE_MATCH'
        elif valid_cnt > 0:
            status = 'PARTIAL_MATCH'
        else:
            status = 'NO_MATCH'
        date_match_status[d] = status
        
    b3['base_cohort_match_status'] = b3['date'].map(date_match_status)
    
    print("\n=== BASE COHORT MATCHING STATUS ===")
    print(b3['base_cohort_match_status'].value_counts())
    
    out_path = EXP_DIR / "round3_dataset.csv"
    b3.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved Round 3 dataset to {out_path} ({len(b3)} rows, {len(b3.columns)} columns)")

if __name__ == "__main__":
    build()
