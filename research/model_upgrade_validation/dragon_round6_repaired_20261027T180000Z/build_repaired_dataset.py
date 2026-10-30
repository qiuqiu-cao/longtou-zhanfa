"""
build_repaired_dataset.py:

Constructs the canonical repaired Round 6 dataset with 100% strict point-in-time integrity:
1. Fixes UNKNOWN_INDUSTRY: Every unknown stock receives a unique isolated dummy identifier (e.g. UNKNOWN_000004),
   guaranteeing they never artificially cluster together or provide fake first-board assist counts.
2. Fixes industry diffusion shift: Reindexes across all official trading sessions from exchange calendar,
   guaranteeing theme_first_board_expansion is strictly relative to the immediately preceding official trading session (t vs t-1).
3. Constructs rigorous point-in-time determined dates for all targets:
   - target_space_leader: determined_date_final
   - target_3to5: date reached 5 boards or determined_date_final
   - target_path_gain_46pct: date high >= 1.464 * p reached; if unreached, matured on 10th official session (or NaN if incomplete)
   - target_rebound_18pct: date high >= 1.18 * p reached (provided tb <= 4); if unreached, matured on 10th session (or NaN if incomplete)
4. Accurately labels incomplete/truncated forward paths as UNKNOWN (NaN), never defaulting to negative/failure.
"""

import sys
from pathlib import Path
import json
from bisect import bisect_right
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(".")
REPAIR_DIR = ROOT / "research/model_upgrade_validation/dragon_round6_repaired_20261027T180000Z"
R3_DATA_PATH = ROOT / "research/model_upgrade_validation/dragon_takeover_20261027T074500Z/round3_dataset.csv"
ATTEMPTS_PATH = ROOT / "results/v215/historical_board_attempts_20260920_v2/historical_closing_and_failed_board_attempts.csv"
CANDIDATES_PATH = ROOT / "data/v16/point_in_time_rebuild_cutoff_20260831_v2/leader_candidates.csv"
CALENDAR_PATH = ROOT / "data/v279/exchange_calendar_audit_20261002_v1/calendar.json"
DAILY_BARS_PATH = ROOT / "data/v3/imported/daily-bars.csv"

def build():
    print("Loading exchange calendar...")
    sessions = json.loads(CALENDAR_PATH.read_text(encoding='utf-8'))['sessions']
    all_sessions_set = set(sessions)
    
    print("Loading stock-to-industry classifications...")
    basics_path = ROOT / "research/first_board_github_industry_qualification_20261020/stock_basics_industry_original.csv"
    df_stocks = pd.read_csv(basics_path, encoding='utf-8')
    df_stocks['code'] = df_stocks['symbol'].astype(str).str.zfill(6)
    ind_map = dict(zip(df_stocks['code'], df_stocks['industry']))
    
    # Load closing limit-up pool across all trading sessions
    print(f"Loading closing limit-up universe from {ATTEMPTS_PATH}...")
    df_att = pd.read_csv(ATTEMPTS_PATH)
    df_att['code_str'] = df_att['code'].astype(str).str.zfill(6)
    df_close = df_att[df_att['pool_assignment'] == 'CLOSING_LIMIT_UP'].copy()
    
    # REPAIR 1: Fix UNKNOWN_INDUSTRY clustering
    # Each unknown stock gets a unique isolated pseudo-industry code so it NEVER pools with others
    mapped_ind = df_close['code_str'].map(ind_map)
    df_close['is_industry_unknown'] = mapped_ind.isna().astype(float)
    df_close['industry'] = np.where(mapped_ind.isna(), "UNKNOWN_" + df_close['code_str'], mapped_ind)
    
    # Load candidate trading amounts
    df_c = pd.read_csv(CANDIDATES_PATH, usecols=['date', 'code', 'signal_amount'])
    df_c['code_str'] = df_c['code'].astype(str).str.zfill(6)
    c_amount_map = dict(zip(zip(df_c['date'], df_c['code_str']), df_c['signal_amount']))
    df_close['signal_amount'] = [c_amount_map.get((d, c), np.nan) for d, c in zip(df_close['signal_date'], df_close['code_str'])]
    
    # Market daily totals
    market_daily = df_close.groupby('signal_date').agg(
        market_total_zt=('code_str', 'count'),
        market_total_first=('board_height', lambda s: (s == 1).sum()),
        market_total_amount=('signal_amount', 'sum')
    ).reset_index()
    
    # Industry daily aggregations
    ind_daily = df_close.groupby(['signal_date', 'industry']).agg(
        theme_cluster_total_zt=('code_str', 'count'),
        theme_first_board_count=('board_height', lambda s: (s == 1).sum()),
        theme_two_board_count=('board_height', lambda s: (s == 2).sum()),
        theme_three_board_count=('board_height', lambda s: (s == 3).sum()),
        theme_max_board=('board_height', 'max'),
        theme_total_amount=('signal_amount', 'sum')
    ).reset_index()
    
    ind_daily = ind_daily.merge(market_daily, on='signal_date', how='left')
    ind_daily['theme_zt_share'] = ind_daily['theme_cluster_total_zt'] / ind_daily['market_total_zt'].replace(0, np.nan)
    ind_daily['theme_first_board_share'] = ind_daily['theme_first_board_count'] / ind_daily['market_total_first'].replace(0, np.nan)
    ind_daily['theme_amount_share'] = ind_daily['theme_total_amount'] / ind_daily['market_total_amount'].replace(0, np.nan)
    ind_daily['theme_ladder_density'] = (
        (ind_daily['theme_first_board_count'] > 0).astype(float) +
        (ind_daily['theme_two_board_count'] > 0).astype(float) +
        (ind_daily['theme_three_board_count'] > 0).astype(float)
    ) / 3.0
    ind_daily['theme_ladder_complete'] = (
        (ind_daily['theme_first_board_count'] > 0) & (ind_daily['theme_two_board_count'] > 0)
    ).astype(float)
    
    # REPAIR 2: Fix industry diffusion shift across official trading calendar
    print("Reindexing industry diffusion grid across official exchange sessions...")
    unique_inds = ind_daily['industry'].unique()
    valid_sessions = sorted([s for s in sessions if s >= ind_daily['signal_date'].min() and s <= ind_daily['signal_date'].max()])
    
    grid_idx = pd.MultiIndex.from_product([valid_sessions, unique_inds], names=['signal_date', 'industry'])
    full_grid = ind_daily.set_index(['signal_date', 'industry']).reindex(grid_idx).fillna({
        'theme_cluster_total_zt': 0.0,
        'theme_first_board_count': 0.0,
        'theme_two_board_count': 0.0,
        'theme_three_board_count': 0.0,
        'theme_max_board': 0.0,
        'theme_total_amount': 0.0,
        'theme_zt_share': 0.0,
        'theme_first_board_share': 0.0,
        'theme_amount_share': 0.0,
        'theme_ladder_density': 0.0,
        'theme_ladder_complete': 0.0
    }).reset_index()
    
    full_grid = full_grid.sort_values(['industry', 'signal_date']).reset_index(drop=True)
    full_grid['theme_first_prev'] = full_grid.groupby('industry')['theme_first_board_count'].shift(1)
    full_grid['theme_first_board_expansion'] = full_grid['theme_first_board_count'] - full_grid['theme_first_prev'].fillna(0.0)
    full_grid['theme_zt_prev'] = full_grid.groupby('industry')['theme_cluster_total_zt'].shift(1)
    full_grid['theme_cluster_expansion'] = full_grid['theme_cluster_total_zt'] - full_grid['theme_zt_prev'].fillna(0.0)
    
    # Load Round 3 dataset
    print(f"Loading Round 3 canonical dataset from {R3_DATA_PATH}...")
    df_r3 = pd.read_csv(R3_DATA_PATH)
    df_r3['code_str'] = df_r3['code'].astype(str).str.zfill(6)
    r3_mapped_ind = df_r3['code_str'].map(ind_map)
    df_r3['is_industry_unknown'] = r3_mapped_ind.isna().astype(float)
    df_r3['industry'] = np.where(r3_mapped_ind.isna(), "UNKNOWN_" + df_r3['code_str'], r3_mapped_ind)
    
    # Merge repaired theme features onto 3-board events
    df_merged = df_r3.merge(
        full_grid,
        left_on=['date', 'industry'],
        right_on=['signal_date', 'industry'],
        how='left'
    )
    
    # Fill defaults for unknown industries / missing matches
    df_merged['theme_cluster_total_zt'] = df_merged['theme_cluster_total_zt'].fillna(1.0)
    df_merged['theme_first_board_count'] = df_merged['theme_first_board_count'].fillna(0.0)
    df_merged['theme_two_board_count'] = df_merged['theme_two_board_count'].fillna(0.0)
    df_merged['theme_three_board_count'] = df_merged['theme_three_board_count'].fillna(1.0)
    df_merged['theme_max_board'] = df_merged['theme_max_board'].fillna(3.0)
    df_merged['theme_zt_share'] = df_merged['theme_zt_share'].fillna(0.0)
    df_merged['theme_first_board_share'] = df_merged['theme_first_board_share'].fillna(0.0)
    df_merged['theme_amount_share'] = df_merged['theme_amount_share'].fillna(0.0)
    df_merged['theme_ladder_density'] = df_merged['theme_ladder_density'].fillna(0.333)
    df_merged['theme_ladder_complete'] = df_merged['theme_ladder_complete'].fillna(0.0)
    df_merged['theme_first_board_expansion'] = df_merged['theme_first_board_expansion'].fillna(0.0)
    df_merged['theme_cluster_expansion'] = df_merged['theme_cluster_expansion'].fillna(0.0)
    
    # Individual relative features within theme
    df_merged['is_theme_sole_leader'] = ((df_merged['theme_max_board'] == 3.0) & (df_merged['theme_three_board_count'] == 1.0)).astype(float)
    df_merged['theme_same_height_rivals'] = (df_merged['theme_three_board_count'] - 1.0).clip(lower=0.0)
    df_merged['is_isolated_lone_wolf'] = ((df_merged['theme_first_board_count'] == 0.0) & (df_merged['theme_two_board_count'] == 0.0)).astype(float)
    df_merged['theme_diffusion_surge'] = ((df_merged['theme_first_board_count'] >= 3.0) & (df_merged['theme_first_board_expansion'] >= 1.0)).astype(float)
    
    # REPAIR 3: Fix 10-session official forward window and exact maturity dates
    print("Loading daily bars for rigorous forward path computation...")
    bars = pd.read_csv(DAILY_BARS_PATH, dtype={'code': str}, usecols=['date', 'code', 'high', 'close'])
    bars['code_str'] = bars['code'].str.zfill(6)
    bars = bars.sort_values(['code_str', 'date']).reset_index(drop=True)
    bars_by_code = {c: g for c, g in bars.groupby('code_str')}
    
    fwd_10d_max_return = []
    fwd_10d_close_return = []
    official10_complete = []
    forward_bars_count = []
    
    target_path_gain_46pct = []
    det_date_gain46 = []
    
    target_rebound_18pct = []
    det_date_reb18 = []
    
    print("Processing official 10-session paths and maturity dates for all 1,851 events...")
    for idx, r in df_merged.iterrows():
        c = r['code_str']
        d = r['date']
        p = r['price']
        tb = r['terminal_board']
        det_final = r['determined_date_final']
        
        i = bisect_right(sessions, d)
        exp_sessions = sessions[i:i+10]
        
        g = bars_by_code.get(c)
        if g is None or len(g) == 0 or pd.isna(p) or p <= 0:
            fwd_10d_max_return.append(np.nan)
            fwd_10d_close_return.append(np.nan)
            official10_complete.append(False)
            forward_bars_count.append(0)
            
            # Gain 46 target
            if tb >= 7.0:
                target_path_gain_46pct.append(1)
                det_date_gain46.append(det_final)
            else:
                target_path_gain_46pct.append(np.nan)
                det_date_gain46.append(None)
                
            # Rebound 18 target
            if tb > 4.0:
                target_rebound_18pct.append(0)
                det_date_reb18.append(det_final)
            else:
                target_rebound_18pct.append(np.nan)
                det_date_reb18.append(None)
            continue
            
        sub = g[g['date'].gt(d)]
        sub_exp = sub[sub['date'].isin(exp_sessions)]
        obs_sessions = sub_exp['date'].tolist()
        
        is_comp = (len(exp_sessions) == 10 and obs_sessions == exp_sessions)
        official10_complete.append(is_comp)
        forward_bars_count.append(len(sub_exp))
        
        if len(sub_exp) > 0:
            max_h = sub_exp['high'].max()
            last_c = sub_exp['close'].iloc[-1]
            fwd_10d_max_return.append((max_h - p) / p)
            fwd_10d_close_return.append((last_c - p) / p)
        else:
            fwd_10d_max_return.append(np.nan)
            fwd_10d_close_return.append(np.nan)
            
        # Target A: Path Gain 46.4%
        hit_g46_date = None
        if tb >= 7.0:
            hit_g46_date = det_final
        else:
            for _, b_row in sub_exp.iterrows():
                if b_row['high'] >= p * 1.464:
                    hit_g46_date = b_row['date']
                    break
                    
        if hit_g46_date is not None:
            target_path_gain_46pct.append(1)
            det_date_gain46.append(hit_g46_date)
        else:
            if is_comp:
                target_path_gain_46pct.append(0)
                det_date_gain46.append(exp_sessions[-1])
            else:
                target_path_gain_46pct.append(np.nan)
                det_date_gain46.append(None)
                
        # Target B: Rebound 18%
        if tb > 4.0:
            target_rebound_18pct.append(0)
            det_date_reb18.append(det_final)
        else:
            hit_r18_date = None
            for _, b_row in sub_exp.iterrows():
                if b_row['high'] >= p * 1.18:
                    hit_r18_date = b_row['date']
                    break
            if hit_r18_date is not None:
                target_rebound_18pct.append(1)
                det_date_reb18.append(hit_r18_date)
            else:
                if is_comp:
                    target_rebound_18pct.append(0)
                    det_date_reb18.append(exp_sessions[-1])
                else:
                    target_rebound_18pct.append(np.nan)
                    det_date_reb18.append(None)
                    
    df_merged['forward_10d_max_return'] = fwd_10d_max_return
    df_merged['forward_10d_close_return'] = fwd_10d_close_return
    df_merged['official10_complete'] = official10_complete
    df_merged['forward_bars_count'] = forward_bars_count
    
    df_merged['target_path_gain_46pct'] = target_path_gain_46pct
    df_merged['det_date_gain46'] = det_date_gain46
    
    df_merged['target_rebound_18pct'] = target_rebound_18pct
    df_merged['det_date_reb18'] = det_date_reb18
    
    # Target 3to5 determined date: use rigorously audited determined_date_3to5 from round3_dataset
    # Fix edge negative case where det_date_3to5 > determined_date_final (failure occurs on terminal_date)
    det_3to5_fixed = np.where(
        (df_r3['is_3to5'] == 0) & (df_r3['determined_date_3to5'] > df_r3['determined_date_final']),
        df_r3['determined_date_final'],
        df_r3['determined_date_3to5']
    )
    df_merged['det_date_3to5'] = det_3to5_fixed
    
    # Save repaired canonical dataset
    out_csv = REPAIR_DIR / "repaired_round6_dataset.csv"
    df_merged.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"\nRepaired canonical dataset successfully created at: {out_csv}")
    print(f"Shape: {df_merged.shape}")
    print("\nSummary of Repaired Labels:")
    print("target_path_gain_46pct:", df_merged['target_path_gain_46pct'].value_counts(dropna=False).to_dict())
    print("target_rebound_18pct:", df_merged['target_rebound_18pct'].value_counts(dropna=False).to_dict())
    print("official10_complete count:", df_merged['official10_complete'].sum())

if __name__ == "__main__":
    build()
