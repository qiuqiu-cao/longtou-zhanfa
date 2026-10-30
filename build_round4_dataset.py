"""build_round4_dataset.py

Constructs the canonical Round 4 dataset incorporating Direction 1:
"题材低位首板成群扩散助攻矩阵" (Theme Low-Board / First-Board Clustered Diffusion Assist Matrix)

Core Innovations & Features:
1. theme_first_board_count: Daily count of 1-board peers in the same theme/industry (diffusion breadth).
2. theme_two_board_count: Daily count of 2-board peers in the same theme/industry (waist support).
3. theme_ladder_density: Proportion of ladder rungs (1, 2, 3 boards) populated in the theme.
4. theme_ladder_complete: Binary indicator for full 1-2-3 board rung structure.
5. theme_zt_share: Theme limit-up count / Total market limit-up count.
6. theme_first_board_share: Theme 1-board count / Total market 1-board count.
7. theme_amount_share: Total turnover in theme limit-ups / Total market limit-up turnover.
8. theme_first_board_expansion: Daily change in 1-board peers (t vs t-1).
9. theme_cluster_expansion: Daily change in total limit-up peers (t vs t-1).
10. is_theme_sole_leader: Stock is the unique highest board in its theme.
11. theme_same_height_rivals: Number of other 3-board stocks in the same theme.
12. is_isolated_lone_wolf: Stock has zero 1-board and zero 2-board peers in its theme.
13. theme_diffusion_surge: High diffusion eruption (>=3 first-boards and positive expansion).

Strict point-in-time calculation as of Day t close; zero future data leakage.
"""

import sys
from pathlib import Path
import glob
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent
EXP_DIR = ROOT / "research/model_upgrade_validation/dragon_theme_matrix_20261027T083500Z"
R3_DATA_PATH = ROOT / "research/model_upgrade_validation/dragon_takeover_20261027T074500Z/round3_dataset.csv"
ATTEMPTS_PATH = ROOT / "results/v215/historical_board_attempts_20260920_v2/historical_closing_and_failed_board_attempts.csv"
CANDIDATES_PATH = ROOT / "data/v16/point_in_time_rebuild_cutoff_20260831_v2/leader_candidates.csv"

def load_industry_mappings():
    print("Loading stock-to-industry classification...")
    # 1. Primary vintage source (4,179 stocks)
    basics_path = ROOT / "research/first_board_github_industry_qualification_20261020/stock_basics_industry_original.csv"
    df_stocks = pd.read_csv(basics_path, encoding='utf-8')
    df_stocks['code'] = df_stocks['symbol'].astype(str).str.zfill(6)
    ind_map = dict(zip(df_stocks['code'], df_stocks['industry']))
    
    # 2. Augment with August 2026 pools for newly listed stocks
    aug_count = 0
    for f in glob.glob(str(ROOT / 'data/2026-08/*.csv')):
        try:
            df = pd.read_csv(f, encoding='utf-8')
            if '代码' in df and '所属行业' in df:
                for _, r in df.iterrows():
                    code = str(r['代码']).zfill(6)
                    ind = str(r['所属行业']).strip()
                    if ind and ind != 'nan' and code not in ind_map:
                        ind_map[code] = ind
                        aug_count += 1
        except Exception:
            pass
    print(f"Total mapped stocks: {len(ind_map)} (including {aug_count} post-2021 additions)")
    return ind_map

def build():
    print(f"Loading Round 3 canonical dataset from {R3_DATA_PATH}...")
    df_r3 = pd.read_csv(R3_DATA_PATH)
    print(f"Round 3 rows: {len(df_r3)}")
    
    ind_map = load_industry_mappings()
    
    # Load complete closing limit-up pool across 681 trading days
    print(f"Loading closing limit-up universe from {ATTEMPTS_PATH}...")
    df_att = pd.read_csv(ATTEMPTS_PATH)
    df_att['code_str'] = df_att['code'].astype(str).str.zfill(6)
    df_close = df_att[df_att['pool_assignment'] == 'CLOSING_LIMIT_UP'].copy()
    df_close['industry'] = df_close['code_str'].map(ind_map).fillna("UNKNOWN_INDUSTRY")
    
    # Load candidate amounts to compute theme_amount_share
    print("Loading candidate trading amounts...")
    df_c = pd.read_csv(CANDIDATES_PATH, usecols=['date', 'code', 'signal_amount'])
    df_c['code_str'] = df_c['code'].astype(str).str.zfill(6)
    c_amount_map = dict(zip(zip(df_c['date'], df_c['code_str']), df_c['signal_amount']))
    df_close['signal_amount'] = [c_amount_map.get((d, c), np.nan) for d, c in zip(df_close['signal_date'], df_close['code_str'])]
    
    # Market-wide daily totals
    market_daily = df_close.groupby('signal_date').agg(
        market_total_zt=('code_str', 'count'),
        market_total_first=('board_height', lambda s: (s == 1).sum()),
        market_total_amount=('signal_amount', 'sum')
    ).reset_index()
    
    # Theme/Industry daily aggregations
    print("Computing daily theme diffusion statistics...")
    ind_daily = df_close.groupby(['signal_date', 'industry']).agg(
        theme_cluster_total_zt=('code_str', 'count'),
        theme_first_board_count=('board_height', lambda s: (s == 1).sum()),
        theme_two_board_count=('board_height', lambda s: (s == 2).sum()),
        theme_three_board_count=('board_height', lambda s: (s == 3).sum()),
        theme_max_board=('board_height', 'max'),
        theme_total_amount=('signal_amount', 'sum')
    ).reset_index()
    
    # Merge market daily totals
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
    
    # Sort by industry and date to compute expansion deltas (strictly historical)
    ind_daily = ind_daily.sort_values(['industry', 'signal_date'])
    ind_daily['theme_first_prev'] = ind_daily.groupby('industry')['theme_first_board_count'].shift(1)
    ind_daily['theme_first_board_expansion'] = ind_daily['theme_first_board_count'] - ind_daily['theme_first_prev'].fillna(0)
    ind_daily['theme_zt_prev'] = ind_daily.groupby('industry')['theme_cluster_total_zt'].shift(1)
    ind_daily['theme_cluster_expansion'] = ind_daily['theme_cluster_total_zt'] - ind_daily['theme_zt_prev'].fillna(0)
    
    # Attach industry to df_r3
    df_r3['code_str'] = df_r3['code'].astype(str).str.zfill(6)
    df_r3['industry'] = df_r3['code_str'].map(ind_map).fillna("UNKNOWN_INDUSTRY")
    
    # Merge theme features onto 3-board candidate dataset
    print("Merging theme diffusion matrix onto 3-board events...")
    df_r4 = df_r3.merge(
        ind_daily,
        left_on=['date', 'industry'],
        right_on=['signal_date', 'industry'],
        how='left'
    )
    
    # Fill defaults for unknown industries / missing matches
    df_r4['theme_cluster_total_zt'] = df_r4['theme_cluster_total_zt'].fillna(1.0)
    df_r4['theme_first_board_count'] = df_r4['theme_first_board_count'].fillna(0.0)
    df_r4['theme_two_board_count'] = df_r4['theme_two_board_count'].fillna(0.0)
    df_r4['theme_three_board_count'] = df_r4['theme_three_board_count'].fillna(1.0)
    df_r4['theme_max_board'] = df_r4['theme_max_board'].fillna(3.0)
    df_r4['theme_zt_share'] = df_r4['theme_zt_share'].fillna(0.0)
    df_r4['theme_first_board_share'] = df_r4['theme_first_board_share'].fillna(0.0)
    df_r4['theme_amount_share'] = df_r4['theme_amount_share'].fillna(0.0)
    df_r4['theme_ladder_density'] = df_r4['theme_ladder_density'].fillna(0.333)
    df_r4['theme_ladder_complete'] = df_r4['theme_ladder_complete'].fillna(0.0)
    df_r4['theme_first_board_expansion'] = df_r4['theme_first_board_expansion'].fillna(0.0)
    df_r4['theme_cluster_expansion'] = df_r4['theme_cluster_expansion'].fillna(0.0)
    
    # Individual relative features within theme
    df_r4['is_theme_sole_leader'] = ((df_r4['theme_max_board'] == 3.0) & (df_r4['theme_three_board_count'] == 1.0)).astype(float)
    df_r4['theme_same_height_rivals'] = (df_r4['theme_three_board_count'] - 1.0).clip(lower=0.0)
    df_r4['is_isolated_lone_wolf'] = ((df_r4['theme_first_board_count'] == 0.0) & (df_r4['theme_two_board_count'] == 0.0)).astype(float)
    df_r4['theme_diffusion_surge'] = ((df_r4['theme_first_board_count'] >= 3.0) & (df_r4['theme_first_board_expansion'] >= 1.0)).astype(float)
    
    # Clean up auxiliary columns
    if 'signal_date' in df_r4:
        df_r4 = df_r4.drop(columns=['signal_date'])
    if 'code_str' in df_r4:
        df_r4 = df_r4.drop(columns=['code_str'])
        
    out_path = EXP_DIR / "round4_dataset.csv"
    df_r4.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"Round 4 dataset successfully built with shape: {df_r4.shape}")
    print(f"Saved to: {out_path}")
    
    # Quick sanity check
    print("\nSanity Check on Key Theme Metrics:")
    print("Mean theme_first_board_count:", df_r4['theme_first_board_count'].mean())
    print("Isolated Lone Wolf fraction:", df_r4['is_isolated_lone_wolf'].mean())
    print("Theme Sole Leader fraction:", df_r4['is_theme_sole_leader'].mean())
    print("Theme Diffusion Surge fraction:", df_r4['theme_diffusion_surge'].mean())
    print("Ladder Complete fraction:", df_r4['theme_ladder_complete'].mean())

if __name__ == "__main__":
    build()
