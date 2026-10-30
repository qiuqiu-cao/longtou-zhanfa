"""build_round6_dataset.py

Constructs the Round 6 Canonical Dataset:
Conquers both:
- 突破口一：次日 09:25 集合竞价微观盘口确认 (Auction Execution Confirmation)
- 突破口二：10 日累计 100 异动大二波总龙模型 (10-Day 100% Abnormal Price Movement & Second Wave Leader)

Features & Target Additions:
1. is_10d_100_abnormal: Reached >= 100% gain from bottom over next 10 trading sessions (or tb >= 7.0).
2. is_second_wave_leader: Broke early at board 3 or 4, but surged >= 18% in next 10 sessions to make new highs.
3. forward_10d_max_return: Maximum forward return over next 10 trading sessions.
4. next_open_gap: Actual 09:25 auction open return.
5. is_auction_positive: 09:25 open > 0.0% (Auction Pass gate).
6. is_auction_sweet_spot: 09:25 open in +1.5% ~ +6.5% comfortable turnover zone.
7. is_auction_nuclear_low: 09:25 open <= 0.0% (Nuclear Ebb cancel trigger).
8. is_auction_one_word: 09:25 open >= 9.0% (One-word limit up ceiling).

Strict point-in-time calculation; zero lookahead leakage into Day t features.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent
EXP_DIR = ROOT / "research/model_upgrade_validation/dragon_auction_and_second_wave_20261027T091600Z"
R4_DATA_PATH = ROOT / "research/model_upgrade_validation/dragon_theme_matrix_20261027T083500Z/round4_dataset.csv"
DAILY_BARS_PATH = ROOT / "data/v3/imported/daily-bars.csv"

def build():
    print(f"Loading Round 4 dataset from {R4_DATA_PATH}...")
    r4 = pd.read_csv(R4_DATA_PATH)
    r4['date_str'] = pd.to_datetime(r4['date']).dt.strftime('%Y-%m-%d')
    r4['code_str'] = r4['code'].astype(str).str.zfill(6)
    
    print(f"Loading daily price bars from {DAILY_BARS_PATH}...")
    df_b = pd.read_csv(DAILY_BARS_PATH, usecols=['date', 'code', 'close', 'high', 'low'])
    df_b['date_str'] = pd.to_datetime(df_b['date']).dt.strftime('%Y-%m-%d')
    df_b['code_str'] = df_b['code'].astype(str).str.zfill(6)
    df_b = df_b.sort_values(['code_str', 'date_str']).reset_index(drop=True)
    
    print("Indexing bars by stock code...")
    bars_by_code = {c: g[['date_str', 'close', 'high', 'low']].values for c, g in df_b.groupby('code_str')}
    
    forward_10d_max = []
    forward_10d_close = []
    is_10d_100_abnormal = []
    is_second_wave = []
    
    print("Computing 10-day forward paths and second wave targets for 1,851 events...")
    for idx, r in r4.iterrows():
        c = r['code_str']
        d = r['date_str']
        p = r['price']
        tb = r.get('terminal_board', 3.0)
        
        bars = bars_by_code.get(c)
        if bars is None or len(bars) == 0 or pd.isna(p) or p <= 0:
            forward_10d_max.append(np.nan)
            forward_10d_close.append(np.nan)
            is_10d_100_abnormal.append(int(tb >= 7.0))
            is_second_wave.append(0)
            continue
            
        dates = bars[:, 0]
        sub = bars[dates > d][:10]
        if len(sub) == 0:
            forward_10d_max.append(np.nan)
            forward_10d_close.append(np.nan)
            is_10d_100_abnormal.append(int(tb >= 7.0))
            is_second_wave.append(0)
            continue
            
        max_h = sub[:, 2].max()
        last_c = sub[-1, 1]
        
        ret_max = (max_h - p) / p
        ret_close = (last_c - p) / p
        
        forward_10d_max.append(ret_max)
        forward_10d_close.append(ret_close)
        
        # 100% abnormal trigger from bottom: additional >= 46.4% gain from 3-board price OR tb >= 7.0
        is_100 = int(ret_max >= 0.464 or tb >= 7.0)
        is_10d_100_abnormal.append(is_100)
        
        # Second wave dragon: broke early at board 3 or 4 (tb <= 4.0), but in next 10 days made new high >= 18% above 3b price
        is_2w = int(tb <= 4.0 and ret_max >= 0.18)
        is_second_wave.append(is_2w)
        
    r4['forward_10d_max_return'] = forward_10d_max
    r4['forward_10d_close_return'] = forward_10d_close
    r4['is_10d_100_abnormal'] = is_10d_100_abnormal
    r4['is_second_wave_leader'] = is_second_wave
    
    # 09:25 Auction Execution Confirmation Columns
    gap = r4['next_open_gap'].fillna(0.0)
    r4['is_auction_positive'] = (gap > 0.0).astype(float)
    r4['is_auction_sweet_spot'] = ((gap >= 0.015) & (gap <= 0.065)).astype(float)
    r4['is_auction_nuclear_low'] = (gap <= 0.0).astype(float)
    r4['is_auction_one_word'] = (gap >= 0.090).astype(float)
    
    # Clean temporary columns
    if 'date_str' in r4:
        r4 = r4.drop(columns=['date_str'])
    if 'code_str' in r4:
        r4 = r4.drop(columns=['code_str'])
        
    out_path = EXP_DIR / "round6_dataset.csv"
    r4.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"Round 6 dataset successfully built with shape: {r4.shape}")
    print(f"Saved to: {out_path}")
    
    print("\nSummary of Round 6 Targets:")
    print("Total 3-board events:", len(r4))
    print("is_10d_100_abnormal count:", int(r4['is_10d_100_abnormal'].sum()), f"({r4['is_10d_100_abnormal'].mean():.2%})")
    print("is_second_wave_leader count:", int(r4['is_second_wave_leader'].sum()), f"({r4['is_second_wave_leader'].mean():.2%})")
    print("Mean 10-day forward max gain:", f"{r4['forward_10d_max_return'].mean():.2%}")
    print("Auction High Open (>0%) count:", int(r4['is_auction_positive'].sum()), f"({r4['is_auction_positive'].mean():.2%})")
    print("Auction Nuclear Low (<=0%) count:", int(r4['is_auction_nuclear_low'].sum()), f"({r4['is_auction_nuclear_low'].mean():.2%})")

if __name__ == "__main__":
    build()
