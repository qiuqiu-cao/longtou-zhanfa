"""build_round5_regime_dataset.py

Constructs the daily market regime dataset for Direction 2:
"大盘退潮期一票禁买闸门" (Market Ebb Regime Gate & Slump Circuit Breaker)

Scope:
- Covers all 599 unique trading sessions with 3-board candidates (2023-11-28 to 2026-08-31).
- Extracts market emotion, height gap, and cohort structure features as of Day t close.
- Defines ground-truth daily regime labels:
  - is_dragon_day: Cohort produced >= 1 Space Leader or >= 1 5-board continuous breakthrough.
  - is_zero_dragon_day: Entire 3-board cohort died (0 Space Leaders AND 0 5-boards).
Strict point-in-time calculation; zero lookahead leakage.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent
EXP_DIR = ROOT / "research/model_upgrade_validation/dragon_regime_gate_20261027T085500Z"
R4_DATA_PATH = ROOT / "research/model_upgrade_validation/dragon_theme_matrix_20261027T083500Z/round4_dataset.csv"

def build():
    print(f"Loading Round 4 candidate data from {R4_DATA_PATH}...")
    r4 = pd.read_csv(R4_DATA_PATH)
    r4['date'] = pd.to_datetime(r4['date']).dt.strftime('%Y-%m-%d')
    
    market_cols = [
        'date', 'emotion_score', 'down_limit_rate', 'failed_rate', 'space_feedback',
        'space_feedback_3d', 'ladder_gap_count', 'breadth_height_divergence',
        'yesterday_premium', 'failed_tail_loss', 'previous_limit_tail_loss',
        'breadth_health', 'ladder_health', 'loss_effect_health', 'crowding_risk',
        'regime_fragility', 'emotion_adverse_next_probability', 'market_max_board',
        'market_total_zt', 'market_total_first', 'market_total_amount'
    ]
    mkt = r4[market_cols].drop_duplicates('date').copy()
    
    # Daily cohort target aggregations
    print("Aggregating daily 3-board cohort metrics...")
    daily_target = r4.groupby('date').agg(
        cohort_space_sum=('is_market_space_leader', 'sum'),
        cohort_5_sum=('is_3to5', 'sum'),
        cohort_dragon_sum=('is_emotion_dragon_proxy', 'sum'),
        cohort_size=('code', 'count'),
        has_lone_wolf=('is_isolated_lone_wolf', lambda s: (s == 1).any()),
        avg_diffusion=('theme_first_board_count', 'mean'),
        max_diffusion=('theme_first_board_count', 'max')
    ).reset_index()
    
    daily_df = mkt.merge(daily_target, on='date')
    daily_df['height_gap_to_max'] = daily_df['market_max_board'] - 3
    
    # Ground truth regime labels
    daily_df['is_dragon_day'] = ((daily_df['cohort_space_sum'] > 0) | (daily_df['cohort_5_sum'] > 0)).astype(int)
    daily_df['is_zero_dragon_day'] = 1 - daily_df['is_dragon_day']
    
    # Derived composite market risk indicators
    daily_df['high_board_monopoly'] = (daily_df['market_max_board'] >= 7).astype(float)
    daily_df['divergence_risk_flag'] = ((daily_df['failed_rate'] >= 0.38) | (daily_df['yesterday_premium'] < -0.015)).astype(float)
    daily_df['adverse_transition_flag'] = (daily_df['emotion_adverse_next_probability'] >= 0.60).astype(float)
    
    out_path = EXP_DIR / "round5_daily_regime_dataset.csv"
    daily_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"Daily regime dataset successfully built: {daily_df.shape} rows/cols")
    print(f"Saved to: {out_path}")
    
    print("\nDataset Summary:")
    print("Total trading days:", len(daily_df))
    print("Dragon Days (>=1 leader):", int(daily_df['is_dragon_day'].sum()), f"({daily_df['is_dragon_day'].mean():.2%})")
    print("Zero-Dragon Slump Days (0 leaders):", int(daily_df['is_zero_dragon_day'].sum()), f"({daily_df['is_zero_dragon_day'].mean():.2%})")

if __name__ == "__main__":
    build()
