"""
build_repaired_regime_dataset.py:

Builds the repaired daily regime dataset with exact point-in-time maturity tracking:
Each date d has an explicit det_date_regime = max(det_date_space, det_date_5b) across its 3-board cohort.
If det_date_regime > train_end, that date CANNOT be trained on at train_end.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(".")
REPAIR_DIR = ROOT / "research/model_upgrade_validation/dragon_round6_repaired_20261027T180000Z"
DATA_PATH = REPAIR_DIR / "repaired_round6_dataset.csv"

def build_regime():
    print("Building repaired regime gate dataset...")
    df = pd.read_csv(DATA_PATH)
    
    daily_records = []
    for d, g in df.groupby('date'):
        # Check if cohort produced space leader or 5-board
        is_dragon = int((g['is_market_space_leader'] == 1).any() or (g['is_3to5'] == 1).any())
        
        # Exact maturity date for this day's regime label:
        # It matures when the final outcome of the 3-board cohort is settled
        det_date = g['determined_date_final'].max()
        
        # Extract market emotion features as of Day t close (from first row)
        r0 = g.iloc[0]
        daily_records.append({
            'date': d,
            'cohort_size': len(g),
            'is_dragon_day': is_dragon,
            'determined_date_regime': det_date,
            'emotion_score': r0['emotion_score'],
            'down_limit_rate': r0['down_limit_rate'],
            'failed_rate': r0['failed_rate'],
            'seal_rate': r0['seal_rate'],
            'space_feedback': r0['space_feedback'],
            'ladder_gap_count': r0['ladder_gap_count'],
            'breadth_height_divergence': r0['breadth_height_divergence'],
            'yesterday_premium': r0['yesterday_premium'],
            'failed_tail_loss': r0['failed_tail_loss'],
            'previous_limit_tail_loss': r0['previous_limit_tail_loss'],
            'breadth_health': r0['breadth_health'],
            'ladder_health': r0['ladder_health'],
            'loss_effect_health': r0['loss_effect_health'],
            'crowding_risk': r0['crowding_risk'],
            'regime_fragility': r0['regime_fragility'],
            'total_limit_up': r0['total_limit_up'],
            'consecutive_count': r0['consecutive_count'],
            'market_max_board': r0['market_max_board'],
            'market_max_board_count': r0['market_max_board_count'],
            'height_gap': r0['height_gap'],
        })
        
    regime_df = pd.DataFrame(daily_records).sort_values('date').reset_index(drop=True)
    out_path = REPAIR_DIR / "repaired_regime_dataset.csv"
    regime_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"Saved repaired regime dataset to {out_path} with {len(regime_df)} sessions.")
    print("Dragon days:", regime_df['is_dragon_day'].sum(), f"({regime_df['is_dragon_day'].mean():.1%})")

if __name__ == "__main__":
    build_regime()
