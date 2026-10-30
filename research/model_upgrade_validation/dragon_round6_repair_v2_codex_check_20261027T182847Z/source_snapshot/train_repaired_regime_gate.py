"""
train_repaired_regime_gate.py:

Trains the repaired market regime gate strictly adhering to point-in-time maturity:
On each fold cutoff train_end, only trains on sessions where determined_date_regime <= train_end.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(".")
REPAIR_DIR = ROOT / "research/model_upgrade_validation/dragon_round6_repaired_20261027T180000Z"
DATA_PATH = REPAIR_DIR / "repaired_regime_dataset.csv"

FOLDS = [
    ("2024-08-31", "2024-09-01", "2024-12-31"),
    ("2024-12-31", "2025-01-01", "2025-05-31"),
    ("2025-05-31", "2025-06-01", "2025-11-30"),
    ("2025-11-30", "2025-12-01", "2026-08-31"),
]

def main():
    print("Training repaired regime gate with strict maturity...")
    df = pd.read_csv(DATA_PATH)
    
    feature_cols = [
        'emotion_score', 'down_limit_rate', 'failed_rate', 'seal_rate', 'space_feedback',
        'ladder_gap_count', 'breadth_height_divergence',
        'yesterday_premium', 'failed_tail_loss', 'previous_limit_tail_loss',
        'breadth_health', 'ladder_health', 'loss_effect_health', 'crowding_risk',
        'regime_fragility', 'total_limit_up', 'consecutive_count',
        'market_max_board', 'market_max_board_count', 'height_gap', 'cohort_size'
    ]
    
    all_oof = []
    models = {}
    
    for fold_idx, (t_end, s_start, s_end) in enumerate(FOLDS):
        # Strict maturity: determined_date_regime <= t_end
        train = df[(df['date'] <= t_end) & (df['determined_date_regime'] <= t_end)].copy()
        test = df[(df['date'] >= s_start) & (df['date'] <= s_end)].copy()
        
        print(f"Fold {fold_idx + 1} ({s_start} to {s_end}): Mature Train Sessions: {len(train)}, Test Sessions: {len(test)}")
        
        clf = HistGradientBoostingClassifier(
            max_iter=100, max_depth=3, min_samples_leaf=12, l2_regularization=3.0,
            learning_rate=0.03, random_state=42 + fold_idx
        )
        clf.fit(train[feature_cols], train['is_dragon_day'])
        test['prob_dragon_day'] = clf.predict_proba(test[feature_cols])[:, 1]
        test['gate_arm_b_open_th55'] = (test['prob_dragon_day'] >= 0.55).astype(int)
        test['fold'] = fold_idx + 1
        
        auc = roc_auc_score(test['is_dragon_day'], test['prob_dragon_day'])
        ap = average_precision_score(test['is_dragon_day'], test['prob_dragon_day'])
        print(f"  Test AUC: {auc:.4f}, AP: {ap:.4f}, Gate Open (th>=0.55): {test['gate_arm_b_open_th55'].sum()}/{len(test)}")
        
        all_oof.append(test)
        models[f"regime_gate_fold_{fold_idx+1}"] = clf
        
    oof_df = pd.concat(all_oof, ignore_index=True)
    joblib.dump(models, REPAIR_DIR / "repaired_regime_gate_models.joblib")
    
    out_csv = REPAIR_DIR / "oof_repaired_regime_gate.csv"
    oof_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"Saved repaired regime gate OOF predictions to {out_csv}")

if __name__ == "__main__":
    main()
