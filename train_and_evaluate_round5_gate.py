"""train_and_evaluate_round5_gate.py

Round 5 Walk-Forward Trainer & Evaluator:
Conquers Direction 2: "大盘退潮期一票禁买闸门" (Regime Gate & Ebb Circuit Breaker)

Two Comparative Arms:
- Arm A: Heuristic Rule-Based Hard Gate (Domain sentiment rules + Candidate score)
- Arm B: Walk-Forward Machine-Learned Probability Gate (HistGradientBoostingClassifier on 18 market indicators)

Strict walk-forward cross validation over 4 folds; zero lookahead leakage.
"""

import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent
EXP_DIR = ROOT / "research/model_upgrade_validation/dragon_regime_gate_20261027T085500Z"
DAILY_REGIME_PATH = EXP_DIR / "round5_daily_regime_dataset.csv"
R4_COMPETITION_PATH = ROOT / "research/model_upgrade_validation/dragon_theme_matrix_20261027T083500Z/daily_competition_round4.csv"

def main():
    print(f"Loading Daily Regime dataset from {DAILY_REGIME_PATH}...")
    daily_df = pd.read_csv(DAILY_REGIME_PATH)
    daily_df['date_dt'] = pd.to_datetime(daily_df['date'])
    daily_df = daily_df.sort_values('date_dt').reset_index(drop=True)
    
    print(f"Loading Round 4 competition predictions from {R4_COMPETITION_PATH}...")
    r4_comp = pd.read_csv(R4_COMPETITION_PATH)
    r4_comp['date'] = pd.to_datetime(r4_comp['date']).dt.strftime('%Y-%m-%d')
    
    # Merge Top-1 metrics from Round 4 onto the test daily dataframe
    merged_daily = daily_df.merge(
        r4_comp[[
            'date', 'top1_code', 'top1_name', 'top1_industry', 'top1_score',
            'top1_is_space_leader', 'top1_is_emotion_dragon', 'top1_is_3to5', 'top1_is_3to4',
            'top1_is_lone_wolf', 'top1_theme_first_count', 'base_top1_space', 'base_top1_5',
            'base_match_status'
        ]],
        on='date',
        how='left'
    )
    
    feature_cols = [
        'market_max_board', 'height_gap_to_max', 'cohort_size', 'emotion_score',
        'down_limit_rate', 'failed_rate', 'space_feedback', 'space_feedback_3d',
        'yesterday_premium', 'breadth_health', 'ladder_health', 'loss_effect_health',
        'crowding_risk', 'regime_fragility', 'emotion_adverse_next_probability',
        'market_total_zt', 'market_total_first', 'avg_diffusion'
    ]
    
    folds = [
        ('2024-08-31', '2024-09-01', '2024-12-31'),
        ('2024-12-31', '2025-01-01', '2025-05-31'),
        ('2025-05-31', '2025-06-01', '2025-11-30'),
        ('2025-11-30', '2025-12-01', '2026-08-31'),
    ]
    
    all_oof_results = []
    fold_metrics = []
    
    for fold_idx, (train_end_str, test_start_str, test_end_str) in enumerate(folds):
        train_end = pd.to_datetime(train_end_str)
        test_start = pd.to_datetime(test_start_str)
        test_end = pd.to_datetime(test_end_str)
        
        train = merged_daily[merged_daily['date_dt'] <= train_end].copy()
        test = merged_daily[(merged_daily['date_dt'] >= test_start) & (merged_daily['date_dt'] <= test_end)].copy()
        
        print(f"\n--- FOLD {fold_idx + 1}: {test_start_str} to {test_end_str} ---")
        print(f"Training Trading Days: {len(train)}, Test Days: {len(test)}")
        
        # 1. ARM A: Heuristic Rule Gate Evaluation on Test Set
        # Rule 1: High board monopoly + low candidate confidence
        r1 = (test['market_max_board'] >= 7) & (test['top1_score'] < 0.32)
        # Rule 2: Panic divergence + low candidate confidence
        r2 = ((test['failed_rate'] >= 0.38) | (test['yesterday_premium'] < -0.015)) & (test['top1_score'] < 0.32)
        # Rule 3: Adverse transition + low candidate confidence
        r3 = (test['emotion_adverse_next_probability'] >= 0.60) & (test['top1_score'] < 0.32)
        # Rule 4: Lone wolf + mid-to-high board
        r4 = (test['top1_is_lone_wolf'] == 1) & (test['market_max_board'] >= 6) & (test['top1_score'] < 0.35)
        
        test['gate_arm_a_close'] = (r1 | r2 | r3 | r4).astype(int)
        test['gate_arm_a_open'] = 1 - test['gate_arm_a_close']
        
        # 2. ARM B: Machine-Learned Walk-Forward Probability Gate
        clf_regime = HistGradientBoostingClassifier(
            max_iter=100, max_depth=3, min_samples_leaf=12, l2_regularization=3.0,
            learning_rate=0.035, random_state=42 + fold_idx
        )
        clf_regime.fit(train[feature_cols], train['is_dragon_day'])
        test['prob_dragon_day'] = clf_regime.predict_proba(test[feature_cols])[:, 1]
        
        auc = roc_auc_score(test['is_dragon_day'], test['prob_dragon_day'])
        brier = brier_score_loss(test['is_dragon_day'], test['prob_dragon_day'])
        
        # Fixed calibrated threshold (0.55 chosen as optimal conservative operational threshold)
        test['gate_arm_b_open_th55'] = (test['prob_dragon_day'] >= 0.55).astype(int)
        test['gate_arm_b_open_th60'] = (test['prob_dragon_day'] >= 0.60).astype(int)
        test['fold'] = fold_idx + 1
        
        fold_metrics.append({
            "fold": fold_idx + 1,
            "test_range": f"{test_start_str} to {test_end_str}",
            "train_days": len(train),
            "test_days": len(test),
            "auc_dragon_regime": float(auc),
            "brier_score": float(brier),
            "arm_a_open_days": int(test['gate_arm_a_open'].sum()),
            "arm_b_open_days_th55": int(test['gate_arm_b_open_th55'].sum()),
            "arm_b_open_days_th60": int(test['gate_arm_b_open_th60'].sum()),
        })
        print(f"Fold {fold_idx + 1} Regime Model AUC: {auc:.4f}, Brier: {brier:.4f}")
        all_oof_results.append(test)
        
    oof_df = pd.concat(all_oof_results, ignore_index=True)
    overall_auc = roc_auc_score(oof_df['is_dragon_day'], oof_df['prob_dragon_day'])
    print(f"\nOverall Walk-Forward Out-of-Fold AUC: {overall_auc:.4f} across {len(oof_df)} trading days")
    
    # Save regime model
    joblib.dump(clf_regime, EXP_DIR / "model_regime_gate_r5.joblib")
    
    # Save daily predictions
    out_cols = [
        'date', 'fold', 'is_dragon_day', 'is_zero_dragon_day',
        'prob_dragon_day',
        'gate_arm_a_open', 'gate_arm_a_close',
        'gate_arm_b_open_th55', 'gate_arm_b_open_th60',
        'market_max_board', 'height_gap_to_max', 'cohort_size',
        'emotion_score', 'failed_rate', 'yesterday_premium', 'down_limit_rate',
        'top1_code', 'top1_name', 'top1_industry', 'top1_score',
        'top1_is_space_leader', 'top1_is_emotion_dragon', 'top1_is_3to5', 'top1_is_3to4',
        'cohort_space_sum', 'cohort_5_sum', 'cohort_dragon_sum',
        'base_top1_space', 'base_top1_5', 'base_match_status'
    ]
    oof_out = oof_df[out_cols].copy()
    oof_out_path = EXP_DIR / "oof_round5_daily_gate_predictions.csv"
    oof_out.to_csv(oof_out_path, index=False, encoding='utf-8-sig')
    print(f"Saved daily gate predictions to: {oof_out_path}")
    
    # Save manifest
    manifest = {
        "experiment": "dragon_regime_gate_20261027T085500Z",
        "conquered_direction": "Direction 2: 大盘退潮期一票禁买闸门",
        "overall_walkforward_auc": float(overall_auc),
        "total_test_days": len(oof_df),
        "fold_metrics": fold_metrics
    }
    with open(EXP_DIR / "manifest_round5.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("Manifest successfully written.")

if __name__ == "__main__":
    main()
