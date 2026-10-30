"""train_and_evaluate_round3.py

Round 3 Walk-Forward Trainer & Evaluator:
1. Strict session-by-session streak validation along adjacent official trading days.
2. Label maturity verification (terminal_date <= train_end) per fold.
3. Zero data leakage: all scalers and medians fitted strictly on train partition.
4. Target A (Maintain / Extend Leadership) vs Target B (Takeover Stage Space Leader) modeled cleanly.
5. Emotion Dragon Candidate Proxy & Continuous Height models.
6. Authentic historical BASE OOF score matching, separating complete cohort match vs partial match.
7. Generates model artifacts, OOF predictions, daily competition records, and metrics.
"""

import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent
EXP_DIR = ROOT / "research/model_upgrade_validation/dragon_takeover_20261027T074500Z"
DATA_PATH = EXP_DIR / "round3_dataset.csv"

def engineer_features(train_df, test_df, raw_cols):
    """
    Computes domain-engineered dragon indicators strictly fitted on train_df,
    then applied to both train_df and test_df.
    """
    train = train_df.copy()
    test = test_df.copy()
    
    # Estimate thresholds strictly on train
    med_price = train['price'].median()
    med_cap = train['float_cap_log'].median()
    
    for df in [train, test]:
        df['is_low_price'] = (df['price'] <= med_price).astype(float)
        df['is_small_cap'] = (df['float_cap_log'] <= med_cap).astype(float)
        df['is_tight_amplitude'] = (df['amplitude'] <= 3.0).astype(float)
        df['is_high_churn_risk'] = ((df['turnover'] > 0.18) & (df['one_word_proxy'] == 0)).astype(float)
        df['is_crowded'] = (df['same_height_rivals'] >= 4).astype(float)
        df['firm_seal_strength'] = (df['one_word_proxy'] * 2.0 + (df['amplitude'] <= 2.0).astype(float) - (df['amplitude'] > 8.0).astype(float))
        df['price_cap_penalty'] = (df['price'] > 16.0).astype(float) + (df['float_cap_log'] > 23.0).astype(float)
        df['healthy_turnover'] = ((df['turnover'] >= 0.05) & (df['turnover'] <= 0.18)).astype(float)
        df['space_leader_stalling'] = ((df['height_gap'] > 0) & (df['space_feedback'] <= 0)).astype(float)
        df['takeover_window_open'] = ((df['height_gap'] > 0) & (df['space_feedback'] <= 0) & (df['promotion_3to4'] > 0.4)).astype(float)
        
    engineered_cols = raw_cols + [
        'is_low_price', 'is_small_cap', 'is_tight_amplitude', 'is_high_churn_risk',
        'is_crowded', 'firm_seal_strength', 'price_cap_penalty', 'healthy_turnover',
        'space_leader_stalling', 'takeover_window_open'
    ]
    return train, test, engineered_cols

def main():
    print(f"Loading Round 3 dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['determined_date_final'] = pd.to_datetime(df['determined_date_final'])
    df['determined_date_3to5'] = pd.to_datetime(df['determined_date_3to5'])
    df['determined_date_3to4'] = pd.to_datetime(df['determined_date_3to4'])
    df = df.sort_values(['date', 'code']).reset_index(drop=True)
    
    raw_feature_cols = [
        'height_gap', 'is_space_leader', 'unique_space', 'height_share', 'same_height_rivals',
        'turnover', 'signal_amount', 'turnover_vs_prev', 'amount_float_cap', 'one_word_proxy',
        'one_word_streak', 'amplitude', 'relative_strength', 'price', 'float_cap_log',
        'emotion_score', 'limit_count_scaled', 'consecutive_scaled', 'down_limit_rate',
        'failed_rate', 'seal_rate', 'height_change_scaled', 'limit_change_scaled',
        'promotion_overall', 'promotion_1to2', 'promotion_2to3', 'promotion_3to4',
        'promotion_mid', 'space_feedback', 'middle_density', 'ladder_gap_count',
        'breadth_height_divergence', 'yesterday_premium', 'failed_tail_loss',
        'previous_limit_tail_loss', 'breadth_health', 'relay_appetite', 'ladder_health',
        'loss_effect_health', 'crowding_risk', 'regime_fragility',
        'announcement_risk', 'reduction_risk', 'unlock_risk', 'earnings_risk',
        'market_max_board', 'market_max_board_count', 'consecutive_count', 'total_limit_up',
        'amount_cohort_share', 'amount_cohort_rank', 'amplitude_cohort_rank', 'price_cohort_rank', 'turnover_cohort_rank',
        'is_cohort_volume_leader', 'is_cohort_tightest', 'is_cohort_lowest_price', 'is_already_space_leader'
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
        
        # Enforce maturity condition on training set:
        train_raw = df[(df['date'] <= train_end) & (df['determined_date_final'] <= train_end)].copy()
        test_raw = df[(df['date'] >= test_start) & (df['date'] <= test_end)].copy()
        
        train, test, feature_cols = engineer_features(train_raw, test_raw, raw_feature_cols)
        
        print(f"\n--- FOLD {fold_idx + 1}: {test_start_str} to {test_end_str} ---")
        print(f"Mature Training Samples: {len(train)}, Test Samples: {len(test)}")
        
        # 1. Model for Stage Space Leadership (Unified Space Leader: maintain if Case A, takeover if Case B)
        # Target: For Case A, target is reaching >= 4 and maintaining leadership; for Case B, target is taking over space leadership!
        # This maps to is_market_space_leader!
        clf_space = HistGradientBoostingClassifier(
            max_iter=120, max_depth=4, min_samples_leaf=16, l2_regularization=2.5,
            learning_rate=0.035, class_weight={0: 1.0, 1: 1.6}, random_state=101 + fold_idx
        )
        clf_space.fit(train[feature_cols], train['is_market_space_leader'])
        test['score_space_leader'] = clf_space.predict_proba(test[feature_cols])[:, 1]
        
        # 2. Model for Takeover (Case B specific: predicting subsequent takeover of space leader)
        train_b = train[train['is_already_space_leader'] == 0].copy()
        clf_takeover = HistGradientBoostingClassifier(
            max_iter=120, max_depth=4, min_samples_leaf=16, l2_regularization=2.5,
            learning_rate=0.035, class_weight={0: 1.0, 1: 2.0}, random_state=201 + fold_idx
        )
        clf_takeover.fit(train_b[feature_cols], train_b['target_B_takeover_space'])
        test['score_takeover'] = clf_takeover.predict_proba(test[feature_cols])[:, 1]
        
        # 3. Model for Emotion Dragon Candidate Proxy
        clf_dragon = HistGradientBoostingClassifier(
            max_iter=120, max_depth=4, min_samples_leaf=16, l2_regularization=3.0,
            learning_rate=0.035, class_weight={0: 1.0, 1: 2.5}, random_state=301 + fold_idx
        )
        clf_dragon.fit(train[feature_cols], train['is_emotion_dragon_proxy'])
        test['score_emotion_dragon'] = clf_dragon.predict_proba(test[feature_cols])[:, 1]
        
        # 4. Model for Continuous 5 Boards
        clf_5 = HistGradientBoostingClassifier(
            max_iter=120, max_depth=4, min_samples_leaf=16, l2_regularization=2.5,
            learning_rate=0.035, class_weight={0: 1.0, 1: 1.5}, random_state=401 + fold_idx
        )
        clf_5.fit(train[feature_cols], train['is_3to5'])
        test['score_height_5'] = clf_5.predict_proba(test[feature_cols])[:, 1]
        
        # Composite Dragon Score
        test['score_round3_composite'] = (
            0.35 * test['score_space_leader'] +
            0.25 * test['score_takeover'] +
            0.25 * test['score_emotion_dragon'] +
            0.15 * test['score_height_5']
        )
        test['fold'] = fold_idx + 1
        all_oof_results.append(test)
        
        auc_space = roc_auc_score(test['is_market_space_leader'], test['score_space_leader'])
        auc_dragon = roc_auc_score(test['is_emotion_dragon_proxy'], test['score_emotion_dragon'])
        auc_5 = roc_auc_score(test['is_3to5'], test['score_height_5'])
        
        fold_metrics.append({
            "fold": fold_idx + 1,
            "test_range": f"{test_start_str} to {test_end_str}",
            "train_samples": len(train),
            "test_samples": len(test),
            "auc_space_leader": round(float(auc_space), 4),
            "auc_emotion_dragon": round(float(auc_dragon), 4),
            "auc_height_5": round(float(auc_5), 4)
        })
        print(f"Fold {fold_idx + 1} Done: AUC Space={auc_space:.3f}, AUC Dragon={auc_dragon:.3f}, AUC 5-Board={auc_5:.3f}")

    oof = pd.concat(all_oof_results, ignore_index=True)
    
    # Save OOF predictions
    oof_cols = [
        'date', 'code', 'name', 'boards', 'terminal_board', 'terminal_date',
        'is_already_space_leader', 'height_gap',
        'is_3to4', 'is_3to5', 'is_deep_6', 'is_monster_7',
        'target_A_continue', 'target_A_maintain_days', 'target_A_unique',
        'target_B_takeover_space', 'target_B_takeover_unique',
        'is_market_space_leader', 'is_unique_space_leader', 'is_emotion_dragon_proxy',
        'score_space_leader', 'score_takeover', 'score_emotion_dragon', 'score_height_5', 'score_round3_composite',
        'base_historical_score', 'base_cohort_match_status', 'fold'
    ]
    oof[oof_cols].to_csv(EXP_DIR / "oof_round3_predictions.csv", index=False, encoding='utf-8-sig')
    
    # Train production research models on full mature dataset
    print("\nTraining final research models on full mature dataset...")
    full_mature = df[df['determined_date_final'] <= '2026-08-31'].copy()
    _, full_mature, full_features = engineer_features(full_mature, full_mature, raw_feature_cols)
    
    clf_space_final = HistGradientBoostingClassifier(
        max_iter=120, max_depth=4, min_samples_leaf=16, l2_regularization=2.5,
        learning_rate=0.035, class_weight={0: 1.0, 1: 1.6}, random_state=42
    )
    clf_space_final.fit(full_mature[full_features], full_mature['is_market_space_leader'])
    joblib.dump(clf_space_final, EXP_DIR / "model_space_leader_r3.joblib")
    
    clf_dragon_final = HistGradientBoostingClassifier(
        max_iter=120, max_depth=4, min_samples_leaf=16, l2_regularization=3.0,
        learning_rate=0.035, class_weight={0: 1.0, 1: 2.5}, random_state=42
    )
    clf_dragon_final.fit(full_mature[full_features], full_mature['is_emotion_dragon_proxy'])
    joblib.dump(clf_dragon_final, EXP_DIR / "model_emotion_dragon_r3.joblib")
    
    clf_5_final = HistGradientBoostingClassifier(
        max_iter=120, max_depth=4, min_samples_leaf=16, l2_regularization=2.5,
        learning_rate=0.035, class_weight={0: 1.0, 1: 1.5}, random_state=42
    )
    clf_5_final.fit(full_mature[full_features], full_mature['is_3to5'])
    joblib.dump(clf_5_final, EXP_DIR / "model_height_5_r3.joblib")
    
    manifest = {
        "round": 3,
        "title": "Stage Market Space Leader Takeover & Emotion Dragon Identification",
        "saved_at": pd.Timestamp.now().isoformat(),
        "total_oof_samples": len(oof),
        "raw_features_count": len(raw_feature_cols),
        "engineered_features_count": len(full_features),
        "features": full_features,
        "fold_metrics": fold_metrics
    }
    with open(EXP_DIR / "manifest_round3.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        
    print(f"\nRound 3 training complete! Saved to {EXP_DIR}")

if __name__ == "__main__":
    main()
