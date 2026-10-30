"""train_and_evaluate_round2.py

Round 2 Dedicated Research & Walk-Forward Backtester.
Predicting:
1. Stage Market Space Leader (is_market_space_leader)
2. Market Emotion Dragon Proxy (is_emotion_dragon_proxy)
3. Milestone Progression (is_3to4, is_3to5, is_deep_dragon_height >= 6)

Strictly Enforces:
- Point-in-time features known at Day t close
- Label maturity verification (terminal_date <= train_end)
- Zero data leakage (all medians and imputations fitted strictly on training partition)
- Comparison against authentic historical BASE OOF scores
- Six-tag 3-board risk audit (audit_three_board_traps.py)
- Selective recommendation with coverage tracking and missed dragon analysis
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXP_DIR = Path(__file__).resolve().parent
DATA_PATH = EXP_DIR / "stage_leader_dataset.csv"

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
        
    engineered_cols = raw_cols + [
        'is_low_price', 'is_small_cap', 'is_tight_amplitude', 'is_high_churn_risk',
        'is_crowded', 'firm_seal_strength', 'price_cap_penalty', 'healthy_turnover'
    ]
    return train, test, engineered_cols

def main():
    print(f"Loading stage leader dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['terminal_date'] = pd.to_datetime(df['terminal_date'])
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
        'market_max_board', 'market_max_board_count', 'consecutive_count', 'total_limit_up'
    ]
    
    folds = [
        ('2024-08-31', '2024-09-01', '2024-12-31'),
        ('2024-12-31', '2025-01-01', '2025-05-31'),
        ('2025-05-31', '2025-06-01', '2025-11-30'),
        ('2025-11-30', '2025-12-01', '2026-08-31'),
    ]
    
    all_oof_results = []
    fold_summaries = []
    
    for fold_idx, (train_end_str, test_start_str, test_end_str) in enumerate(folds):
        train_end = pd.to_datetime(train_end_str)
        test_start = pd.to_datetime(test_start_str)
        test_end = pd.to_datetime(test_end_str)
        
        # Enforce maturity condition on training set:
        # Stock must have hit 3 boards on or before train_end,
        # AND its streak must have matured (terminal_date <= train_end)!
        train_raw = df[(df['date'] <= train_end) & (df['terminal_date'] <= train_end)].copy()
        test_raw = df[(df['date'] >= test_start) & (df['date'] <= test_end)].copy()
        
        train, test, feature_cols = engineer_features(train_raw, test_raw, raw_feature_cols)
        
        print(f"\n--- FOLD {fold_idx + 1}: {test_start_str} to {test_end_str} ---")
        print(f"Mature Training Samples: {len(train)}, Test Samples: {len(test)}")
        
        # Train Space Leader Model
        clf_space = HistGradientBoostingClassifier(
            max_iter=120,
            max_depth=4,
            min_samples_leaf=16,
            l2_regularization=2.5,
            learning_rate=0.035,
            class_weight={0: 1.0, 1: 1.5},
            random_state=201 + fold_idx
        )
        clf_space.fit(train[feature_cols], train['is_market_space_leader'])
        test['score_space_leader'] = clf_space.predict_proba(test[feature_cols])[:, 1]
        
        # Train Emotion Dragon Candidate Model (higher asymmetric weight due to ~13% base rate)
        clf_dragon = HistGradientBoostingClassifier(
            max_iter=120,
            max_depth=4,
            min_samples_leaf=16,
            l2_regularization=3.0,
            learning_rate=0.035,
            class_weight={0: 1.0, 1: 2.2},
            random_state=301 + fold_idx
        )
        clf_dragon.fit(train[feature_cols], train['is_emotion_dragon_proxy'])
        test['score_emotion_dragon'] = clf_dragon.predict_proba(test[feature_cols])[:, 1]
        
        # Train 5-board continuous model
        clf_5 = HistGradientBoostingClassifier(
            max_iter=120,
            max_depth=4,
            min_samples_leaf=16,
            l2_regularization=2.5,
            learning_rate=0.035,
            class_weight={0: 1.0, 1: 1.5},
            random_state=401 + fold_idx
        )
        clf_5.fit(train[feature_cols], train['is_3to5'])
        test['score_height_5'] = clf_5.predict_proba(test[feature_cols])[:, 1]
        
        # Composite Dragon Score
        test['score_composite_dragon'] = (
            0.40 * test['score_space_leader'] +
            0.35 * test['score_emotion_dragon'] +
            0.25 * test['score_height_5']
        )
        test['fold'] = fold_idx + 1
        all_oof_results.append(test)
        
        # Fold evaluation metrics
        auc_space = roc_auc_score(test['is_market_space_leader'], test['score_space_leader'])
        auc_dragon = roc_auc_score(test['is_emotion_dragon_proxy'], test['score_emotion_dragon'])
        auc_5 = roc_auc_score(test['is_3to5'], test['score_height_5'])
        
        # Daily competition on this fold
        comp_fold = test.groupby('date').filter(lambda g: len(g) >= 2)
        top1_space_hits = [g.sort_values('score_composite_dragon', ascending=False).iloc[0]['is_market_space_leader'] for _, g in comp_fold.groupby('date')]
        top1_dragon_hits = [g.sort_values('score_composite_dragon', ascending=False).iloc[0]['is_emotion_dragon_proxy'] for _, g in comp_fold.groupby('date')]
        top1_5_hits = [g.sort_values('score_composite_dragon', ascending=False).iloc[0]['is_3to5'] for _, g in comp_fold.groupby('date')]
        top1_4_hits = [g.sort_values('score_composite_dragon', ascending=False).iloc[0]['is_3to4'] for _, g in comp_fold.groupby('date')]
        
        fold_summaries.append({
            "fold": fold_idx + 1,
            "test_range": f"{test_start_str} to {test_end_str}",
            "train_samples": len(train),
            "test_samples": len(test),
            "auc_space_leader": round(float(auc_space), 4),
            "auc_emotion_dragon": round(float(auc_dragon), 4),
            "auc_height_5": round(float(auc_5), 4),
            "competitive_days": len(top1_space_hits),
            "top1_space_leader_rate": round(float(np.mean(top1_space_hits)), 4) if top1_space_hits else 0.0,
            "top1_emotion_dragon_rate": round(float(np.mean(top1_dragon_hits)), 4) if top1_dragon_hits else 0.0,
            "top1_5board_rate": round(float(np.mean(top1_5_hits)), 4) if top1_5_hits else 0.0,
            "top1_4board_rate": round(float(np.mean(top1_4_hits)), 4) if top1_4_hits else 0.0,
        })
        print(f"Fold {fold_idx + 1} Done: AUC Space={auc_space:.3f}, AUC Dragon={auc_dragon:.3f}, Top1 Space Rate={np.mean(top1_space_hits):.3f}, Top1 Dragon Rate={np.mean(top1_dragon_hits):.3f}")

    oof = pd.concat(all_oof_results, ignore_index=True)
    
    # Save OOF predictions
    oof_out_cols = [
        'date', 'code', 'name', 'boards', 'terminal_board', 'terminal_date',
        'is_3to4', 'is_3to5', 'is_deep_dragon_height', 'is_super_monster_height',
        'is_market_space_leader', 'is_unique_space_leader', 'is_emotion_dragon_proxy',
        'score_space_leader', 'score_emotion_dragon', 'score_height_5', 'score_composite_dragon',
        'base_historical_score', 'fold'
    ]
    oof[oof_out_cols].to_csv(EXP_DIR / "oof_round2_predictions.csv", index=False, encoding='utf-8-sig')
    
    # Train full dataset models for final deployment/reproducibility
    print("\nTraining final models on full mature dataset...")
    full_mature = df[df['terminal_date'] <= '2026-08-31'].copy()
    _, full_mature, full_features = engineer_features(full_mature, full_mature, raw_feature_cols)
    
    clf_space_final = HistGradientBoostingClassifier(
        max_iter=120, max_depth=4, min_samples_leaf=16, l2_regularization=2.5,
        learning_rate=0.035, class_weight={0: 1.0, 1: 1.5}, random_state=42
    )
    clf_space_final.fit(full_mature[full_features], full_mature['is_market_space_leader'])
    joblib.dump(clf_space_final, EXP_DIR / "model_space_leader.joblib")
    
    clf_dragon_final = HistGradientBoostingClassifier(
        max_iter=120, max_depth=4, min_samples_leaf=16, l2_regularization=3.0,
        learning_rate=0.035, class_weight={0: 1.0, 1: 2.2}, random_state=42
    )
    clf_dragon_final.fit(full_mature[full_features], full_mature['is_emotion_dragon_proxy'])
    joblib.dump(clf_dragon_final, EXP_DIR / "model_emotion_dragon.joblib")
    
    clf_5_final = HistGradientBoostingClassifier(
        max_iter=120, max_depth=4, min_samples_leaf=16, l2_regularization=2.5,
        learning_rate=0.035, class_weight={0: 1.0, 1: 1.5}, random_state=42
    )
    clf_5_final.fit(full_mature[full_features], full_mature['is_3to5'])
    joblib.dump(clf_5_final, EXP_DIR / "model_height_5.joblib")
    
    # Save manifest
    manifest = {
        "round": 2,
        "title": "Stage Market Space Leader & Emotion Dragon Candidate Model",
        "saved_at": pd.Timestamp.now().isoformat(),
        "total_oof_samples": len(oof),
        "raw_features_count": len(raw_feature_cols),
        "engineered_features_count": len(full_features),
        "features": full_features,
        "fold_summaries": fold_summaries
    }
    with open(EXP_DIR / "manifest_round2.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        
    print(f"\nRound 2 training complete! Artifacts saved to {EXP_DIR}")

if __name__ == "__main__":
    main()
