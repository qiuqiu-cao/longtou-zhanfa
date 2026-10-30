"""
train_and_evaluate_repaired_system.py:

Rigorous Walk-Forward Model Training & OOF Evaluation for Repaired Round 6:
1. Trains 5 explicit heads across all 4 walk-forward folds:
   - Head 1: 3进4 核心晋级头 (target: is_3to4)
   - Head 2: 空间总高标头 (target: is_market_space_leader)
   - Head 3: 连续 5 板突破头 (target: is_3to5)
   - Head 4: 自定义路径上涨代理 (target: target_path_gain_46pct)
   - Head 5: 早期断板反抽代理 (target: target_rebound_18pct)
2. Strict preprocessing consistency:
   - Each head computes its own training medians on its own mature training subset.
   - Test features for each head are engineered/imputed using that head's EXACT training medians.
3. Code-report formula alignment (100% identical):
   - Arm 0: 1.0 * score_3to4
   - Arm 1: 0.45 * score_3to4 + 0.25 * score_height_5 + 0.15 * score_path_gain46 + 0.15 * score_rebound18
   - Arm 2: 0.60 * score_space_leader + 0.40 * score_height_5
4. Feature integrity:
   - Exactly 80 engineered features recorded in feature_order.json.
5. All 20 models (5 heads x 4 folds) and fold medians persisted to disk.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import joblib

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(".")
REPAIR_DIR = ROOT / "research/model_upgrade_validation/dragon_round6_repaired_20261027T180000Z"
DATASET_PATH = REPAIR_DIR / "repaired_round6_dataset.csv"
REGIME_GATE_PATH = REPAIR_DIR / "oof_repaired_regime_gate.csv"

FOLDS = [
    ("2024-08-31", "2024-09-01", "2024-12-31"),
    ("2024-12-31", "2025-01-01", "2025-05-31"),
    ("2025-05-31", "2025-06-01", "2025-11-30"),
    ("2025-11-30", "2025-12-01", "2026-08-31"),
]

RAW_FEATURE_COLS = [
    # Space & height
    'height_gap', 'is_space_leader', 'unique_space', 'height_share', 'same_height_rivals',
    # Price & volume
    'turnover', 'signal_amount', 'turnover_vs_prev', 'amount_float_cap', 'one_word_proxy',
    'one_word_streak', 'amplitude', 'relative_strength', 'price', 'float_cap_log',
    # Market emotion
    'emotion_score', 'limit_count_scaled', 'consecutive_scaled', 'down_limit_rate',
    'failed_rate', 'seal_rate', 'height_change_scaled', 'limit_change_scaled',
    'promotion_overall', 'promotion_1to2', 'promotion_2to3', 'promotion_3to4',
    'promotion_mid', 'space_feedback', 'middle_density', 'ladder_gap_count',
    'breadth_height_divergence', 'yesterday_premium', 'failed_tail_loss',
    'previous_limit_tail_loss', 'breadth_health', 'relay_appetite', 'ladder_health',
    'loss_effect_health', 'crowding_risk', 'regime_fragility',
    # Cohort relative
    'market_max_board', 'market_max_board_count', 'consecutive_count', 'total_limit_up',
    'amount_cohort_share', 'amount_cohort_rank', 'amplitude_cohort_rank', 'price_cohort_rank', 'turnover_cohort_rank',
    'is_cohort_volume_leader', 'is_cohort_tightest', 'is_cohort_lowest_price', 'is_already_space_leader',
    # Cleaned theme diffusion matrix
    'theme_cluster_total_zt', 'theme_first_board_count', 'theme_two_board_count', 'theme_three_board_count',
    'theme_zt_share', 'theme_first_board_share', 'theme_amount_share',
    'theme_ladder_density', 'theme_ladder_complete',
    'theme_first_board_expansion', 'theme_cluster_expansion',
    'is_theme_sole_leader', 'theme_same_height_rivals',
    'is_isolated_lone_wolf', 'theme_diffusion_surge', 'is_industry_unknown'
]

def engineer_features_for_head(train_df, test_df, raw_cols):
    train = train_df.copy()
    test = test_df.copy()
    
    med_price = train['price'].median()
    med_cap = train['float_cap_log'].median()
    
    for df in [train, test]:
        df['is_low_price'] = (df['price'] < med_price).astype(float)
        df['is_small_cap'] = (df['float_cap_log'] < med_cap).astype(float)
        df['is_tight_amplitude'] = (df['amplitude'] < 5.0).astype(float)
        df['is_high_churn_risk'] = (df['turnover'] > 0.20).astype(float)
        df['is_crowded'] = ((df['signal_amount'] > 1e9) & (df['turnover'] > 0.15)).astype(float)
        df['firm_seal_strength'] = (df['amplitude'] < 3.0).astype(float)
        df['healthy_turnover'] = ((df['turnover'] >= 0.08) & (df['turnover'] <= 0.20)).astype(float)
        df['theme_assisted_dragon'] = ((df['theme_first_board_count'] >= 2.0) & (df['theme_zt_share'] >= 0.3)).astype(float)
        df['lone_wolf_penalty'] = (df['is_isolated_lone_wolf'] * (1.0 - df['unique_space'])).astype(float)
        df['theme_diffusion_heating'] = ((df['theme_first_board_expansion'] > 0.0) & (df['theme_cluster_expansion'] > 0.0)).astype(float)
        
    feat_cols = raw_cols + [
        'is_low_price', 'is_small_cap', 'is_tight_amplitude', 'is_high_churn_risk',
        'is_crowded', 'firm_seal_strength', 'healthy_turnover', 'theme_assisted_dragon',
        'lone_wolf_penalty', 'theme_diffusion_heating'
    ]
    
    # Compute training medians for this specific head
    medians = train[feat_cols].median().to_dict()
    medians['_head_med_price'] = float(med_price)
    medians['_head_med_cap'] = float(med_cap)
    
    # Strictly impute using this head's training medians
    col_meds = {k: v for k, v in medians.items() if not k.startswith('_')}
    train[feat_cols] = train[feat_cols].fillna(col_meds)
    test[feat_cols] = test[feat_cols].fillna(col_meds)
    
    return train, test, feat_cols, medians

def main():
    print(f"Loading canonical repaired dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset Shape: {df.shape}")
    
    print(f"Loading regime gate predictions from {REGIME_GATE_PATH}...")
    df_gate = pd.read_csv(REGIME_GATE_PATH)
    reg_map = dict(zip(df_gate['date'], df_gate['prob_dragon_day']))
    gate_map = dict(zip(df_gate['date'], df_gate['gate_arm_b_open_th55']))
    
    all_models = {}
    saved_medians = {}
    oof_records = []
    fold_summaries = []
    
    # Strict temporal walk-forward cross validation
    for fold_idx, (t_end, s_start, s_end) in enumerate(FOLDS):
        print(f"\n==========================================")
        print(f"FOLD {fold_idx + 1}: Train <= {t_end} | Test {s_start} to {s_end}")
        print(f"==========================================")
        
        # Test slice
        test_raw = df[(df['date'] >= s_start) & (df['date'] <= s_end)].copy()
        
        # Train slices with STRICT mature target filtering for each head:
        # Head 1: 3进4 核心晋级头 (mature when determined_date_3to4 <= t_end)
        train_3to4_raw = df[(df['date'] <= t_end) & (df['determined_date_3to4'] <= t_end)].copy()
        # Head 2: 空间总高标头 (mature when determined_date_final <= t_end)
        train_space_raw = df[(df['date'] <= t_end) & (df['determined_date_final'] <= t_end)].copy()
        # Head 3: 5板突破头 (mature when det_date_3to5 <= t_end)
        train_5b_raw = df[(df['date'] <= t_end) & (df['det_date_3to5'] <= t_end)].copy()
        # Head 4: 自定义路径上涨代理 (mature when target is non-null and det_date_gain46 <= t_end)
        train_g46_raw = df[(df['date'] <= t_end) & (df['target_path_gain_46pct'].notna()) & (df['det_date_gain46'] <= t_end)].copy()
        # Head 5: 早期断板反抽代理 (mature when target is non-null and det_date_reb18 <= t_end)
        train_r18_raw = df[(df['date'] <= t_end) & (df['target_rebound_18pct'].notna()) & (df['det_date_reb18'] <= t_end)].copy()
        
        print(f"Mature Training Samples -> 3to4: {len(train_3to4_raw)}, Space: {len(train_space_raw)}, 5B: {len(train_5b_raw)}, Gain46: {len(train_g46_raw)}, Reb18: {len(train_r18_raw)}")
        print(f"Test Samples: {len(test_raw)}")
        
        # Preprocess each head strictly with its OWN training medians (fixes Problem 4!)
        tr_3to4, test_3to4, feat_cols, medians_3to4 = engineer_features_for_head(train_3to4_raw, test_raw, RAW_FEATURE_COLS)
        tr_space, test_space, _, medians_space = engineer_features_for_head(train_space_raw, test_raw, RAW_FEATURE_COLS)
        tr_5b, test_5b, _, medians_5b = engineer_features_for_head(train_5b_raw, test_raw, RAW_FEATURE_COLS)
        tr_g46, test_g46, _, medians_g46 = engineer_features_for_head(train_g46_raw, test_raw, RAW_FEATURE_COLS)
        tr_r18, test_r18, _, medians_r18 = engineer_features_for_head(train_r18_raw, test_raw, RAW_FEATURE_COLS)
        
        saved_medians[f"fold_{fold_idx+1}_3to4"] = medians_3to4
        saved_medians[f"fold_{fold_idx+1}_space"] = medians_space
        saved_medians[f"fold_{fold_idx+1}_5b"] = medians_5b
        saved_medians[f"fold_{fold_idx+1}_gain46"] = medians_g46
        saved_medians[f"fold_{fold_idx+1}_reb18"] = medians_r18
        
        # Train Head 1: 3进4 核心晋级头 (fixes Problem 2!)
        clf_3to4 = HistGradientBoostingClassifier(
            max_iter=130, max_depth=4, min_samples_leaf=16, l2_regularization=2.5,
            learning_rate=0.035, class_weight={0: 1.0, 1: 1.2}, random_state=51 + fold_idx
        )
        clf_3to4.fit(tr_3to4[feat_cols], tr_3to4['is_3to4'])
        score_3to4 = clf_3to4.predict_proba(test_3to4[feat_cols])[:, 1]
        
        # Train Head 2: 空间总高标头
        clf_space = HistGradientBoostingClassifier(
            max_iter=130, max_depth=4, min_samples_leaf=16, l2_regularization=2.5,
            learning_rate=0.035, class_weight={0: 1.0, 1: 1.6}, random_state=101 + fold_idx
        )
        clf_space.fit(tr_space[feat_cols], tr_space['is_market_space_leader'])
        score_space = clf_space.predict_proba(test_space[feat_cols])[:, 1]
        
        # Train Head 3: 5板突破头
        clf_5b = HistGradientBoostingClassifier(
            max_iter=130, max_depth=4, min_samples_leaf=16, l2_regularization=2.5,
            learning_rate=0.035, class_weight={0: 1.0, 1: 1.5}, random_state=201 + fold_idx
        )
        clf_5b.fit(tr_5b[feat_cols], tr_5b['is_3to5'])
        score_5b = clf_5b.predict_proba(test_5b[feat_cols])[:, 1]
        
        # Train Head 4: 自定义路径上涨代理 (46.4%)
        clf_g46 = HistGradientBoostingClassifier(
            max_iter=130, max_depth=4, min_samples_leaf=16, l2_regularization=2.5,
            learning_rate=0.035, class_weight={0: 1.0, 1: 2.2}, random_state=301 + fold_idx
        )
        clf_g46.fit(tr_g46[feat_cols], tr_g46['target_path_gain_46pct'].astype(int))
        score_g46 = clf_g46.predict_proba(test_g46[feat_cols])[:, 1]
        
        # Train Head 5: 早期断板反抽代理 (18%)
        clf_r18 = HistGradientBoostingClassifier(
            max_iter=130, max_depth=4, min_samples_leaf=16, l2_regularization=2.5,
            learning_rate=0.035, class_weight={0: 1.0, 1: 1.4}, random_state=401 + fold_idx
        )
        clf_r18.fit(tr_r18[feat_cols], tr_r18['target_rebound_18pct'].astype(int))
        score_r18 = clf_r18.predict_proba(test_r18[feat_cols])[:, 1]
        
        # Save models for this fold
        all_models[f"fold_{fold_idx+1}_3to4"] = clf_3to4
        all_models[f"fold_{fold_idx+1}_space"] = clf_space
        all_models[f"fold_{fold_idx+1}_5b"] = clf_5b
        all_models[f"fold_{fold_idx+1}_gain46"] = clf_g46
        all_models[f"fold_{fold_idx+1}_reb18"] = clf_r18
        
        # Assemble test dataframe (includes all engineered features)
        test_out = test_3to4.copy()
        test_out['score_3to4'] = score_3to4
        test_out['score_space_leader'] = score_space
        test_out['score_height_5'] = score_5b
        test_out['score_path_gain46'] = score_g46
        test_out['score_rebound18'] = score_r18
        
        # Construct the 3 comparison arms (100% strictly aligned with report!):
        # Arm 0: Repaired Round 4 Baseline (Pure 3to4 Promotion Head)
        test_out['score_arm0_r4'] = 1.0 * score_3to4
        
        # Arm 1: Repaired Round 6 Full-Horizon (4-Head Full Composite)
        test_out['score_arm1_full'] = (
            0.45 * score_3to4 +
            0.25 * score_5b +
            0.15 * score_g46 +
            0.15 * score_r18
        )
        
        # Arm 2: Dual-Track Architecture Primary Space Score (Space Core)
        test_out['score_arm2_primary'] = (
            0.60 * score_space +
            0.40 * score_5b
        )
        
        test_out['fold'] = fold_idx + 1
        test_out['prob_regime_dragon'] = test_out['date'].map(reg_map).fillna(0.5)
        test_out['gate_open_55'] = test_out['date'].map(gate_map).fillna(1).astype(int)
        
        # Record metrics on OOF test set
        auc_3to4 = roc_auc_score(test_out['is_3to4'], score_3to4)
        auc_sp = roc_auc_score(test_out['is_market_space_leader'], score_space)
        auc_5b = roc_auc_score(test_out['is_3to5'], score_5b)
        
        known_g46 = test_out[test_out['target_path_gain_46pct'].notna()]
        auc_g46 = roc_auc_score(known_g46['target_path_gain_46pct'].astype(int), known_g46['score_path_gain46']) if len(known_g46['target_path_gain_46pct'].unique()) > 1 else np.nan
        
        known_r18 = test_out[test_out['target_rebound_18pct'].notna()]
        auc_r18 = roc_auc_score(known_r18['target_rebound_18pct'].astype(int), known_r18['score_rebound18']) if len(known_r18['target_rebound_18pct'].unique()) > 1 else np.nan
        
        print(f"Fold {fold_idx + 1} AUCs -> 3to4: {auc_3to4:.4f}, Space: {auc_sp:.4f}, 5B: {auc_5b:.4f}, Gain46: {auc_g46:.4f}, Reb18: {auc_r18:.4f}")
        
        oof_records.append(test_out)
        fold_summaries.append({
            "fold": fold_idx + 1,
            "test_range": f"{s_start} to {s_end}",
            "auc_3to4": float(auc_3to4),
            "auc_space": float(auc_sp),
            "auc_5b": float(auc_5b),
            "auc_gain46": float(auc_g46),
            "auc_reb18": float(auc_r18),
            "test_samples": len(test_out)
        })
        
    df_oof = pd.concat(oof_records, ignore_index=True)
    oof_csv = REPAIR_DIR / "oof_repaired_predictions.csv"
    df_oof.to_csv(oof_csv, index=False, encoding='utf-8-sig')
    print(f"\nAll OOF predictions saved to: {oof_csv} (Total Rows: {len(df_oof)})")
    
    # Save all fold models and medians
    joblib.dump(all_models, REPAIR_DIR / "repaired_all_fold_models.joblib")
    with open(REPAIR_DIR / "feature_medians.json", "w", encoding="utf-8") as f:
        json.dump(saved_medians, f, indent=2)
    with open(REPAIR_DIR / "feature_order.json", "w", encoding="utf-8") as f:
        json.dump(feat_cols, f, indent=2)
        
    print(f"All 20 fold models saved to: {REPAIR_DIR / 'repaired_all_fold_models.joblib'}")
    print(f"Feature order ({len(feat_cols)} features) saved to: {REPAIR_DIR / 'feature_order.json'}")

if __name__ == "__main__":
    main()
