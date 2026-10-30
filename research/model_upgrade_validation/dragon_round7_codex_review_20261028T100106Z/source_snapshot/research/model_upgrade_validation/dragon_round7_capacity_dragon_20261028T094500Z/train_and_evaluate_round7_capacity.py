"""
train_and_evaluate_round7_capacity.py:

Trains the Round 7 Capacity Dragon Upgraded Model (大容量换手真龙升级版):
1. Overhauls the 5 self-destructive features from Round 6:
   - Eliminates blind punishment on high turnover (turnover > 0.20): re-engineered as healthy absorption unless accompanied by afternoon wobble.
   - Converts 'is_crowded' from negative penalty into positive 'is_capacity_dragon' (volume >= 8e8 and turnover 12%~28% with tight seal).
   - Replaces small-cap bias with 'is_capacity_volume_leader' (turnover or amount rank #1 in cohort).
   - Broadens 'healthy_turnover' to 10%~28% (comfortably capturing 20.57% turnover dragons like 襄阳轴承).
2. Preserves the exact 80-feature dimensionality and 4-fold temporal structure.
3. Fits all 5 independent heads: 3to4, space, 5b, gain46, reb18 (20 models in total) using HistGradientBoostingClassifier.
4. Persists models, training medians, feature order, OOF predictions, and manifest.
"""

import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import joblib

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(".")
R7_DIR = ROOT / "research/model_upgrade_validation/dragon_round7_capacity_dragon_20261028T094500Z"
DATASET_PATH = R7_DIR / "round7_dataset.csv"
REGIME_GATE_PATH = R7_DIR / "round7_regime_gate.csv"

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
        # R7 Surgery 1: Replace small-cap bias with capacity volume leadership
        df['is_capacity_volume_leader'] = ((df['turnover_cohort_rank'] == 1.0) | (df['is_cohort_volume_leader'] == 1.0)).astype(float)
        df['is_tight_amplitude'] = (df['amplitude'] < 5.0).astype(float)
        # R7 Surgery 2: High churn risk ONLY applies if accompanied by tail loss / wobble
        df['is_high_churn_risk'] = ((df['turnover'] > 0.25) & (df['previous_limit_tail_loss'] > 0.0)).astype(float)
        # R7 Surgery 3: Positive Capacity Dragon identifier (>=8e8 volume and 12%~28% turnover with tight seal)
        df['is_capacity_dragon'] = ((df['signal_amount'] >= 8e8) & (df['turnover'] >= 0.12) & (df['turnover'] <= 0.28) & (df['amplitude'] < 5.0)).astype(float)
        df['firm_seal_strength'] = (df['amplitude'] < 3.0).astype(float)
        # R7 Surgery 4: Broaden healthy turnover range to 10%~28% (capturing 襄阳轴承's 20.57%)
        df['healthy_turnover'] = ((df['turnover'] >= 0.10) & (df['turnover'] <= 0.28)).astype(float)
        df['theme_assisted_dragon'] = ((df['theme_first_board_count'] >= 2.0) & (df['theme_zt_share'] >= 0.3)).astype(float)
        df['lone_wolf_penalty'] = (df['is_isolated_lone_wolf'] * (1.0 - df['unique_space'])).astype(float)
        df['theme_diffusion_heating'] = ((df['theme_first_board_expansion'] > 0.0) & (df['theme_cluster_expansion'] > 0.0)).astype(float)
        
    feat_cols = raw_cols + [
        'is_low_price', 'is_capacity_volume_leader', 'is_tight_amplitude', 'is_high_churn_risk',
        'is_capacity_dragon', 'firm_seal_strength', 'healthy_turnover', 'theme_assisted_dragon',
        'lone_wolf_penalty', 'theme_diffusion_heating'
    ]
    
    medians = train[feat_cols].median().to_dict()
    medians['_head_med_price'] = float(med_price)
    medians['_head_med_cap'] = float(med_cap)
    
    col_meds = {k: v for k, v in medians.items() if not k.startswith('_')}
    train[feat_cols] = train[feat_cols].fillna(col_meds)
    test[feat_cols] = test[feat_cols].fillna(col_meds)
    
    return train, test, feat_cols, medians

def main():
    print(f"Loading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset Shape: {df.shape}")
    
    print(f"Loading regime gate predictions from {REGIME_GATE_PATH}...")
    df_gate = pd.read_csv(REGIME_GATE_PATH)
    reg_map = dict(zip(df_gate['date'], df_gate['prob_dragon_day']))
    gate_map = dict(zip(df_gate['date'], df_gate['gate_arm_b_open_th55']))
    
    all_models = {}
    all_medians = {}
    fold_metrics = []
    oof_dfs = []
    
    for fold_idx, (t_cut, v_start, v_end) in enumerate(FOLDS, 1):
        print(f"\n==================== FOLD {fold_idx}: Cutoff={t_cut}, Test={v_start} to {v_end} ====================")
        test_raw = df[(df['date'] >= v_start) & (df['date'] <= v_end)].sort_values(['date', 'code']).copy()
        
        # 1. Head 3to4: Target is_3to4
        train_3to4_raw = df[(df['date'] <= t_cut) & (df['is_3to4'].notna()) & (df['determined_date_3to4'] <= t_cut)].copy()
        tr_3to4, te_3to4, feat_cols, med_3to4 = engineer_features_for_head(train_3to4_raw, test_raw, RAW_FEATURE_COLS)
        clf_3to4 = HistGradientBoostingClassifier(
            max_iter=130, max_depth=4, min_samples_leaf=16, l2_regularization=2.5,
            learning_rate=0.035, class_weight={0: 1.0, 1: 1.2}, random_state=51 + fold_idx
        )
        clf_3to4.fit(tr_3to4[feat_cols], tr_3to4['is_3to4'])
        preds_3to4 = clf_3to4.predict_proba(te_3to4[feat_cols])[:, 1]
        auc_3to4 = float(roc_auc_score(te_3to4['is_3to4'], preds_3to4))
        all_models[f'fold_{fold_idx}_3to4'] = clf_3to4
        all_medians[f'fold_{fold_idx}_3to4'] = med_3to4
        print(f"Fold {fold_idx} 3to4 Head: Train N={len(tr_3to4)}, Test N={len(te_3to4)}, Test AUC={auc_3to4:.5f}")
        
        # 2. Head Space: Target is_market_space_leader
        train_space_raw = df[(df['date'] <= t_cut) & (df['is_market_space_leader'].notna()) & (df['determined_date_final'] <= t_cut)].copy()
        tr_space, te_space, _, med_space = engineer_features_for_head(train_space_raw, test_raw, RAW_FEATURE_COLS)
        clf_space = HistGradientBoostingClassifier(
            max_iter=130, max_depth=4, min_samples_leaf=16, l2_regularization=2.5,
            learning_rate=0.035, class_weight={0: 1.0, 1: 1.6}, random_state=101 + fold_idx
        )
        clf_space.fit(tr_space[feat_cols], tr_space['is_market_space_leader'])
        preds_space = clf_space.predict_proba(te_space[feat_cols])[:, 1]
        auc_space = float(roc_auc_score(te_space['is_market_space_leader'], preds_space))
        all_models[f'fold_{fold_idx}_space'] = clf_space
        all_medians[f'fold_{fold_idx}_space'] = med_space
        print(f"Fold {fold_idx} Space Head: Train N={len(tr_space)}, Test N={len(te_space)}, Test AUC={auc_space:.5f}")
        
        # 3. Head 5b: Target is_3to5
        train_5b_raw = df[(df['date'] <= t_cut) & (df['is_3to5'].notna()) & (df['det_date_3to5'] <= t_cut)].copy()
        tr_5b, te_5b, _, med_5b = engineer_features_for_head(train_5b_raw, test_raw, RAW_FEATURE_COLS)
        clf_5b = HistGradientBoostingClassifier(
            max_iter=130, max_depth=4, min_samples_leaf=16, l2_regularization=2.5,
            learning_rate=0.035, class_weight={0: 1.0, 1: 1.5}, random_state=201 + fold_idx
        )
        clf_5b.fit(tr_5b[feat_cols], tr_5b['is_3to5'])
        preds_5b = clf_5b.predict_proba(te_5b[feat_cols])[:, 1]
        auc_5b = float(roc_auc_score(te_5b['is_3to5'], preds_5b))
        all_models[f'fold_{fold_idx}_5b'] = clf_5b
        all_medians[f'fold_{fold_idx}_5b'] = med_5b
        print(f"Fold {fold_idx} 5B Head: Train N={len(tr_5b)}, Test N={len(te_5b)}, Test AUC={auc_5b:.5f}")
        
        # 4. Head Gain46: Target target_path_gain_46pct
        train_g46_raw = df[(df['date'] <= t_cut) & (df['target_path_gain_46pct'].notna()) & (df['det_date_gain46'] <= t_cut)].copy()
        tr_g46, te_g46, _, med_g46 = engineer_features_for_head(train_g46_raw, test_raw, RAW_FEATURE_COLS)
        clf_g46 = HistGradientBoostingClassifier(
            max_iter=130, max_depth=4, min_samples_leaf=16, l2_regularization=2.5,
            learning_rate=0.035, class_weight={0: 1.0, 1: 2.2}, random_state=301 + fold_idx
        )
        clf_g46.fit(tr_g46[feat_cols], tr_g46['target_path_gain_46pct'])
        preds_g46 = clf_g46.predict_proba(te_g46[feat_cols])[:, 1]
        valid_te_g46 = te_g46[te_g46['target_path_gain_46pct'].notna()]
        valid_preds_g46 = preds_g46[te_g46['target_path_gain_46pct'].notna()]
        auc_g46 = float(roc_auc_score(valid_te_g46['target_path_gain_46pct'], valid_preds_g46))
        all_models[f'fold_{fold_idx}_gain46'] = clf_g46
        all_medians[f'fold_{fold_idx}_gain46'] = med_g46
        print(f"Fold {fold_idx} Gain46 Head: Train N={len(tr_g46)}, Test N={len(valid_te_g46)}, Test AUC={auc_g46:.5f}")
        
        # 5. Head Reb18: Target target_rebound_18pct
        train_r18_raw = df[(df['date'] <= t_cut) & (df['target_rebound_18pct'].notna()) & (df['det_date_reb18'] <= t_cut)].copy()
        tr_r18, te_r18, _, med_r18 = engineer_features_for_head(train_r18_raw, test_raw, RAW_FEATURE_COLS)
        clf_r18 = HistGradientBoostingClassifier(
            max_iter=130, max_depth=4, min_samples_leaf=16, l2_regularization=2.5,
            learning_rate=0.035, class_weight={0: 1.0, 1: 1.4}, random_state=401 + fold_idx
        )
        clf_r18.fit(tr_r18[feat_cols], tr_r18['target_rebound_18pct'])
        preds_r18 = clf_r18.predict_proba(te_r18[feat_cols])[:, 1]
        valid_te_r18 = te_r18[te_r18['target_rebound_18pct'].notna()]
        valid_preds_r18 = preds_r18[te_r18['target_rebound_18pct'].notna()]
        auc_r18 = float(roc_auc_score(valid_te_r18['target_rebound_18pct'], valid_preds_r18))
        all_models[f'fold_{fold_idx}_reb18'] = clf_r18
        all_medians[f'fold_{fold_idx}_reb18'] = med_r18
        print(f"Fold {fold_idx} Reb18 Head: Train N={len(tr_r18)}, Test N={len(valid_te_r18)}, Test AUC={auc_r18:.5f}")
        
        # Construct Out-Of-Fold DataFrame
        test_out = te_3to4.copy()
        test_out['fold'] = fold_idx
        test_out['score_3to4'] = preds_3to4
        test_out['score_space_leader'] = preds_space
        test_out['score_height_5'] = preds_5b
        test_out['score_path_gain46'] = preds_g46
        test_out['score_rebound18'] = preds_r18
        
        # Arms formulas:
        # Arm 0: Clean 3to4 baseline
        test_out['score_arm0_r4'] = test_out['score_3to4']
        # Arm 1: Full-horizon combo
        test_out['score_arm1_full'] = (
            0.45 * test_out['score_3to4'] +
            0.25 * test_out['score_height_5'] +
            0.15 * test_out['score_path_gain46'] +
            0.15 * test_out['score_rebound18']
        )
        # Arm 2: Space Core
        test_out['score_arm2_primary'] = (
            0.60 * test_out['score_space_leader'] +
            0.40 * test_out['score_height_5']
        )
        # Arm 3: Round 7 Capacity Dragon Dual-Track (大容量换手真龙核心轨)
        test_out['score_arm3_capacity'] = (
            0.40 * test_out['score_space_leader'] +
            0.35 * test_out['score_height_5'] +
            0.25 * test_out['score_3to4']
        )
        
        test_out['prob_dragon_day'] = [reg_map.get(d, 0.5) for d in test_out['date']]
        test_out['gate_open_55'] = [gate_map.get(d, 1) for d in test_out['date']]
        
        oof_dfs.append(test_out)
        
        fold_metrics.append({
            "fold": fold_idx,
            "test_range": f"{v_start} to {v_end}",
            "auc_3to4": auc_3to4,
            "auc_space": auc_space,
            "auc_5b": auc_5b,
            "auc_gain46": auc_g46,
            "auc_reb18": auc_r18,
            "test_samples": len(test_out)
        })
        
    oof_df = pd.concat(oof_dfs, axis=0).reset_index(drop=True)
    
    print("\nSaving Round 7 models, medians, feature order, and OOF predictions...")
    joblib.dump(all_models, R7_DIR / "round7_all_fold_models.joblib")
    (R7_DIR / "feature_medians.json").write_text(json.dumps(all_medians, indent=2, ensure_ascii=False), encoding='utf-8')
    (R7_DIR / "feature_order.json").write_text(json.dumps(feat_cols, indent=2, ensure_ascii=False), encoding='utf-8')
    oof_df.to_csv(R7_DIR / "oof_round7_predictions.csv", index=False)
    
    # Generate manifest
    m_joblib = hashlib.sha256((R7_DIR / "round7_all_fold_models.joblib").read_bytes()).hexdigest()
    m_oof = hashlib.sha256((R7_DIR / "oof_round7_predictions.csv").read_bytes()).hexdigest()
    manifest = {
        "experiment": "dragon_round7_capacity_dragon_20261028T094500Z",
        "version": "v1_capacity_dragon_upgrade",
        "heads_trained": ["3to4", "space", "5b", "gain46", "reb18"],
        "total_models": 20,
        "total_oof_rows": len(oof_df),
        "total_test_days": int(oof_df['date'].nunique()),
        "feature_count": len(feat_cols),
        "fold_metrics": fold_metrics,
        "artifacts_sha256": {
            "round7_all_fold_models.joblib": m_joblib,
            "oof_round7_predictions.csv": m_oof
        }
    }
    (R7_DIR / "round7_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print("Training complete! Manifest and artifacts successfully exported.")

if __name__ == "__main__":
    main()
