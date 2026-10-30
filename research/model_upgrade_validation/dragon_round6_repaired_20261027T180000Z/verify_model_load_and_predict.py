"""
verify_model_load_and_predict.py:

Verifies that loading the persisted models and feature medians
reproduces the exact predictions and rankings with zero discrepancy
across all 4 folds and all 5 heads (20 models in total).
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

ROOT = Path(".")
REPAIR_DIR = ROOT / "research/model_upgrade_validation/dragon_round6_repaired_20261027T180000Z"
RAW_DATA_PATH = REPAIR_DIR / "repaired_round6_dataset.csv"

FOLDS = [
    (1, "2024-09-01", "2024-12-31"),
    (2, "2025-01-01", "2025-05-31"),
    (3, "2025-06-01", "2025-11-30"),
    (4, "2025-12-01", "2026-08-31"),
]

HEADS = [
    ('3to4', 'score_3to4'),
    ('space', 'score_space_leader'),
    ('5b', 'score_height_5'),
    ('gain46', 'score_path_gain46'),
    ('reb18', 'score_rebound18'),
]

def transform_features_with_medians(df, medians, raw_cols):
    res = df.copy()
    med_price = medians['_head_med_price']
    med_cap = medians['_head_med_cap']
    
    res['is_low_price'] = (res['price'] < med_price).astype(float)
    res['is_small_cap'] = (res['float_cap_log'] < med_cap).astype(float)
    res['is_tight_amplitude'] = (res['amplitude'] < 5.0).astype(float)
    res['is_high_churn_risk'] = (res['turnover'] > 0.20).astype(float)
    res['is_crowded'] = ((res['signal_amount'] > 1e9) & (res['turnover'] > 0.15)).astype(float)
    res['firm_seal_strength'] = (res['amplitude'] < 3.0).astype(float)
    res['healthy_turnover'] = ((res['turnover'] >= 0.08) & (res['turnover'] <= 0.20)).astype(float)
    res['theme_assisted_dragon'] = ((res['theme_first_board_count'] >= 2.0) & (res['theme_zt_share'] >= 0.3)).astype(float)
    res['lone_wolf_penalty'] = (res['is_isolated_lone_wolf'] * (1.0 - res['unique_space'])).astype(float)
    res['theme_diffusion_heating'] = ((res['theme_first_board_expansion'] > 0.0) & (res['theme_cluster_expansion'] > 0.0)).astype(float)
    
    feat_cols = raw_cols + [
        'is_low_price', 'is_small_cap', 'is_tight_amplitude', 'is_high_churn_risk',
        'is_crowded', 'firm_seal_strength', 'healthy_turnover', 'theme_assisted_dragon',
        'lone_wolf_penalty', 'theme_diffusion_heating'
    ]
    col_meds = {k: v for k, v in medians.items() if not k.startswith('_')}
    res[feat_cols] = res[feat_cols].fillna(col_meds)
    return res, feat_cols

def verify():
    print("Verifying saved model persistence and reproducibility across ALL 4 FOLDS and 20 HEADS...")
    oof_df = pd.read_csv(REPAIR_DIR / "oof_repaired_predictions.csv", dtype={'code': str})
    raw_df = pd.read_csv(RAW_DATA_PATH, dtype={'code': str})
    models = joblib.load(REPAIR_DIR / "repaired_all_fold_models.joblib")
    medians = json.loads((REPAIR_DIR / "feature_medians.json").read_text(encoding='utf-8'))
    feat_cols_raw = json.loads((REPAIR_DIR / "feature_order.json").read_text(encoding='utf-8'))
    raw_cols = feat_cols_raw[:70] # original raw feature columns
    
    # Check that all 5 heads for all 4 folds exist (20 models)
    for fold, _, _ in FOLDS:
        for h, _ in HEADS:
            key = f"fold_{fold}_{h}"
            assert key in models, f"Missing model: {key}"
            assert key in medians, f"Missing medians: {key}"
            
    print("All 20 fold estimators present in repaired_all_fold_models.joblib.")
    
    max_global_diff = 0.0
    for fold, start, end in FOLDS:
        fold_test_raw = raw_df[(raw_df['date'] >= start) & (raw_df['date'] <= end)].sort_values(['date', 'code']).reset_index(drop=True)
        fold_oof = oof_df[oof_df['fold'] == fold].sort_values(['date', 'code']).reset_index(drop=True)
        assert len(fold_test_raw) == len(fold_oof), f"Row count mismatch in fold {fold}"
        
        for h, col in HEADS:
            clf = models[f'fold_{fold}_{h}']
            h_meds = medians[f'fold_{fold}_{h}']
            test_h, cols_h = transform_features_with_medians(fold_test_raw, h_meds, raw_cols)
            preds = clf.predict_proba(test_h[cols_h])[:, 1]
            diff = float(np.abs(preds - fold_oof[col].values).max())
            if diff > max_global_diff:
                max_global_diff = diff
            assert diff < 1e-12, f"Discrepancy detected for fold {fold} {h}: {diff}"
            print(f"Fold {fold} {h:6s} max diff: {diff:.2e}")
            
    print(f"\nAll 20 estimators verified! Global max difference: {max_global_diff:.2e} (< 1e-12)")
    print("Model persistence and reproducibility test PASSED 100% across all 4 folds!")

if __name__ == "__main__":
    verify()
