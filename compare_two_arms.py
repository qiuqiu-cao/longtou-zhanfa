"""compare_two_arms.py

Formal Two-Arm Comparison:
Arm A: BASE_CONTROL / BASELINE_HEURISTIC
Arm B: DRAGON_3TO5_CHALLENGER

Evaluates on the exact 4-Fold Temporal Walk-Forward Out-of-Sample set (1,480 samples, 2024-09 to 2026-08).
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parent
EXP_DIR = ROOT / "research/model_upgrade_validation/dragon_3to5_breakthrough_20261027T064500Z"
DATA_PATH = ROOT / "data/v16/point_in_time_rebuild_cutoff_20260831_v2/leader_candidates.csv"
BASE_MODEL_PATH = ROOT / "research/model_upgrade_validation/fixed_20261007T112017Z/models/relay/BASE/model.joblib"

# Load data
df = pd.read_csv(DATA_PATH)
b3 = df[df['boards'] == 3].copy()
b3['date'] = pd.to_datetime(b3['date'])
b3 = b3.sort_values(['date', 'code']).reset_index(drop=True)
b3['target_5'] = (b3['terminal_board'] >= 5).astype(int)
b3['target_4'] = (b3['terminal_board'] >= 4).astype(int)

# Load existing OOF predictions from DRAGON_3TO5_CHALLENGER
oof_b = pd.read_csv(EXP_DIR / "oof_predictions.csv")
oof_b['date'] = pd.to_datetime(oof_b['date'])

# Load BASE model
base_pipe = joblib.load(BASE_MODEL_PATH)
base_features = list(base_pipe.feature_names_in_)

# Compute missing BASE features
for k in ["price", "signal_amount", "turnover", "turnover_vs_prev"]:
    b3[k + "_same_height_percentile"] = b3.groupby(["date", "boards"])[k].rank(pct=True)
b3["signal_one_word"] = b3["one_word_proxy"]
b3["log_signal_amount"] = np.log1p(b3["signal_amount"].clip(lower=0))

# Filter rows matching oof_b
b3_oos = b3[b3['date'] >= '2024-09-01'].copy().reset_index(drop=True)
b3_oos = pd.merge(b3_oos, oof_b[['date', 'code', 'pred_prob', 'fold', 'score_tier']], on=['date', 'code'])
b3_oos.rename(columns={'pred_prob': 'score_arm_b'}, inplace=True)

# Generate Arm A1 (BASE model score)
# Note: BASE was trained on all heights predicting next day progression (target_next_limit)
b3_oos['score_base'] = base_pipe.predict_proba(b3_oos[base_features])[:, 1]

# Generate Arm A_linear (Walk-forward Logistic Regression trained specifically on target_5)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Engineered features matching Arm B
b3['is_low_price'] = (b3['price'] <= b3['price'].median()).astype(float)
b3['is_small_cap'] = (b3['float_cap_log'] <= b3['float_cap_log'].median()).astype(float)
b3['is_tight_amplitude'] = (b3['amplitude'] <= 3.0).astype(float)
b3['is_high_churn_risk'] = ((b3['turnover'] > 0.18) & (b3['one_word_proxy'] == 0)).astype(float)
b3['is_crowded'] = (b3['same_height_rivals'] >= 4).astype(float)
b3['firm_seal_strength'] = (b3['one_word_proxy'] * 2.0 + (b3['amplitude'] <= 2.0).astype(float) - (b3['amplitude'] > 8.0).astype(float))
b3['price_cap_penalty'] = (b3['price'] > 16.0).astype(float) + (b3['float_cap_log'] > 23.0).astype(float)
b3['healthy_turnover'] = ((b3['turnover'] >= 0.05) & (b3['turnover'] <= 0.18)).astype(float)

feature_cols = [
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
    'is_low_price', 'is_small_cap', 'is_tight_amplitude', 'is_high_churn_risk',
    'is_crowded', 'firm_seal_strength', 'price_cap_penalty', 'healthy_turnover'
]

folds = [
    ('2024-08-31', '2024-09-01', '2024-12-31'),
    ('2024-12-31', '2025-01-01', '2025-05-31'),
    ('2025-05-31', '2025-06-01', '2025-11-30'),
    ('2025-11-30', '2025-12-01', '2026-08-31'),
]

arm_a_lin_scores = []
for fold_idx, (train_end, test_start, test_end) in enumerate(folds):
    train = b3[b3['date'] <= train_end].copy()
    test = b3[(b3['date'] >= test_start) & (b3['date'] <= test_end)].copy()
    
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=0.1, class_weight='balanced', random_state=42))
    ])
    pipe.fit(train[feature_cols], train['target_5'])
    test['score_arm_a_lin'] = pipe.predict_proba(test[feature_cols])[:, 1]
    arm_a_lin_scores.append(test[['date', 'code', 'score_arm_a_lin']])

arm_a_df = pd.concat(arm_a_lin_scores, axis=0)
b3_oos = pd.merge(b3_oos, arm_a_df, on=['date', 'code'])

print(f"Total OOS evaluation candidates: {len(b3_oos)}")
base_target5_rate = b3_oos['target_5'].mean()
base_target4_rate = b3_oos['target_4'].mean()
print(f"OOS Base rate 3->4: {base_target4_rate:.4f} ({b3_oos['target_4'].sum()}/{len(b3_oos)})")
print(f"OOS Base rate 3->5: {base_target5_rate:.4f} ({b3_oos['target_5'].sum()}/{len(b3_oos)})")

# Evaluation metrics for:
# 1. BASE Model (Arm A_BASE)
# 2. Linear Model (Arm A_LIN)
# 3. Challenger Model (Arm B_CHALLENGER)

def evaluate_predictions(df, score_col, target_col='target_5'):
    auc = roc_auc_score(df[target_col], df[score_col])
    pr_auc = average_precision_score(df[target_col], df[score_col])
    brier = brier_score_loss(df[target_col], df[score_col])
    
    # Competition metrics
    comp = df.groupby('date').filter(lambda g: len(g) >= 2)
    top1_5_hits = []
    top2_5_hits = []
    top1_4_hits = []
    bottom1_5_hits = []
    
    for _, g in comp.groupby('date'):
        g_sorted = g.sort_values(score_col, ascending=False)
        top1_5_hits.append(g_sorted.iloc[0]['target_5'])
        top1_4_hits.append(g_sorted.iloc[0]['target_4'])
        top2_5_hits.extend(g_sorted.iloc[:2]['target_5'].tolist())
        bottom1_5_hits.append(g_sorted.iloc[-1]['target_5'])
        
    return {
        "auc_roc": round(float(auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "brier": round(float(brier), 4),
        "comp_days": len(top1_5_hits),
        "comp_baseline_5": round(float(comp['target_5'].mean()), 4),
        "top1_target5_hit_rate": round(float(np.mean(top1_5_hits)), 4),
        "top1_target4_hit_rate": round(float(np.mean(top1_4_hits)), 4),
        "top2_target5_hit_rate": round(float(np.mean(top2_5_hits)), 4),
        "bottom1_target5_hit_rate": round(float(np.mean(bottom1_5_hits)), 4),
        "alpha_over_comp_base": round(float(np.mean(top1_5_hits) - comp['target_5'].mean()), 4),
        "top1_bottom1_spread": round(float(np.mean(top1_5_hits) - np.mean(bottom1_5_hits)), 4)
    }

metrics_base = evaluate_predictions(b3_oos, 'score_base')
metrics_arm_a = evaluate_predictions(b3_oos, 'score_arm_a_lin')
metrics_arm_b = evaluate_predictions(b3_oos, 'score_arm_b')

print("\n" + "=" * 70)
print(f"{'Metric':<30} | {'BASE (Arm A1)':<12} | {'Linear (Arm A2)':<15} | {'Challenger (Arm B)':<15}")
print("-" * 70)
for k in ["auc_roc", "pr_auc", "top1_target5_hit_rate", "top1_target4_hit_rate", "top2_target5_hit_rate", "bottom1_target5_hit_rate", "alpha_over_comp_base", "top1_bottom1_spread"]:
    print(f"{k:<30} | {metrics_base[k]:<12} | {metrics_arm_a[k]:<15} | {metrics_arm_b[k]:<15}")
print("=" * 70)

# Save results
out_data = {
    "evaluation_samples": len(b3_oos),
    "date_range": "2024-09-01 to 2026-08-31",
    "base_target4_rate": round(float(base_target4_rate), 4),
    "base_target5_rate": round(float(base_target5_rate), 4),
    "arm_a_base": metrics_base,
    "arm_a_linear": metrics_arm_a,
    "arm_b_challenger": metrics_arm_b
}

with open(EXP_DIR / "two_arm_comparison.json", "w", encoding="utf-8") as f:
    json.dump(out_data, f, ensure_ascii=False, indent=2)

print(f"\nTwo-arm comparison saved to {EXP_DIR / 'two_arm_comparison.json'}")
