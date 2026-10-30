"""run_two_arm_experiment.py

Runs the frozen two-arm comparison:
Arm A: LIN_3TO5_CONTROL (L2 Regularized Logistic Regression)
Arm B: DRAGON_3TO5_CHALLENGER (Asymmetric Cost HistGradientBoosting)

Outputs full comparative metrics across all 4 temporal walk-forward folds.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parent
EXP_DIR = ROOT / "research/model_upgrade_validation/dragon_3to5_breakthrough_20261027T064500Z"
DATA_PATH = ROOT / "data/v16/point_in_time_rebuild_cutoff_20260831_v2/leader_candidates.csv"

def get_data():
    df = pd.read_csv(DATA_PATH)
    b3 = df[df['boards'] == 3].copy()
    b3['date'] = pd.to_datetime(b3['date'])
    b3 = b3.sort_values(['date', 'code']).reset_index(drop=True)
    b3['target_5'] = (b3['terminal_board'] >= 5).astype(int)
    b3['target_4'] = (b3['terminal_board'] >= 4).astype(int) # 3-to-4 success
    
    # Engineered features
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
    return b3, feature_cols


def evaluate_arm(oof_df, prob_col):
    auc = roc_auc_score(oof_df['target_5'], oof_df[prob_col])
    pr_auc = average_precision_score(oof_df['target_5'], oof_df[prob_col])
    brier = brier_score_loss(oof_df['target_5'], oof_df[prob_col])
    
    # Deciles / Tiers
    oof_df['tier'] = pd.qcut(oof_df[prob_col], 5, labels=['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4', 'Tier 5'])
    t_summary = oof_df.groupby('tier', observed=False)['target_5'].agg(
        count='count', reach_5='sum', rate='mean'
    ).reset_index().to_dict(orient='records')
    
    # Head to head competitive days
    comp = oof_df.groupby('date').filter(lambda g: len(g) >= 2)
    top1_5_hits = []
    top1_4_hits = []
    top2_5_hits = []
    bot_5_hits = []
    
    for d, g in comp.groupby('date'):
        sorted_g = g.sort_values(prob_col, ascending=False)
        top1 = sorted_g.iloc[0]
        top1_5_hits.append(int(top1['target_5']))
        top1_4_hits.append(int(top1['target_4']))
        
        top2 = sorted_g.iloc[:2]
        top2_5_hits.extend(top2['target_5'].tolist())
        
        bot = sorted_g.iloc[-1]
        bot_5_hits.append(int(bot['target_5']))
        
    return {
        "auc_roc": round(float(auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "brier": round(float(brier), 4),
        "tier1_rate": round(float(t_summary[0]['rate']), 4),
        "tier5_rate": round(float(t_summary[-1]['rate']), 4),
        "tier_spread": round(float(t_summary[-1]['rate'] - t_summary[0]['rate']), 4),
        "competitive_days": len(top1_5_hits),
        "top1_reach_5_rate": round(float(np.mean(top1_5_hits)), 4),
        "top1_reach_4_rate": round(float(np.mean(top1_4_hits)), 4),
        "top2_reach_5_rate": round(float(np.mean(top2_5_hits)), 4),
        "bottom1_reach_5_rate": round(float(np.mean(bot_5_hits)), 4),
        "top1_alpha_spread": round(float(np.mean(top1_5_hits) - comp['target_5'].mean()), 4),
        "separation_spread": round(float(np.mean(top1_5_hits) - np.mean(bot_5_hits)), 4),
        "tier_details": t_summary
    }


def main():
    b3, feature_cols = get_data()
    print(f"Total 3-board samples: {len(b3)}")
    
    folds = [
        ('2024-08-31', '2024-09-01', '2024-12-31'),
        ('2024-12-31', '2025-01-01', '2025-05-31'),
        ('2025-05-31', '2025-06-01', '2025-11-30'),
        ('2025-11-30', '2025-12-01', '2026-08-31'),
    ]
    
    oof_rows = []
    
    for fold_idx, (train_end, test_start, test_end) in enumerate(folds):
        train = b3[b3['date'] <= train_end].copy()
        test = b3[(b3['date'] >= test_start) & (b3['date'] <= test_end)].copy()
        
        # Arm A: Linear Control (Logistic Regression Pipeline)
        arm_a_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=0.25, class_weight={0: 1.0, 1: 1.5}, random_state=42, max_iter=2000))
        ])
        arm_a_pipe.fit(train[feature_cols], train['target_5'])
        test['prob_arm_a'] = arm_a_pipe.predict_proba(test[feature_cols])[:, 1]
        
        # Arm B: Challenger (HistGradientBoosting with Asymmetric Weighting)
        arm_b_model = HistGradientBoostingClassifier(
            max_iter=110,
            max_depth=4,
            min_samples_leaf=16,
            l2_regularization=2.5,
            learning_rate=0.04,
            class_weight={0: 1.0, 1: 1.5},
            random_state=100 + fold_idx
        )
        arm_b_model.fit(train[feature_cols], train['target_5'])
        test['prob_arm_b'] = arm_b_model.predict_proba(test[feature_cols])[:, 1]
        
        test['fold'] = fold_idx + 1
        oof_rows.append(test)
        
    full_oof = pd.concat(oof_rows, axis=0)
    
    res_a = evaluate_arm(full_oof.copy(), 'prob_arm_a')
    res_b = evaluate_arm(full_oof.copy(), 'prob_arm_b')
    
    print("\n" + "=" * 65)
    print("=== TWO-ARM FROZEN COMPARATIVE EXPERIMENT RESULTS ===")
    print(f"Total Out-of-Sample Cohort: {len(full_oof)}")
    print(f"Natural Base Rate (Reach >= 5): {full_oof['target_5'].mean():.4f}")
    print(f"Natural 3-to-4 Advance Rate:   {full_oof['target_4'].mean():.4f}")
    print("\n--- Arm A: LIN_3TO5_CONTROL (Linear Baseline) ---")
    print(f"  AUC-ROC: {res_a['auc_roc']}")
    print(f"  PR-AUC:  {res_a['pr_auc']}")
    print(f"  Tier 5 Rate: {res_a['tier5_rate']} vs Tier 1: {res_a['tier1_rate']} (Spread: {res_a['tier_spread']})")
    print(f"  Top-1 5-Board Hit Rate: {res_a['top1_reach_5_rate']} (Alpha: {res_a['top1_alpha_spread']:+})")
    print(f"  Top-1 3-to-4 Hit Rate:  {res_a['top1_reach_4_rate']}")
    print(f"  Top-1 vs Bottom-1 Separation: {res_a['separation_spread']:+}")
    
    print("\n--- Arm B: DRAGON_3TO5_CHALLENGER (Challenger Model) ---")
    print(f"  AUC-ROC: {res_b['auc_roc']} (Diff: {res_b['auc_roc'] - res_a['auc_roc']:+.4f})")
    print(f"  PR-AUC:  {res_b['pr_auc']} (Diff: {res_b['pr_auc'] - res_a['pr_auc']:+.4f})")
    print(f"  Tier 5 Rate: {res_b['tier5_rate']} vs Tier 1: {res_b['tier1_rate']} (Spread: {res_b['tier_spread']})")
    print(f"  Top-1 5-Board Hit Rate: {res_b['top1_reach_5_rate']} (Alpha: {res_b['top1_alpha_spread']:+})")
    print(f"  Top-1 3-to-4 Hit Rate:  {res_b['top1_reach_4_rate']}")
    print(f"  Top-1 vs Bottom-1 Separation: {res_b['separation_spread']:+}")
    
    # Save full comparative json
    comparison = {
        "timestamp": Path(__file__).stat().st_mtime,
        "sample_size": len(full_oof),
        "natural_base_rate_5": round(float(full_oof['target_5'].mean()), 4),
        "natural_base_rate_4": round(float(full_oof['target_4'].mean()), 4),
        "arm_a_control": res_a,
        "arm_b_challenger": res_b,
        "delta": {
            "auc_roc_gain": round(res_b['auc_roc'] - res_a['auc_roc'], 4),
            "pr_auc_gain": round(res_b['pr_auc'] - res_a['pr_auc'], 4),
            "top1_reach_5_gain": round(res_b['top1_reach_5_rate'] - res_a['top1_reach_5_rate'], 4),
            "top1_reach_4_gain": round(res_b['top1_reach_4_rate'] - res_a['top1_reach_4_rate'], 4),
            "separation_gain": round(res_b['separation_spread'] - res_a['separation_spread'], 4)
        }
    }
    
    with open(EXP_DIR / "two_arm_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
        
    print(f"\nComparative results saved to {EXP_DIR / 'two_arm_comparison.json'}")


if __name__ == "__main__":
    main()
