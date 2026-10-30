import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, average_precision_score

df = pd.read_csv('data/v16/point_in_time_rebuild_cutoff_20260831_v2/leader_candidates.csv')
b3 = df[df['boards'] == 3].copy()
b3['date'] = pd.to_datetime(b3['date'])
b3 = b3.sort_values('date').reset_index(drop=True)
b3['target_5'] = (b3['terminal_board'] >= 5).astype(int)

# 1. Feature Engineering
b3['log_float_cap'] = b3['float_cap_log']
b3['is_low_price'] = (b3['price'] <= b3['price'].median()).astype(float)
b3['is_small_cap'] = (b3['float_cap_log'] <= b3['float_cap_log'].median()).astype(float)
b3['is_tight_amplitude'] = (b3['amplitude'] <= 3.0).astype(float)
b3['is_high_churn_risk'] = ((b3['turnover'] > 0.18) & (b3['one_word_proxy'] == 0)).astype(float)
b3['is_crowded'] = (b3['same_height_rivals'] >= 4).astype(float)
b3['firm_seal_strength'] = (b3['one_word_proxy'] * 2.0 + (b3['amplitude'] <= 2.0).astype(float) - (b3['amplitude'] > 8.0).astype(float))
b3['price_cap_penalty'] = (b3['price'] > 15.0).astype(float) + (b3['float_cap_log'] > 23.0).astype(float)

features = [
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
    'is_crowded', 'firm_seal_strength', 'price_cap_penalty'
]

# 2. Strict Walk-Forward Evaluation (Expanding Window 4-Fold CV)
# Fold 1: Train <= 2024-08, Test 2024-09 to 2024-12
# Fold 2: Train <= 2024-12, Test 2025-01 to 2025-05
# Fold 3: Train <= 2025-05, Test 2025-06 to 2025-11
# Fold 4: Train <= 2025-11, Test 2025-12 to 2026-08

folds = [
    ('2024-08-31', '2024-09-01', '2024-12-31'),
    ('2024-12-31', '2025-01-01', '2025-05-31'),
    ('2025-05-31', '2025-06-01', '2025-11-30'),
    ('2025-11-30', '2025-12-01', '2026-08-31'),
]

all_oof = []

for fold_idx, (train_end, test_start, test_end) in enumerate(folds):
    train = b3[b3['date'] <= train_end].copy()
    test = b3[(b3['date'] >= test_start) & (b3['date'] <= test_end)].copy()
    
    # Model
    model = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=4,
        min_samples_leaf=15,
        l2_regularization=3.0,
        learning_rate=0.04,
        class_weight={0: 1.0, 1: 1.5}, # Asymmetric penalty: reward finding 5-board dragons
        random_state=42 + fold_idx
    )
    model.fit(train[features], train['target_5'])
    
    test['pred_prob'] = model.predict_proba(test[features])[:, 1]
    test['fold'] = fold_idx + 1
    all_oof.append(test)

oof_df = pd.concat(all_oof, axis=0)

print("=== OUT-OF-SAMPLE WALK-FORWARD BACKTEST RESULTS ===")
print(f"Total Out-of-Sample 3-Board Candidates: {len(oof_df)}")
overall_base_rate = oof_df['target_5'].mean()
print(f"Overall Natural Base Rate (Reach >= 5): {overall_base_rate:.4f} ({oof_df['target_5'].sum()}/{len(oof_df)})")

overall_auc = roc_auc_score(oof_df['target_5'], oof_df['pred_prob'])
overall_pr_auc = average_precision_score(oof_df['target_5'], oof_df['pred_prob'])
print(f"Overall OOS AUC-ROC: {overall_auc:.4f}")
print(f"Overall OOS PR-AUC: {overall_pr_auc:.4f} vs Baseline: {overall_base_rate:.4f}")

# Threshold analysis
print("\n--- Precision by Score Deciles / Tiers ---")
oof_df['score_tier'] = pd.qcut(oof_df['pred_prob'], 5, labels=['Tier 1(Lowest)', 'Tier 2', 'Tier 3', 'Tier 4', 'Tier 5(Highest)'])
print(oof_df.groupby('score_tier', observed=False)['target_5'].agg(['count', 'sum', 'mean']))

# Competitive Days: When multiple 3-board stocks exist on the same day
comp_days = oof_df.groupby('date').filter(lambda g: len(g) >= 2)
top1_list = []
top2_list = []
bottom1_list = []

for d, g in comp_days.groupby('date'):
    sorted_g = g.sort_values('pred_prob', ascending=False)
    top1 = sorted_g.iloc[0]
    top1_list.append(top1['target_5'])
    
    top2 = sorted_g.iloc[:2]
    top2_list.extend(top2['target_5'].tolist())
    
    bottom1 = sorted_g.iloc[-1]
    bottom1_list.append(bottom1['target_5'])

print("\n--- Head-to-Head Daily Selection Backtest (Competitive Days) ---")
print(f"Competitive Trading Days: {len(top1_list)}")
print(f"Baseline Hit Rate on Competitive Days: {comp_days['target_5'].mean():.4f}")
print(f"Model Top-1 Pick 5-Board Hit Rate: {np.mean(top1_list):.4f} ({sum(top1_list)}/{len(top1_list)})")
print(f"Model Top-2 Pick 5-Board Hit Rate: {np.mean(top2_list):.4f} ({sum(top2_list)}/{len(top2_list)})")
print(f"Model Bottom-1 (Lowest Scored) 5-Board Rate: {np.mean(bottom1_list):.4f} ({sum(bottom1_list)}/{len(bottom1_list)})")
print(f"Top-1 vs Bottom-1 Separation Spread: +{(np.mean(top1_list) - np.mean(bottom1_list))*100:.2f} percentage points!")
print(f"Top-1 vs Baseline Alpha Spread: +{(np.mean(top1_list) - comp_days['target_5'].mean())*100:.2f} percentage points!")
