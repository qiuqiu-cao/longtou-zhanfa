import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

df = pd.read_csv('data/v16/point_in_time_rebuild_cutoff_20260831_v2/leader_candidates.csv')
b3 = df[df['boards'] == 3].copy()
b3['date'] = pd.to_datetime(b3['date'])
b3 = b3.sort_values('date').reset_index(drop=True)
b3['target_5'] = (b3['terminal_board'] >= 5).astype(int)

# Hard exclusion rule based on empirical death zones:
# 1. Large float cap (top 15% largest caps)
# 2. Extreme high price (> 20 yuan)
# 3. High churn (> 22% turnover without space leadership)
# 4. Chaotic amplitude (> 10%)
cap_threshold = b3['float_cap_log'].quantile(0.85)
b3['hard_veto_filter'] = (
    (b3['float_cap_log'] > cap_threshold) |
    (b3['price'] > 20.0) |
    ((b3['turnover'] > 0.22) & (b3['is_space_leader'] == 0)) |
    (b3['amplitude'] > 10.0)
).astype(int)

print(f"Total 3-board rows: {len(b3)}")
print("Hard Vetoed rows:", b3['hard_veto_filter'].sum())
print("Success rate of hard-vetoed rows:", b3[b3['hard_veto_filter'] == 1]['target_5'].mean())
print("Success rate of non-vetoed rows:", b3[b3['hard_veto_filter'] == 0]['target_5'].mean())

# Feature engineering
b3['log_float_cap'] = b3['float_cap_log']
b3['is_low_price'] = (b3['price'] <= b3['price'].median()).astype(float)
b3['is_small_cap'] = (b3['float_cap_log'] <= b3['float_cap_log'].median()).astype(float)
b3['is_tight_amplitude'] = (b3['amplitude'] <= 3.0).astype(float)
b3['firm_seal_strength'] = (b3['one_word_proxy'] * 2.0 + (b3['amplitude'] <= 2.0).astype(float) - (b3['amplitude'] > 7.0).astype(float))

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
    'is_low_price', 'is_small_cap', 'is_tight_amplitude', 'firm_seal_strength'
]

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
    
    # Train on non-vetoed data or full data with weight
    sample_weights = np.where(train['hard_veto_filter'] == 1, 0.5, 1.0)
    
    model = HistGradientBoostingClassifier(
        max_iter=120,
        max_depth=4,
        min_samples_leaf=15,
        l2_regularization=2.5,
        learning_rate=0.04,
        class_weight={0: 1.0, 1: 1.6},
        random_state=42 + fold_idx
    )
    model.fit(train[features], train['target_5'], sample_weight=sample_weights)
    
    test['pred_prob'] = model.predict_proba(test[features])[:, 1]
    # Penalize hard-vetoed candidates in test score
    test['adjusted_score'] = np.where(test['hard_veto_filter'] == 1, test['pred_prob'] * 0.4, test['pred_prob'])
    all_oof.append(test)

res = pd.concat(all_oof, axis=0)

comp_days = res.groupby('date').filter(lambda g: len(g) >= 2)
top1_list = []
top2_list = []

for d, g in comp_days.groupby('date'):
    sorted_g = g.sort_values('adjusted_score', ascending=False)
    top1 = sorted_g.iloc[0]
    top1_list.append(top1['target_5'])
    top2 = sorted_g.iloc[:2]
    top2_list.extend(top2['target_5'].tolist())

print(f"\nCompetitive Days: {len(top1_list)}")
print(f"Baseline on Competitive Days: {comp_days['target_5'].mean():.4f}")
print(f"Top-1 Pick 5-Board Hit Rate: {np.mean(top1_list):.4f} ({sum(top1_list)}/{len(top1_list)})")
print(f"Top-2 Pick 5-Board Hit Rate: {np.mean(top2_list):.4f} ({sum(top2_list)}/{len(top2_list)})")
print(f"Alpha over Baseline: +{(np.mean(top1_list) - comp_days['target_5'].mean())*100:.2f} percentage points!")

# Precision when Top-1 score > 0.45
confident_top1 = []
for d, g in comp_days.groupby('date'):
    sorted_g = g.sort_values('adjusted_score', ascending=False)
    top1 = sorted_g.iloc[0]
    if top1['adjusted_score'] >= 0.40:
        confident_top1.append(top1['target_5'])

print(f"\nHigh-Confidence Top-1 Picks (Score >= 0.40): {len(confident_top1)} days")
print(f"High-Confidence Top-1 5-Board Hit Rate: {np.mean(confident_top1):.4f} ({sum(confident_top1)}/{len(confident_top1)})")
print(f"High-Confidence Alpha: +{(np.mean(confident_top1) - comp_days['target_5'].mean())*100:.2f} percentage points!")
