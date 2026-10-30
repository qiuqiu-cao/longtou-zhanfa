import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, brier_score_loss

df = pd.read_csv('data/v16/point_in_time_rebuild_cutoff_20260831_v2/leader_candidates.csv')
b3 = df[df['boards'] == 3].copy()
b3['date'] = pd.to_datetime(b3['date'])
b3 = b3.sort_values('date').reset_index(drop=True)
b3['target_5'] = (b3['terminal_board'] >= 5).astype(int)

# Feature selection
base_features = [
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
    'state_start', 'state_warming', 'state_acceleration', 'state_consensus',
    'state_divergence', 'state_retreat', 'state_repair',
    'announcement_risk', 'reduction_risk', 'unlock_risk', 'earnings_risk'
]

# Add engineered features
b3['is_crowded_cohort'] = (b3['same_height_rivals'] >= 3).astype(float)
b3['is_space_isolated'] = (b3['is_space_leader'] * b3['unique_space']).astype(float)
b3['healthy_turnover'] = ((b3['turnover'] >= 0.08) & (b3['turnover'] <= 0.22)).astype(float)
b3['turnover_surge'] = (b3['turnover_vs_prev'] > 2.5).astype(float)
b3['risk_composite'] = (b3['announcement_risk'] + b3['earnings_risk'] + b3['reduction_risk']).astype(float)

all_features = base_features + ['is_crowded_cohort', 'is_space_isolated', 'healthy_turnover', 'turnover_surge', 'risk_composite']

# Train/Test split: Train on first 60% of dates, test on last 40% (temporal split)
dates = b3['date'].drop_duplicates().sort_values()
split_idx = int(len(dates) * 0.6)
split_date = dates.iloc[split_idx]

train = b3[b3['date'] < split_date].copy()
test = b3[b3['date'] >= split_date].copy()

t_min = train['date'].min().strftime('%Y-%m-%d')
t_max = train['date'].max().strftime('%Y-%m-%d')
te_min = test['date'].min().strftime('%Y-%m-%d')
te_max = test['date'].max().strftime('%Y-%m-%d')

print(f"Train period: {t_min} to {t_max}, N={len(train)}, reach_5={train['target_5'].mean():.3f}")
print(f"Test period:  {te_min} to {te_max}, N={len(test)}, reach_5={test['target_5'].mean():.3f}")

clf = HistGradientBoostingClassifier(
    max_iter=120, 
    max_depth=4, 
    min_samples_leaf=20, 
    l2_regularization=2.0, 
    learning_rate=0.05,
    random_state=42
)
clf.fit(train[all_features], train['target_5'])

test['pred_prob'] = clf.predict_proba(test[all_features])[:, 1]
test['pred_binary'] = (test['pred_prob'] >= 0.30).astype(int)

auc = roc_auc_score(test['target_5'], test['pred_prob'])
prec = precision_score(test['target_5'], test['pred_binary'])
rec = recall_score(test['target_5'], test['pred_binary'])

print(f"Test AUC-ROC: {auc:.4f}")
print(f"Test Precision (prob >= 0.30): {prec:.4f} vs Baseline: {test['target_5'].mean():.4f}")
print(f"Test Recall: {rec:.4f}")

# Check Top-1 Precision on competitive dates with >= 2 candidates at 3 boards
mult_dates = test.groupby('date').filter(lambda g: len(g) >= 2)
top1_hits = []
for d, g in mult_dates.groupby('date'):
    top1 = g.sort_values('pred_prob', ascending=False).iloc[0]
    top1_hits.append(top1['target_5'])

baseline_mult = mult_dates['target_5'].mean()
top1_prec = np.mean(top1_hits)
print(f"Competitive 3-board days: {len(top1_hits)}")
print(f"Top-1 Precision on competitive days: {top1_prec:.4f} ({sum(top1_hits)}/{len(top1_hits)})")
print(f"Baseline rate on competitive days: {baseline_mult:.4f}")
print(f"Improvement over baseline: +{(top1_prec - baseline_mult)*100:.2f} percentage points!")
