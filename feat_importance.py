import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, brier_score_loss

df = pd.read_csv('data/v16/point_in_time_rebuild_cutoff_20260831_v2/leader_candidates.csv')
b3 = df[df['boards'] == 3].copy()
b3['date'] = pd.to_datetime(b3['date'])
b3 = b3.sort_values('date').reset_index(drop=True)
b3['target_5'] = (b3['terminal_board'] >= 5).astype(int)
b3['died_at_3'] = (b3['terminal_board'] == 3).astype(int)

# Check feature importances using RandomForest
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
    'state_start', 'state_warming', 'state_acceleration', 'state_consensus',
    'state_divergence', 'state_retreat', 'state_repair',
    'announcement_risk', 'reduction_risk', 'unlock_risk', 'earnings_risk'
]

# Clean NaNs
X = b3[features].fillna(b3[features].median())
y = b3['target_5']

rf = RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=15, random_state=42)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("=== TOP 20 FEATURE IMPORTANCES FOR REACHING 5 BOARDS ===")
print(importances.head(20))
