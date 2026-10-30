import sys
import joblib
import pandas as pd
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

EXP_DIR = Path('research/model_upgrade_validation/dragon_stage_leader_20261027T071500Z')
m_space = joblib.load(EXP_DIR / 'model_space_leader.joblib')
m_dragon = joblib.load(EXP_DIR / 'model_emotion_dragon.joblib')

features = pd.read_csv(EXP_DIR / 'stage_leader_dataset.csv').columns

# Inspect permutation or tree importances if available
from sklearn.inspection import permutation_importance
test_df = pd.read_csv(EXP_DIR / 'stage_leader_dataset.csv')

# Compute engineered features
med_price = test_df['price'].median()
med_cap = test_df['float_cap_log'].median()
test_df['is_low_price'] = (test_df['price'] <= med_price).astype(float)
test_df['is_small_cap'] = (test_df['float_cap_log'] <= med_cap).astype(float)
test_df['is_tight_amplitude'] = (test_df['amplitude'] <= 3.0).astype(float)
test_df['is_high_churn_risk'] = ((test_df['turnover'] > 0.18) & (test_df['one_word_proxy'] == 0)).astype(float)
test_df['is_crowded'] = (test_df['same_height_rivals'] >= 4).astype(float)
test_df['firm_seal_strength'] = (test_df['one_word_proxy'] * 2.0 + (test_df['amplitude'] <= 2.0).astype(float) - (test_df['amplitude'] > 8.0).astype(float))
test_df['price_cap_penalty'] = (test_df['price'] > 16.0).astype(float) + (test_df['float_cap_log'] > 23.0).astype(float)
test_df['healthy_turnover'] = ((test_df['turnover'] >= 0.05) & (test_df['turnover'] <= 0.18)).astype(float)

eval_df = test_df[test_df['date'] >= '2025-06-01']

print("Evaluating feature importance on Space Leader Model...")
raw_cols = [
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
    'market_max_board', 'market_max_board_count', 'consecutive_count', 'total_limit_up',
    'is_low_price', 'is_small_cap', 'is_tight_amplitude', 'is_high_churn_risk',
    'is_crowded', 'firm_seal_strength', 'price_cap_penalty', 'healthy_turnover'
]

# Calculate proxy importance using HGBT's internal properties if any, or permutation
r_space = permutation_importance(m_space, eval_df[raw_cols], eval_df['is_market_space_leader'], n_repeats=5, random_state=42)
top_space_idx = r_space.importances_mean.argsort()[::-1][:15]
print("\nTop 15 Features for STAGE MARKET SPACE LEADER:")
for idx in top_space_idx:
    print(f"  {raw_cols[idx]:<30}: {r_space.importances_mean[idx]:.4f} +/- {r_space.importances_std[idx]:.4f}")

r_dragon = permutation_importance(m_dragon, eval_df[raw_cols], eval_df['is_emotion_dragon_proxy'], n_repeats=5, random_state=42)
top_dragon_idx = r_dragon.importances_mean.argsort()[::-1][:15]
print("\nTop 15 Features for MARKET EMOTION DRAGON PROXY:")
for idx in top_dragon_idx:
    print(f"  {raw_cols[idx]:<30}: {r_dragon.importances_mean[idx]:.4f} +/- {r_dragon.importances_std[idx]:.4f}")
