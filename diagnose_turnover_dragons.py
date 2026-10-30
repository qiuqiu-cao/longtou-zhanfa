"""
diagnose_turnover_dragons.py:
Deep dive into non-one-word (换手板) 3-board stocks with auction open in +2.0% ~ +9.0%.
Why do some succeed (True Dragons like 深圳华强) while others collapse (False Pumps)?
"""

import numpy as np
import pandas as pd

oof = pd.read_csv('research/model_upgrade_validation/dragon_auction_and_second_wave_20261027T091600Z/oof_round6_predictions.csv')
daily = pd.read_csv('research/model_upgrade_validation/dragon_auction_and_second_wave_20261027T091600Z/daily_competition_round6.csv')

oof['code_str'] = oof['code'].astype(str).str.zfill(6)
oof['date_str'] = pd.to_datetime(oof['date']).dt.strftime('%Y-%m-%d')
daily['code_str'] = daily['top1_code'].astype(str).str.zfill(6)
daily['date_str'] = pd.to_datetime(daily['date']).dt.strftime('%Y-%m-%d')

merged = daily.merge(oof, left_on=['date_str', 'code_str'], right_on=['date_str', 'code_str'], suffixes=('', '_oof'))

# Filter to non-one-word open (+2.0% <= open_gap < +9.2%)
turnover_pool = merged[(merged['top1_open_gap'] >= 0.02) & (merged['top1_open_gap'] < 0.092)].copy()

print(f"Total Non-One-Word Open Top-1 (+2%~+9.2%): {len(turnover_pool)}")
succ = turnover_pool[turnover_pool['top1_is_3to4'] == 1]
fail = turnover_pool[turnover_pool['top1_is_3to4'] == 0]
print(f"Successes: {len(succ)} ({len(succ)/len(turnover_pool):.1%})")
print(f"Failures: {len(fail)} ({len(fail)/len(turnover_pool):.1%})")

# Compare features between success and failure in this turnover pool
feature_candidates = [
    'top1_score', 'prob_regime_dragon', 'relative_strength', 'amplitude',
    'turnover', 'turnover_vs_prev', 'amount_float_cap', 'float_cap_log',
    'theme_cluster_total_zt', 'theme_first_board_count', 'theme_zt_share',
    'same_height_rivals', 'emotion_score', 'promotion_3to4', 'space_feedback'
]

print("\n--- Feature Comparison: Success vs Failure in Turnover Group ---")
comp = []
for f in feature_candidates:
    if f in turnover_pool.columns:
        mean_s = succ[f].mean()
        mean_f = fail[f].mean()
        diff = mean_s - mean_f
        comp.append({
            'feature': f,
            'success_mean': mean_s,
            'failure_mean': mean_f,
            'diff': diff,
            'pct_diff': diff / (abs(mean_f) + 1e-6)
        })
comp_df = pd.DataFrame(comp).sort_values('pct_diff', ascending=False)
print(comp_df.to_string())

# Test multi-factor rule for turnover group:
print("\n--- Testing Combined Filters on Turnover Group ---")
# Filter A: Score >= 0.25
sub_a = turnover_pool[turnover_pool['top1_score'] >= 0.25]
print(f"Filter A (Score >= 0.25, N={len(sub_a)}): 3进4胜率={sub_a['top1_is_3to4'].mean():.1%}, 空间高标={sub_a['top1_is_space_leader'].mean():.1%}")

# Filter B: Regime >= 0.55 + Score >= 0.25
sub_b = turnover_pool[(turnover_pool['prob_regime_dragon'] >= 0.55) & (turnover_pool['top1_score'] >= 0.25)]
print(f"Filter B (Regime >= 0.55 + Score >= 0.25, N={len(sub_b)}): 3进4胜率={sub_b['top1_is_3to4'].mean():.1%}, 空间高标={sub_b['top1_is_space_leader'].mean():.1%}")

# Filter C: Regime >= 0.55 + Score >= 0.25 + Healthy Turnover (turnover <= 0.20)
sub_c = turnover_pool[(turnover_pool['prob_regime_dragon'] >= 0.55) & (turnover_pool['top1_score'] >= 0.25) & (turnover_pool['turnover'] <= 0.20)]
print(f"Filter C (+ Turnover <= 20%, N={len(sub_c)}): 3进4胜率={sub_c['top1_is_3to4'].mean():.1%}, 空间高标={sub_c['top1_is_space_leader'].mean():.1%}")

# Filter D: Golden Master for Turnover Dragons: Regime >= 0.55 + Score >= 0.30 + Turnover <= 0.20
sub_d = turnover_pool[(turnover_pool['prob_regime_dragon'] >= 0.55) & (turnover_pool['top1_score'] >= 0.30) & (turnover_pool['turnover'] <= 0.20)]
print(f"Filter D (Master Turnover: Regime>=0.55 + Score>=0.30 + Churn<=20%, N={len(sub_d)}): 3进4胜率={sub_d['top1_is_3to4'].mean():.1%}, 空间高标={sub_d['top1_is_space_leader'].mean():.1%}")
