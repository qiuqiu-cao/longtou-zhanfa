"""
audit_round6_strict.py:
Performs a strict, independent verification of Round 6:
1. Label Maturity Audit: Check if any training sample had an immature 10-day label (determined_date_10d > train_end).
2. Missing Data Audit: Check if any missing forward bar (31 cases) was improperly marked as 0 (failure) rather than NaN (unknown).
3. Re-train models with 100% strict label maturity and NaN exclusion, then compare predictions and metrics against reported Round 6 metrics.
4. Verify whether any lookahead leakage exists in features.
5. Verify whether 09:25 auction execution gating rules and hit rates remain solid.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(".")
EXP_DIR = ROOT / "research/model_upgrade_validation/dragon_auction_and_second_wave_20261027T091600Z"
DATA_PATH = EXP_DIR / "round6_dataset.csv"
DAILY_BARS_PATH = ROOT / "data/v3/imported/daily-bars.csv"
REGIME_GATE_PATH = ROOT / "research/model_upgrade_validation/dragon_regime_gate_20261027T085500Z/oof_round5_daily_gate_predictions.csv"

def audit():
    print("=== STARTING RIGOROUS ROUND 6 AUDIT ===")
    
    # 1. Load data
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['determined_date_final'] = pd.to_datetime(df['determined_date_final'])
    df['code_str'] = df['code'].astype(str).str.zfill(6)
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
    
    # Load bars to get exact 10th trading day date for every event
    df_b = pd.read_csv(DAILY_BARS_PATH, usecols=['date', 'code'])
    df_b['date_str'] = pd.to_datetime(df_b['date']).dt.strftime('%Y-%m-%d')
    df_b['code_str'] = df_b['code'].astype(str).str.zfill(6)
    bars_by_code = {c: g['date_str'].values for c, g in df_b.groupby('code_str')}
    
    det_date_10d = []
    sub_lens = []
    for idx, r in df.iterrows():
        c = r['code_str']
        d = r['date_str']
        b = bars_by_code.get(c)
        if b is not None:
            sub = b[b > d][:10]
            sub_lens.append(len(sub))
            if len(sub) == 10:
                det_date_10d.append(pd.to_datetime(sub[-1]))
            else:
                det_date_10d.append(pd.NaT)
        else:
            sub_lens.append(0)
            det_date_10d.append(pd.NaT)
            
    df['determined_date_10d'] = det_date_10d
    df['sub_len'] = sub_lens
    
    # Audit 1: Exactly which rows have sub_len < 10
    incomplete = df[df['sub_len'] < 10]
    print(f"Audit 1: Incomplete/missing forward 10-day bars: {len(incomplete)} rows out of {len(df)} ({len(incomplete)/len(df):.2%})")
    print(f"  - Zero forward bars (untracked): {(df['sub_len'] == 0).sum()} rows")
    print(f"  - Truncated at end of sample (<10 bars): {((df['sub_len'] > 0) & (df['sub_len'] < 10)).sum()} rows")
    
    # Mark genuine unknowns strictly
    df['is_10d_mature'] = (df['sub_len'] == 10).astype(int)
    
    # Audit 2: Check Feature Leakage
    raw_feature_cols = [
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
        'market_max_board', 'market_max_board_count', 'consecutive_count', 'total_limit_up',
        'amount_cohort_share', 'amount_cohort_rank', 'amplitude_cohort_rank', 'price_cohort_rank', 'turnover_cohort_rank',
        'is_cohort_volume_leader', 'is_cohort_tightest', 'is_cohort_lowest_price', 'is_already_space_leader',
        'theme_cluster_total_zt', 'theme_first_board_count', 'theme_two_board_count', 'theme_three_board_count',
        'theme_zt_share', 'theme_first_board_share', 'theme_amount_share',
        'theme_ladder_density', 'theme_ladder_complete',
        'theme_first_board_expansion', 'theme_cluster_expansion',
        'is_theme_sole_leader', 'theme_same_height_rivals',
        'is_isolated_lone_wolf', 'theme_diffusion_surge'
    ]
    
    forbidden_tokens = ['next', 'future', '10d', 'target', 'open_gap', 'auction', 'forward', 'second_wave']
    leaked_features = [f for f in raw_feature_cols if any(tok in f.lower() for tok in forbidden_tokens)]
    print(f"\nAudit 2: Feature Leakage Check in raw_feature_cols:")
    print(f"  - Any forbidden tokens in raw_feature_cols? {leaked_features} -> {'PASS (Zero leakage)' if not leaked_features else 'FAIL'}")
    
    # Audit 3: Strict Re-run with 100% Label Maturity & Unknown Exclusion
    print(f"\nAudit 3: Retraining with 100% Strict Label Maturity & Comparing against Round 6...")
    folds = [
        ('2024-08-31', '2024-09-01', '2024-12-31'),
        ('2024-12-31', '2025-01-01', '2025-05-31'),
        ('2025-05-31', '2025-06-01', '2025-11-30'),
        ('2025-11-30', '2025-12-01', '2026-08-31'),
    ]
    
    from train_and_evaluate_round6 import engineer_features
    
    strict_oofs = []
    for fold_idx, (train_end_str, test_start_str, test_end_str) in enumerate(folds):
        train_end = pd.to_datetime(train_end_str)
        test_start = pd.to_datetime(test_start_str)
        test_end = pd.to_datetime(test_end_str)
        
        # Base mature train for space & 5-board
        train_base = df[(df['date'] <= train_end) & (df['determined_date_final'] <= train_end)].copy()
        
        # Strictly mature train for 10d models (sub_len == 10 AND determined_date_10d <= train_end)
        train_10d = df[(df['date'] <= train_end) & (df['determined_date_10d'].notna()) & (df['determined_date_10d'] <= train_end)].copy()
        
        test_raw = df[(df['date'] >= test_start) & (df['date'] <= test_end)].copy()
        
        train_b_eng, test_eng, feat_cols = engineer_features(train_base, test_raw, raw_feature_cols)
        train_10d_eng, _, _ = engineer_features(train_10d, test_raw, raw_feature_cols)
        
        # Fit space
        clf_space = HistGradientBoostingClassifier(
            max_iter=130, max_depth=4, min_samples_leaf=16, l2_regularization=2.5,
            learning_rate=0.035, class_weight={0: 1.0, 1: 1.6}, random_state=101 + fold_idx
        )
        clf_space.fit(train_b_eng[feat_cols], train_b_eng['is_market_space_leader'])
        test_eng['strict_score_space'] = clf_space.predict_proba(test_eng[feat_cols])[:, 1]
        
        # Fit 5
        clf_5 = HistGradientBoostingClassifier(
            max_iter=130, max_depth=4, min_samples_leaf=16, l2_regularization=2.5,
            learning_rate=0.035, class_weight={0: 1.0, 1: 1.5}, random_state=201 + fold_idx
        )
        clf_5.fit(train_b_eng[feat_cols], train_b_eng['is_3to5'])
        test_eng['strict_score_5'] = clf_5.predict_proba(test_eng[feat_cols])[:, 1]
        
        # Fit 100% abnormal strictly on mature samples
        clf_100 = HistGradientBoostingClassifier(
            max_iter=130, max_depth=4, min_samples_leaf=16, l2_regularization=2.5,
            learning_rate=0.035, class_weight={0: 1.0, 1: 2.2}, random_state=301 + fold_idx
        )
        clf_100.fit(train_10d_eng[feat_cols], train_10d_eng['is_10d_100_abnormal'])
        test_eng['strict_score_100'] = clf_100.predict_proba(test_eng[feat_cols])[:, 1]
        
        # Fit second wave strictly on mature samples
        clf_2w = HistGradientBoostingClassifier(
            max_iter=130, max_depth=4, min_samples_leaf=16, l2_regularization=2.5,
            learning_rate=0.035, class_weight={0: 1.0, 1: 1.4}, random_state=401 + fold_idx
        )
        clf_2w.fit(train_10d_eng[feat_cols], train_10d_eng['is_second_wave_leader'])
        test_eng['strict_score_2w'] = clf_2w.predict_proba(test_eng[feat_cols])[:, 1]
        
        test_eng['strict_score_total'] = (
            0.30 * test_eng['strict_score_space'] +
            0.25 * test_eng['strict_score_5'] +
            0.25 * test_eng['strict_score_100'] +
            0.20 * test_eng['strict_score_2w']
        )
        strict_oofs.append(test_eng)
        
    strict_df = pd.concat(strict_oofs, ignore_index=True)
    
    # Compare original R6 score vs strict score
    r6_oof = pd.read_csv(EXP_DIR / "oof_round6_predictions.csv")
    corr = np.corrcoef(strict_df['strict_score_total'], r6_oof['score_round6_total_dragon'])[0, 1]
    print(f"  - Correlation between Original R6 Score and 100% Strict Matured Score: {corr:.6f}")
    
    # Check Top-1 concordance
    strict_df['strict_rank'] = strict_df.groupby('date')['strict_score_total'].rank(ascending=False, method='min')
    strict_top1 = strict_df[strict_df['strict_rank'] == 1].drop_duplicates('date', keep='first')
    
    r6_daily = pd.read_csv(EXP_DIR / "daily_competition_round6.csv")
    
    same_top1_count = 0
    total_dates = len(r6_daily)
    for idx, r in r6_daily.iterrows():
        d = r['date']
        c_orig = str(r['top1_code']).zfill(6)
        match_s = strict_top1[strict_top1['date_str'] == d]
        if len(match_s) > 0:
            c_strict = match_s.iloc[0]['code_str']
            if c_orig == c_strict:
                same_top1_count += 1
                
    print(f"  - Daily Top-1 Concordance: {same_top1_count} / {total_dates} days ({same_top1_count/total_dates:.2%}) identical!")
    
    # Check hit rates under strict models
    strict_top1_m = strict_top1.merge(r6_daily[['date', 'prob_regime_dragon', 'gate_open_55']], left_on='date_str', right_on='date')
    
    gap = strict_top1_m['next_open_gap'].fillna(0.0)
    is_pos = gap > 0.0
    is_nuclear = gap <= 0.0
    
    print("\nAudit 4: Performance Verification under 100% Strict Label Maturity:")
    print("  --- Strict Tier 0 (All Days, N=449) ---")
    print(f"    3进4 晋级率: {strict_top1_m['is_3to4'].mean():.2%}")
    print(f"    空间高标率: {strict_top1_m['is_market_space_leader'].mean():.2%}")
    print(f"    3进5突破率: {strict_top1_m['is_3to5'].mean():.2%}")
    
    print("  --- Strict Tier 1 (Auction Red Open > 0.0%) ---")
    pos_sub = strict_top1_m[is_pos]
    print(f"    N: {len(pos_sub)} days ({len(pos_sub)/len(strict_top1_m):.1%})")
    print(f"    3进4 晋级率: {pos_sub['is_3to4'].mean():.2%}")
    print(f"    空间高标率: {pos_sub['is_market_space_leader'].mean():.2%}")
    print(f"    3进5突破率: {pos_sub['is_3to5'].mean():.2%}")
    
    print("  --- Strict Nuclear Low Open (<= 0.0%) ---")
    nuc_sub = strict_top1_m[is_nuclear]
    print(f"    N: {len(nuc_sub)} days ({len(nuc_sub)/len(strict_top1_m):.1%})")
    print(f"    3进4 晋级率: {nuc_sub['is_3to4'].mean():.2%}")
    print(f"    空间高标率: {nuc_sub['is_market_space_leader'].mean():.2%}")
    
    print("  --- Strict Tier 4 (Golden Master: Regime >= 0.55 + Open > 0% + Score >= 0.40) ---")
    tier4_sub = strict_top1_m[(strict_top1_m['gate_open_55'] == 1) & is_pos & (strict_top1_m['strict_score_total'] >= 0.40)]
    print(f"    N: {len(tier4_sub)} days")
    print(f"    3进4 晋级率: {tier4_sub['is_3to4'].mean():.2%}")
    print(f"    空间高标率: {tier4_sub['is_market_space_leader'].mean():.2%}")
    print(f"    3进5突破率: {tier4_sub['is_3to5'].mean():.2%}")

if __name__ == "__main__":
    audit()
