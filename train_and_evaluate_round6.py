"""train_and_evaluate_round6.py

Round 6 Walk-Forward Trainer & Evaluator:
Simultaneously conquers:
- 突破口一：次日 09:25 集合竞价盘口与微观挂单结构确认 (Auction Execution Confirmation)
- 突破口二：10 日累计 100 异动大二波总龙模型 (10-Day 100% Abnormal Price Movement & Second Wave Leader)

Multi-Objective Framework:
1. Space Leadership (is_market_space_leader)
2. 5-Board Breakthrough (is_3to5)
3. 10-Day 100% Abnormal Runner (is_10d_100_abnormal)
4. Second Wave Rebound Dragon (is_second_wave_leader)
5. 09:25 Auction Execution Gating (Sweet spot vs Nuclear Low Open Cancellation)

Strict walk-forward cross validation; zero lookahead leakage into Day t features.
"""

import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent
EXP_DIR = ROOT / "research/model_upgrade_validation/dragon_auction_and_second_wave_20261027T091600Z"
DATA_PATH = EXP_DIR / "round6_dataset.csv"
REGIME_GATE_PATH = ROOT / "research/model_upgrade_validation/dragon_regime_gate_20261027T085500Z/oof_round5_daily_gate_predictions.csv"

def engineer_features(train_df, test_df, raw_cols):
    train = train_df.copy()
    test = test_df.copy()
    
    med_price = train['price'].median()
    med_cap = train['float_cap_log'].median()
    
    for df in [train, test]:
        df['is_low_price'] = (df['price'] <= med_price).astype(float)
        df['is_small_cap'] = (df['float_cap_log'] <= med_cap).astype(float)
        df['is_tight_amplitude'] = (df['amplitude'] <= 3.0).astype(float)
        df['is_high_churn_risk'] = ((df['turnover'] > 0.18) & (df['one_word_proxy'] == 0)).astype(float)
        df['is_crowded'] = (df['same_height_rivals'] >= 4).astype(float)
        df['firm_seal_strength'] = (df['one_word_proxy'] * 2.0 + (df['amplitude'] <= 2.0).astype(float) - (df['amplitude'] > 8.0).astype(float))
        df['healthy_turnover'] = ((df['turnover'] >= 0.05) & (df['turnover'] <= 0.18)).astype(float)
        df['theme_assisted_dragon'] = ((df['theme_first_board_count'] >= 2.0) & (df['is_theme_sole_leader'] == 1.0)).astype(float)
        df['lone_wolf_penalty'] = (df['is_isolated_lone_wolf'] == 1.0).astype(float)
        df['theme_diffusion_heating'] = ((df['theme_first_board_expansion'] > 0) & (df['theme_zt_share'] >= 0.05)).astype(float)
        
    engineered_cols = raw_cols + [
        'is_low_price', 'is_small_cap', 'is_tight_amplitude', 'is_high_churn_risk',
        'is_crowded', 'firm_seal_strength', 'healthy_turnover',
        'theme_assisted_dragon', 'lone_wolf_penalty', 'theme_diffusion_heating'
    ]
    return train, test, engineered_cols

def main():
    print(f"Loading Round 6 dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['determined_date_final'] = pd.to_datetime(df['determined_date_final'])
    df = df.sort_values(['date', 'code']).reset_index(drop=True)
    
    raw_feature_cols = [
        # Space & height
        'height_gap', 'is_space_leader', 'unique_space', 'height_share', 'same_height_rivals',
        # Price & volume
        'turnover', 'signal_amount', 'turnover_vs_prev', 'amount_float_cap', 'one_word_proxy',
        'one_word_streak', 'amplitude', 'relative_strength', 'price', 'float_cap_log',
        # Market emotion
        'emotion_score', 'limit_count_scaled', 'consecutive_scaled', 'down_limit_rate',
        'failed_rate', 'seal_rate', 'height_change_scaled', 'limit_change_scaled',
        'promotion_overall', 'promotion_1to2', 'promotion_2to3', 'promotion_3to4',
        'promotion_mid', 'space_feedback', 'middle_density', 'ladder_gap_count',
        'breadth_height_divergence', 'yesterday_premium', 'failed_tail_loss',
        'previous_limit_tail_loss', 'breadth_health', 'relay_appetite', 'ladder_health',
        'loss_effect_health', 'crowding_risk', 'regime_fragility',
        # Cohort relative
        'market_max_board', 'market_max_board_count', 'consecutive_count', 'total_limit_up',
        'amount_cohort_share', 'amount_cohort_rank', 'amplitude_cohort_rank', 'price_cohort_rank', 'turnover_cohort_rank',
        'is_cohort_volume_leader', 'is_cohort_tightest', 'is_cohort_lowest_price', 'is_already_space_leader',
        # Direction 1: Theme Low-board diffusion matrix
        'theme_cluster_total_zt', 'theme_first_board_count', 'theme_two_board_count', 'theme_three_board_count',
        'theme_zt_share', 'theme_first_board_share', 'theme_amount_share',
        'theme_ladder_density', 'theme_ladder_complete',
        'theme_first_board_expansion', 'theme_cluster_expansion',
        'is_theme_sole_leader', 'theme_same_height_rivals',
        'is_isolated_lone_wolf', 'theme_diffusion_surge'
    ]
    
    folds = [
        ('2024-08-31', '2024-09-01', '2024-12-31'),
        ('2024-12-31', '2025-01-01', '2025-05-31'),
        ('2025-05-31', '2025-06-01', '2025-11-30'),
        ('2025-11-30', '2025-12-01', '2026-08-31'),
    ]
    
    all_oof_results = []
    fold_metrics = []
    
    for fold_idx, (train_end_str, test_start_str, test_end_str) in enumerate(folds):
        train_end = pd.to_datetime(train_end_str)
        test_start = pd.to_datetime(test_start_str)
        test_end = pd.to_datetime(test_end_str)
        
        train_raw = df[(df['date'] <= train_end) & (df['determined_date_final'] <= train_end)].copy()
        test_raw = df[(df['date'] >= test_start) & (df['date'] <= test_end)].copy()
        
        train, test, feature_cols = engineer_features(train_raw, test_raw, raw_feature_cols)
        
        print(f"\n--- FOLD {fold_idx + 1}: {test_start_str} to {test_end_str} ---")
        print(f"Training Samples: {len(train)}, Test Samples: {len(test)}")
        
        # 1. Model for Space Leadership
        clf_space = HistGradientBoostingClassifier(
            max_iter=130, max_depth=4, min_samples_leaf=16, l2_regularization=2.5,
            learning_rate=0.035, class_weight={0: 1.0, 1: 1.6}, random_state=101 + fold_idx
        )
        clf_space.fit(train[feature_cols], train['is_market_space_leader'])
        test['score_space_leader'] = clf_space.predict_proba(test[feature_cols])[:, 1]
        
        # 2. Model for 5-Board Breakthrough
        clf_5 = HistGradientBoostingClassifier(
            max_iter=130, max_depth=4, min_samples_leaf=16, l2_regularization=2.5,
            learning_rate=0.035, class_weight={0: 1.0, 1: 1.5}, random_state=201 + fold_idx
        )
        clf_5.fit(train[feature_cols], train['is_3to5'])
        test['score_height_5'] = clf_5.predict_proba(test[feature_cols])[:, 1]
        
        # 3. Model for 10-Day 100% Abnormal Runner (突破口二 A)
        clf_100 = HistGradientBoostingClassifier(
            max_iter=130, max_depth=4, min_samples_leaf=16, l2_regularization=2.5,
            learning_rate=0.035, class_weight={0: 1.0, 1: 2.2}, random_state=301 + fold_idx
        )
        clf_100.fit(train[feature_cols], train['is_10d_100_abnormal'])
        test['score_10d_100_abnormal'] = clf_100.predict_proba(test[feature_cols])[:, 1]
        
        # 4. Model for Second Wave Rebound Dragon (突破口二 B)
        clf_2w = HistGradientBoostingClassifier(
            max_iter=130, max_depth=4, min_samples_leaf=16, l2_regularization=2.5,
            learning_rate=0.035, class_weight={0: 1.0, 1: 1.4}, random_state=401 + fold_idx
        )
        clf_2w.fit(train[feature_cols], train['is_second_wave_leader'])
        test['score_second_wave'] = clf_2w.predict_proba(test[feature_cols])[:, 1]
        
        # Total Dragon Leader Composite Score (incorporating 100% abnormal & second wave)
        test['score_round6_total_dragon'] = (
            0.30 * test['score_space_leader'] +
            0.25 * test['score_height_5'] +
            0.25 * test['score_10d_100_abnormal'] +
            0.20 * test['score_second_wave']
        )
        test['fold'] = fold_idx + 1
        all_oof_results.append(test)
        
        auc_space = roc_auc_score(test['is_market_space_leader'], test['score_space_leader'])
        auc_5 = roc_auc_score(test['is_3to5'], test['score_height_5'])
        auc_100 = roc_auc_score(test['is_10d_100_abnormal'], test['score_10d_100_abnormal'])
        auc_2w = roc_auc_score(test['is_second_wave_leader'], test['score_second_wave'])
        
        fold_metrics.append({
            "fold": fold_idx + 1,
            "test_range": f"{test_start_str} to {test_end_str}",
            "auc_space_leader": float(auc_space),
            "auc_height_5": float(auc_5),
            "auc_100_abnormal": float(auc_100),
            "auc_second_wave": float(auc_2w),
        })
        print(f"Fold {fold_idx + 1} Metrics -> Space AUC: {auc_space:.4f}, 5-Board AUC: {auc_5:.4f}, 100% Abnormal AUC: {auc_100:.4f}, Second Wave AUC: {auc_2w:.4f}")
        
    oof_df = pd.concat(all_oof_results, ignore_index=True)
    
    # Save models
    joblib.dump(clf_100, EXP_DIR / "model_10d_100_abnormal_r6.joblib")
    joblib.dump(clf_2w, EXP_DIR / "model_second_wave_r6.joblib")
    
    # Daily rankings
    oof_df['rank_total_dragon'] = oof_df.groupby('date')['score_round6_total_dragon'].rank(ascending=False, method='min')
    
    # Save full OOF predictions
    oof_out_path = EXP_DIR / "oof_round6_predictions.csv"
    oof_df.to_csv(oof_out_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved Round 6 full OOF predictions to {oof_out_path}")
    
    # Merge with Regime Gate decisions from Round 5
    print("Merging with Round 5 Regime Gate decisions...")
    regime_df = pd.read_csv(REGIME_GATE_PATH)
    regime_df['date'] = pd.to_datetime(regime_df['date'])
    
    daily_comp = []
    for d, g in oof_df.groupby('date'):
        top1 = g[g['rank_total_dragon'] == 1].iloc[0]
        top2 = g[g['rank_total_dragon'] <= 2]
        
        reg_match = regime_df[regime_df['date'] == d]
        prob_regime = float(reg_match.iloc[0]['prob_dragon_day']) if len(reg_match) else 0.5
        gate_open_55 = int(reg_match.iloc[0]['gate_arm_b_open_th55']) if len(reg_match) else 1
        
        open_gap = float(top1.get('next_open_gap', 0.0))
        is_auction_pos = int(open_gap > 0.0)
        is_auction_nuclear = int(open_gap <= 0.0)
        is_auction_sweet = int(0.015 <= open_gap <= 0.065)
        
        daily_comp.append({
            'date': str(d.date()),
            'cohort_size': len(g),
            'top1_code': str(top1['code']).zfill(6),
            'top1_name': top1['name'],
            'top1_industry': top1.get('industry', ''),
            'top1_score': float(top1['score_round6_total_dragon']),
            'top1_open_gap': open_gap,
            'is_auction_pos': is_auction_pos,
            'is_auction_nuclear': is_auction_nuclear,
            'is_auction_sweet': is_auction_sweet,
            'prob_regime_dragon': prob_regime,
            'gate_open_55': gate_open_55,
            'top1_is_space_leader': int(top1['is_market_space_leader']),
            'top1_is_3to5': int(top1['is_3to5']),
            'top1_is_3to4': int(top1['is_3to4']),
            'top1_is_100_abnormal': int(top1['is_10d_100_abnormal']),
            'top1_is_second_wave': int(top1['is_second_wave_leader']),
            'top1_10d_max_return': float(top1['forward_10d_max_return']) if pd.notna(top1['forward_10d_max_return']) else 0.0,
            'cohort_space_sum': int(g['is_market_space_leader'].sum()),
            'cohort_5_sum': int(g['is_3to5'].sum()),
            'cohort_100_sum': int(g['is_10d_100_abnormal'].sum()),
            'cohort_2w_sum': int(g['is_second_wave_leader'].sum())
        })
        
    daily_comp_df = pd.DataFrame(daily_comp)
    daily_comp_path = EXP_DIR / "daily_competition_round6.csv"
    daily_comp_df.to_csv(daily_comp_path, index=False, encoding='utf-8-sig')
    print(f"Saved daily competition results to {daily_comp_path}")
    
    # Save manifest
    manifest = {
        "experiment": "dragon_auction_and_second_wave_20261027T091600Z",
        "conquered_directions": [
            "Direction 3: 次日 09:25 集合竞价盘口与微观挂单结构确认",
            "Direction 5: 10 日累计 100 异动大二波总龙模型"
        ],
        "total_oof_rows": len(oof_df),
        "total_test_days": len(daily_comp_df),
        "fold_metrics": fold_metrics
    }
    with open(EXP_DIR / "manifest_round6.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("Manifest successfully written.")

if __name__ == "__main__":
    main()
