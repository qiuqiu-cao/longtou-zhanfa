"""train_and_evaluate_round4.py

Round 4 Walk-Forward Trainer & Evaluator:
Conquers Direction 1: "题材低位首板成群扩散助攻矩阵" (Theme Low-Board Diffusion Matrix)

Key Enhancements vs Round 3:
1. Incorporates 14 theme diffusion & ladder features into training.
2. Learns cluster dragon support vs isolated lone wolf penalties.
3. Strict session-by-session label maturity verification (determined_date_final <= train_end).
4. Zero lookahead leakage: all medians/thresholds fitted strictly on train fold.
5. Saves all OOF predictions, trained model bundles, daily competition results, and metrics.
"""

import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent
EXP_DIR = ROOT / "research/model_upgrade_validation/dragon_theme_matrix_20261027T083500Z"
DATA_PATH = EXP_DIR / "round4_dataset.csv"

def engineer_features(train_df, test_df, raw_cols):
    """
    Computes domain-engineered dragon indicators strictly fitted on train_df,
    then applied to both train_df and test_df.
    """
    train = train_df.copy()
    test = test_df.copy()
    
    # Estimate thresholds strictly on train
    med_price = train['price'].median()
    med_cap = train['float_cap_log'].median()
    med_diffusion = train['theme_first_board_count'].median()
    
    for df in [train, test]:
        df['is_low_price'] = (df['price'] <= med_price).astype(float)
        df['is_small_cap'] = (df['float_cap_log'] <= med_cap).astype(float)
        df['is_tight_amplitude'] = (df['amplitude'] <= 3.0).astype(float)
        df['is_high_churn_risk'] = ((df['turnover'] > 0.18) & (df['one_word_proxy'] == 0)).astype(float)
        df['is_crowded'] = (df['same_height_rivals'] >= 4).astype(float)
        df['firm_seal_strength'] = (df['one_word_proxy'] * 2.0 + (df['amplitude'] <= 2.0).astype(float) - (df['amplitude'] > 8.0).astype(float))
        df['price_cap_penalty'] = (df['price'] > 16.0).astype(float) + (df['float_cap_log'] > 23.0).astype(float)
        df['healthy_turnover'] = ((df['turnover'] >= 0.05) & (df['turnover'] <= 0.18)).astype(float)
        df['space_leader_stalling'] = ((df['height_gap'] > 0) & (df['space_feedback'] <= 0)).astype(float)
        df['takeover_window_open'] = ((df['height_gap'] > 0) & (df['space_feedback'] <= 0) & (df['promotion_3to4'] > 0.4)).astype(float)
        
        # New Theme-Diffusion Domain Features
        df['theme_assisted_dragon'] = ((df['theme_first_board_count'] >= 2.0) & (df['is_theme_sole_leader'] == 1.0)).astype(float)
        df['lone_wolf_penalty'] = (df['is_isolated_lone_wolf'] == 1.0).astype(float)
        df['theme_diffusion_heating'] = ((df['theme_first_board_expansion'] > 0) & (df['theme_zt_share'] >= 0.05)).astype(float)
        
    engineered_cols = raw_cols + [
        'is_low_price', 'is_small_cap', 'is_tight_amplitude', 'is_high_churn_risk',
        'is_crowded', 'firm_seal_strength', 'price_cap_penalty', 'healthy_turnover',
        'space_leader_stalling', 'takeover_window_open',
        'theme_assisted_dragon', 'lone_wolf_penalty', 'theme_diffusion_heating'
    ]
    return train, test, engineered_cols

def main():
    print(f"Loading Round 4 dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['determined_date_final'] = pd.to_datetime(df['determined_date_final'])
    df['determined_date_3to5'] = pd.to_datetime(df['determined_date_3to5'])
    df['determined_date_3to4'] = pd.to_datetime(df['determined_date_3to4'])
    df = df.sort_values(['date', 'code']).reset_index(drop=True)
    
    raw_feature_cols = [
        # 1. Base space & height
        'height_gap', 'is_space_leader', 'unique_space', 'height_share', 'same_height_rivals',
        # 2. Price & volume
        'turnover', 'signal_amount', 'turnover_vs_prev', 'amount_float_cap', 'one_word_proxy',
        'one_word_streak', 'amplitude', 'relative_strength', 'price', 'float_cap_log',
        # 3. Market emotion context
        'emotion_score', 'limit_count_scaled', 'consecutive_scaled', 'down_limit_rate',
        'failed_rate', 'seal_rate', 'height_change_scaled', 'limit_change_scaled',
        'promotion_overall', 'promotion_1to2', 'promotion_2to3', 'promotion_3to4',
        'promotion_mid', 'space_feedback', 'middle_density', 'ladder_gap_count',
        'breadth_height_divergence', 'yesterday_premium', 'failed_tail_loss',
        'previous_limit_tail_loss', 'breadth_health', 'relay_appetite', 'ladder_health',
        'loss_effect_health', 'crowding_risk', 'regime_fragility',
        # 4. Fundamental & risk
        'announcement_risk', 'reduction_risk', 'unlock_risk', 'earnings_risk',
        # 5. Cohort relative
        'market_max_board', 'market_max_board_count', 'consecutive_count', 'total_limit_up',
        'amount_cohort_share', 'amount_cohort_rank', 'amplitude_cohort_rank', 'price_cohort_rank', 'turnover_cohort_rank',
        'is_cohort_volume_leader', 'is_cohort_tightest', 'is_cohort_lowest_price', 'is_already_space_leader',
        # 6. DIRECTION 1: Theme Low-Board Diffusion Matrix
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
        
        # Enforce maturity condition on training set
        train_raw = df[(df['date'] <= train_end) & (df['determined_date_final'] <= train_end)].copy()
        test_raw = df[(df['date'] >= test_start) & (df['date'] <= test_end)].copy()
        
        train, test, feature_cols = engineer_features(train_raw, test_raw, raw_feature_cols)
        
        print(f"\n--- FOLD {fold_idx + 1}: {test_start_str} to {test_end_str} ---")
        print(f"Mature Training Samples: {len(train)}, Test Samples: {len(test)}")
        print(f"Total Features Evaluated: {len(feature_cols)}")
        
        # 1. Model for Stage Space Leadership
        clf_space = HistGradientBoostingClassifier(
            max_iter=130, max_depth=4, min_samples_leaf=16, l2_regularization=2.5,
            learning_rate=0.035, class_weight={0: 1.0, 1: 1.6}, random_state=101 + fold_idx
        )
        clf_space.fit(train[feature_cols], train['is_market_space_leader'])
        test['score_space_leader'] = clf_space.predict_proba(test[feature_cols])[:, 1]
        
        # 2. Model for Takeover (Case B specific)
        train_b = train[train['is_already_space_leader'] == 0].copy()
        clf_takeover = HistGradientBoostingClassifier(
            max_iter=130, max_depth=4, min_samples_leaf=16, l2_regularization=2.5,
            learning_rate=0.035, class_weight={0: 1.0, 1: 2.0}, random_state=201 + fold_idx
        )
        clf_takeover.fit(train_b[feature_cols], train_b['target_B_takeover_space'])
        test['score_takeover'] = clf_takeover.predict_proba(test[feature_cols])[:, 1]
        
        # 3. Model for Emotion Dragon Candidate Proxy
        clf_dragon = HistGradientBoostingClassifier(
            max_iter=130, max_depth=4, min_samples_leaf=16, l2_regularization=3.0,
            learning_rate=0.035, class_weight={0: 1.0, 1: 2.5}, random_state=301 + fold_idx
        )
        clf_dragon.fit(train[feature_cols], train['is_emotion_dragon_proxy'])
        test['score_emotion_dragon'] = clf_dragon.predict_proba(test[feature_cols])[:, 1]
        
        # 4. Model for Continuous 5 Boards
        clf_5 = HistGradientBoostingClassifier(
            max_iter=130, max_depth=4, min_samples_leaf=16, l2_regularization=2.5,
            learning_rate=0.035, class_weight={0: 1.0, 1: 1.5}, random_state=401 + fold_idx
        )
        clf_5.fit(train[feature_cols], train['is_3to5'])
        test['score_height_5'] = clf_5.predict_proba(test[feature_cols])[:, 1]
        
        # Composite Dragon Score
        test['score_round4_composite'] = (
            0.35 * test['score_space_leader'] +
            0.25 * test['score_takeover'] +
            0.25 * test['score_emotion_dragon'] +
            0.15 * test['score_height_5']
        )
        test['fold'] = fold_idx + 1
        all_oof_results.append(test)
        
        auc_space = roc_auc_score(test['is_market_space_leader'], test['score_space_leader'])
        auc_dragon = roc_auc_score(test['is_emotion_dragon_proxy'], test['score_emotion_dragon'])
        auc_5 = roc_auc_score(test['is_3to5'], test['score_height_5'])
        
        fold_metrics.append({
            "fold": fold_idx + 1,
            "test_range": f"{test_start_str} to {test_end_str}",
            "train_samples": len(train),
            "test_samples": len(test),
            "auc_space_leader": float(auc_space),
            "auc_emotion_dragon": float(auc_dragon),
            "auc_height_5": float(auc_5),
        })
        print(f"Fold {fold_idx + 1} Metrics -> Space Leader AUC: {auc_space:.4f}, Dragon AUC: {auc_dragon:.4f}, 5-Board AUC: {auc_5:.4f}")
        
    # Combine OOF predictions
    oof_df = pd.concat(all_oof_results, ignore_index=True)
    print(f"\nTotal OOF Predictions: {len(oof_df)} across {oof_df['date'].nunique()} trading days")
    
    # Save models from final fold
    joblib.dump(clf_space, EXP_DIR / "model_space_leader_r4.joblib")
    joblib.dump(clf_takeover, EXP_DIR / "model_takeover_r4.joblib")
    joblib.dump(clf_dragon, EXP_DIR / "model_emotion_dragon_r4.joblib")
    joblib.dump(clf_5, EXP_DIR / "model_height_5_r4.joblib")
    
    # Compute daily rankings
    oof_df['rank_composite'] = oof_df.groupby('date')['score_round4_composite'].rank(ascending=False, method='min')
    oof_df['rank_space'] = oof_df.groupby('date')['score_space_leader'].rank(ascending=False, method='min')
    oof_df['rank_dragon'] = oof_df.groupby('date')['score_emotion_dragon'].rank(ascending=False, method='min')
    oof_df['rank_5'] = oof_df.groupby('date')['score_height_5'].rank(ascending=False, method='min')
    
    # Also rank BASE if present
    if 'base_historical_score' in oof_df.columns:
        oof_df['rank_base'] = oof_df.groupby('date')['base_historical_score'].rank(ascending=False, method='min')
        
    oof_cols = [
        'date', 'code', 'name', 'boards', 'industry', 'fold',
        'is_already_space_leader', 'target_A_continue', 'target_B_takeover_space',
        'is_market_space_leader', 'is_unique_space_leader', 'is_emotion_dragon_proxy',
        'is_3to4', 'is_3to5', 'is_deep_6', 'is_monster_7',
        'score_round4_composite', 'rank_composite',
        'score_space_leader', 'rank_space',
        'score_takeover',
        'score_emotion_dragon', 'rank_dragon',
        'score_height_5', 'rank_5',
        'theme_first_board_count', 'theme_ladder_density', 'theme_ladder_complete',
        'theme_zt_share', 'theme_amount_share', 'theme_first_board_expansion',
        'is_theme_sole_leader', 'is_isolated_lone_wolf', 'theme_diffusion_surge',
        'base_historical_score', 'base_cohort_match_status'
    ]
    oof_out = oof_df[[c for c in oof_cols if c in oof_df.columns]].copy()
    oof_out_path = EXP_DIR / "oof_round4_predictions.csv"
    oof_out.to_csv(oof_out_path, index=False, encoding='utf-8-sig')
    print(f"Saved OOF predictions to {oof_out_path}")
    
    # Save daily competition summary
    daily_comp = []
    for d, g in oof_df.groupby('date'):
        top1 = g[g['rank_composite'] == 1].iloc[0]
        top2 = g[g['rank_composite'] <= 2]
        
        base_top1_space = np.nan
        base_top1_5 = np.nan
        if 'rank_base' in g.columns and g['rank_base'].notna().any():
            base_cand = g[g['rank_base'] == 1]
            if len(base_cand) > 0:
                base_top1_space = base_cand.iloc[0]['is_market_space_leader']
                base_top1_5 = base_cand.iloc[0]['is_3to5']
                
        daily_comp.append({
            'date': str(d.date()),
            'cohort_size': len(g),
            'top1_code': str(top1['code']).zfill(6),
            'top1_name': top1['name'],
            'top1_industry': top1.get('industry', ''),
            'top1_theme_first_count': top1.get('theme_first_board_count', 0),
            'top1_is_lone_wolf': top1.get('is_isolated_lone_wolf', 0),
            'top1_score': float(top1['score_round4_composite']),
            'top1_is_space_leader': int(top1['is_market_space_leader']),
            'top1_is_emotion_dragon': int(top1['is_emotion_dragon_proxy']),
            'top1_is_3to5': int(top1['is_3to5']),
            'top1_is_3to4': int(top1['is_3to4']),
            'top2_space_hit': int(top2['is_market_space_leader'].any()),
            'top2_dragon_hit': int(top2['is_emotion_dragon_proxy'].any()),
            'top2_5_hit': int(top2['is_3to5'].any()),
            'cohort_space_sum': int(g['is_market_space_leader'].sum()),
            'cohort_dragon_sum': int(g['is_emotion_dragon_proxy'].sum()),
            'cohort_5_sum': int(g['is_3to5'].sum()),
            'base_top1_space': base_top1_space,
            'base_top1_5': base_top1_5,
            'base_match_status': top1.get('base_cohort_match_status', 'UNKNOWN')
        })
    daily_comp_df = pd.DataFrame(daily_comp)
    daily_comp_path = EXP_DIR / "daily_competition_round4.csv"
    daily_comp_df.to_csv(daily_comp_path, index=False, encoding='utf-8-sig')
    print(f"Saved daily competition records to {daily_comp_path}")
    
    # Save manifest
    manifest = {
        "experiment": "dragon_theme_matrix_20261027T083500Z",
        "conquered_direction": "Direction 1: 题材低位首板成群扩散助攻矩阵",
        "features_evaluated": len(feature_cols),
        "total_oof_rows": len(oof_df),
        "unique_trading_days": int(oof_df['date'].nunique()),
        "fold_metrics": fold_metrics
    }
    with open(EXP_DIR / "manifest_round4.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("Manifest successfully written.")

if __name__ == "__main__":
    main()
