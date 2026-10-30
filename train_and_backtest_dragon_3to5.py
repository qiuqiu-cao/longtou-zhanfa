"""train_and_backtest_dragon_3to5.py

Dedicated research trainer and walk-forward backtester for predicting:
Given a stock is at 3 boards at Day t close (boards == 3),
can it advance through 4 boards to successfully reach >= 5 boards (走出5板成妖)?

Target:
y = (terminal_board >= 5).astype(int)

Preserves production BASE model 100% untouched.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parent
EXP_DIR = ROOT / "research/model_upgrade_validation/dragon_3to5_breakthrough_20261027T064500Z"
DATA_PATH = ROOT / "data/v16/point_in_time_rebuild_cutoff_20260831_v2/leader_candidates.csv"

def prepare_features(df):
    b3 = df[df['boards'] == 3].copy()
    b3['date'] = pd.to_datetime(b3['date'])
    b3 = b3.sort_values(['date', 'code']).reset_index(drop=True)
    b3['target_5'] = (b3['terminal_board'] >= 5).astype(int)
    
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


def main():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    b3, feature_cols = prepare_features(df)
    print(f"Total 3-board events: {len(b3)}")
    print(f"Natural base rate of reaching >= 5 boards: {b3['target_5'].mean():.4f} ({b3['target_5'].sum()}/{len(b3)})")
    
    # 4-Fold Temporal Expanding Window Walk-Forward Backtest
    folds = [
        ('2024-08-31', '2024-09-01', '2024-12-31'),
        ('2024-12-31', '2025-01-01', '2025-05-31'),
        ('2025-05-31', '2025-06-01', '2025-11-30'),
        ('2025-11-30', '2025-12-01', '2026-08-31'),
    ]
    
    all_oof = []
    fold_metrics = []
    
    for fold_idx, (train_end, test_start, test_end) in enumerate(folds):
        train = b3[b3['date'] <= train_end].copy()
        test = b3[(b3['date'] >= test_start) & (b3['date'] <= test_end)].copy()
        
        clf = HistGradientBoostingClassifier(
            max_iter=110,
            max_depth=4,
            min_samples_leaf=16,
            l2_regularization=2.5,
            learning_rate=0.04,
            class_weight={0: 1.0, 1: 1.5},
            random_state=100 + fold_idx
        )
        clf.fit(train[feature_cols], train['target_5'])
        
        test['pred_prob'] = clf.predict_proba(test[feature_cols])[:, 1]
        test['fold'] = fold_idx + 1
        all_oof.append(test)
        
        f_auc = roc_auc_score(test['target_5'], test['pred_prob'])
        f_pr_auc = average_precision_score(test['target_5'], test['pred_prob'])
        f_base = test['target_5'].mean()
        
        # Calculate daily competition top1 for this fold
        comp = test.groupby('date').filter(lambda g: len(g) >= 2)
        top1_hits = [g.sort_values('pred_prob', ascending=False).iloc[0]['target_5'] for _, g in comp.groupby('date')]
        top1_acc = np.mean(top1_hits) if top1_hits else 0.0
        
        fold_metrics.append({
            "fold": fold_idx + 1,
            "train_end": train_end,
            "test_range": f"{test_start} to {test_end}",
            "train_samples": len(train),
            "test_samples": len(test),
            "base_rate": round(float(f_base), 4),
            "auc_roc": round(float(f_auc), 4),
            "pr_auc": round(float(f_pr_auc), 4),
            "competitive_days": len(top1_hits),
            "top1_accuracy": round(float(top1_acc), 4),
            "top1_alpha_spread": round(float(top1_acc - comp['target_5'].mean()), 4) if len(comp) > 0 else 0.0
        })
        print(f"Fold {fold_idx + 1} ({test_start} to {test_end}): N={len(test)}, Base={f_base:.3f}, AUC={f_auc:.3f}, Top1={top1_acc:.3f}")
        
    oof = pd.concat(all_oof, axis=0)
    
    # Overall OOS Metrics
    total_oos = len(oof)
    base_rate_oos = float(oof['target_5'].mean())
    auc_oos = float(roc_auc_score(oof['target_5'], oof['pred_prob']))
    pr_auc_oos = float(average_precision_score(oof['target_5'], oof['pred_prob']))
    brier_oos = float(brier_score_loss(oof['target_5'], oof['pred_prob']))
    
    # Decile / Tier Analysis
    oof['score_tier'] = pd.qcut(oof['pred_prob'], 5, labels=['Tier 1 (Lowest)', 'Tier 2', 'Tier 3', 'Tier 4', 'Tier 5 (Highest)'])
    tier_summary = oof.groupby('score_tier', observed=False)['target_5'].agg(
        total='count',
        reached_5='sum',
        hit_rate='mean'
    ).reset_index().to_dict(orient='records')
    
    # Head-to-Head Daily Competitions
    comp_days = oof.groupby('date').filter(lambda g: len(g) >= 2)
    daily_results = []
    top1_hits, top2_hits, bottom1_hits = [], [], []
    
    for d, g in comp_days.groupby('date'):
        sorted_g = g.sort_values('pred_prob', ascending=False)
        top1 = sorted_g.iloc[0]
        top1_hits.append(int(top1['target_5']))
        
        top2 = sorted_g.iloc[:2]
        top2_hits.extend(top2['target_5'].tolist())
        
        bot = sorted_g.iloc[-1]
        bottom1_hits.append(int(bot['target_5']))
        
        daily_results.append({
            "date": d.strftime("%Y-%m-%d"),
            "cohort_size": len(g),
            "top1_code": str(top1['code']),
            "top1_name": str(top1['name']),
            "top1_score": round(float(top1['pred_prob']), 4),
            "top1_reached_5": bool(top1['target_5'] == 1),
            "actual_reached_count": int(g['target_5'].sum()),
            "bottom1_code": str(bot['code']),
            "bottom1_name": str(bot['name']),
            "bottom1_score": round(float(bot['pred_prob']), 4),
            "bottom1_reached_5": bool(bot['target_5'] == 1)
        })
        
    comp_base = float(comp_days['target_5'].mean())
    top1_rate = float(np.mean(top1_hits))
    top2_rate = float(np.mean(top2_hits))
    bottom1_rate = float(np.mean(bottom1_hits))
    
    print("\n" + "=" * 60)
    print("=== SUMMARY OF WALK-FORWARD OUT-OF-SAMPLE BACKTEST ===")
    print(f"Total Out-of-Sample 3-Board Candidates: {total_oos}")
    print(f"Natural Baseline Rate (3-to-5): {base_rate_oos:.4f}")
    print(f"Model OOS AUC-ROC: {auc_oos:.4f}")
    print(f"Model OOS PR-AUC: {pr_auc_oos:.4f} (vs. Base: {base_rate_oos:.4f}, Relative +{(pr_auc_oos/base_rate_oos - 1)*100:.1f}%)")
    print(f"Tier 5 (Top Decile) Success Rate: {tier_summary[-1]['hit_rate']:.4f} vs Tier 1: {tier_summary[0]['hit_rate']:.4f}")
    print(f"Competitive Trading Days: {len(top1_hits)}")
    print(f"Daily Top-1 Pick 5-Board Rate: {top1_rate:.4f} ({sum(top1_hits)}/{len(top1_hits)})")
    print(f"Competitive Baseline Rate: {comp_base:.4f}")
    print(f"Top-1 Alpha Spread over Baseline: +{(top1_rate - comp_base)*100:.2f} percentage points!")
    print(f"Top-1 vs Bottom-1 Separation Spread: +{(top1_rate - bottom1_rate)*100:.2f} percentage points!")
    
    # Train the final production research model on full dataset
    print("\nTraining final production research model DRAGON_3TO5_CHALLENGER on full dataset...")
    final_model = HistGradientBoostingClassifier(
        max_iter=110,
        max_depth=4,
        min_samples_leaf=16,
        l2_regularization=2.5,
        learning_rate=0.04,
        class_weight={0: 1.0, 1: 1.5},
        random_state=42
    )
    final_model.fit(b3[feature_cols], b3['target_5'])
    
    # Save Model Artifact
    joblib.dump(final_model, EXP_DIR / "model.joblib")
    
    # Save Out-of-Sample Predictions
    export_cols = ['date', 'code', 'name', 'boards', 'terminal_board', 'target_5', 'pred_prob', 'fold', 'score_tier']
    oof[export_cols].to_csv(EXP_DIR / "oof_predictions.csv", index=False, encoding='utf-8-sig')
    pd.DataFrame(daily_results).to_csv(EXP_DIR / "competition_daily_summary.csv", index=False, encoding='utf-8-sig')
    
    # Save Metrics & Manifest
    manifest = {
        "model_name": "DRAGON_3TO5_CHALLENGER",
        "purpose": "Predict probability of a 3-board stock reaching >= 5 boards at Day t close",
        "target": "y = (terminal_board >= 5).astype(int)",
        "training_samples": len(b3),
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "base_model_preserved": True,
        "base_model_path": "research/model_upgrade_validation/fixed_20261007T112017Z/models/relay/BASE/model.joblib",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "HistGradientBoostingClassifier(asymmetric_cost=1.5, max_depth=4, l2_reg=2.5)",
        "backtest_evaluation": {
            "evaluation_type": "4-Fold Temporal Expanding Window Walk-Forward Backtest",
            "out_of_sample_samples": total_oos,
            "baseline_natural_rate": base_rate_oos,
            "auc_roc": auc_oos,
            "pr_auc": pr_auc_oos,
            "brier_score": brier_oos,
            "tier_summary": tier_summary,
            "head_to_head_competition": {
                "competitive_trading_days": len(top1_hits),
                "baseline_competition_rate": comp_base,
                "top1_pick_5board_rate": top1_rate,
                "top2_pick_5board_rate": top2_rate,
                "bottom1_pick_5board_rate": bottom1_rate,
                "top1_alpha_over_baseline": round(top1_rate - comp_base, 4),
                "top1_vs_bottom1_separation": round(top1_rate - bottom1_rate, 4)
            },
            "fold_details": fold_metrics
        }
    }
    
    with open(EXP_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        
    print(f"Artifacts successfully written to {EXP_DIR}")


if __name__ == "__main__":
    main()
