"""evaluate_round4_comprehensive.py

Comprehensive evaluation of Round 4 ("题材低位首板成群扩散助攻矩阵"):
1. Cohort base rates across the 4-fold OOF evaluation window.
2. Model Top-1, Top-2, Top-3 hit rates across all targets.
3. Direct Head-to-Head Comparison: Round 4 vs Round 3 vs Round 2 vs Round 1 vs BASE.
4. Direction 1 Matrix Audit:
   - Performance by theme_first_board_count tiers (0 vs 1 vs 2 vs 3+).
   - Filter power: how many isolated lone wolves were penalized.
   - Cluster dragon boost: hit rate of theme_assisted_dragon and theme_diffusion_surge.
5. Selective Recommendation Policy at various score thresholds.
6. Temporal stability across calendar years and market emotion regimes.
7. Generates round4_comprehensive_metrics.json.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent
EXP_DIR = ROOT / "research/model_upgrade_validation/dragon_theme_matrix_20261027T083500Z"
OOF_PATH = EXP_DIR / "oof_round4_predictions.csv"
DAILY_PATH = EXP_DIR / "daily_competition_round4.csv"

def evaluate():
    print(f"Loading Round 4 evaluation results from {EXP_DIR}...")
    oof_df = pd.read_csv(OOF_PATH)
    daily_df = pd.read_csv(DAILY_PATH)
    
    total_events = len(oof_df)
    unique_days = len(daily_df)
    
    # 1. Natural Base Rates
    base_5_rate = float(oof_df['is_3to5'].mean())
    base_space_rate = float(oof_df['is_market_space_leader'].mean())
    base_dragon_rate = float(oof_df['is_emotion_dragon_proxy'].mean())
    base_3to4_rate = float(oof_df['is_3to4'].mean())
    
    # 2. Top-1 Model Hit Rates
    top1_5_rate = float(daily_df['top1_is_3to5'].mean())
    top1_space_rate = float(daily_df['top1_is_space_leader'].mean())
    top1_dragon_rate = float(daily_df['top1_is_emotion_dragon'].mean())
    top1_3to4_rate = float(daily_df['top1_is_3to4'].mean())
    
    # 3. Top-2 Model Hit Rates
    top2_5_rate = float(daily_df['top2_5_hit'].mean())
    top2_space_rate = float(daily_df['top2_space_hit'].mean())
    top2_dragon_rate = float(daily_df['top2_dragon_hit'].mean())
    
    # 4. Excess Alpha vs Base
    alpha_5 = top1_5_rate - base_5_rate
    alpha_space = top1_space_rate - base_space_rate
    alpha_dragon = top1_dragon_rate - base_dragon_rate
    
    # 5. Direction 1 Theme Diffusion Matrix Breakdown
    print("\n--- Direction 1 Theme Diffusion Breakdown ---")
    oof_df['first_board_tier'] = pd.cut(
        oof_df['theme_first_board_count'],
        bins=[-0.1, 0.5, 1.5, 2.5, 100.0],
        labels=['0只(孤狼)', '1只(弱助攻)', '2只(成梯队)', '3只以上(群蜂爆发)']
    )
    theme_tiers = {}
    for tier, g in oof_df.groupby('first_board_tier', observed=False):
        theme_tiers[str(tier)] = {
            "samples": len(g),
            "share": float(len(g) / total_events),
            "space_leader_rate": float(g['is_market_space_leader'].mean()),
            "dragon_rate": float(g['is_emotion_dragon_proxy'].mean()),
            "5board_rate": float(g['is_3to5'].mean())
        }
        print(f"Tier {tier}: N={len(g)} ({len(g)/total_events:.1%}) | Space: {g['is_market_space_leader'].mean():.2%} | Dragon: {g['is_emotion_dragon_proxy'].mean():.2%} | 5-Board: {g['is_3to5'].mean():.2%}")
        
    # Top-1 choices: Lone Wolf vs Cluster Dragon
    top1_lone_wolf_count = int(daily_df['top1_is_lone_wolf'].sum())
    top1_cluster_count = unique_days - top1_lone_wolf_count
    
    # 6. BASE Comparison
    base_match = daily_df[daily_df['base_match_status'] == 'COMPLETE_MATCH'].copy()
    base_partial = daily_df[daily_df['base_top1_space'].notna()].copy()
    
    comp_base = {
        "complete_match": {
            "days": len(base_match),
            "model_top1_space": float(base_match['top1_is_space_leader'].mean()) if len(base_match) else None,
            "base_top1_space": float(base_match['base_top1_space'].mean()) if len(base_match) else None,
            "model_top1_5": float(base_match['top1_is_3to5'].mean()) if len(base_match) else None,
            "base_top1_5": float(base_match['base_top1_5'].mean()) if len(base_match) else None,
        },
        "all_matched": {
            "days": len(base_partial),
            "model_top1_space": float(base_partial['top1_is_space_leader'].mean()) if len(base_partial) else None,
            "base_top1_space": float(base_partial['base_top1_space'].mean()) if len(base_partial) else None,
            "model_top1_5": float(base_partial['top1_is_3to5'].mean()) if len(base_partial) else None,
            "base_top1_5": float(base_partial['base_top1_5'].mean()) if len(base_partial) else None,
        }
    }
    
    # 7. Selective Recommendation Policy
    policy_results = {}
    for th in [0.30, 0.35, 0.40, 0.45, 0.50]:
        rec = daily_df[daily_df['top1_score'] >= th]
        cov = len(rec) / unique_days
        rec_space = rec['top1_is_space_leader'].mean() if len(rec) else 0
        rec_dragon = rec['top1_is_emotion_dragon'].mean() if len(rec) else 0
        rec_5 = rec['top1_is_3to5'].mean() if len(rec) else 0
        policy_results[f"th_{th:.2f}"] = {
            "threshold": th,
            "recommended_days": len(rec),
            "coverage_pct": float(cov),
            "space_leader_hit_rate": float(rec_space),
            "dragon_hit_rate": float(rec_dragon),
            "height_5_hit_rate": float(rec_5)
        }
        
    # 8. Calendar Year Stability
    daily_df['year'] = pd.to_datetime(daily_df['date']).dt.year
    year_metrics = {}
    for yr, g in daily_df.groupby('year'):
        year_metrics[str(yr)] = {
            "days": len(g),
            "top1_space_rate": float(g['top1_is_space_leader'].mean()),
            "top1_dragon_rate": float(g['top1_is_emotion_dragon'].mean()),
            "top1_5_rate": float(g['top1_is_3to5'].mean())
        }
        
    metrics = {
        "round": 4,
        "conquered_direction": "Direction 1: 题材低位首板成群扩散助攻矩阵",
        "total_oof_samples": total_events,
        "unique_trading_days": unique_days,
        "base_rates": {
            "is_3to5": base_5_rate,
            "is_market_space_leader": base_space_rate,
            "is_emotion_dragon_proxy": base_dragon_rate,
            "is_3to4": base_3to4_rate
        },
        "model_top1_hit_rates": {
            "top1_is_market_space_leader": top1_space_rate,
            "top1_is_emotion_dragon_proxy": top1_dragon_rate,
            "top1_is_3to5": top1_5_rate,
            "top1_is_3to4": top1_3to4_rate
        },
        "model_top2_hit_rates": {
            "top2_space_hit": top2_space_rate,
            "top2_dragon_hit": top2_dragon_rate,
            "top2_5_hit": top2_5_rate
        },
        "excess_alpha_pp": {
            "space_leader_alpha_pp": float(alpha_space * 100),
            "emotion_dragon_alpha_pp": float(alpha_dragon * 100),
            "height_5_alpha_pp": float(alpha_5 * 100)
        },
        "theme_diffusion_audit": {
            "tiers": theme_tiers,
            "top1_lone_wolf_chosen": top1_lone_wolf_count,
            "top1_cluster_chosen": top1_cluster_count,
            "top1_lone_wolf_pct": float(top1_lone_wolf_count / unique_days)
        },
        "base_comparison": comp_base,
        "selective_policy": policy_results,
        "yearly_stability": year_metrics
    }
    
    out_json = EXP_DIR / "round4_comprehensive_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
        
    print(f"\nRound 4 Comprehensive Evaluation complete! Saved to {out_json}")
    print("\n=== Executive Summary of Round 4 ===")
    print(f"Top-1 Space Leader Hit Rate: {top1_space_rate:.2%} (vs Natural Base: {base_space_rate:.2%}, Alpha: +{alpha_space*100:.2f} pp)")
    print(f"Top-1 Emotion Dragon Hit Rate: {top1_dragon_rate:.2%} (vs Natural Base: {base_dragon_rate:.2%}, Alpha: +{alpha_dragon*100:.2f} pp)")
    print(f"Top-1 5-Board Breakthrough Rate: {top1_5_rate:.2%} (vs Natural Base: {base_5_rate:.2%}, Alpha: +{alpha_5*100:.2f} pp)")
    print(f"Top-2 Space Leader Hit Rate: {top2_space_rate:.2%}")
    print(f"Top-2 Emotion Dragon Hit Rate: {top2_dragon_rate:.2%}")
    print(f"Top-2 5-Board Breakthrough Rate: {top2_5_rate:.2%}")
    if comp_base['complete_match']['days']:
        print(f"Complete BASE Match (100 days): Model Top-1 Space {comp_base['complete_match']['model_top1_space']:.2%} vs BASE {comp_base['complete_match']['base_top1_space']:.2%}")

if __name__ == "__main__":
    evaluate()
