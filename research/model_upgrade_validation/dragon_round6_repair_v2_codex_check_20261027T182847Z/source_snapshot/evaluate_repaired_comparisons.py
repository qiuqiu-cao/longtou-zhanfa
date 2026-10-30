"""
evaluate_repaired_comparisons.py:

Executes a comprehensive, transparent, reproducible evaluation comparing:
1. Original Round 4
2. Original Round 6
3. Repaired Arm 0 (Clean R4 Baseline: Space 0.55 + 5B 0.45)
4. Repaired Arm 1 (Clean R6 Full-Horizon: Space 0.50 + 5B 0.35 + Gain46 0.15)
5. Repaired Arm 2 (Dual-Track Space Core: Space 0.60 + 5B 0.40)
6. BASE Historical Model (on complete competitive dates)

Computes full execution funnels, Top-1, Top-2, non-one-word subsets, and risk metrics.
"""

import sys
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(".")
REPAIR_DIR = ROOT / "research/model_upgrade_validation/dragon_round6_repaired_20261027T180000Z"
R6_ORIG_DIR = ROOT / "research/model_upgrade_validation/dragon_auction_and_second_wave_20261027T091600Z"
R4_ORIG_DIR = ROOT / "research/model_upgrade_validation/dragon_theme_matrix_20261027T083500Z"

def wilson_interval(hits, n):
    if n == 0:
        return [0.0, 0.0]
    p = hits / n
    z = 1.959963984540054
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [round(center - half, 4), round(center + half, 4)]

def compute_arm_stats(top_df, name):
    n = len(top_df)
    ans = {
        "arm_name": name,
        "n_days": n,
        "is_3to4": {
            "hits": int(top_df['is_3to4'].sum()),
            "rate": round(float(top_df['is_3to4'].mean()), 4),
            "wilson95": wilson_interval(int(top_df['is_3to4'].sum()), n)
        },
        "is_3to5": {
            "hits": int(top_df['is_3to5'].sum()),
            "rate": round(float(top_df['is_3to5'].mean()), 4),
            "wilson95": wilson_interval(int(top_df['is_3to5'].sum()), n)
        },
        "is_market_space_leader": {
            "hits": int(top_df['is_market_space_leader'].sum()),
            "rate": round(float(top_df['is_market_space_leader'].mean()), 4),
            "wilson95": wilson_interval(int(top_df['is_market_space_leader'].sum()), n)
        },
        "is_emotion_dragon_proxy": {
            "hits": int(top_df['is_emotion_dragon_proxy'].sum()),
            "rate": round(float(top_df['is_emotion_dragon_proxy'].mean()), 4),
            "wilson95": wilson_interval(int(top_df['is_emotion_dragon_proxy'].sum()), n)
        },
        # Execution & Market Microstructure
        "whole_day_one_word_up": int(top_df.get('trade_one_word_up', pd.Series(0, index=top_df.index)).eq(1).sum()),
        "open_gap_ge_9p5pct": int(top_df.get('next_open_gap', pd.Series(0, index=top_df.index)).ge(0.095).sum()),
        "open_gap_pos_below_9p5pct": int((top_df.get('next_open_gap', pd.Series(0, index=top_df.index)).gt(0.0) & top_df.get('next_open_gap', pd.Series(0, index=top_df.index)).lt(0.095)).sum()),
        "open_gap_non_positive": int(top_df.get('next_open_gap', pd.Series(0, index=top_df.index)).le(0.0).sum()),
        "base_lt50": int(top_df.get('base_historical_score', pd.Series(np.nan, index=top_df.index)).lt(0.5).sum()),
        "base_missing": int(top_df.get('base_historical_score', pd.Series(np.nan, index=top_df.index)).isna().sum()),
        "execution_status_breakdown": top_df.get('execution_status', pd.Series('unknown', index=top_df.index)).value_counts(dropna=False).to_dict(),
        "mean_10d_max_return": round(float(top_df.get('forward_10d_max_return', pd.Series(np.nan, index=top_df.index)).dropna().mean()), 4) if len(top_df.get('forward_10d_max_return', pd.Series(np.nan, index=top_df.index)).dropna()) else None,
        "negative_10th_close_count": int(top_df.get('forward_10d_close_return', pd.Series(np.nan, index=top_df.index)).lt(0).sum()),
        "negative_10th_close_rate": round(float(top_df.get('forward_10d_close_return', pd.Series(np.nan, index=top_df.index)).lt(0).mean()), 4) if len(top_df) else None,
        "positive_max_but_negative_10th_close": int((top_df.get('forward_10d_max_return', pd.Series(0, index=top_df.index)).gt(0) & top_df.get('forward_10d_close_return', pd.Series(0, index=top_df.index)).lt(0)).sum())
    }
    
    # Non-one-word sub-cohort stats (True actionable liquidity)
    sub_actionable = top_df[top_df.get('trade_one_word_up', pd.Series(0, index=top_df.index)) != 1]
    n_act = len(sub_actionable)
    ans["actionable_non_one_word"] = {
        "n_days": n_act,
        "is_3to4_hits": int(sub_actionable['is_3to4'].sum()),
        "is_3to4_rate": round(float(sub_actionable['is_3to4'].mean()), 4) if n_act else 0,
        "is_space_hits": int(sub_actionable['is_market_space_leader'].sum()),
        "is_space_rate": round(float(sub_actionable['is_market_space_leader'].mean()), 4) if n_act else 0,
        "is_5b_hits": int(sub_actionable['is_3to5'].sum()),
        "is_5b_rate": round(float(sub_actionable['is_3to5'].mean()), 4) if n_act else 0
    }
    return ans

def main():
    print("Loading datasets for definitive comparison...")
    oof_rep = pd.read_csv(REPAIR_DIR / "oof_repaired_predictions.csv", dtype={'code': str})
    oof_rep['code_str'] = oof_rep['code'].str.zfill(6)
    
    # Load original R4 and R6 OOF
    oof_r4 = pd.read_csv(R4_ORIG_DIR / "oof_round4_predictions.csv", dtype={'code': str})
    oof_r4['code_str'] = oof_r4['code'].str.zfill(6)
    
    oof_r6 = pd.read_csv(R6_ORIG_DIR / "oof_round6_predictions.csv", dtype={'code': str})
    oof_r6['code_str'] = oof_r6['code'].str.zfill(6)
    
    # Pick Top-1 for each arm (tie-breaker: code ascending)
    def pick_top1(df_oof, col):
        return df_oof.sort_values(['date', col, 'code_str'], ascending=[True, False, True]).groupby('date', sort=True).head(1).copy()
        
    def pick_top2(df_oof, col):
        return df_oof.sort_values(['date', col, 'code_str'], ascending=[True, False, True]).groupby('date', sort=True).head(2).copy()

    top_r4 = pick_top1(oof_r4, 'score_round4_composite')
    top_r6 = pick_top1(oof_r6, 'score_round6_total_dragon')
    top_arm0 = pick_top1(oof_rep, 'score_arm0_r4')
    top_arm1 = pick_top1(oof_rep, 'score_arm1_full')
    top_arm2 = pick_top1(oof_rep, 'score_arm2_primary')

    stats_r4 = compute_arm_stats(top_r4, "Original Round 4")
    stats_r6 = compute_arm_stats(top_r6, "Original Round 6")
    stats_arm0 = compute_arm_stats(top_arm0, "Repaired Arm 0 (Clean R4 Baseline)")
    stats_arm1 = compute_arm_stats(top_arm1, "Repaired Arm 1 (Clean R6 Full-Horizon)")
    stats_arm2 = compute_arm_stats(top_arm2, "Repaired Arm 2 (Dual-Track Space Core)")

    # Top-2 Evaluation
    top2_arm0 = pick_top2(oof_rep, 'score_arm0_r4')
    top2_arm2 = pick_top2(oof_rep, 'score_arm2_primary')
    
    def top2_stats(top2_df, name):
        agg = top2_df.groupby('date').agg(
            hit_3to4=('is_3to4', 'max'),
            hit_space=('is_market_space_leader', 'max'),
            hit_5b=('is_3to5', 'max')
        )
        return {
            "name": name,
            "days": len(agg),
            "top2_3to4_hits": int(agg['hit_3to4'].sum()),
            "top2_3to4_rate": round(float(agg['hit_3to4'].mean()), 4),
            "top2_space_hits": int(agg['hit_space'].sum()),
            "top2_space_rate": round(float(agg['hit_space'].mean()), 4),
            "top2_5b_hits": int(agg['hit_5b'].sum()),
            "top2_5b_rate": round(float(agg['hit_5b'].mean()), 4),
        }
        
    top2_arm0_stats = top2_stats(top2_arm0, "Repaired Arm 0 Top-2")
    top2_arm2_stats = top2_stats(top2_arm2, "Repaired Arm 2 Top-2")

    # Complete Competitive Dates BASE vs Repaired Arm 2
    comp = oof_rep.groupby("date").filter(lambda g: len(g) >= 2)
    complete = comp[comp['base_cohort_match_status'].eq("COMPLETE_MATCH")]
    complete_arm2 = pick_top1(complete, 'score_arm2_primary')
    complete_base = pick_top1(complete, 'base_historical_score')
    
    stats_comp_arm2 = compute_arm_stats(complete_arm2, "Complete Competitive Dates: Repaired Arm 2")
    stats_comp_base = compute_arm_stats(complete_base, "Complete Competitive Dates: BASE Historical")

    # Full Execution Funnel for Arm 2 Top-1
    # 1. Non-cumulative Independent Checks
    f_total = len(top_arm2)
    top_arm2_base_pass = top_arm2[top_arm2['base_historical_score'] >= 0.50]
    f_base_pass = len(top_arm2_base_pass)
    top_arm2_auction_pass = top_arm2[top_arm2['next_open_gap'] > 0.0]
    f_auction_pass = len(top_arm2_auction_pass)
    top_arm2_buyable = top_arm2[(top_arm2['next_open_gap'] > 0.0) & (top_arm2['trade_one_word_up'] != 1)]
    f_buyable = len(top_arm2_buyable)
    top_arm2_filled = top_arm2_buyable[top_arm2_buyable['execution_status'] == 'inferred_daily_limit_touch_fill']
    f_filled = len(top_arm2_filled)

    # 2. Strict Cumulative Cascading Funnel (Cascade Filter addressing Audit Problem 1)
    c_s1 = top_arm2
    c_s2 = c_s1[(c_s1['base_historical_score'] >= 0.50) & (c_s1['base_cohort_match_status'] == 'COMPLETE_MATCH')]
    c_s3 = c_s2[c_s2['next_open_gap'] > 0.0]
    c_s4 = c_s3[c_s3['trade_one_word_up'] != 1]
    c_s5 = c_s4[c_s4['execution_status'] == 'inferred_daily_limit_touch_fill']

    cumulative_cascading_funnel_arm2 = {
        "step_1_total_candidates": len(c_s1),
        "step_2_base_qualified_complete_ge50": {
            "n": len(c_s2),
            "share": round(len(c_s2) / f_total, 4),
            "3to4_hits": int(c_s2['is_3to4'].sum()),
            "3to4_rate": round(float(c_s2['is_3to4'].mean()), 4) if len(c_s2) else 0
        },
        "step_3_cumulative_auction_pass_open_gt0": {
            "n": len(c_s3),
            "share": round(len(c_s3) / f_total, 4),
            "3to4_hits": int(c_s3['is_3to4'].sum()),
            "3to4_rate": round(float(c_s3['is_3to4'].mean()), 4) if len(c_s3) else 0
        },
        "step_4_cumulative_actionable_liquid": {
            "n": len(c_s4),
            "share": round(len(c_s4) / f_total, 4),
            "3to4_hits": int(c_s4['is_3to4'].sum()),
            "3to4_rate": round(float(c_s4['is_3to4'].mean()), 4) if len(c_s4) else 0
        },
        "step_5_cumulative_limit_fill_proxy": {
            "n": len(c_s5),
            "share": round(len(c_s5) / f_total, 4),
            "3to4_hits": int(c_s5['is_3to4'].sum()),
            "3to4_rate": round(float(c_s5['is_3to4'].mean()), 4) if len(c_s5) else 0,
            "space_hits": int(c_s5['is_market_space_leader'].sum()),
            "space_rate": round(float(c_s5['is_market_space_leader'].mean()), 4) if len(c_s5) else 0,
            "caveat": "Only daily high limit-touch proxy; micro-queue order and 6-labels remain unverified at tick level"
        }
    }

    # Breakdown of the 206 non-cumulative filled cases:
    filled_206_base_lt50 = int(((top_arm2_filled['base_historical_score'] < 0.50) & top_arm2_filled['base_historical_score'].notna()).sum())
    filled_206_base_missing = int((top_arm2_filled['base_historical_score'].isna() | (top_arm2_filled['base_cohort_match_status'] != 'COMPLETE_MATCH')).sum())
    filled_206_base_qualified = int(((top_arm2_filled['base_historical_score'] >= 0.50) & (top_arm2_filled['base_cohort_match_status'] == 'COMPLETE_MATCH')).sum())

    execution_funnel_arm2 = {
        "step_1_total_top1_signals": f_total,
        "step_2_base_qualified_ge_50": {
            "n": f_base_pass,
            "share": round(f_base_pass / f_total, 4),
            "3to4_hits": int(top_arm2_base_pass['is_3to4'].sum()),
            "3to4_rate": round(float(top_arm2_base_pass['is_3to4'].mean()), 4) if f_base_pass else 0,
            "space_hits": int(top_arm2_base_pass['is_market_space_leader'].sum()),
            "space_rate": round(float(top_arm2_base_pass['is_market_space_leader'].mean()), 4) if f_base_pass else 0
        },
        "step_3_auction_pass_open_gt_0": {
            "n": f_auction_pass,
            "share": round(f_auction_pass / f_total, 4),
            "3to4_hits": int(top_arm2_auction_pass['is_3to4'].sum()),
            "3to4_rate": round(float(top_arm2_auction_pass['is_3to4'].mean()), 4) if f_auction_pass else 0,
            "space_hits": int(top_arm2_auction_pass['is_market_space_leader'].sum()),
            "space_rate": round(float(top_arm2_auction_pass['is_market_space_leader'].mean()), 4) if f_auction_pass else 0
        },
        "step_4_actionable_liquid_open_gt_0_not_oneword": {
            "n": f_buyable,
            "share": round(f_buyable / f_total, 4),
            "3to4_hits": int(top_arm2_buyable['is_3to4'].sum()),
            "3to4_rate": round(float(top_arm2_buyable['is_3to4'].mean()), 4) if f_buyable else 0,
            "space_hits": int(top_arm2_buyable['is_market_space_leader'].sum()),
            "space_rate": round(float(top_arm2_buyable['is_market_space_leader'].mean()), 4) if f_buyable else 0
        },
        "step_5_filled_daily_limit_touch": {
            "n": f_filled,
            "share": round(f_filled / f_total, 4),
            "3to4_hits": int(top_arm2_filled['is_3to4'].sum()),
            "3to4_rate": round(float(top_arm2_filled['is_3to4'].mean()), 4) if f_filled else 0,
            "space_hits": int(top_arm2_filled['is_market_space_leader'].sum()),
            "space_rate": round(float(top_arm2_filled['is_market_space_leader'].mean()), 4) if f_filled else 0,
            "audit_breakdown_of_206": {
                "base_lt50": filled_206_base_lt50,
                "base_missing_or_incomplete": filled_206_base_missing,
                "base_qualified_complete": filled_206_base_qualified
            }
        },
        "cumulative_cascading_funnel": cumulative_cascading_funnel_arm2
    }

    # Consolidated comparison payload
    comparison_summary = {
        "arms_full_449_days": {
            "r4_original": stats_r4,
            "r6_original": stats_r6,
            "arm0_repaired_r4_baseline": stats_arm0,
            "arm1_repaired_r6_full_horizon": stats_arm1,
            "arm2_repaired_dual_track_space_core": stats_arm2
        },
        "top2_evaluation": {
            "arm0": top2_arm0_stats,
            "arm2": top2_arm2_stats
        },
        "complete_competitive_dates_100_days": {
            "arm2": stats_comp_arm2,
            "base": stats_comp_base
        },
        "execution_funnel_arm2": execution_funnel_arm2
    }

    out_file = REPAIR_DIR / "repaired_comprehensive_evaluation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved repaired comprehensive evaluation to: {out_file}")

    # Print Clean Console Summary Table
    print("\n" + "="*110)
    print("DEFINITIVE 449-DAY COMPARISON TABLE (IDENTICAL TEST POOL)")
    print("="*110)
    print(f"{'模型/方案':<35} | {'3进4 命中率':<18} | {'空间总高标率':<18} | {'连续5板突破率':<18} | {'非一字3进4胜率':<16}")
    print("-"*110)
    for k, v in comparison_summary["arms_full_449_days"].items():
        name = v["arm_name"]
        h_34 = f"{v['is_3to4']['hits']}/449 ({v['is_3to4']['rate']*100:.2f}%)"
        h_sp = f"{v['is_market_space_leader']['hits']}/449 ({v['is_market_space_leader']['rate']*100:.2f}%)"
        h_5b = f"{v['is_3to5']['hits']}/449 ({v['is_3to5']['rate']*100:.2f}%)"
        act = v["actionable_non_one_word"]
        h_act = f"{act['is_3to4_hits']}/{act['n_days']} ({act['is_3to4_rate']*100:.2f}%)"
        print(f"{name:<33} | {h_34:<18} | {h_sp:<18} | {h_5b:<18} | {h_act:<16}")
    print("="*110)

    print("\n--- TOP-2 EVALUATION (COVERAGE GAIN) ---")
    print(f"Repaired Arm 0 Top-2: 3进4={top2_arm0_stats['top2_3to4_hits']}/449 ({top2_arm0_stats['top2_3to4_rate']*100:.2f}%), Space={top2_arm0_stats['top2_space_hits']}/449 ({top2_arm0_stats['top2_space_rate']*100:.2f}%)")
    print(f"Repaired Arm 2 Top-2: 3进4={top2_arm2_stats['top2_3to4_hits']}/449 ({top2_arm2_stats['top2_3to4_rate']*100:.2f}%), Space={top2_arm2_stats['top2_space_hits']}/449 ({top2_arm2_stats['top2_space_rate']*100:.2f}%)")

    print("\n--- COMPLETE COMPETITIVE DATES (100 DAYS) vs BASE ---")
    b_34 = f"{stats_comp_base['is_3to4']['hits']}/100 ({stats_comp_base['is_3to4']['rate']*100:.2f}%)"
    a2_34 = f"{stats_comp_arm2['is_3to4']['hits']}/100 ({stats_comp_arm2['is_3to4']['rate']*100:.2f}%)"
    b_sp = f"{stats_comp_base['is_market_space_leader']['hits']}/100 ({stats_comp_base['is_market_space_leader']['rate']*100:.2f}%)"
    a2_sp = f"{stats_comp_arm2['is_market_space_leader']['hits']}/100 ({stats_comp_arm2['is_market_space_leader']['rate']*100:.2f}%)"
    print(f"BASE Historical: 3进4={b_34}, Space Leader={b_sp}")
    print(f"Repaired Arm 2 : 3进4={a2_34}, Space Leader={a2_sp}")

    print("\n--- EXECUTION FUNNEL (ARM 2) ---")
    for step, data in execution_funnel_arm2.items():
        if step == "cumulative_cascading_funnel":
            print("\n  >>> STRICT CUMULATIVE CASCADING FUNNEL (AUDIT FIX 1) <<<")
            for c_step, c_data in data.items():
                if isinstance(c_data, dict):
                    print(f"    {c_step}: N={c_data['n']} ({c_data['share']*100:.1f}%), 3进4={c_data['3to4_hits']}/{c_data['n']} ({c_data['3to4_rate']*100:.1f}%)")
                else:
                    print(f"    {c_step}: N={c_data}")
        elif isinstance(data, dict):
            if 'n' in data:
                sp_hit = data.get('space_hits', '-')
                sp_rate = f"{data['space_rate']*100:.1f}%" if 'space_rate' in data else '-'
                print(f"  {step}: N={data['n']} ({data['share']*100:.1f}%), 3进4={data['3to4_hits']}/{data['n']} ({data['3to4_rate']*100:.1f}%), Space={sp_hit}/{data['n']} ({sp_rate})")
            else:
                print(f"  {step}: {data}")
        else:
            print(f"  {step}: N={data}")

if __name__ == "__main__":
    main()
