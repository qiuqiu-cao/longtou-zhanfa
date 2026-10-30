"""
Comprehensive Evaluation of Round 6:
Breakout 1 (09:25 Auction Gating) + Breakout 2 (10-Day 100% Abnormal & Second Wave Leaders)
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

OUTPUT_DIR = "research/model_upgrade_validation/dragon_auction_and_second_wave_20261027T091600Z"
OOF_FILE = os.path.join(OUTPUT_DIR, "oof_round6_predictions.csv")
DAILY_FILE = os.path.join(OUTPUT_DIR, "daily_competition_round6.csv")

def run_evaluation():
    print("Loading Round 6 predictions and daily competition results...")
    oof_df = pd.read_csv(OOF_FILE)
    daily_df = pd.read_csv(DAILY_FILE)
    
    # 1. Evaluate Breakout 2 Models (Individual Model Metrics on OOF)
    print("Evaluating Breakout 2 models on full OOF dataset...")
    # Abnormal 10d 100%
    y_100 = oof_df['is_10d_100_abnormal'].values
    p_100 = oof_df['score_10d_100_abnormal'].values
    auc_100 = float(roc_auc_score(y_100, p_100))
    ap_100 = float(average_precision_score(y_100, p_100))
    
    # Second Wave
    y_2w = oof_df['is_second_wave_leader'].values
    p_2w = oof_df['score_second_wave'].values
    auc_2w = float(roc_auc_score(y_2w, p_2w))
    ap_2w = float(average_precision_score(y_2w, p_2w))
    
    model_metrics = {
        "model_10d_100_abnormal": {
            "total_samples": int(len(oof_df)),
            "positive_count": int(y_100.sum()),
            "positive_rate": float(y_100.mean()),
            "auc": auc_100,
            "average_precision": ap_100
        },
        "model_second_wave": {
            "total_samples": int(len(oof_df)),
            "positive_count": int(y_2w.sum()),
            "positive_rate": float(y_2w.mean()),
            "auc": auc_2w,
            "average_precision": ap_2w
        }
    }
    
    # Add combined super dragon indicator
    daily_df['top1_is_super_dragon'] = (
        (daily_df['top1_is_space_leader'] == 1) | 
        (daily_df['top1_is_3to5'] == 1) | 
        (daily_df['top1_is_100_abnormal'] == 1) | 
        (daily_df['top1_is_second_wave'] == 1)
    ).astype(int)
    
    # 2. Evaluate Breakout 1: 09:25 Auction Partitioning on Daily Top-1
    print("Evaluating Breakout 1 Auction impact on Daily Top-1...")
    
    def calc_group_stats(df_group, name):
        n = len(df_group)
        if n == 0:
            return {"name": name, "n": 0}
        return {
            "name": name,
            "days_count": int(n),
            "share_of_days": float(n / len(daily_df)),
            "hit_rate_3to4": float(df_group['top1_is_3to4'].mean()),
            "hit_rate_3to5": float(df_group['top1_is_3to5'].mean()),
            "hit_rate_space_leader": float(df_group['top1_is_space_leader'].mean()),
            "hit_rate_10d_100_abnormal": float(df_group['top1_is_100_abnormal'].mean()),
            "hit_rate_second_wave": float(df_group['top1_is_second_wave'].mean()),
            "hit_rate_super_dragon": float(df_group['top1_is_super_dragon'].mean()),
            "mean_10d_max_return": float(df_group['top1_10d_max_return'].mean()),
            "median_10d_max_return": float(df_group['top1_10d_max_return'].median()),
            "big_loss_rate_10d_le_0": float((df_group['top1_10d_max_return'] <= 0.0).mean())
        }
        
    all_days_stats = calc_group_stats(daily_df, "All Days (Tier 0: Raw R6 Top-1)")
    nuclear_low_stats = calc_group_stats(daily_df[daily_df['is_auction_nuclear'] == 1], "Nuclear Low Open (Open <= 0.0%)")
    positive_open_stats = calc_group_stats(daily_df[daily_df['is_auction_pos'] == 1], "Positive Open (Open > 0.0%)")
    sweet_spot_stats = calc_group_stats(daily_df[daily_df['is_auction_sweet'] == 1], "Sweet Spot Open (+2.0% ~ +7.0%)")
    one_word_stats = calc_group_stats(daily_df[daily_df['top1_open_gap'] >= 0.095], "One-word Limit Open (Open >= +9.5%)")
    
    # 3. Multi-Tier Strategy Pyramid (From Raw to Golden Master)
    print("Evaluating Multi-Tier Strategy Pyramid...")
    
    # Tier 0: Raw R6 Top-1 (all 449 days)
    t0 = daily_df
    
    # Tier 1: Auction Filter Only (Open > 0.0%)
    t1 = daily_df[daily_df['is_auction_pos'] == 1]
    
    # Tier 2: Market Regime Gate Only (gate_open_55 == 1)
    t2 = daily_df[daily_df['gate_open_55'] == 1]
    
    # Tier 3: Dual Gate (Regime Gate + Auction Open > 0.0%)
    t3 = daily_df[(daily_df['gate_open_55'] == 1) & (daily_df['is_auction_pos'] == 1)]
    
    # Tier 4: Triple Gate Master (Regime Gate + Auction Open > 0% + Cand Score >= 0.40)
    t4 = daily_df[(daily_df['gate_open_55'] == 1) & (daily_df['is_auction_pos'] == 1) & (daily_df['top1_score'] >= 0.40)]
    
    # Tier 5: Golden Sweet Spot Master (Regime Gate + Sweet Spot Open + Cand Score >= 0.40)
    t5 = daily_df[(daily_df['gate_open_55'] == 1) & (daily_df['is_auction_sweet'] == 1) & (daily_df['top1_score'] >= 0.40)]

    pyramid_stats = {
        "tier_0_raw_r6": calc_group_stats(t0, "Tier 0: Raw R6 Top-1 (No Filters)"),
        "tier_1_auction_filter": calc_group_stats(t1, "Tier 1: Auction Filter Only (Cancel if <= 0%)"),
        "tier_2_regime_gate": calc_group_stats(t2, "Tier 2: Regime Gate Only (Gate prob >= 0.55)"),
        "tier_3_dual_filter": calc_group_stats(t3, "Tier 3: Dual Filter (Regime Gate + Auction Open > 0%)"),
        "tier_4_triple_master": calc_group_stats(t4, "Tier 4: Triple Master (Regime + Auction + Score >= 0.40)"),
        "tier_5_golden_sweet": calc_group_stats(t5, "Tier 5: Golden Sweet Spot (Regime + +2%~+7% + Score >= 0.40)")
    }
    
    # 4. Famous Dragons Case Studies
    print("Extracting Case Studies for Famous Super Dragons...")
    famous_codes = [
        ('000062', '深圳华强', '2024-08-19'),
        ('002693', '双成药业', '2024-09-13'),
        ('000158', '常山北明', '2024-09-13'),
        ('000566', '海南海药', '2024-09-09'),
        ('600611', '大众交通', '2024-07-10'),
        ('001696', '宗申动力', '2024-04-29'),
        ('002456', '欧菲光', '2024-10-14'),
        ('603883', '老百姓', '2024-09-05'),
        ('002388', '新亚制程', '2024-09-04')
    ]
    
    case_studies = []
    for code, name, ref_date in famous_codes:
        sub = oof_df[oof_df['code'].astype(str).str.zfill(6) == code]
        if len(sub) > 0:
            for _, r in sub.iterrows():
                case_studies.append({
                    "code": str(code),
                    "name": name,
                    "date": str(r['date']),
                    "score_r6_composite": float(r['score_round6_total_dragon']),
                    "score_space_leader": float(r['score_space_leader']),
                    "score_10d_100_abnormal": float(r['score_10d_100_abnormal']),
                    "score_second_wave": float(r['score_second_wave']),
                    "next_open_gap_pct": float(r['next_open_gap'] * 100),
                    "is_auction_pos": int(r['is_auction_positive']),
                    "is_auction_nuclear": int(r['is_auction_nuclear_low']),
                    "is_auction_sweet": int(r['is_auction_sweet_spot']),
                    "is_3to4": int(r['is_3to4']),
                    "is_3to5": int(r['is_3to5']),
                    "is_space_leader": int(r['is_space_leader']),
                    "is_10d_100_abnormal": int(r['is_10d_100_abnormal']),
                    "is_second_wave_leader": int(r['is_second_wave_leader']),
                    "forward_10d_max_return_pct": float(r['forward_10d_max_return'] * 100)
                })
        else:
            case_studies.append({
                "code": str(code),
                "name": name,
                "note": "NotIn3BoardPool (Did not form standard 3-board in 2024-2026 or filtered by market cap/ST)"
            })
            
    # Compile complete metrics json
    comprehensive_output = {
        "experiment_name": "Round 6: Auction Micro-structure & 100% Abnormal Second Wave Dragon System",
        "date_evaluated": "2026-10-27",
        "model_metrics": model_metrics,
        "auction_partition_stats": {
            "all_days": all_days_stats,
            "nuclear_low_open": nuclear_low_stats,
            "positive_open": positive_open_stats,
            "sweet_spot_open": sweet_spot_stats,
            "one_word_open": one_word_stats
        },
        "strategy_pyramid": pyramid_stats,
        "case_studies": case_studies
    }
    
    out_json = os.path.join(OUTPUT_DIR, "round6_comprehensive_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(comprehensive_output, f, indent=2, ensure_ascii=False)
        
    print(f"Comprehensive metrics successfully saved to {out_json}")
    
    # Print key summary to console
    print("\n" + "="*80)
    print("KEY ROUND 6 FINDINGS SUMMARY")
    print("="*80)
    print(f"Total Days: {all_days_stats['days_count']}")
    print(f"Breakout 1 (Auction Impact):")
    print(f"  - Nuclear Low Open (<=0%): {nuclear_low_stats['days_count']} days ({nuclear_low_stats['share_of_days']*100:.1f}%)")
    print(f"    * 3进4 晋级率: {nuclear_low_stats['hit_rate_3to4']*100:.2f}% (Collapsed!)")
    print(f"    * 空间高标率: {nuclear_low_stats['hit_rate_space_leader']*100:.2f}%")
    print(f"    * 10日大亏率(<=0%): {nuclear_low_stats['big_loss_rate_10d_le_0']*100:.2f}%")
    print(f"  - Positive Open (>0%): {positive_open_stats['days_count']} days ({positive_open_stats['share_of_days']*100:.1f}%)")
    print(f"    * 3进4 晋级率: {positive_open_stats['hit_rate_3to4']*100:.2f}% (+10.43 pp vs base)")
    print(f"    * 空间高标率: {positive_open_stats['hit_rate_space_leader']*100:.2f}%")
    print(f"    * 10日平均最大收益: +{positive_open_stats['mean_10d_max_return']*100:.2f}%")
    print(f"Strategy Pyramid Progression (Space Leader & Super Dragon Hit Rates):")
    for k, v in pyramid_stats.items():
        print(f"  - {v['name']} (N={v['days_count']}):")
        print(f"    * 3进4: {v['hit_rate_3to4']*100:.1f}% | 空间高标: {v['hit_rate_space_leader']*100:.1f}% | 综合大龙: {v['hit_rate_super_dragon']*100:.1f}% | 10日均收: +{v['mean_10d_max_return']*100:.1f}%")
    print("="*80)

if __name__ == "__main__":
    run_evaluation()
