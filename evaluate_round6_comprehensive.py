"""evaluate_round6_comprehensive.py

Comprehensive evaluation of Round 6:
Conquering both:
- 突破口一：次日 09:25 集合竞价微观盘口确认 (Auction Execution Confirmation)
- 突破口二：10 日累计 100 异动大二波总龙模型 (10-Day 100% Abnormal Price Movement & Second Wave Leader)

Metrics Evaluated:
1. Executive Multi-Objective Matrix:
   - Space Leadership
   - 5-Board Breakthrough
   - 3-to-4 Advancement
   - 10-Day 100% Abnormal Gain Runner
   - Second Wave Rebound Dragon
   - Average 10-Day Maximum Forward Return
2. 09:25 Auction Confirmation Uplift & Nuclear Low-Open Cancellation Power
3. Triple-Filter Golden Master System:
   - Regime Gate OPEN (prob >= 0.55)
   - Auction High Open Confirmed (> 0.0%)
   - Candidate Confidence Score >= 0.40
4. Generates round6_comprehensive_metrics.json.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent
EXP_DIR = ROOT / "research/model_upgrade_validation/dragon_auction_and_second_wave_20261027T091600Z"
DAILY_PATH = EXP_DIR / "daily_competition_round6.csv"
OOF_PATH = EXP_DIR / "oof_round6_predictions.csv"

def evaluate():
    print(f"Loading Round 6 competition results from {DAILY_PATH}...")
    daily = pd.read_csv(DAILY_PATH)
    oof = pd.read_csv(OOF_PATH)
    
    total_days = len(daily)
    total_events = len(oof)
    
    # 1. Natural Base Rates in 3-board cohort
    base_space = float(oof['is_market_space_leader'].mean())
    base_5b = float(oof['is_3to5'].mean())
    base_3to4 = float(oof['is_3to4'].mean())
    base_100 = float(oof['is_10d_100_abnormal'].mean())
    base_2w = float(oof['is_second_wave_leader'].mean())
    base_10d_ret = float(oof['forward_10d_max_return'].mean())
    
    # 2. Mode A: Baseline (EOD Top-1 Selection, No Gate, No Auction Filter)
    top1_space = float(daily['top1_is_space_leader'].mean())
    top1_5b = float(daily['top1_is_3to5'].mean())
    top1_3to4 = float(daily['top1_is_3to4'].mean())
    top1_100 = float(daily['top1_is_100_abnormal'].mean())
    top1_2w = float(daily['top1_is_second_wave'].mean())
    top1_10d_ret = float(daily['top1_10d_max_return'].mean())
    top1_combined_dragon = float(((daily['top1_is_100_abnormal'] == 1) | (daily['top1_is_second_wave'] == 1)).mean())
    
    mode_baseline = {
        "days": total_days,
        "coverage_pct": 1.0,
        "space_leader_hit": top1_space,
        "height_5b_hit": top1_5b,
        "advancement_3to4_hit": top1_3to4,
        "abnormal_100_hit": top1_100,
        "second_wave_hit": top1_2w,
        "combined_dragon_hit": top1_combined_dragon,
        "avg_10d_max_return": top1_10d_ret
    }
    
    # 3. Mode B: 09:25 Auction Filter Alone (High Open > 0.0%, Nuclear Low Cancelled)
    auction_pass = daily[daily['is_auction_pos'] == 1].copy()
    auction_nuclear = daily[daily['is_auction_nuclear'] == 1].copy()
    
    mode_auction_pass = {
        "days": len(auction_pass),
        "coverage_pct": float(len(auction_pass) / total_days),
        "nuclear_days_cancelled": len(auction_nuclear),
        "nuclear_days_cancelled_pct": float(len(auction_nuclear) / total_days),
        "nuclear_3to4_fail_rate": float(1.0 - auction_nuclear['top1_is_3to4'].mean()),
        "space_leader_hit": float(auction_pass['top1_is_space_leader'].mean()),
        "height_5b_hit": float(auction_pass['top1_is_3to5'].mean()),
        "advancement_3to4_hit": float(auction_pass['top1_is_3to4'].mean()),
        "abnormal_100_hit": float(auction_pass['top1_is_100_abnormal'].mean()),
        "second_wave_hit": float(auction_pass['top1_is_second_wave'].mean()),
        "combined_dragon_hit": float(((auction_pass['top1_is_100_abnormal'] == 1) | (auction_pass['top1_is_second_wave'] == 1)).mean()),
        "avg_10d_max_return": float(auction_pass['top1_10d_max_return'].mean())
    }
    
    # 4. Mode C: Regime Gate Alone (prob >= 0.55)
    regime_pass = daily[daily['gate_open_55'] == 1].copy()
    mode_regime_pass = {
        "days": len(regime_pass),
        "coverage_pct": float(len(regime_pass) / total_days),
        "space_leader_hit": float(regime_pass['top1_is_space_leader'].mean()),
        "height_5b_hit": float(regime_pass['top1_is_3to5'].mean()),
        "advancement_3to4_hit": float(regime_pass['top1_is_3to4'].mean()),
        "abnormal_100_hit": float(regime_pass['top1_is_100_abnormal'].mean()),
        "second_wave_hit": float(regime_pass['top1_is_second_wave'].mean()),
        "combined_dragon_hit": float(((regime_pass['top1_is_100_abnormal'] == 1) | (regime_pass['top1_is_second_wave'] == 1)).mean()),
        "avg_10d_max_return": float(regime_pass['top1_10d_max_return'].mean())
    }
    
    # 5. Mode D: Dual Gate (Regime Gate + Auction High Open)
    dual_pass = daily[(daily['gate_open_55'] == 1) & (daily['is_auction_pos'] == 1)].copy()
    mode_dual_pass = {
        "days": len(dual_pass),
        "coverage_pct": float(len(dual_pass) / total_days),
        "space_leader_hit": float(dual_pass['top1_is_space_leader'].mean()),
        "height_5b_hit": float(dual_pass['top1_is_3to5'].mean()),
        "advancement_3to4_hit": float(dual_pass['top1_is_3to4'].mean()),
        "abnormal_100_hit": float(dual_pass['top1_is_100_abnormal'].mean()),
        "second_wave_hit": float(dual_pass['top1_is_second_wave'].mean()),
        "combined_dragon_hit": float(((dual_pass['top1_is_100_abnormal'] == 1) | (dual_pass['top1_is_second_wave'] == 1)).mean()),
        "avg_10d_max_return": float(dual_pass['top1_10d_max_return'].mean())
    }
    
    # 6. Mode E: Triple Golden Master System (Regime + Auction + Candidate Score >= 0.40)
    golden_master = daily[(daily['gate_open_55'] == 1) & (daily['is_auction_pos'] == 1) & (daily['top1_score'] >= 0.40)].copy()
    mode_golden_master = {
        "days": len(golden_master),
        "coverage_pct": float(len(golden_master) / total_days),
        "space_leader_hit": float(golden_master['top1_is_space_leader'].mean()),
        "height_5b_hit": float(golden_master['top1_is_3to5'].mean()),
        "advancement_3to4_hit": float(golden_master['top1_is_3to4'].mean()),
        "abnormal_100_hit": float(golden_master['top1_is_100_abnormal'].mean()),
        "second_wave_hit": float(golden_master['top1_is_second_wave'].mean()),
        "combined_dragon_hit": float(((golden_master['top1_is_100_abnormal'] == 1) | (golden_master['top1_is_second_wave'] == 1)).mean()),
        "avg_10d_max_return": float(golden_master['top1_10d_max_return'].mean())
    }
    
    metrics = {
        "round": 6,
        "conquered_directions": [
            "Direction 3: 次日 09:25 集合竞价盘口与微观挂单结构确认",
            "Direction 5: 10 日累计 100 异动大二波总龙模型"
        ],
        "total_test_days": total_days,
        "cohort_natural_base_rates": {
            "is_market_space_leader": base_space,
            "is_3to5": base_5b,
            "is_3to4": base_3to4,
            "is_10d_100_abnormal": base_100,
            "is_second_wave_leader": base_2w,
            "avg_10d_max_return": base_10d_ret
        },
        "mode_baseline": mode_baseline,
        "mode_auction_pass": mode_auction_pass,
        "mode_regime_pass": mode_regime_pass,
        "mode_dual_pass": mode_dual_pass,
        "mode_golden_master": mode_golden_master
    }
    
    out_json = EXP_DIR / "round6_comprehensive_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
        
    print(f"\nEvaluation successfully completed and saved to {out_json}")
    print("\n=== Executive Summary of Round 6 Dual Breakthrough ===")
    print(f"1. Baseline (All 449 Days): Space={top1_space:.2%}, 5b={top1_5b:.2%}, 3-4={top1_3to4:.2%}, 100% Abnormal={top1_100:.2%}, 2nd Wave={top1_2w:.2%}")
    print(f"2. Auction Pass (High Open >0%, 353 Days): Space={mode_auction_pass['space_leader_hit']:.2%}, 5b={mode_auction_pass['height_5b_hit']:.2%}, 3-4={mode_auction_pass['advancement_3to4_hit']:.2%}")
    print(f"   (Nuclear low-open cancellation eliminated {mode_auction_pass['nuclear_days_cancelled']} traps with {mode_auction_pass['nuclear_3to4_fail_rate']:.1%} failure rate!)")
    print(f"3. Dual Pass (Regime + Auction, 232 Days): Space={mode_dual_pass['space_leader_hit']:.2%}, 5b={mode_dual_pass['height_5b_hit']:.2%}, 3-4={mode_dual_pass['advancement_3to4_hit']:.2%}, Combined Dragon={mode_dual_pass['combined_dragon_hit']:.2%}")
    print(f"4. Triple Golden Master (Regime + Auction + Score >= 0.40, {len(golden_master)} Days):")
    print(f"   Space Leader Hit: {mode_golden_master['space_leader_hit']:.2%} (vs Base 19.86%, Alpha: +{(mode_golden_master['space_leader_hit']-base_space)*100:.2f} pp)")
    print(f"   5-Board Breakthrough: {mode_golden_master['height_5b_hit']:.2%} (vs Base 22.50%, Alpha: +{(mode_golden_master['height_5b_hit']-base_5b)*100:.2f} pp)")
    print(f"   3-to-4 Advancement: {mode_golden_master['advancement_3to4_hit']:.2%} (vs Base 47.84%, Alpha: +{(mode_golden_master['advancement_3to4_hit']-base_3to4)*100:.2f} pp)")
    print(f"   10-Day 100% Abnormal Runner: {mode_golden_master['abnormal_100_hit']:.2%}")
    print(f"   Second Wave Rebound Dragon: {mode_golden_master['second_wave_hit']:.2%}")
    print(f"   Combined Dragon Hit Rate: {mode_golden_master['combined_dragon_hit']:.2%}")
    print(f"   Average 10-Day Max Return: {mode_golden_master['avg_10d_max_return']:.2%}")

if __name__ == "__main__":
    evaluate()
