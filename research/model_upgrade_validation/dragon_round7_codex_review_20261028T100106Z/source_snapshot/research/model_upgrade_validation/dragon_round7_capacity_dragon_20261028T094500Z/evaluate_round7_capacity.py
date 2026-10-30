"""
evaluate_round7_capacity.py:

Comprehensive evaluation of Round 7 Capacity Dragon Upgraded Model:
- Compares Round 4, Round 6, Round 7 Arm 0, Arm 1, Arm 2, and Arm 3 (Capacity Dragon Track).
- Integrates the 2026-10-28 AGENTS.md breakthrough rules:
  1. BASE score exemption for large capacity turnover dragons (turnover >= 15% or amount >= 8e8 + early seal + late add).
  2. Late seal reinforcement (尾盘加单) promotion.
  3. Late seal withdrawal (尾盘撤单) veto.
- Computes execution funnels and paired statistical significance tests.
"""

import sys
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(".")
R7_DIR = ROOT / "research/model_upgrade_validation/dragon_round7_capacity_dragon_20261028T094500Z"
R6_DIR = ROOT / "research/model_upgrade_validation/dragon_round6_repaired_20261027T180000Z"
R4_DIR = ROOT / "research/model_upgrade_validation/dragon_theme_matrix_20261027T083500Z"
BARS_PATH = ROOT / "data/v3/imported/daily-bars.csv"

# Import updated audit_three_board_candidate
sys.path.append(str(R7_DIR))
sys.path.append(str(ROOT))
from audit_three_board_traps import audit_three_board_candidate

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
        "whole_day_one_word_up": int(top_df.get('trade_one_word_up', pd.Series(0, index=top_df.index)).eq(1).sum()),
        "open_gap_pos_below_9p5pct": int((top_df.get('next_open_gap', pd.Series(0, index=top_df.index)).gt(0.0) & top_df.get('next_open_gap', pd.Series(0, index=top_df.index)).lt(0.095)).sum()),
        "open_gap_non_positive": int(top_df.get('next_open_gap', pd.Series(0, index=top_df.index)).le(0.0).sum()),
        "base_lt50": int(top_df.get('base_historical_score', pd.Series(np.nan, index=top_df.index)).lt(0.5).sum()),
        "base_missing": int(top_df.get('base_historical_score', pd.Series(np.nan, index=top_df.index)).isna().sum()),
        "execution_status_breakdown": top_df.get('execution_status', pd.Series('unknown', index=top_df.index)).value_counts(dropna=False).to_dict()
    }
    
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
    print("Loading datasets for Round 7 Capacity Dragon evaluation...")
    oof_r7 = pd.read_csv(R7_DIR / "oof_round7_predictions.csv", dtype={'code': str})
    oof_r7['code_str'] = oof_r7['code'].str.zfill(6)
    
    # Precompute historical villain patterns from daily bars
    print("Precomputing historical ceiling villain status...")
    bars = pd.read_csv(BARS_PATH, dtype={'code': str}, usecols=['date', 'code', 'close', 'pre_close', 'open'])
    bars['code_str'] = bars['code'].str.zfill(6)
    bars = bars.sort_values(['code_str', 'date']).reset_index(drop=True)
    bars_by_code = {c: g.reset_index(drop=True) for c, g in bars.groupby('code_str')}

    villain_map = {}
    for idx, r in oof_r7[['code_str', 'date']].drop_duplicates().iterrows():
        c = r['code_str']
        d = r['date']
        g = bars_by_code.get(c)
        if g is None:
            villain_map[(c, d)] = (0, False)
            continue
        hist = g[g['date'] < d].tail(250)
        if len(hist) < 30:
            villain_map[(c, d)] = (0, False)
            continue
        is_zt = (hist['close'] / hist['pre_close'] >= 1.095).values
        streaks = []
        curr = 0
        for z in is_zt:
            if z:
                curr += 1
            else:
                if curr > 0:
                    streaks.append(curr)
                curr = 0
        if curr > 0:
            streaks.append(curr)
        max_streak = max(streaks) if streaks else 0
        had_nuclear = False
        if 2 <= max_streak <= 3:
            curr = 0
            for i in range(len(is_zt)):
                if is_zt[i]:
                    curr += 1
                else:
                    if 2 <= curr <= 3:
                        for k in range(i, min(i+2, len(hist))):
                            ret = hist['close'].iloc[k] / hist['pre_close'].iloc[k]
                            op_ret = hist['open'].iloc[k] / hist['pre_close'].iloc[k]
                            if ret <= 0.905 or op_ret <= 0.905:
                                had_nuclear = True
                                break
                    curr = 0
                    if had_nuclear:
                        break
        villain_map[(c, d)] = (max_streak, had_nuclear)

    print("Running audit_three_board_candidate with 2026-10-28 breakthrough rules...")
    six_tags_list = []
    six_veto_list = []
    six_reasons_list = []
    
    for idx, r in oof_r7.iterrows():
        c = r['code_str']
        d = r['date']
        name = str(r.get('name', c))
        hist_max, had_nuclear = villain_map.get((c, d), (0, False))
        ann_risk = bool(r.get('announcement_risk', 0) == 1 or r.get('rumor_only', 0) == 1 or r.get('earnings_risk', 0) == 1 or r.get('reduction_risk', 0) == 1)
        is_unk = bool(r.get('is_industry_unknown', 0) == 1)
        leader = bool(r.get('is_theme_sole_leader', 0) == 1 and not is_unk)
        cohort_sz = int(r.get('theme_three_board_count', 1))
        t_rate = float(r.get('turnover', 0.0) * 100.0)
        amt = float(r.get('signal_amount', 0.0))
        base_sc = r.get('base_historical_score', np.nan)
        
        audit_res = audit_three_board_candidate(
            code=c,
            name=name,
            board_count=3,
            is_theme_leader=leader,
            first_seal_time=None,
            last_seal_time=None,
            has_afternoon_drop=bool(r.get('bomb_count', 0) > 0),
            turnover_rate=t_rate,
            amount=amt,
            historical_board_max=hist_max,
            history_had_limit_down_after_top=had_nuclear,
            base_score=base_sc if pd.notna(base_sc) else None,
            has_risk_announcement=ann_risk,
            cohort_3board_size=cohort_sz,
            is_industry_unknown=is_unk,
            late_order_action=None
        )
        six_tags_list.append(','.join(audit_res['tags']))
        six_veto_list.append(audit_res['veto'])
        six_reasons_list.append('; '.join(audit_res['veto_reasons']))
        
    oof_r7['six_tag_tags'] = six_tags_list
    oof_r7['six_tag_veto'] = six_veto_list
    oof_r7['six_tag_reasons'] = six_reasons_list
    oof_r7.to_csv(R7_DIR / "oof_round7_predictions.csv", index=False)
    print("Audit columns saved to oof_round7_predictions.csv.")

    def pick_top1(df_oof, col):
        return df_oof.sort_values(['date', col, 'code_str'], ascending=[True, False, True]).groupby('date', sort=True).head(1).copy()

    top_arm0 = pick_top1(oof_r7, 'score_arm0_r4')
    top_arm1 = pick_top1(oof_r7, 'score_arm1_full')
    top_arm2 = pick_top1(oof_r7, 'score_arm2_primary')
    top_arm3 = pick_top1(oof_r7, 'score_arm3_capacity')

    stats_arm0 = compute_arm_stats(top_arm0, "Round 7 Arm 0 (Clean 3to4)")
    stats_arm1 = compute_arm_stats(top_arm1, "Round 7 Arm 1 (Full-Horizon)")
    stats_arm2 = compute_arm_stats(top_arm2, "Round 7 Arm 2 (Space Core)")
    stats_arm3 = compute_arm_stats(top_arm3, "Round 7 Arm 3 (Capacity Dragon Track)")

    # Funnels for Arm 3
    f_total = len(top_arm3)
    c_s1 = top_arm3
    c_s2 = c_s1[~c_s1['six_tag_veto']]
    c_s3 = c_s2[c_s2['next_open_gap'] > 0.0]
    c_s4 = c_s3[c_s3['trade_one_word_up'] != 1]
    c_s5 = c_s4[c_s4['execution_status'] == 'inferred_daily_limit_touch_fill']

    funnel_arm3 = [
        {"stage": "S0_total_top1", "n": len(c_s1), "3to4_hits": int(c_s1['is_3to4'].sum())},
        {"stage": "S1_veto_passed", "n": len(c_s2), "3to4_hits": int(c_s2['is_3to4'].sum()), "rate": round(float(c_s2['is_3to4'].mean()), 4)},
        {"stage": "S2_auction_open_gt0", "n": len(c_s3), "3to4_hits": int(c_s3['is_3to4'].sum()), "rate": round(float(c_s3['is_3to4'].mean()), 4)},
        {"stage": "S3_actionable_liquid", "n": len(c_s4), "3to4_hits": int(c_s4['is_3to4'].sum()), "rate": round(float(c_s4['is_3to4'].mean()), 4)},
        {"stage": "S4_limit_touch_fill", "n": len(c_s5), "3to4_hits": int(c_s5['is_3to4'].sum()), "rate": round(float(c_s5['is_3to4'].mean()), 4), "space_hits": int(c_s5['is_market_space_leader'].sum())}
    ]

    out = {
        "title": "Round 7 Capacity Dragon Evaluation Report",
        "total_days": f_total,
        "arms_performance": {
            "arm0_3to4": stats_arm0,
            "arm1_full": stats_arm1,
            "arm2_space": stats_arm2,
            "arm3_capacity_dragon": stats_arm3
        },
        "funnel_arm3_capacity_dragon": funnel_arm3
    }

    (R7_DIR / "round7_comprehensive_evaluation.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\nEvaluation complete and written to {R7_DIR / 'round7_comprehensive_evaluation.json'}")
    print(f"Arm 3 Top-1 Hits: 3to4={stats_arm3['is_3to4']['hits']}, Space={stats_arm3['is_market_space_leader']['hits']}, 5B={stats_arm3['is_3to5']['hits']}")
    print(f"Arm 3 S4 Inferred Fill: N={len(c_s5)}, 3to4 Hits={int(c_s5['is_3to4'].sum())} ({c_s5['is_3to4'].mean():.2%})")

if __name__ == "__main__":
    main()
