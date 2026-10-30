"""Test the three methods step-by-step for raising Top-3 1-to-2 hit rate towards 40%.
Strict protocol:
- 286 contracted dates, exactly 3 picks per day (858 slots).
- Evaluation via evaluate_selection: report known hits, known precision, confirmed/slots.
- No dropping retreat days, no reducing slots, no expanding float cap > 93.27亿.
"""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

# Load data
features_path = HERE / "results/FULL_POOL_EXISTING_EVIDENCE_NO_LABELS.csv"
outcomes_path = HERE / "results/OUTCOMES_DIAGNOSTIC_ONLY.csv"

features = pd.read_csv(features_path, dtype={"code": str}, low_memory=False)
outcomes = pd.read_csv(outcomes_path, dtype={"code": str}, low_memory=False)

df = features.merge(outcomes, on=["signal_date", "code"])
assert len(df) == 7254 and df["signal_date"].nunique() == 286

dates = sorted(df["signal_date"].unique())
P2 = "promotion_next_LABEL_ONLY"
P4 = "reach4_LABEL_ONLY"

def evaluate_selection(pool, selected_df, dates, k=3, target=0.4):
    picks = selected_df[["signal_date", "code"]].merge(pool[["signal_date", "code", P2, P4]], on=["signal_date", "code"], how="left")
    slots = k * len(dates)
    hits = int(picks[P2].eq(1).sum())
    known = int(picks[P2].notna().sum())
    unknown = len(picks) - known
    r4_hits = int(picks[P4].eq(1).sum())
    r4_known = int(picks[P4].notna().sum())
    precision = hits / known if known else None
    r4_prec = r4_hits / r4_known if r4_known else None
    
    return {
        "dates": len(dates),
        "slots": slots,
        "selected": len(picks),
        "known": known,
        "unknown": unknown,
        "hits": hits,
        "known_precision": precision,
        "confirmed_slots_rate": hits / slots,
        "r4_hits": r4_hits,
        "r4_precision": r4_prec,
        "extra_hits_needed_40pct": max(0, math.ceil(target * known) - hits),
    }

# Compute sector peer counts
known_ind = df[df["legacy_industry_group_PROXY"].notna()].copy()
peer_counts = known_ind.groupby(["signal_date", "legacy_industry_group_PROXY"])["code"].transform("count")
df.loc[known_ind.index, "peer_count"] = peer_counts
df["peer_count"] = df["peer_count"].fillna(0)

# 0. Baseline: Pure S1 Rank
df["rank_base"] = df.sort_values(["signal_date", "s1_score", "code"], ascending=[True, False, True]).groupby("signal_date").cumcount() + 1
base_top3 = df[df["rank_base"].le(3)]
res_base = evaluate_selection(df, base_top3, dates)

print("="*85)
print(f"基线 (S1原始分): 命中 {res_base['hits']} / {res_base['known']} = {res_base['known_precision']*100:.2f}% (距40%还差 {res_base['extra_hits_needed_40pct']} 命中)")
print("="*85)

# =========================================================================
# 方法一：换手率与成交量重塑 (惩罚超缩量板/一字板，提拔5%~15%健康换手与大成交额)
# =========================================================================
# 规则：
# 1. 超缩量 (<3% 或 一字板): 减分 -0.04
# 2. 5%~15% 充分换手: 加分 +0.02
# 3. 成交额 >= 2.5亿 (大容量承接): 加分 +0.015
m1_adj = np.zeros(len(df))
m1_adj += np.where((df["turnover"] < 3.0) | (df["one_price"] == 1), -0.04, 0.0)
m1_adj += np.where(df["turnover"].between(5.0, 15.0), 0.02, 0.0)
m1_adj += np.where(df["amount"] >= 2.5e8, 0.015, 0.0)

df["score_m1"] = df["s1_score"] + m1_adj
df["rank_m1"] = df.sort_values(["signal_date", "score_m1", "code"], ascending=[True, False, True]).groupby("signal_date").cumcount() + 1
top3_m1 = df[df["rank_m1"].le(3)]
res_m1 = evaluate_selection(df, top3_m1, dates)

# =========================================================================
# 方法二：题材身位与板块共振 (剔除独苗与过度高潮，奖励2~6只适度扩散)
# =========================================================================
# 规则：
# 1. 同题材 2~6 只 (良性协同扩散): 加分 +0.025
# 2. 孤勇者 (同题材仅1只): 减分 -0.025
# 3. 过度拥挤高潮 (同题材 >= 8只): 减分 -0.015
m2_adj = np.zeros(len(df))
m2_adj += np.where(df["peer_count"].between(2, 6), 0.025, 0.0)
m2_adj += np.where(df["peer_count"].le(1), -0.025, 0.0)
m2_adj += np.where(df["peer_count"].ge(8), -0.015, 0.0)

df["score_m2"] = df["s1_score"] + m2_adj
df["rank_m2"] = df.sort_values(["signal_date", "score_m2", "code"], ascending=[True, False, True]).groupby("signal_date").cumcount() + 1
top3_m2 = df[df["rank_m2"].le(3)]
res_m2 = evaluate_selection(df, top3_m2, dates)

# =========================================================================
# 方法三：图形形态过滤 (驱逐深跌熊坑超跌反抽，奖励平台突破与多头排列)
# =========================================================================
# 规则：
# 1. 阴跌熊坑超跌反抽 (position_60 < 0.25 & close_ma60 < -0.10): 减分 -0.05
# 2. 平台突破 (breakout_20 >= 0): 加分 +0.02
# 3. 均线多头排列 (close_ma5 > close_ma20 > close_ma60): 加分 +0.015
is_ma_bull = (df["close_ma5"] > df["close_ma20"]) & (df["close_ma20"] > df["close_ma60"])
is_oversold_trap = (df["position_60"] < 0.25) & (df["close_ma60"] < -0.10)
is_breakout = df["breakout_20"] >= 0

m3_adj = np.zeros(len(df))
m3_adj += np.where(is_oversold_trap, -0.05, 0.0)
m3_adj += np.where(is_breakout, 0.02, 0.0)
m3_adj += np.where(is_ma_bull, 0.015, 0.0)

df["score_m3"] = df["s1_score"] + m3_adj
df["rank_m3"] = df.sort_values(["signal_date", "score_m3", "code"], ascending=[True, False, True]).groupby("signal_date").cumcount() + 1
top3_m3 = df[df["rank_m3"].le(3)]
res_m3 = evaluate_selection(df, top3_m3, dates)

# =========================================================================
# 方法四：传智式小市值弹性 (< 35亿小盘轻装上阵)
# =========================================================================
m4_adj = np.where(df["float_cap"] <= 35e8, 0.025, 0.0)
df["score_m4"] = df["s1_score"] + m4_adj
df["rank_m4"] = df.sort_values(["signal_date", "score_m4", "code"], ascending=[True, False, True]).groupby("signal_date").cumcount() + 1
top3_m4 = df[df["rank_m4"].le(3)]
res_m4 = evaluate_selection(df, top3_m4, dates)

# =========================================================================
# 组合叠加测试：逐级叠加
# =========================================================================
# 组合 A: 方法一 + 方法三 (换手容量 + 图形突破)
df["score_combo_1_3"] = df["s1_score"] + m1_adj + m3_adj
df["rank_combo_1_3"] = df.sort_values(["signal_date", "score_combo_1_3", "code"], ascending=[True, False, True]).groupby("signal_date").cumcount() + 1
top3_c13 = df[df["rank_combo_1_3"].le(3)]
res_c13 = evaluate_selection(df, top3_c13, dates)

# 组合 B: 方法一 + 方法二 + 方法三 (换手容量 + 题材身位 + 图形突破)
df["score_combo_1_2_3"] = df["s1_score"] + m1_adj + m2_adj + m3_adj
df["rank_combo_1_2_3"] = df.sort_values(["signal_date", "score_combo_1_2_3", "code"], ascending=[True, False, True]).groupby("signal_date").cumcount() + 1
top3_c123 = df[df["rank_combo_1_2_3"].le(3)]
res_c123 = evaluate_selection(df, top3_c123, dates)

# 组合 C: 全维度融合 (换手容量 + 题材身位 + 图形突破 + 小市值传智弹性)
df["score_all"] = df["s1_score"] + m1_adj + m2_adj + m3_adj + m4_adj
df["rank_all"] = df.sort_values(["signal_date", "score_all", "code"], ascending=[True, False, True]).groupby("signal_date").cumcount() + 1
top3_all = df[df["rank_all"].le(3)]
res_all = evaluate_selection(df, top3_all, dates)

# 输出汇总表格
runs = [
    ("基线: S1原始分", res_base),
    ("方法一: 换手率与容量重塑 (驱逐缩量一字/提拔充分换手)", res_m1),
    ("方法二: 题材身位与板块共振 (剔除孤苗/奖励2~6只扩散)", res_m2),
    ("方法三: 图形形态过滤 (剔除超跌阴跌/奖励突破多头)", res_m3),
    ("方法四: 小市值传智弹性 (<=35亿)", res_m4),
    ("组合 A: 方法一 + 方法三 (换手 + 图形)", res_c13),
    ("组合 B: 方法一 + 方法二 + 方法三 (换手 + 题材 + 图形)", res_c123),
    ("组合 C: 全维度四合一 (换手 + 题材 + 图形 + 小盘)", res_all),
]

summary_rows = []
for name, r in runs:
    summary_rows.append({
        "方案": name,
        "Top3命中数": r["hits"],
        "已知总数": r["known"],
        "Top3一进二命中率": f"{r['known_precision']*100:.2f}%",
        "相比基线净增命中": r["hits"] - res_base["hits"],
        "距40%目标差距": r["extra_hits_needed_40pct"],
        "四板成妖命中数": r["r4_hits"],
        "四板成妖率": f"{r['r4_precision']*100:.2f}%" if r["r4_precision"] else "N/A"
    })

res_summary = pd.DataFrame(summary_rows)
res_summary.to_csv(HERE / "results/THREE_METHODS_STEP_BY_STEP_RESULTS.csv", index=False, encoding="utf-8-sig")

print("\n" + "="*95)
print(f"{'方案名称':<38} | {'Top3命中':<9} | {'Top3命中率':<10} | {'净增命中':<8} | {'距40%差距':<8}")
print("-" * 95)
for _, r in res_summary.iterrows():
    print(f"{r['方案']:<36} | {r['Top3命中数']:<9} | {r['Top3一进二命中率']:<10} | {r['相比基线净增命中']:<8} | {r['距40%目标差距']:<8}")
print("="*95)
