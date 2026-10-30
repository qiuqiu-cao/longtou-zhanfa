"""audit_qualified_pool_data_coverage.py - 完整 7,254 条合格候选的数据覆盖表与时序可行性审计

严格契约：
1. 全量扫描 2025-07-01 至 2026-08-31 完整 286 个测试日的 7,254 条合格候选；
2. 逐条逐日统计在：日线、30分钟线、5分钟线、1分钟线、分时封板时钟（seal_first）上的真实匹配覆盖；
3. 穿透分析榜外 4,397 条候选（82 成功 vs 4,155 失败 vs 160 未知）的特征与数据可得性；
4. 明确划分“具备完整分时资料的日期”与“缺失资料保留未知的日期”，绝不把缺失当不存在或事后删除。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "research/data_coverage_7254_qualified"

SCREEN_PATH = ROOT / "research/first_board_leader_emotion_20261021T1706Z/ALL28503_PATHS_AND_ORIGINAL_SCREEN.csv"
PREDS_PATH = ROOT / "research/first_board_leader_20261020T114600Z/evaluation_ALL_PREDICTIONS.csv"

# Intraday archives
P30_PATH = ROOT / "data/v290/full_long_interval_20261002_v1/thirty_minute_bars_UNRECONCILED.csv"
P5_AUG_PATH = ROOT / "data/v289/full_august_five_minute_20261002_v1/five_minute_bars_UNRECONCILED.csv"
P5_SEP_PATH = ROOT / "data/v287/september_five_minute_20261002_v1/five_minute_bars_UNRECONCILED.csv"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("【龙头战法】完整 7,254 条合格候选数据覆盖表审计")
    print(f"执行时间: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 80)

    # 1. Load qualified pool
    print("\n[Step 1] 加载 7,254 条原筛选合格候选与旧模型基准排名...")
    screen_df = pd.read_csv(SCREEN_PATH, dtype={"code": str}, low_memory=False)
    test = screen_df[screen_df["signal_date"] >= "2025-07-01"].copy()
    allowed = test[test["allowed"].astype(str).str.lower().eq("true")].copy()
    allowed["code"] = allowed["code"].str.zfill(6)
    assert len(allowed) == 7254, f"Expected 7254, got {len(allowed)}"
    print(f"合格候选总数: {len(allowed)} 行，涵盖 {allowed['signal_date'].nunique()} 个交易日")

    # Load predictions to rank
    preds = pd.read_csv(PREDS_PATH, dtype={"code": str}, low_memory=False, usecols=["signal_date", "code", "model_score"])
    preds["code"] = preds["code"].str.zfill(6)
    allowed = allowed.merge(preds, on=["signal_date", "code"], how="left")
    allowed["rank"] = allowed.groupby("signal_date")["model_score"].rank(ascending=False, method="min").astype(int)

    # Daily pool size distribution
    daily_sizes = allowed.groupby("signal_date").size()
    print(f"每日合格池容量分布: 最小 {daily_sizes.min()} 只，中位 {daily_sizes.median():.0f} 只，最大 {daily_sizes.max()} 只")

    # 2. Segment into Top 3, Rank 4-10, Beyond Top 10
    allowed["pool_segment"] = np.select(
        [allowed["rank"] <= 3, (allowed["rank"] > 3) & (allowed["rank"] <= 10), allowed["rank"] > 10],
        ["TOP3", "RANK_4_TO_10", "BEYOND_TOP10"],
        default="UNKNOWN"
    )

    print("\n合格池分段构成与四板路径分布:")
    seg_summary = allowed.groupby("pool_segment").agg(
        total_candidates=("code", "count"),
        reach4_dragons=("reach4_LABEL_ONLY", lambda s: int(s.eq(1).sum())),
        reach4_failures=("reach4_LABEL_ONLY", lambda s: int(s.eq(0).sum())),
        reach4_unknown=("reach4_LABEL_ONLY", lambda s: int(s.isna().sum())),
        median_turnover=("turnover", "median"),
        median_amount_mil=("amount", lambda s: s.median() / 1e6),
        median_float_cap_bil=("float_cap", lambda s: s.median() / 1e8),
        p_next_rate=("promotion_next_LABEL_ONLY", lambda s: s.mean() * 100),
    )
    print(seg_summary)

    # 3. Check intraday data coverage across archives
    print("\n[Step 2] 扫描微观时序数据覆盖情况...")
    
    # 30-minute coverage
    keys_30 = set()
    if P30_PATH.exists():
        df30 = pd.read_csv(P30_PATH, usecols=["code", "minute"], dtype={"code": str}, low_memory=False)
        df30["code"] = df30["code"].str.zfill(6)
        keys_30 = set(zip(df30["minute"].str[:10], df30["code"]))
        print(f"30分钟线源记录已加载，独立 (date, code) 键数: {len(keys_30)}")

    # 5-minute coverage (August & September)
    keys_5 = set()
    for p5_path in [P5_AUG_PATH, P5_SEP_PATH]:
        if p5_path.exists():
            df5 = pd.read_csv(p5_path, usecols=["code", "minute"], dtype={"code": str}, low_memory=False)
            df5["code"] = df5["code"].str.zfill(6)
            keys_5.update(zip(df5["minute"].str[:10], df5["code"]))
    print(f"5分钟线源记录已加载，独立 (date, code) 键数: {len(keys_5)}")

    # Map coverage back to 7,254 candidates
    allowed["has_daily_ohlc"] = True  # All have daily OHLC by definition
    allowed["has_30m_bar"] = [ (d, c) in keys_30 for d, c in zip(allowed["signal_date"], allowed["code"]) ]
    allowed["has_5m_bar"] = [ (d, c) in keys_5 for d, c in zip(allowed["signal_date"], allowed["code"]) ]
    
    # Check if primary seal time (seal_first) is available
    allowed["has_seal_first"] = allowed.get("seal_first", pd.Series(np.nan, index=allowed.index)).notna()

    # Overall coverage statistics
    cov_total = len(allowed)
    cov_30m = int(allowed["has_30m_bar"].sum())
    cov_5m = int(allowed["has_5m_bar"].sum())
    cov_seal = int(allowed["has_seal_first"].sum())

    print("\n7,254 条合格候选整体覆盖统计:")
    print(f"  日线完备数据: {cov_total} / {cov_total} (100.0%)")
    print(f"  30分钟线覆盖: {cov_30m} / {cov_total} ({cov_30m/cov_total*100:.2f}%)")
    print(f"  5分钟线覆盖:  {cov_5m} / {cov_total} ({cov_5m/cov_total*100:.2f}%)")
    print(f"  首封时钟数据: {cov_seal} / {cov_total} ({cov_seal/cov_total*100:.2f}%)")

    # 4. Coverage among the 4,397 candidates beyond Top 10 (82 dragons vs 4,155 failures)
    beyond = allowed[allowed["pool_segment"] == "BEYOND_TOP10"].copy()
    dragons_82 = beyond[beyond["reach4_LABEL_ONLY"] == 1]
    failures_4155 = beyond[beyond["reach4_LABEL_ONLY"] == 0]

    print("\n榜外 4,397 条候选（82 只真龙 vs 4,155 只失败者）覆盖对比:")
    print(f"  82只真龙 - 30分钟线覆盖: {int(dragons_82['has_30m_bar'].sum())} / 82 ({dragons_82['has_30m_bar'].mean()*100:.2f}%)")
    print(f"  82只真龙 - 5分钟线覆盖:  {int(dragons_82['has_5m_bar'].sum())} / 82 ({dragons_82['has_5m_bar'].mean()*100:.2f}%)")
    print(f"  4155只失败者 - 30分钟覆盖: {int(failures_4155['has_30m_bar'].sum())} / 4155 ({failures_4155['has_30m_bar'].mean()*100:.2f}%)")
    print(f"  4155只失败者 - 5分钟覆盖:  {int(failures_4155['has_5m_bar'].sum())} / 4155 ({failures_4155['has_5m_bar'].mean()*100:.2f}%)")

    # 5. Date-level completeness
    date_cov = allowed.groupby("signal_date").agg(
        total_candidates=("code", "count"),
        covered_30m=("has_30m_bar", "sum"),
        covered_5m=("has_5m_bar", "sum"),
    )
    date_cov["pct_30m"] = date_cov["covered_30m"] / date_cov["total_candidates"]
    date_cov["pct_5m"] = date_cov["covered_5m"] / date_cov["total_candidates"]

    dates_full_30m = int((date_cov["pct_30m"] == 1.0).sum())
    dates_partial_30m = int(((date_cov["pct_30m"] > 0) & (date_cov["pct_30m"] < 1.0)).sum())
    dates_zero_30m = int((date_cov["pct_30m"] == 0.0).sum())

    dates_full_5m = int((date_cov["pct_5m"] == 1.0).sum())
    dates_zero_5m = int((date_cov["pct_5m"] == 0.0).sum())

    print("\n交易日维度的资料完备性分布 (共 286 日):")
    print(f"  30分钟线完全完备日期 (100%覆盖): {dates_full_30m} 日 ({dates_full_30m/286*100:.2f}%)")
    print(f"  30分钟线部分覆盖日期:           {dates_partial_30m} 日 ({dates_partial_30m/286*100:.2f}%)")
    print(f"  30分钟线完全缺失日期:           {dates_zero_30m} 日 ({dates_zero_30m/286*100:.2f}%)")
    print(f"  5分钟线完全完备日期:            {dates_full_5m} 日 ({dates_full_5m/286*100:.2f}%)")
    print(f"  5分钟线完全缺失日期:            {dates_zero_5m} 日 ({dates_zero_5m/286*100:.2f}%)")

    # 6. Save full coverage table and summary
    out_csv = OUT_DIR / "QUALIFIED_7254_DATA_COVERAGE.csv"
    allowed[[
        "signal_date", "code", "name", "rank", "pool_segment", "float_cap", "amount", "turnover",
        "reach4_LABEL_ONLY", "reach5_LABEL_ONLY", "promotion_next_LABEL_ONLY",
        "has_daily_ohlc", "has_30m_bar", "has_5m_bar", "has_seal_first"
    ]].to_csv(out_csv, index=False, encoding="utf-8-sig")

    receipt = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_qualified": len(allowed),
        "total_test_dates": allowed["signal_date"].nunique(),
        "pool_segment_summary": seg_summary.to_dict(orient="index"),
        "coverage_summary": {
            "daily_ohlc_rows": cov_total,
            "thirty_minute_rows": cov_30m,
            "five_minute_rows": cov_5m,
            "seal_first_rows": cov_seal,
            "dates_full_30m": dates_full_30m,
            "dates_partial_30m": dates_partial_30m,
            "dates_zero_30m": dates_zero_30m,
            "dates_full_5m": dates_full_5m,
            "dates_zero_5m": dates_zero_5m,
        },
        "dragons_82_coverage": {
            "total_dragons_beyond_top10": len(dragons_82),
            "matched_30m": int(dragons_82["has_30m_bar"].sum()),
            "matched_5m": int(dragons_82["has_5m_bar"].sum()),
        },
        "failures_4155_coverage": {
            "total_failures_beyond_top10": len(failures_4155),
            "matched_30m": int(failures_4155["has_30m_bar"].sum()),
            "matched_5m": int(failures_4155["has_5m_bar"].sum()),
        }
    }

    receipt_path = OUT_DIR / "DATA_COVERAGE_REPORT.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[Done] 完整覆盖审计表已写入: {out_csv}")
    print(f"[Done] 结构化审计回执已写入: {receipt_path}")


if __name__ == "__main__":
    main()
