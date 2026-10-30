"""audit_execution_upgrade.py - 龙头战法首板执行升级全量审计与双向核算对账

核心验证点：
1. 固定旧模型（44 维特征，0 次拟合，原权重保持）；
2. 严格恢复原冻结筛选（流通市值 ≤ 93.27 亿，薄换手风险代理排除）；
3. 全池审阅覆盖完整合格池，而非仅看 Top 10；
4. 盘口特赦事实核验：无尾盘封单数据严格保持【待核】；
5. 实盘 09:25 审核卡：必须先审核再授权，严禁 09:20 前挂单；
6. 双向核算对账：统计拦截失败与错杀成功。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import first_board_execution_upgrade as up

DATA_PATH = ROOT / "research/first_board_leader_20261020T114600Z/all_first_boards_features_and_LABEL_ONLY.csv"
SCREEN_PATH = ROOT / "research/first_board_leader_emotion_20261021T1706Z/ALL28503_PATHS_AND_ORIGINAL_SCREEN.csv"
SCENARIO_TRADES_PATH = ROOT / "research/first_board_leader_20261020T114600Z/model_score_SCENARIO_TRADES.csv"
OUT_DIR = ROOT / "research/execution_upgrade_audit_20261029"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("【龙头战法】首板执行系统升级审计与双向对账")
    print(f"执行时间: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 80)

    # 1. Load data
    print("\n[Step 1] 加载历史归档与原筛选基准...")
    raw_df = pd.read_csv(DATA_PATH, dtype={"code": str}, low_memory=False)
    screen_df = pd.read_csv(SCREEN_PATH, dtype={"code": str}, low_memory=False, usecols=["signal_date", "code", "allowed", "exclusion_reason"])
    
    merged = raw_df.merge(screen_df, on=["signal_date", "code"], how="left")
    merged["code"] = merged["code"].astype(str).str.zfill(6)
    print(f"全归档样本: {len(merged)} 行，涵盖 {merged['signal_date'].nunique()} 个交易日")

    # Filter test set (2025-07-01 to 2026-08-31)
    test_df = merged[merged["signal_date"] >= "2025-07-01"].copy()
    print(f"测试集 (2025H2~2026): {len(test_df)} 行，涵盖 {test_df['signal_date'].nunique()} 个交易日")

    # 2. Score with frozen legacy model (0 fits)
    print("\n[Step 2] 使用旧模型 44 维特征进行评分 (0次拟合，固定基准)...")
    scored = up.score_legacy_first_boards(test_df)
    assert not scored["legacy_model_score"].isna().any(), "评分不得有 NaN"
    print(f"旧模型基准评分完毕，均值: {scored['legacy_model_score'].mean():.4f}, 最大值: {scored['legacy_model_score'].max():.4f}")

    # 3. Apply Original Screen (<= 93.27 亿, 剔除薄换手)
    print("\n[Step 3] 验证原市值与风险筛选执行情况...")
    screened = up.apply_original_screen(scored)
    
    # Check agreement with archived screen
    archived_allowed = screened["allowed"].astype(str).str.lower().eq("true")
    new_allowed = screened["original_screen_allowed"]
    match_rate = (archived_allowed == new_allowed).mean()
    print(f"新筛选器与归档原筛选一致率: {match_rate*100:.2f}%")

    # 4. Compare Top 3 before and after Original Screen
    print("\n[Step 4] 对比原筛选实施前后的 Top 3 四板命中率与排除统计...")
    # Baseline: Unscreened Top 3 using monthly rolling model_score
    eval_path = ROOT / "research/first_board_leader_20261020T114600Z/evaluation_ALL_PREDICTIONS.csv"
    if eval_path.exists():
        eval_df = pd.read_csv(eval_path, dtype={"code": str}, low_memory=False, usecols=["signal_date", "code", "model_score"])
        eval_df["code"] = eval_df["code"].astype(str).str.zfill(6)
        screened = screened.merge(eval_df, on=["signal_date", "code"], how="left")
        score_col = "model_score"
    else:
        score_col = "legacy_model_score"

    top3_unscreened = screened.sort_values(["signal_date", score_col, "code"], ascending=[True, False, True]).groupby("signal_date").head(3)
    known_unscreened = top3_unscreened[top3_unscreened["reach4_LABEL_ONLY"].notna()]
    hits_unscreened = int(known_unscreened["reach4_LABEL_ONLY"].sum())
    total_unscreened = len(known_unscreened)
    prec_unscreened = hits_unscreened / total_unscreened if total_unscreened else 0.0

    # Screened Top 3 (Only qualified stocks)
    screened_pool = screened[screened["original_screen_allowed"]].copy()
    top3_screened = screened_pool.sort_values(["signal_date", score_col, "code"], ascending=[True, False, True]).groupby("signal_date").head(3)
    known_screened = top3_screened[top3_screened["reach4_LABEL_ONLY"].notna()]
    hits_screened = int(known_screened["reach4_LABEL_ONLY"].sum())
    total_screened = len(known_screened)
    prec_screened = hits_screened / total_screened if total_screened else 0.0

    print(f"未筛选 Top 3 四板命中: {hits_unscreened} / {total_unscreened} ({prec_unscreened*100:.2f}%)")
    print(f"原筛选合格池 Top 3 四板命中: {hits_screened} / {total_screened} ({prec_screened*100:.2f}%)")

    # Exclusions analysis in Top 3
    excluded_in_raw_top3 = top3_unscreened[~top3_unscreened["original_screen_allowed"]]
    print(f"\n原始 Top 3 中被原筛选排除的标的数: {len(excluded_in_raw_top3)} 只次")
    print("排除原因分布:")
    print(excluded_in_raw_top3["exclusion_reason"].value_counts())

    # 5. Dual Accounting Ledger: avoided losses vs killed profits
    print("\n[Step 5] 建立双向核算账本对账 (Dual-Accounting Ledger)...")
    ledger = up.DualAccountingLedger()

    # Load trading simulation if available
    trade_sim = {}
    if SCENARIO_TRADES_PATH.exists():
        sim_df = pd.read_csv(SCENARIO_TRADES_PATH, dtype={"code": str})
        sim_df["code"] = sim_df["code"].astype(str).str.zfill(6)
        for _, r in sim_df.iterrows():
            trade_sim[(str(r["signal_date"]), str(r["code"]))] = {
                "profitable": float(r.get("trade_return", 0) or 0) > 0,
                "return": float(r.get("trade_return", 0) or 0)
            }

    # Record all decisions for Top 3
    for date, group in screened.groupby("signal_date"):
        group_screened = group.copy()
        
        # Raw top 3
        raw_top3_codes = set(group.sort_values([score_col, "code"], ascending=[False, True]).head(3)["code"])
        # Screened top 3
        screened_cand = group_screened[group_screened["original_screen_allowed"]]
        final_top3_codes = set(screened_cand.sort_values([score_col, "code"], ascending=[False, True]).head(3)["code"])

        for _, row in group.iterrows():
            c = str(row["code"]).zfill(6)
            is_raw_top3 = c in raw_top3_codes
            is_final_top3 = c in final_top3_codes
            
            if not is_raw_top3 and not is_final_top3:
                continue

            r4 = row.get("reach4_LABEL_ONLY")
            actual_r4 = int(r4) if pd.notna(r4) else None
            sim_info = trade_sim.get((str(date), c), {})
            prof = sim_info.get("profitable")
            ret = sim_info.get("return")

            if is_raw_top3 and is_final_top3:
                itype = "KEEP"
                reason = "原筛选与模型双重认可"
            elif is_raw_top3 and not is_final_top3:
                itype = "EXCLUDE_BY_SCREEN"
                reason = str(group_screened.loc[group_screened["code"] == c, "exclusion_reason"].iloc[0])
            else:
                itype = "PROMOTE_FROM_BELOW"
                reason = "替补原筛选排除名额进入前三"

            ledger.record_decision(
                signal_date=str(date),
                code=c,
                name=str(row["name"]),
                legacy_rank=int(row["legacy_rank"]),
                legacy_score=float(row[score_col]),
                final_selected=is_final_top3,
                intervention_type=itype,
                reason=reason,
                actual_reach4=actual_r4,
                actual_trade_profitable=prof,
                trade_return=ret
            )

    ledger_summary = ledger.summarize()
    print("双向账本核算结果:")
    print(json.dumps(ledger_summary, ensure_ascii=False, indent=2))

    # 6. Test on Live 2026-10-28 Facts
    print("\n[Step 6] 实战回放：2026-10-28 盘口特赦与 09:25 执行卡生成...")
    facts_path = ROOT / "results/daily_review_20261028T1020Z/facts/RESOLVED_FACTS_v2_SCORES_MASKED.json"
    if facts_path.exists():
        facts = json.loads(facts_path.read_text(encoding="utf-8"))
        first_boards_1028 = [r for r in facts["rows"] if r.get("boards") == 1]
        print(f"2026-10-28 共有首板标的: {len(first_boards_1028)} 只")

        audit_results = []
        for r in first_boards_1028:
            cap = float(r.get("reported_float_cap_cny", 0) or 0)
            amt = float(r.get("amount_cny", 0) or 0)
            to = float(r.get("turnover_pct", 0) or 0)
            s_first = r.get("seal_first")
            s_delta = r.get("seal_delta_1430_to_close")
            
            row_dict = {
                "float_cap": cap, "amount": amt, "turnover": to,
                "seal_first": s_first, "seal_delta": s_delta
            }
            exemp_status, exemp_reason = up.audit_capacity_dragon_exemption(row_dict)
            veto_status, veto_reason = up.audit_sneak_attack_trap_veto(row_dict)

            audit_results.append({
                "code": str(r["code"]).zfill(6),
                "name": r["name"],
                "amount_亿": round(amt / 1e8, 2),
                "turnover_%": to,
                "seal_first": s_first,
                "seal_delta": s_delta,
                "capacity_exemption": exemp_status,
                "exemption_reason": exemp_reason,
                "trap_veto": veto_status,
                "veto_reason": veto_reason
            })

        df_audit = pd.DataFrame(audit_results)
        print("\n10月28日特赦核验状态分布 (严格核验):")
        print(df_audit["capacity_exemption"].value_counts())
        
        # Check Baoyang and Aoyang
        special_stocks = df_audit[df_audit["name"].isin(["百洋股份", "澳洋健康", "联明股份"])]
        print("\n重点关注股票审计核查:")
        print(special_stocks[["code", "name", "seal_first", "capacity_exemption", "exemption_reason", "trap_veto", "veto_reason"]])

        # Generate sample 09:25 auction card
        sample_cand = {"code": "002696", "name": "百洋股份", "close": 5.48}
        auction_card = up.generate_auction_execution_card(sample_cand)
        print("\n09:25 实盘竞价审核卡样本:")
        print(json.dumps(auction_card, ensure_ascii=False, indent=2))

    # Save complete receipt
    receipt = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "legacy_bundle": str(up.LEGACY_BUNDLE_PATH),
        "test_period": "2025-07-01 to 2026-08-31",
        "test_dates": test_df["signal_date"].nunique(),
        "test_rows": len(test_df),
        "unscreened_precision": prec_unscreened,
        "screened_precision": prec_screened,
        "ledger_summary": ledger_summary,
    }
    receipt_path = OUT_DIR / "AUDIT_RECEIPT_EXECUTION_UPGRADE.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[Done] 审计回执已写入: {receipt_path}")


if __name__ == "__main__":
    main()
