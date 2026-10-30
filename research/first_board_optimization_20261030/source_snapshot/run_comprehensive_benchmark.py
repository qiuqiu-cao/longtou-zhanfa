"""Retrospective overlay comparison and daily-open scenario ledger.
Executes the Four Levers across 286 trading days (7,254 qualified candidates)
and tests on the 2026-10-28 live facts.
Zero model retraining, full pool dynamic processing (7~72 stocks/day), dual-accounting ledger.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd

from first_board_comprehensive_engine import (
    EngineConfig,
    ComprehensiveFirstBoardEngine,
    DualAccountingLedger,
    parse_seal_time_to_minutes,
    to_number,
    FROZEN_UPPER_CAP_CNY,
)
from legacy_industry_snapshot import attach_legacy_groups

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "research"
OUT_DIR = ROOT / "research/four_levers_benchmark_20261029_corrected"

# Source paths
SCREEN_PATH = ROOT / "research/first_board_leader_emotion_20261021T1706Z/ALL28503_PATHS_AND_ORIGINAL_SCREEN.csv"
HEIGHT_PREDS_PATH = ROOT / "research/first_board_leader_20261020T114600Z/evaluation_ALL_PREDICTIONS.csv"
S1_PREDS_PATH = ROOT / "research/first_second_stage_20261020_run1/S1_ALL_PREDICTIONS.csv"
SCENARIO_TRADES_PATH = ROOT / "research/first_board_leader_20261020T114600Z/model_score_SCENARIO_TRADES.csv"
CALENDAR_PATH = ROOT / "data/v279/exchange_calendar_audit_20261002_v1/calendar.json"
BARS_SOURCE = ROOT / "data/v16/point_in_time_rebuild_cutoff_20260831_v2"
FACTS_1028_PATH = ROOT / "results/daily_review_20261028T1020Z/facts/RESOLVED_FACTS_v2_SCORES_MASKED.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_all_bars(calendar_sessions: list[str]) -> tuple[pd.DataFrame, dict]:
    pos = {d: i for i, d in enumerate(calendar_sessions)}
    cols = [
        "date", "code", "name", "open", "high", "low", "close", "pre_close",
        "volume", "amount", "turnover", "float_cap", "st", "new_listing", "status_verified"
    ]
    b1 = pd.read_csv(BARS_SOURCE / "point_in_time_current_daily_bars.csv", usecols=cols, dtype={"code": str}, low_memory=False)
    b2 = pd.read_csv(BARS_SOURCE / "verified_recovered_daily_bars.csv", usecols=cols, dtype={"code": str}, low_memory=False)
    bars = pd.concat([b1, b2], ignore_index=True)
    bars["code"] = bars["code"].str.zfill(6)
    bars["date"] = bars["date"].astype(str).str[:10]
    bars = bars.drop_duplicates(["date", "code"], keep="last")
    
    num_cols = ["open", "high", "low", "close", "pre_close", "volume", "amount", "turnover", "float_cap"]
    bars[num_cols] = bars[num_cols].apply(pd.to_numeric, errors="coerce")
    bars["scope_known"] = bars["status_verified"].astype(str).str.lower().isin(["true", "1", "1.0"])
    bars["eligible"] = (
        bars["scope_known"]
        & ~bars["st"].astype(str).str.lower().isin(["true", "1", "1.0"])
        & ~bars["new_listing"].astype(str).str.lower().isin(["true", "1", "1.0"])
        & bars["code"].str.startswith(("000", "001", "002", "003", "600", "601", "603", "605"))
    )
    bars["p"] = bars["date"].map(pos)
    bars = bars[bars["p"].notna()].sort_values(["code", "p"]).reset_index(drop=True)
    bars["p"] = bars["p"].astype(int)
    bars["valid_price"] = (
        bars[["open", "high", "low", "close", "pre_close"]].gt(0).all(axis=1)
        & bars["high"].ge(bars[["open", "close", "low"]].max(axis=1))
        & bars["low"].le(bars[["open", "close"]].min(axis=1))
    )
    bars["cap"] = np.floor(bars["pre_close"].values * 110 + 0.5 + 1e-8)
    bars["floor"] = np.floor(bars["pre_close"].values * 90 + 0.5 + 1e-8)
    
    lookup = bars.set_index(["date", "code"]).to_dict("index")
    return bars, lookup


def selection_metrics(rows: pd.DataFrame) -> dict:
    result = {"selected_count": len(rows), "selected_dates": int(rows.signal_date.nunique())}
    for label, short in [("promotion_next_LABEL_ONLY", "promotion"), ("reach4_LABEL_ONLY", "reach4")]:
        known = int(rows[label].notna().sum())
        hits = int(rows[label].eq(1).sum())
        result[short] = {"hits": hits, "known": known, "unknown": len(rows)-known,
                         "precision": hits/known if known else None}
    return result


def generate_auction_card(candidate: dict) -> dict:
    code = str(candidate.get("code", "")).zfill(6)
    close = to_number(candidate.get("close"))
    if not re.fullmatch(r"\d{6}", code) or close is None or close <= 0:
        raise ValueError(f"Invalid code or close: {code}, {close}")
    upper = float((Decimal(str(close)) * Decimal("1.10")).quantize(Decimal(".01"), rounding=ROUND_HALF_UP))
    target = candidate.get("target_date", "2026-10-29")
    return {
        "code": code,
        "name": str(candidate.get("name", "")),
        "pre_close": close,
        "limit_ref": upper,
        "signal_date": candidate.get("signal_date", "2026-10-28"),
        "target_date": target,
        "status": "PENDING_SCREENSHOT_REVIEW_NOT_AN_EXECUTED_AUDIT",
        "formed_at": datetime.now(timezone.utc).isoformat(),
        "is_new_prospective_prediction": False,
        "model_scored": False,
        "original_screen_status": "NOT_EVALUATED_BY_THIS_CARD",
        "exchange_rule_warning": "09:20-09:25不接受撤单；09:25已成交不能撤回。审核失败禁用尚未触发条件，已提交订单按实际状态处理。",
        "stop_loss_execution_note": "止损仅是触发条件，T+1、跳空和无法成交可能使实际亏损超过阈值。",
        "visual_checklist": [
            "09:20~09:25 撮合价格曲线：核验价格与量的变化；审核完成前不提交待审核买单",
            "09:20~09:25 买一委比与挂单：委比必须厚实正向，买一挂单不能空心（如茂业商业委比-91%买一仅13手直接一票否决）",
            "分时竞价匹配量：核验09:20后匹配量、价格与未匹配量；截图不能证明资金身份或次日必涨",
            "开盘确认：按已记录预期核验；不达标时禁用未触发条件，不新增委托，不能假定已提交或已成交买单已撤销",
        ],
        "authorized_action": "NONE_PENDING_REVIEW",
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=False)
    
    print("=== [龙头战法] 四项研究规则复算；日线开盘情景不是实盘撮合 ===")
    print("1. 校验数据源哈希与环境依赖...")
    sources = {
        "screen": SCREEN_PATH,
        "height_preds": HEIGHT_PREDS_PATH,
        "s1_preds": S1_PREDS_PATH,
        "scenario_trades": SCENARIO_TRADES_PATH,
        "calendar": CALENDAR_PATH,
        "engine": ROOT / "first_board_comprehensive_engine.py",
        "benchmark": Path(__file__).resolve(),
        "facts_1028": FACTS_1028_PATH,
    }
    hashes = {k: sha256_file(p) for k, p in sources.items()}
    
    calendar = json.loads(CALENDAR_PATH.read_text(encoding="utf-8"))["sessions"]
    print(f"Calendar sessions loaded: {len(calendar)} sessions ({calendar[0]} to {calendar[-1]})")
    
    print("2. 加载日线价格，供开盘情景对照...")
    bars, bars_lookup = load_all_bars(calendar)
    print(f"Bars loaded: {len(bars)} unique bar rows.")
    
    print("3. 加载首板样本并严格执行原冻结筛选条件 (7,254只合格首板, 286个交易日)...")
    screen_df = pd.read_csv(SCREEN_PATH, dtype={"code": str}, low_memory=False)
    test_screen = screen_df[screen_df["signal_date"].between("2025-07-01", "2026-08-31")].copy()
    qualified_df = test_screen[test_screen["allowed"].astype(str).str.lower().isin(["true", "1", "1.0"])].copy()
    
    # Merge predictions
    height_preds = pd.read_csv(HEIGHT_PREDS_PATH, dtype={"code": str}, low_memory=False)
    s1_preds = pd.read_csv(S1_PREDS_PATH, dtype={"code": str}, low_memory=False)
    
    merged = qualified_df.merge(
        height_preds[["signal_date", "code", "model_score"]].rename(columns={"model_score": "height_score"}),
        on=["signal_date", "code"],
        validate="one_to_one"
    )
    merged = merged.merge(
        s1_preds[["signal_date", "code", "model_score"]].rename(columns={"model_score": "s1_score"}),
        on=["signal_date", "code"],
        validate="one_to_one"
    )
    
    print(f"Qualified merged rows: {len(merged)}, Dates: {merged['signal_date'].nunique()}")
    assert len(merged) == 7254, f"Expected exactly 7,254 qualified rows, got {len(merged)}"
    
    print("4. 挂载2021静态行业代理；不是当期题材或梯队真值...")
    merged = attach_legacy_groups(merged)
    
    # 5. Define benchmark configurations
    print("5. 运行四大抓手全面基准对照与实战模拟...")
    
    benchmark_results = []
    
    # Config 1: Legacy unscreened reference (Raw model_score from height predictions)
    eng_ref = ComprehensiveFirstBoardEngine(EngineConfig(enable_emotion_defense=False, enable_sector_resonance=False, enable_seal_divergence=False))
    # We load raw height predictions for unscreened reference
    raw_test = screen_df[screen_df["signal_date"].between("2025-07-01", "2026-08-31")].copy()
    raw_merged = raw_test.merge(height_preds[["signal_date", "code", "model_score"]].rename(columns={"model_score": "height_score"}), on=["signal_date", "code"])
    raw_merged["s1_score"] = raw_merged["height_score"]
    _, raw_trades, raw_report = eng_ref.run_simulation(raw_merged, bars_lookup, calendar)
    raw_top1 = raw_merged.sort_values(["signal_date", "height_score", "code"], ascending=[True, False, True]).groupby("signal_date").head(1)
    raw_top3 = raw_merged.sort_values(["signal_date", "height_score", "code"], ascending=[True, False, True]).groupby("signal_date").head(3)
    
    benchmark_results.append({
        "name": "1. 旧高度模型未筛选 (基准对照)",
        "config_desc": "未筛选高度排序，日线开盘情景，单笔上限20%权益；无竞价审核或打板成交证据",
        "top1_1to2_pct": float(raw_top1["promotion_next_LABEL_ONLY"].mean() * 100.0) if "promotion_next_LABEL_ONLY" in raw_top1 else np.nan,
        "top1_reach4_pct": float(raw_top1["reach4_LABEL_ONLY"].mean() * 100.0),
        "top3_1to2_pct": float(raw_top3["promotion_next_LABEL_ONLY"].mean() * 100.0) if "promotion_next_LABEL_ONLY" in raw_top3 else np.nan,
        "top3_reach4_pct": float(raw_top3["reach4_LABEL_ONLY"].mean() * 100.0),
        "top3_reach4_hits": int(raw_top3["reach4_LABEL_ONLY"].sum()),
        "trades_count": raw_report["closed_trades"],
        "win_rate_pct": raw_report["win_rate_pct"],
        "total_return_pct": raw_report["total_return_pct"],
        "max_drawdown_pct": raw_report["max_drawdown_pct"],
        "profit_factor": raw_report["profit_factor"],
        "total_net_pnl_cny": raw_report["total_net_pnl_cny"],
        "blocked_exit_count": raw_report["blocked_exit_count"],
        "average_win_loss_ratio": raw_report["average_win_loss_ratio"],
    })
    
    # Matrix of 4-Lever settings to benchmark
    test_specs = [
        ("2. 原筛选纯高度 (w=0.0)", EngineConfig(fusion_weight_s1=0.0, enable_sector_resonance=False, enable_seal_divergence=False, enable_emotion_defense=False), "仅原筛选，纯高度模型得分"),
        ("3. 原筛选纯晋级S1 (w=1.0)", EngineConfig(fusion_weight_s1=1.0, enable_sector_resonance=False, enable_seal_divergence=False, enable_emotion_defense=False), "仅原筛选，纯S1晋级模型得分"),
        ("4. 抓手1：双目标融合 (w=0.5)", EngineConfig(fusion_weight_s1=0.5, enable_sector_resonance=False, enable_seal_divergence=False, enable_emotion_defense=False), "双目标等权融合：0.5*S1 + 0.5*Height"),
        ("5. 抓手1：双目标融合 (w=0.7)", EngineConfig(fusion_weight_s1=0.7, enable_sector_resonance=False, enable_seal_divergence=False, enable_emotion_defense=False), "双目标偏晋级融合：0.7*S1 + 0.3*Height"),
        ("6. 抓手1+4：双目标(w=0.5)+情绪防守", EngineConfig(fusion_weight_s1=0.5, enable_sector_resonance=False, enable_seal_divergence=False, enable_emotion_defense=True), "双目标融合 + 极端退潮期空仓防守"),
        ("7. 抓手1+4：双目标(w=0.7)+情绪防守", EngineConfig(fusion_weight_s1=0.7, enable_sector_resonance=False, enable_seal_divergence=False, enable_emotion_defense=True), "双目标融合(0.7) + 极端退潮期空仓防守"),
        ("8. 四项规则研究方案 (未通过生产验收)", EngineConfig(fusion_weight_s1=0.5, enable_sector_resonance=True, enable_seal_divergence=True, enable_emotion_defense=True), "融合0.5 + 静态行业代理调整 + 封板规则(历史缺字段) + 防守；研究用"),
    ]
    
    last_scored_df = None
    last_trades_df = None
    method_metrics = {}
    method_reports = {}
    original_ranks = merged.sort_values(["signal_date", "height_score", "code"], ascending=[True, False, True]).copy()
    original_ranks["height_rank"] = original_ranks.groupby("signal_date").cumcount() + 1
    original_rank_lookup = original_ranks.set_index(["signal_date", "code"])["height_rank"].to_dict()
    
    for name, cfg, desc in test_specs:
        eng = ComprehensiveFirstBoardEngine(cfg)
        scored, trades, rep = eng.run_simulation(merged, bars_lookup, calendar)
        last_scored_df = scored
        last_trades_df = trades
        
        t1 = scored[~scored["sneak_veto"]].sort_values(["signal_date", "final_rank", "code"]).groupby("signal_date").head(1)
        t3 = scored[~scored["sneak_veto"]].sort_values(["signal_date", "final_rank", "code"]).groupby("signal_date").head(3)
        stem = f"METHOD_{name.split('.')[0]}"
        scored.to_csv(out_dir / f"{stem}_ALL_SCORED.csv", index=False, encoding="utf-8-sig")
        trades.to_csv(out_dir / f"{stem}_OPEN_SCENARIO_EVENTS.csv", index=False, encoding="utf-8-sig")
        active = scored[scored["selected"]]
        method_metrics[name] = {
            "all_signal_days": int(scored.signal_date.nunique()),
            "full_date_top1_ranking": selection_metrics(t1),
            "full_date_top3_ranking": selection_metrics(t3),
            "post_defense_selected": selection_metrics(active),
            "abstained_dates": int(scored.signal_date.nunique() - active.signal_date.nunique()),
            "unfilled_top3_slots": int(scored.signal_date.nunique()*3-len(active)),
            "seal_first_observed_rows": int(scored.apply(lambda r: parse_seal_time_to_minutes(r.get("seal_first")) is not None, axis=1).sum()),
            "vetoed_rows": int(scored.sneak_veto.sum()),
            "exempt_rows": int(scored.capacity_exemption.sum()),
        }
        method_reports[name] = rep
        
        benchmark_results.append({
            "name": name,
            "config_desc": desc,
            "top1_1to2_pct": float(t1["promotion_next_LABEL_ONLY"].mean() * 100.0),
            "top1_reach4_pct": float(t1["reach4_LABEL_ONLY"].mean() * 100.0),
            "top3_1to2_pct": float(t3["promotion_next_LABEL_ONLY"].mean() * 100.0),
            "top3_reach4_pct": float(t3["reach4_LABEL_ONLY"].mean() * 100.0),
            "top3_reach4_hits": int(t3["reach4_LABEL_ONLY"].sum()),
            "trades_count": rep["closed_trades"],
            "win_rate_pct": rep["win_rate_pct"],
            "total_return_pct": rep["total_return_pct"],
            "max_drawdown_pct": rep["max_drawdown_pct"],
            "profit_factor": rep["profit_factor"],
            "total_net_pnl_cny": rep["total_net_pnl_cny"],
            "blocked_exit_count": rep["blocked_exit_count"],
            "average_win_loss_ratio": rep["average_win_loss_ratio"],
        })
        print(f"Done: {name} -> Win Rate: {rep['win_rate_pct']:.2f}%, Total Ret: {rep['total_return_pct']:.2f}%, MaxDD: {rep['max_drawdown_pct']:.2f}%, Top1 1-to-2: {t1['promotion_next_LABEL_ONLY'].mean()*100:.2f}%, Top3 R4 Hits: {int(t3['reach4_LABEL_ONLY'].sum())}")
        
    benchmark_df = pd.DataFrame(benchmark_results)
    benchmark_df.to_csv(out_dir / "FOUR_LEVERS_BENCHMARK_COMPARISON.csv", index=False, encoding="utf-8-sig")
    
    # 6. Build the 4-Dimensional Accounting Ledger
    print("6. 编制四维双向实战核算账本 (Dual Accounting Ledger)...")
    ledger = DualAccountingLedger()
    
    # Find all defense dates
    defense_dates = set(last_scored_df[last_scored_df["is_defense_day"]]["signal_date"].unique())
    print(f"Total defense dates identified: {len(defense_dates)} out of 286 trading days")
    
    # Hold ranking fixed. Compare this same overlay with defense disabled.
    counter_cfg = EngineConfig(fusion_weight_s1=0.5, enable_sector_resonance=True,
                              enable_seal_divergence=True, enable_emotion_defense=False)
    counter_scored, counter_trades, counter_report = ComprehensiveFirstBoardEngine(counter_cfg).run_simulation(merged, bars_lookup, calendar)
    assert counter_scored[["signal_date", "code", "final_rank", "final_score"]].equals(last_scored_df[["signal_date", "code", "final_rank", "final_score"]])
    counter_trades.to_csv(out_dir / "SAME_RANKING_DEFENSE_OFF_SCENARIO_EVENTS.csv", index=False, encoding="utf-8-sig")
    counter_lookup = {(r["signal_date"], r["code"]): {
        "trade_status": r["status"], "trade_return": to_number(r.get("net_return")),
        "trade_net_pnl": to_number(r.get("net_pnl"))} for r in counter_trades.to_dict("records")}
            
    # For every candidate in the top ranks, record decision
    for r in last_scored_df.itertuples():
        key = (r.signal_date, r.code)
        height_rank = int(original_rank_lookup[key])
        intervention = "KEEP" if r.final_rank <= 3 else "NOT_SELECTED"
        reason = "入选排序前三" if r.final_rank <= 3 else "完整池留档"
        if r.sneak_veto:
            intervention = "EXCLUDE_BY_VETO"
            reason = r.sneak_veto_reason
        elif r.is_defense_day and r.final_rank <= 3:
            intervention = "DEFENSE_EMPTY_CASH"
            reason = f"情绪防守空仓 ({r.defense_flags})"
        elif r.capacity_exemption:
            intervention = "PROMOTE_CAPACITY_EXEMPTION"
            reason = "换手真龙特赦提拔"
        elif r.final_rank <= 3 and height_rank > 3:
            intervention = "PROMOTE_FROM_BELOW"
            reason = "相对原高度前三换入"
        elif r.final_rank > 3 and height_rank <= 3:
            intervention = "EXCLUDE_BY_RERANK"
            reason = "相对原高度前三换出"
            
        trade_info = counter_lookup.get(key, {"trade_status": "NOT_TOP1_SCENARIO_CANDIDATE"})
        
        ledger.record_decision(
            signal_date=r.signal_date,
            code=r.code,
            name=r.name,
            legacy_rank=height_rank,
            legacy_score=float(r.height_score),
            final_rank=int(r.final_rank),
            final_score=float(r.final_score),
            selected=bool(r.selected),
            intervention=intervention,
            reason=reason,
            reach4_outcome=int(r.reach4_LABEL_ONLY) if pd.notna(r.reach4_LABEL_ONLY) else None,
            promotion_outcome=int(r.promotion_next_LABEL_ONLY) if pd.notna(r.promotion_next_LABEL_ONLY) else None,
            **trade_info
        )
        
    ledger_summary = ledger.summarize()
    final_report = method_reports[test_specs[-1][0]]
    ledger_summary["paired_portfolio_defense_effect"] = {
        "control": "SAME_COMPREHENSIVE_RANKING_DEFENSE_OFF",
        "off_closed_trades": counter_report["closed_trades"],
        "on_closed_trades": final_report["closed_trades"],
        "off_selected_dates": counter_report["selected_count"],
        "on_selected_dates": final_report["selected_count"],
        "off_total_return_pct": counter_report["total_return_pct"],
        "on_total_return_pct": final_report["total_return_pct"],
        "equity_change_cny": (final_report["total_return_pct"]-counter_report["total_return_pct"])*10000,
        "drawdown_change_pp": final_report["max_drawdown_pct"]-counter_report["max_drawdown_pct"],
        "interpretation": "配对组合期末权益差，包含防守导致的现金与后续仓位变化；不等于过滤交易金额简单加总",
    }
    pd.DataFrame(ledger.decisions).to_csv(out_dir / "FOUR_DIMENSIONAL_DECISION_LEDGER.csv", index=False, encoding="utf-8-sig")
    
    # 7. Live Fact Micro-Audit on 2026-10-28
    print("7. 用10月28日收盘事实生成事后待核卡；本次没有竞价截图审核...")
    live_cards = []
    live_audits = []
    
    if FACTS_1028_PATH.exists():
        facts_data = json.loads(FACTS_1028_PATH.read_text(encoding="utf-8"))
        rows_1028 = facts_data.get("rows", [])
        
        for r in rows_1028:
            if r.get("kind") != "limit_up" or r.get("boards") != 1:
                continue
            code = r.get("code")
            name = r.get("name")
            close = r.get("close")
            card = generate_auction_card({
                "code": code,
                "name": name,
                "close": close,
                "signal_date": "2026-10-28",
                "target_date": "2026-10-29",
            })
            live_cards.append(card)
            
            # Audit sneak and capacity
            row_dict = {
                "float_cap": r.get("reported_float_cap_cny"),
                "amount": r.get("amount_cny"),
                "turnover": r.get("turnover_pct"),
                "seal_first": r.get("seal_first"),
                "seal_delta": r.get("seal_delta_1430_to_close"),
                "seal_delta_verified": r.get("seal_delta_verified", False),
                "has_bad_history": r.get("has_bad_history"),
            }
            veto, v_reason = ComprehensiveFirstBoardEngine.audit_sneak_attack_trap(row_dict)
            exempt, e_reason = ComprehensiveFirstBoardEngine.audit_capacity_dragon_exemption(row_dict)
            
            live_audits.append({
                "code": code,
                "name": name,
                "turnover_pct": r.get("turnover_pct"),
                "amount_cny": r.get("amount_cny"),
                "seal_first": r.get("seal_first"),
                "seal_delta_1430_to_close": r.get("seal_delta_1430_to_close"),
                "sneak_veto": veto,
                "sneak_veto_reason": v_reason,
                "capacity_exemption": exempt,
                "capacity_reason": e_reason,
                "status": "RETROSPECTIVE_CLOSE_FACT_RULE_CHECK_NOT_AUCTION_REVIEW",
                "seal_delta_verified": row_dict["seal_delta_verified"],
                "has_bad_history": row_dict["has_bad_history"],
            })
            
        with (out_dir / "1028_LIVE_AUCTION_CARDS.json").open("w", encoding="utf-8") as f:
            json.dump(live_cards, f, ensure_ascii=False, indent=2)
        with (out_dir / "1028_LIVE_MICROSTRUCTURE_AUDIT.json").open("w", encoding="utf-8") as f:
            json.dump(live_audits, f, ensure_ascii=False, indent=2)
            
    # 8. Compile Comprehensive Receipt
    receipt = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "RESEARCH_ONLY_NOT_VALIDATED_FOR_PRODUCTION",
        "fits": 0,
        "models_used": [
            "research/first_board_leader_20261020T114600Z/first_board_leader_bundle.joblib (44-feature legacy height model)",
            "research/first_second_stage_20261020_run1/first_second_stage_bundle.joblib (S1 promotion model)",
        ],
        "qualified_candidates_count": 7254,
        "trading_days_count": 286,
        "daily_pool_range": "7 to 72 stocks (median 24)",
        "benchmark_summary": benchmark_results,
        "dual_accounting_summary": ledger_summary,
        "method_metrics_with_coverage": method_metrics,
        "scenario_reports": method_reports,
        "same_ranking_defense_off_report": counter_report,
        "sector_source": "STATIC_2021_VINTAGE_PROXY_COUNTED_IN_QUALIFIED_FIRST_BOARD_POOL_NOT_CURRENT_THEME_OR_FULL_LADDER",
        "historical_seal_evidence_available": any(v["seal_first_observed_rows"] > 0 for v in method_metrics.values()),
        "scenario_type": "DAILY_OPEN_ENTRY_B_PLUS_2_CLOSE_NOT_AUCTION_OR_BOARD_TRIGGER_FILL",
        "out_of_sample_upgrade_validated": False,
        "production_changed": False,
        "source_hashes": hashes,
        "live_1028_cards_generated": len(live_cards),
        "actual_auction_reviews": 0,
    }
    
    with (out_dir / "COMPREHENSIVE_FOUR_LEVERS_RECEIPT.json").open("w", encoding="utf-8") as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2)
        
    print("\n=== 基准测试完成！汇总摘要 ===")
    print(benchmark_df[["name", "win_rate_pct", "total_return_pct", "max_drawdown_pct", "top1_1to2_pct", "top3_reach4_hits"]].to_string())
    print("\n四维双向实战核算账本:")
    print(json.dumps(ledger_summary["four_dimensional_defense_audit"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
