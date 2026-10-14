#!/usr/bin/env python3
"""Fail-closed preflight for the next-session public live monitor."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

import public_live_capture_v1 as live
import public_live_monitor_v1 as monitor


ROOT = Path(__file__).resolve().parent
SHANGHAI = ZoneInfo("Asia/Shanghai")
VERSION = "v17-public-live-preflight-1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _review_codes(review: dict[str, Any]) -> list[str]:
    rows = list(review.get("first_board_candidates", [])) + list(review.get("relay_candidates", []))
    codes = [live.code6(row.get("code")) for row in rows]
    if len(codes) != len(set(codes)):
        raise ValueError("daily review contains duplicate candidate codes")
    return codes


def evaluate(candidates: pd.DataFrame, review: dict[str, Any], universe: pd.DataFrame,
             trade_date: str, prior_session: str) -> dict[str, Any]:
    candidate_codes = set(candidates["code"].map(live.code6))
    review_codes = set(_review_codes(review))
    universe_codes = set(universe["code"].map(live.code6)) if not universe.empty else set()
    exact_pool = candidate_codes == review_codes
    missing_from_universe = sorted(candidate_codes - universe_codes)
    selection = review.get("selection", {})
    primary = selection.get("conditional_primary")
    primary_code = live.code6(primary.get("code")) if isinstance(primary, dict) and primary.get("code") else None
    maximum = pd.to_numeric(selection.get("maximum_acceptable_price"), errors="coerce")
    shares = pd.to_numeric(selection.get("position_shares"), errors="coerce")
    holding_plan = selection.get("holding_plan")
    primary_row = None
    for row in list(review.get("first_board_candidates", [])) + list(review.get("relay_candidates", [])):
        if primary_code and live.code6(row.get("code")) == primary_code:
            primary_row = row
            break
    plan_checks = {
        "conditional_primary_frozen": primary_code in candidate_codes if primary_code else False,
        "maximum_acceptable_price_frozen": bool(pd.notna(maximum) and float(maximum) > 0),
        "position_frozen_100_share_lot": bool(pd.notna(shares) and int(shares) > 0 and int(shares) % 100 == 0),
        "holding_plan_frozen": bool(holding_plan),
        "auction_conditions_frozen": bool(primary_row and primary_row.get("auction_conditions")),
        "opening_conditions_frozen": bool(primary_row and primary_row.get("opening_conditions")),
        "cancel_conditions_frozen": bool(primary_row and primary_row.get("cancel_conditions")),
    }
    collection_checks = {
        "prior_session_bound": str(prior_session) == str(review.get("signal_date")),
        "candidate_review_pool_exact": exact_pool,
        "candidate_pool_nonempty": bool(candidate_codes),
        "public_universe_file_nonempty": bool(universe_codes),
        "all_candidates_in_public_universe": not missing_from_universe,
    }
    collection_ready = all(collection_checks.values())
    conditional_plan_ready = collection_ready and all(plan_checks.values())
    status = ("COLLECTION_AND_CONDITIONAL_PLAN_READY" if conditional_plan_ready else
              "COLLECTION_READY_DECISION_CASH" if collection_ready else
              "PREFLIGHT_FAILED")
    return {
        "version": VERSION,
        "status": status,
        "trade_date": pd.Timestamp(trade_date).strftime("%Y-%m-%d"),
        "signal_date": str(review.get("signal_date")),
        "prior_session": str(prior_session),
        "candidate_rows": len(candidate_codes),
        "first_board_rows": len(review.get("first_board_candidates", [])),
        "relay_rows": len(review.get("relay_candidates", [])),
        "collection_checks": collection_checks,
        "plan_checks": plan_checks,
        "missing_from_public_universe": missing_from_universe,
        "primary_code": primary_code,
        "collection_ready": collection_ready,
        "conditional_plan_ready": conditional_plan_ready,
        "monitor_effect": ("capture all registered points; decision remains CASH/WATCH"
                           if collection_ready and not conditional_plan_ready else
                           "capture registered points and evaluate the frozen conditional plan"
                           if conditional_plan_ready else "do not start monitor"),
        "direct_entry_authorized": False,
        "guarantee_of_premium": False,
    }


def make_report(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['trade_date']} 公开早盘监控预检", "",
        f"- 状态：**{result['status']}**",
        f"- 复盘日/前一交易日：{result['signal_date']} / {result['prior_session']}",
        f"- 全池：{result['candidate_rows']}只（首板{result['first_board_rows']}、连板{result['relay_rows']}）",
        f"- 冻结首选：{result['primary_code'] or '无'}", "",
        "## 采集检查", "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in result["collection_checks"].items())
    lines += ["", "## 交易计划检查", ""]
    lines.extend(f"- {key}: {value}" for key, value in result["plan_checks"].items())
    lines += ["", f"结论：{result['monitor_effect']}。公开行情不会单独证明排队成交，也不保证溢价。", ""]
    return "\n".join(lines)


def run(candidate_path: Path, review_path: Path, universe_path: Path, market_dates_path: Path,
        trade_date: str, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("choose a new immutable output directory")
    prior = monitor.validate_prior_session_binding(candidate_path, review_path, trade_date, market_dates_path)
    candidates = live.load_candidates(candidate_path)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    universe = live.load_universe(universe_path)
    result = evaluate(candidates, review, universe, trade_date, prior)
    result["created_at"] = datetime.now(SHANGHAI).isoformat()
    result["source_hashes"] = {str(path.resolve()): sha256_file(path) for path in
                               (candidate_path, review_path, universe_path, market_dates_path, Path(__file__))}
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "report.md"
    report_path.write_text(make_report(result), encoding="utf-8")
    result["report_path"] = str(report_path.resolve())
    result["report_sha256"] = sha256_file(report_path)
    result["monitor_command"] = (
        f"python public_live_monitor_v1.py --trade-date {result['trade_date']} "
        f"--candidates \"{candidate_path.resolve()}\" --review \"{review_path.resolve()}\""
    )
    (output / "preflight.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--universe", type=Path, default=live.DEFAULT_UNIVERSE)
    parser.add_argument("--market-dates", type=Path, default=monitor.DEFAULT_MARKET_DATES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.candidates or live.discover_candidates(),
                         args.review or monitor.discover_review(), args.universe,
                         args.market_dates, args.trade_date, args.output), ensure_ascii=False, indent=2))
