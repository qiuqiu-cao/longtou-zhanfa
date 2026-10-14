#!/usr/bin/env python3
"""Fuse the frozen close review with public quote/minute evidence.

The bridge is the formal hand-off from automatic data capture to the daily
decision package.  Public evidence can preserve CASH, cancel a plan, or keep a
candidate on WATCH.  It cannot upgrade a plan to ENTER.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

import public_live_capture_v1 as live
from absorption_forward_protocol_v139 import classify as absorption_shadow_gate


VERSION = "v16-live-decision-bridge-1"


def code6(value: Any) -> str:
    return live.code6(value)


def flag(value: Any) -> bool:
    return live.truthy(value)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_rows(review: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(review.get("first_board_candidates", [])) + list(review.get("relay_candidates", []))
    normalized = []
    seen = set()
    for row in rows:
        code = code6(row.get("code"))
        if code in seen:
            raise ValueError("daily review contains duplicate candidate code")
        seen.add(code)
        normalized.append({**row, "code": code})
    return normalized


def _primary(review: dict[str, Any]) -> dict[str, Any] | None:
    selection = review.get("selection", {})
    candidate = selection.get("conditional_primary")
    if candidate is None:
        candidate = review.get("roles", {}).get("current_best_buy_opportunity")
    if candidate is None:
        return None
    if isinstance(candidate, str):
        return {"code": code6(candidate)}
    if isinstance(candidate, dict) and candidate.get("code"):
        return {**candidate, "code": code6(candidate["code"])}
    raise ValueError("daily review primary choice has invalid shape")


def build(
    review: dict[str, Any], quote_manifest: dict[str, Any], quotes: pd.DataFrame,
    minute_manifest: dict[str, Any] | None = None, measurements: pd.DataFrame | None = None,
) -> dict[str, Any]:
    signal_date = str(review.get("signal_date", ""))
    if not signal_date or quote_manifest.get("signal_date") != signal_date:
        raise ValueError("close review and public quote capture signal_date differ")
    trade_date, asof = quote_manifest.get("trade_date"), quote_manifest.get("asof")
    if not trade_date or not asof:
        raise ValueError("public quote manifest lacks trade_date/asof")
    if quote_manifest.get("direct_entry_authorized") is not False:
        raise ValueError("public quote source must not claim entry authority")
    candidates = _candidate_rows(review)
    expected_codes = {row["code"] for row in candidates}
    quote_frame = quotes.copy()
    quote_frame["code"] = quote_frame["code"].map(code6)
    if set(quote_frame["code"]) != expected_codes or quote_frame["code"].duplicated().any():
        raise ValueError("public quote capture does not cover the exact close-review pool")
    quote_map = quote_frame.set_index("code").to_dict("index")

    measurement_map: dict[str, dict[str, Any]] = {}
    if minute_manifest is not None:
        if measurements is None:
            raise ValueError("minute manifest supplied without measurements")
        if (minute_manifest.get("signal_date") != signal_date
                or minute_manifest.get("trade_date") != trade_date
                or minute_manifest.get("asof") != asof
                or minute_manifest.get("direct_entry_authorized") is not False):
            raise ValueError("minute evidence does not bind to the quote/review session")
        minute_frame = measurements.copy()
        minute_frame["code"] = minute_frame["code"].map(code6)
        if set(minute_frame["code"]) != expected_codes or minute_frame["code"].duplicated().any():
            raise ValueError("public minute capture does not cover the exact close-review pool")
        measurement_map = minute_frame.set_index("code").to_dict("index")

    merged = []
    for candidate in candidates:
        code = candidate["code"]
        quote = quote_map[code]
        minute_row = measurement_map.get(code, {})
        shadow_state = absorption_shadow_gate(
            asof, minute_row.get("minute_data_status"), minute_row.get("close_to_session_vwap_proxy"))
        merged.append({
            "code": code, "name": candidate.get("name"), "boards": candidate.get("boards"),
            "board_group": candidate.get("board_group"),
            "eligibility_status": candidate.get("eligibility_status"),
            "active_hard_risks": candidate.get("active_hard_risks", []),
            "quote_data_status": quote.get("quote_data_status"),
            "quote_age_seconds": quote.get("quote_age_seconds"),
            "open_gap_proxy": quote.get("auction_gap_proxy"), "last": quote.get("last"),
            "at_limit_price_proxy": flag(quote.get("at_limit_price_proxy", False)),
            "buyability": quote.get("public_level1_buyability"),
            "public_price_decision": quote.get("public_feed_decision"),
            "minute_data_status": minute_row.get("minute_data_status"),
            "close_to_vwap_proxy": minute_row.get("close_to_session_vwap_proxy"),
            "vwap_slope_5m_proxy": minute_row.get("vwap_slope_5m_proxy"),
            "failed_seal_twice_proxy": flag(minute_row.get("failed_seal_twice_proxy", False)),
            "below_vwap_expanding_proxy": flag(minute_row.get("below_vwap_expanding_proxy", False)),
            "reseal_minute_proxy": minute_row.get("reseal_minute_proxy"),
            "absorption_shadow_gate_v139": shadow_state,
            "absorption_shadow_gate_effect": "NONE_RESEARCH_ONLY",
            "direct_entry_authorized": False,
        })

    blocking = list(review.get("blocking_reasons", []))
    primary = _primary(review)
    primary_live = None
    if primary is not None:
        primary_live = next((row for row in merged if row["code"] == primary["code"]), None)
        if primary_live is None:
            raise ValueError("primary choice is absent from exact candidate pool")

    breadth = quote_manifest.get("market_breadth", {})
    live_risk_off = (breadth.get("status") == "INFERRED_FROM_FRESH_PUBLIC_QUOTES"
                     and isinstance(breadth.get("decline_ratio"), (int, float))
                     and breadth["decline_ratio"] >= .70)
    if live_risk_off:
        blocking.append("LIVE_BREADTH_RISK_OFF_DECLINERS_AT_LEAST_70_PERCENT")
    if quote_manifest.get("status") != "CONDITIONAL_PUBLIC_PRICE_FRESH":
        blocking.append("PUBLIC_QUOTE_CAPTURE_NOT_FRESH_AND_ON_TIME")

    if review.get("status") == "CASH" or primary is None:
        final = "CASH"
        blocking.append("NO_FROZEN_CONDITIONAL_PRIMARY_FROM_CLOSE_REVIEW")
    elif primary_live["quote_data_status"] != "VERIFIED_PUBLIC_QUOTE_FRESH":
        final = "CASH"
        blocking.append("PRIMARY_PUBLIC_QUOTE_NOT_FRESH")
    elif primary_live["public_price_decision"] in {"CANCEL_WEAK_OPEN", "CASH_DATA_NOT_FRESH"}:
        final = "CANCEL"
        blocking.append("PRIMARY_PUBLIC_PRICE_CANCEL")
    elif primary_live["below_vwap_expanding_proxy"] or primary_live["failed_seal_twice_proxy"]:
        final = "CANCEL"
        blocking.append("PRIMARY_PUBLIC_MINUTE_RISK_PROXY")
    elif primary_live["at_limit_price_proxy"]:
        final = "WATCH_AT_LIMIT_BUYABILITY_UNKNOWN"
        blocking.append("LEVEL2_QUEUE_AND_FILL_UNAVAILABLE")
    elif minute_manifest is None:
        final = "WATCH_PUBLIC_PRICE_ONLY"
        blocking.append("PUBLIC_MINUTE_EVIDENCE_NOT_SUPPLIED")
    elif primary_live["minute_data_status"] != "VERIFIED_PUBLIC_MINUTE_BAR_FRESH":
        final = "WATCH_PUBLIC_MINUTE_NOT_FRESH"
        blocking.append("PUBLIC_MINUTE_EVIDENCE_NOT_FRESH")
    else:
        final = "WATCH_PUBLIC_EVIDENCE_ONLY"
        blocking.append("LEVEL2_QUEUE_AUCTION_CANCEL_AND_ACTUAL_FILL_UNAVAILABLE")

    return {
        "version": VERSION, "status": final, "signal_date": signal_date,
        "trade_date": trade_date, "asof": asof,
        "close_review_status": review.get("status"),
        "public_quote_status": quote_manifest.get("status"),
        "public_minute_status": minute_manifest.get("status") if minute_manifest else None,
        "market": {"close_emotion": review.get("market"), "live_breadth": breadth,
                   "live_breadth_risk_off": live_risk_off},
        "roles": review.get("roles", {}), "primary_choice": primary,
        "primary_live_evidence": primary_live,
        "first_board_candidates": [row for row in merged if row["board_group"] == "FIRST_BOARD"],
        "relay_candidates": [row for row in merged if row["board_group"] == "CONSECUTIVE_BOARD"],
        "blocking_reasons": sorted(set(blocking)),
        "data_quality": {
            "verified": "timestamped public Level-1 price fields when quote_data_status is fresh",
            "inferred": "market breadth and all public 1-minute support/reseal/VWAP measurements",
            "unavailable": "exact auction cancel sequence, active tick orders, Level-2 queue and actual fill",
            "execution_limitation": "public evidence can only preserve WATCH/CANCEL/CASH",
        },
        "direct_entry_authorized": False, "order_submitted": False,
        "automatic_backup_allowed": False, "guarantee_of_premium": False,
        "absorption_shadow_gate_live_effect": "NONE_RESEARCH_ONLY",
    }


def _bound_file(manifest_path: Path, manifest: dict[str, Any], key: str, fallback: str) -> Path:
    block = manifest.get("files", {}).get(key, {})
    path = Path(block.get("path", "")) if block.get("path") else manifest_path.parent / fallback
    if not path.exists():
        path = manifest_path.parent / fallback
    if not path.exists():
        raise FileNotFoundError(f"bound {key} file missing")
    expected = block.get("sha256")
    if expected and sha256_file(path) != expected:
        raise ValueError(f"bound {key} hash mismatch")
    return path


def make_report(result: dict[str, Any]) -> str:
    quality = result["data_quality"]
    lines = [
        f"# {result['trade_date']} {result['asof']} 龙头实时决策桥", "",
        f"- 最终状态：**{result['status']}**",
        f"- 收盘复盘：{result['close_review_status']}",
        f"- 公开价格：{result['public_quote_status']}",
        f"- 公开分钟：{result['public_minute_status'] or '未提供'}",
        f"- 已核验：{quality['verified']}", f"- 推断：{quality['inferred']}",
        f"- 不可用：{quality['unavailable']}", f"- 执行限制：{quality['execution_limitation']}", "",
        "## 阻断条件", "",
    ]
    lines.extend([f"- {reason}" for reason in result["blocking_reasons"]] or ["- 无"])
    lines += ["", "## 首板与连板实时证据", "",
              "| 代码 | 名称 | 板位 | 价格状态 | 分钟状态 | 价格判断 | 可买性 |", "|---|---|---:|---|---|---|---|"]
    rows = result["relay_candidates"] + result["first_board_candidates"]
    for row in rows:
        lines.append(f"| {row['code']} | {row['name']} | {row['boards']} | {row['quote_data_status']} | "
                     f"{row['minute_data_status'] or '未提供'} | {row['public_price_decision']} | {row['buyability']} |")
    lines += ["", "公开行情不会把首选失效后临时排名第二的股票递补为买入候选。", ""]
    return "\n".join(lines)


def run(
    review_path: Path, quote_manifest_path: Path, output: Path,
    minute_manifest_path: Path | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("choose a new immutable live decision output directory")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    quote_manifest = json.loads(quote_manifest_path.read_text(encoding="utf-8"))
    quotes_path = _bound_file(quote_manifest_path, quote_manifest, "candidate_quotes", "candidate_quotes.csv")
    minute_manifest, measurements = None, None
    source_paths = [review_path, quote_manifest_path, quotes_path, Path(__file__)]
    if minute_manifest_path is not None:
        minute_manifest = json.loads(minute_manifest_path.read_text(encoding="utf-8"))
        measurements_path = _bound_file(minute_manifest_path, minute_manifest, "measurements", "measurements.csv")
        measurements = pd.read_csv(measurements_path, dtype={"code": "string"})
        source_paths += [minute_manifest_path, measurements_path]
    result = build(
        review, quote_manifest, pd.read_csv(quotes_path, dtype={"code": "string"}),
        minute_manifest, measurements,
    )
    output.mkdir(parents=True, exist_ok=False)
    result["source_hashes"] = {str(path.resolve()): sha256_file(path) for path in source_paths}
    json_path, report_path = output / "decision.json", output / "report.md"
    report_path.write_text(make_report(result), encoding="utf-8")
    result["report_path"] = str(report_path.resolve())
    result["report_sha256"] = sha256_file(report_path)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**result, "decision_path": str(json_path.resolve())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse close review with timestamped public quote/minute evidence")
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--quote-manifest", type=Path, required=True)
    parser.add_argument("--minute-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run(args.review, args.quote_manifest, args.output, args.minute_manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))
