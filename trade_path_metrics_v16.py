#!/usr/bin/env python3
"""Compute B+1/B+2/B+3 net marks and path risk from verified daily bars."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

import broker_execution_import_v16 as entry
import broker_exit_import_v16 as lifecycle


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "v16" / "trade_path_metrics"
VERSION = "v16-trade-path-metrics-1"
VERIFIED_STATUSES = {"verified_terminal_export", "verified_broker_export", "verified_official_export"}
SELLABILITY_STATES = {"NORMAL", "LOCKED_LIMIT_DOWN", "SUSPENDED", "NO_SELL_LIQUIDITY", "UNKNOWN"}
MARK_FIELDS = {
    "date", "code", "open", "high", "low", "close", "source", "observed_at",
    "available_at", "data_status", "price_basis", "sellability_state",
}
COST_FIELDS = {
    "sell_commission_rate", "sell_min_commission_cash", "sell_stamp_tax_rate",
    "sell_transfer_fee_rate", "sell_other_fee_rate", "currency", "policy_id",
}


def validate_cost_policy(plan: dict[str, Any], first_buy: pd.Timestamp) -> dict[str, Any]:
    if pd.Timestamp(plan.get("frozen_at")) >= first_buy:
        raise ValueError("exit cost policy was not frozen before entry")
    policy = plan.get("cost_policy", {})
    missing = sorted(COST_FIELDS - set(policy))
    if missing:
        raise ValueError(f"frozen exit cost policy missing fields: {missing}")
    result = dict(policy)
    for field in COST_FIELDS - {"currency", "policy_id"}:
        value = pd.to_numeric(result[field], errors="coerce")
        if pd.isna(value) or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"invalid frozen cost policy field: {field}")
        result[field] = float(value)
    if result["currency"] != "CNY" or not str(result["policy_id"]).strip():
        raise ValueError("cost policy requires CNY and a nonempty policy_id")
    return result


def estimated_sell_fee(notional: float, policy: dict[str, Any]) -> float:
    commission = max(notional * policy["sell_commission_rate"], policy["sell_min_commission_cash"])
    rates = policy["sell_stamp_tax_rate"] + policy["sell_transfer_fee_rate"] + policy["sell_other_fee_rate"]
    return commission + notional * rates


def validate_marks(frame: pd.DataFrame, code: str, asof: Any) -> pd.DataFrame:
    missing = sorted(MARK_FIELDS - set(frame.columns))
    if missing:
        raise ValueError(f"daily marks missing fields: {missing}")
    cutoff = pd.Timestamp(asof)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize(entry.SHANGHAI)
    else:
        cutoff = cutoff.tz_convert(entry.SHANGHAI)
    result = frame.copy()
    result["code"] = result["code"].map(entry.code6)
    result = result[result["code"].eq(entry.code6(code))].copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if result.empty or result["date"].isna().any() or result["date"].duplicated().any():
        raise ValueError("daily marks must contain unique valid dates for the entry code")
    for field in ("open", "high", "low", "close"):
        result[field] = pd.to_numeric(result[field], errors="coerce")
    if result[["open", "high", "low", "close"]].isna().any().any() or (result[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("daily mark OHLC must be finite and positive")
    if (result["high"] < result[["open", "low", "close"]].max(axis=1)).any() or (result["low"] > result[["open", "high", "close"]].min(axis=1)).any():
        raise ValueError("daily mark OHLC relationship is invalid")
    if not result["data_status"].isin(VERIFIED_STATUSES).all() or result["source"].astype(str).str.strip().eq("").any():
        raise ValueError("daily marks require a verified source status")
    if not result["price_basis"].eq("unadjusted_no_corporate_action").all():
        raise ValueError("daily marks require an unadjusted no-corporate-action price basis")
    if not result["sellability_state"].isin(SELLABILITY_STATES).all():
        raise ValueError("daily mark sellability_state is invalid")
    observed = pd.to_datetime(result["observed_at"], errors="coerce", utc=True).dt.tz_convert(entry.SHANGHAI)
    available = pd.to_datetime(result["available_at"], errors="coerce", utc=True).dt.tz_convert(entry.SHANGHAI)
    expected_dates = pd.to_datetime(result["date"]).dt.date
    if observed.isna().any() or available.isna().any():
        raise ValueError("daily mark timestamps are invalid")
    if any(obs.date() != day or obs.time() < pd.Timestamp("15:00").time()
           for obs, day in zip(observed, expected_dates)):
        raise ValueError("daily mark observed_at must be at or after that session close")
    if (observed > available).any() or (available > cutoff).any():
        raise ValueError("daily mark was not available by the audit cutoff")
    result["observed_at"] = observed.map(lambda value: value.isoformat())
    result["available_at"] = available.map(lambda value: value.isoformat())
    return result.sort_values("date").reset_index(drop=True)


def calculate(
    entry_report: dict[str, Any], entry_fills: pd.DataFrame, marks: pd.DataFrame,
    sessions: list[str], exit_plan: dict[str, Any], asof: Any,
) -> dict[str, Any]:
    buys = entry_fills[entry_fills["side"].eq("BUY")].copy()
    if buys.empty:
        raise ValueError("entry report contains no filled BUY")
    codes = set(buys["code"].map(entry.code6))
    if len(codes) != 1:
        raise ValueError("trade path requires one entry code")
    code = next(iter(codes))
    if entry.code6(exit_plan.get("code")) != code:
        raise ValueError("exit plan code differs from entry")
    first_buy = pd.to_datetime(buys["fill_time"]).min()
    policy = validate_cost_policy(exit_plan, first_buy)
    if not buys["fee_data_complete"].map(lambda value: str(value).lower() in {"true", "1", "1.0"}).all():
        raise ValueError("B+ path requires complete actual entry fees")
    quantity = int(pd.to_numeric(buys["fill_quantity"]).sum())
    entry_cost = float((pd.to_numeric(buys["fill_price"]) * pd.to_numeric(buys["fill_quantity"])).sum()
                       + pd.to_numeric(buys["fee_cash"]).sum())
    b_date = str(entry_report["trade_date"])
    if b_date not in sessions:
        raise ValueError("entry date is absent from the supplied session calendar")
    origin = sessions.index(b_date)
    expected = {offset: sessions[origin + offset] if origin + offset < len(sessions) else None for offset in (1, 2, 3)}
    mark_map = marks.set_index("date").to_dict("index")
    metric_rows = []
    for offset in (1, 2, 3):
        day = expected[offset]
        row = mark_map.get(day) if day else None
        if row is None:
            metric_rows.append({"session_offset": offset, "date": day, "status": "NOT_YET_AVAILABLE_OR_MISSING",
                                "net_close_return": None, "sellability_state": None})
            continue
        close_notional = float(row["close"]) * quantity
        fee = estimated_sell_fee(close_notional, policy)
        metric_rows.append({
            "session_offset": offset, "date": day, "status": "VERIFIED_CLOSE_MARK",
            "open": float(row["open"]), "high": float(row["high"]), "low": float(row["low"]),
            "close": float(row["close"]), "estimated_sell_fee_cash": fee,
            "net_close_return": (close_notional - fee) / entry_cost - 1,
            "sellability_state": row["sellability_state"],
            "realisable_at_close": row["sellability_state"] == "NORMAL",
            "source": row["source"], "observed_at": row["observed_at"], "available_at": row["available_at"],
        })
    complete = all(row["status"] == "VERIFIED_CLOSE_MARK" for row in metric_rows)
    mfe = mae = None
    if complete:
        highs = [float(mark_map[expected[offset]]["high"]) for offset in (1, 2, 3)]
        lows = [float(mark_map[expected[offset]]["low"]) for offset in (1, 2, 3)]
        high_notional, low_notional = max(highs) * quantity, min(lows) * quantity
        mfe = (high_notional - estimated_sell_fee(high_notional, policy)) / entry_cost - 1
        mae = (low_notional - estimated_sell_fee(low_notional, policy)) / entry_cost - 1
    by_offset = {row["session_offset"]: row["net_close_return"] for row in metric_rows}
    order_flags = {str(row["order_id"]): bool(row.get("strategy_sample_eligible")) for row in entry_report.get("orders", [])}
    strategy_entry = all(order_flags.get(str(order_id), False) for order_id in buys["order_id"])
    return {
        "version": VERSION, "code": code, "entry_date_B": b_date, "entry_quantity": quantity,
        "actual_entry_cost_cash": entry_cost, "actual_entry_cost_per_share": entry_cost / quantity,
        "cost_policy": policy, "marks": metric_rows,
        "B_plus_1_net_close_return": by_offset[1],
        "B_plus_2_net_close_return": by_offset[2],
        "B_plus_3_net_close_return": by_offset[3],
        "B_plus_2_and_B_plus_3_both_positive": (
            by_offset[2] > 0 and by_offset[3] > 0 if by_offset[2] is not None and by_offset[3] is not None else None
        ),
        "window_B1_B3_complete": complete, "maximum_favourable_excursion_net": mfe,
        "maximum_adverse_excursion_net": mae,
        "sellability_blocked_offsets": [row["session_offset"] for row in metric_rows
                                        if row.get("sellability_state") not in {None, "NORMAL"}],
        "strategy_entry_eligible": strategy_entry,
        "strategy_path_metric_eligible": strategy_entry and complete,
        "warning": "close marks are valuation metrics, not assumed sale fills; intraday highs/lows are MFE/MAE only",
    }


def run(
    entry_report_path: Path, marks_path: Path, sessions_path: Path,
    exit_plan_path: Path, asof: Any, output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    entry_report_path = Path(entry_report_path)
    entry_report, entry_fills = lifecycle._verify_entry_report(entry_report_path)
    sessions = lifecycle._load_sessions(Path(sessions_path))
    plan = json.loads(Path(exit_plan_path).read_text(encoding="utf-8"))
    raw_marks = pd.read_csv(marks_path, dtype={"code": "string"})
    code = entry.code6(plan.get("code"))
    marks = validate_marks(raw_marks, code, asof)
    result = calculate(entry_report, entry_fills, marks, sessions, plan, asof)
    sources = [entry_report_path, Path(marks_path), Path(sessions_path), Path(exit_plan_path),
               Path(__file__), Path(entry.__file__)]
    result["asof"] = pd.Timestamp(asof).isoformat()
    result["source_hashes"] = {str(path.resolve()): entry.sha256_file(path) for path in sources}
    identity = hashlib.sha256(json.dumps(result["source_hashes"], sort_keys=True).encode()).hexdigest()[:20]
    target = Path(output) / result["entry_date_B"] / identity
    if target.exists():
        raise FileExistsError("immutable trade-path metric package already exists")
    target.mkdir(parents=True, exist_ok=False)
    normalized = target / "verified_marks.csv"
    marks.to_csv(normalized, index=False, encoding="utf-8-sig")
    result["normalized_marks"] = {"path": str(normalized.resolve()), "sha256": entry.sha256_file(normalized)}
    result["metric_id"] = identity
    report = target / "report.json"
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**result, "report_path": str(report.resolve()), "report_sha256": entry.sha256_file(report)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute verified B+1/B+2/B+3 path metrics")
    parser.add_argument("--entry-report", type=Path, required=True)
    parser.add_argument("--marks", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--exit-plan", type=Path, required=True)
    parser.add_argument("--asof", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.entry_report, args.marks, args.sessions, args.exit_plan, args.asof, args.output),
                     ensure_ascii=False, indent=2))
