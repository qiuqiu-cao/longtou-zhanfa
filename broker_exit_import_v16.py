#!/usr/bin/env python3
"""Close one imported broker entry and audit T+1/frozen exit compliance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

import broker_execution_import_v16 as entry
import exit_plan_audit


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "v16" / "broker_exit"
VERSION = "v16-broker-exit-import-1"


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "1.0", "yes"}


def _verify_entry_report(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("version") != entry.VERSION:
        raise ValueError("entry report is not the current broker import version")
    for item in report.get("source_hashes", {}).items():
        source, expected = item
        source_path = Path(source)
        if not source_path.exists() or entry.sha256_file(source_path) != expected:
            raise ValueError("entry source hash mismatch")
    block = report.get("normalized_files", {}).get("fills", {})
    normalized = Path(str(block.get("path", "")))
    if not normalized.exists() or entry.sha256_file(normalized) != block.get("sha256"):
        raise ValueError("normalized entry fills hash mismatch")
    fills = pd.read_csv(normalized, dtype={"code": "string", "order_id": "string", "fill_id": "string"})
    return report, fills


def _load_sessions(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    field = "date" if "date" in frame else "trade_date" if "trade_date" in frame else None
    if field is None:
        raise ValueError("session file requires date or trade_date")
    dates = pd.to_datetime(frame[field], errors="coerce")
    if dates.isna().any():
        raise ValueError("session file contains invalid dates")
    result = sorted(set(dates.dt.strftime("%Y-%m-%d")))
    if not result:
        raise ValueError("session file is empty")
    return result


def reconcile_exit(
    entry_report: dict[str, Any], entry_fills: pd.DataFrame,
    exit_orders: pd.DataFrame, exit_fills: pd.DataFrame,
    plan: dict[str, Any], confirmations: list[dict[str, Any]],
    sessions: list[str], captured_at: Any,
) -> dict[str, Any]:
    captured = entry._timestamp(captured_at, pd.Timestamp(captured_at).strftime("%Y-%m-%d"))
    if plan.get("plan_id") is None or plan.get("code") is None:
        raise ValueError("frozen exit plan requires plan_id and code")
    code = entry.code6(plan["code"])
    if str(plan.get("kind")) != "user_confirmed":
        raise ValueError("broker lifecycle exit plan kind must be user_confirmed")
    buys = entry_fills[entry_fills["side"].eq("BUY")].copy()
    if buys.empty or set(buys["code"].map(entry.code6)) != {code}:
        raise ValueError("entry fills do not uniquely match exit plan code")
    if not exit_orders.empty and (set(exit_orders["code"]) != {code} or set(exit_orders["side"]) != {"SELL"}):
        raise ValueError("exit orders must be SELL orders for the entry code")
    if not exit_fills.empty:
        if set(exit_fills["code"]) != {code} or set(exit_fills["side"]) != {"SELL"}:
            raise ValueError("exit fills must be SELL fills for the entry code")
        if not set(exit_fills["order_id"]).issubset(set(exit_orders["order_id"])):
            raise ValueError("exit fill references an absent order")
    for order in exit_orders.to_dict("records"):
        filled = int(exit_fills.loc[exit_fills["order_id"].eq(order["order_id"]), "fill_quantity"].sum()) if not exit_fills.empty else 0
        if filled > int(order["order_quantity"]):
            raise ValueError("exit filled quantity exceeds order quantity")
    first_buy = pd.to_datetime(buys["fill_time"]).min()
    if any(pd.Timestamp(value) <= first_buy for value in exit_fills.get("fill_time", [])):
        raise ValueError("T+1 violation: sell fill is not after the entry session")
    if any(pd.Timestamp(value).date() == first_buy.date() for value in exit_fills.get("fill_time", [])):
        raise ValueError("T+1 violation: same-day sell is not permitted")
    if any(pd.Timestamp(value) > captured for value in list(exit_orders.get("order_time", [])) + list(exit_fills.get("fill_time", []))):
        raise ValueError("exit export contains records after captured_at")

    order_flags = {
        str(row["order_id"]): bool(row.get("strategy_sample_eligible"))
        for row in entry_report.get("orders", [])
    }
    buy_quantity = int(pd.to_numeric(buys["fill_quantity"]).sum())
    strategy_buy_quantity = int(pd.to_numeric(
        buys[buys["order_id"].map(lambda value: order_flags.get(str(value), False))]["fill_quantity"]
    ).sum())
    sell_quantity = int(pd.to_numeric(exit_fills["fill_quantity"]).sum()) if not exit_fills.empty else 0
    if sell_quantity > buy_quantity:
        raise ValueError("cumulative exit quantity exceeds entry quantity")

    confirmation_by_fill = {str(row.get("fill_id")): row.get("exit_confirmation_id") for row in confirmations}
    audit_fills = []
    trade_id = str(entry_report.get("import_id"))
    for row in buys.to_dict("records"):
        audit_fills.append({
            "plan_id": plan["plan_id"], "trade_id": trade_id, "code": code,
            "kind": "user_confirmed", "side": "BUY", "shares": int(row["fill_quantity"]),
            "filled_at": row["fill_time"], "evidence_id": str(row["fill_id"]),
        })
    for row in exit_fills.to_dict("records"):
        audit_fills.append({
            "plan_id": plan["plan_id"], "trade_id": trade_id, "code": code,
            "kind": "user_confirmed", "side": "SELL", "shares": int(row["fill_quantity"]),
            "filled_at": row["fill_time"], "evidence_id": str(row["fill_id"]),
            "exit_confirmation_id": confirmation_by_fill.get(str(row["fill_id"])),
        })
    audit_result = exit_plan_audit.audit(audit_fills, [plan], confirmations, sessions, captured.isoformat())
    trade_audit = audit_result["trades"][0] if audit_result.get("trades") else None
    fully_closed = buy_quantity > 0 and sell_quantity == buy_quantity
    buy_fee_complete = bool(not buys.empty and buys["fee_data_complete"].map(_truthy).all())
    sell_fee_complete = bool(not exit_fills.empty and exit_fills["fee_data_complete"].map(_truthy).all())
    fee_complete = buy_fee_complete and sell_fee_complete
    buy_cost = None
    sell_proceeds = None
    actual_net_return = None
    if fully_closed and fee_complete:
        buy_cost = float((pd.to_numeric(buys["fill_price"]) * pd.to_numeric(buys["fill_quantity"])).sum()
                         + pd.to_numeric(buys["fee_cash"]).sum())
        sell_proceeds = float((pd.to_numeric(exit_fills["fill_price"]) * pd.to_numeric(exit_fills["fill_quantity"])).sum()
                              - pd.to_numeric(exit_fills["fee_cash"]).sum())
        actual_net_return = sell_proceeds / buy_cost - 1
    all_entry_strategy_eligible = strategy_buy_quantity == buy_quantity and buy_quantity > 0
    exit_compliant = bool(audit_result.get("status") == "DECLARED_EXIT_RULES_MATCH" and trade_audit
                          and trade_audit.get("terminal_state") == "CLOSED")
    strategy_complete = bool(fully_closed and fee_complete and all_entry_strategy_eligible and exit_compliant)
    primary_choice = entry_report.get("primary_choice") if isinstance(entry_report.get("primary_choice"), dict) else {}
    return {
        "version": VERSION, "entry_trade_date": entry_report.get("trade_date"),
        "exit_asof": captured.isoformat(), "code": code, "plan_id": plan["plan_id"],
        "board_group": primary_choice.get("board_group"),
        "candidate_role": primary_choice.get("candidate_role") or primary_choice.get("role"),
        "entry_quantity": buy_quantity, "strategy_eligible_entry_quantity": strategy_buy_quantity,
        "sold_quantity": sell_quantity, "remaining_quantity": buy_quantity - sell_quantity,
        "fully_closed": fully_closed, "fee_data_complete": fee_complete,
        "buy_cost_cash": buy_cost, "sell_proceeds_cash": sell_proceeds,
        "actual_complete_trade_net_return": actual_net_return,
        "actual_complete_trade_large_loss": None if actual_net_return is None else actual_net_return <= -0.07,
        "strategy_complete_trade_eligible": strategy_complete,
        "strategy_net_return": actual_net_return if strategy_complete else None,
        "strategy_large_loss": (actual_net_return <= -0.07) if strategy_complete else None,
        "exit_audit": audit_result,
        "exit_order_count": len(exit_orders), "exit_fill_count": len(exit_fills),
        "partial_exit": 0 < sell_quantity < buy_quantity,
        "unresolved_open_position": sell_quantity < buy_quantity,
        "execution_independently_verified": False,
        "source_limit": "broker/Tonghuashun CSV provenance is user-confirmed; authenticated API verification is unavailable",
    }


def run(
    entry_report_path: Path, exit_orders_path: Path, exit_fills_path: Path,
    exit_trade_date: str, plan_path: Path, confirmations_path: Path,
    sessions_path: Path, captured_at: Any, source: str,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    entry_report_path = Path(entry_report_path)
    entry_report, entry_fills = _verify_entry_report(entry_report_path)
    exit_orders = entry.normalize_orders(pd.read_csv(exit_orders_path, dtype="string"), exit_trade_date)
    exit_fills = entry.normalize_fills(pd.read_csv(exit_fills_path, dtype="string"), exit_trade_date)
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    confirmations_payload = json.loads(Path(confirmations_path).read_text(encoding="utf-8"))
    confirmations = confirmations_payload if isinstance(confirmations_payload, list) else confirmations_payload.get("exit_confirmations", [])
    sessions = _load_sessions(Path(sessions_path))
    result = reconcile_exit(entry_report, entry_fills, exit_orders, exit_fills, plan, confirmations, sessions, captured_at)
    sources = [entry_report_path, Path(exit_orders_path), Path(exit_fills_path), Path(plan_path),
               Path(confirmations_path), Path(sessions_path), Path(__file__), Path(exit_plan_audit.__file__)]
    result.update({
        "source": source, "data_status": "USER_CONFIRMED_TERMINAL_EXPORT",
        "source_hashes": {str(path.resolve()): entry.sha256_file(path) for path in sources},
    })
    identity = hashlib.sha256(json.dumps(result["source_hashes"], sort_keys=True).encode()).hexdigest()[:20]
    target = Path(output) / exit_trade_date / identity
    if target.exists():
        raise FileExistsError("immutable broker exit import already exists")
    target.mkdir(parents=True, exist_ok=False)
    normalized_orders = target / "normalized_exit_orders.csv"
    normalized_fills = target / "normalized_exit_fills.csv"
    exit_orders.to_csv(normalized_orders, index=False, encoding="utf-8-sig")
    exit_fills.to_csv(normalized_fills, index=False, encoding="utf-8-sig")
    result["normalized_files"] = {
        "orders": {"path": str(normalized_orders.resolve()), "sha256": entry.sha256_file(normalized_orders)},
        "fills": {"path": str(normalized_fills.resolve()), "sha256": entry.sha256_file(normalized_fills)},
    }
    result["exit_import_id"] = identity
    report_path = target / "report.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**result, "report_path": str(report_path.resolve()), "report_sha256": entry.sha256_file(report_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import broker exits and audit a complete trade")
    parser.add_argument("--entry-report", type=Path, required=True)
    parser.add_argument("--exit-orders", type=Path, required=True)
    parser.add_argument("--exit-fills", type=Path, required=True)
    parser.add_argument("--exit-trade-date", required=True)
    parser.add_argument("--exit-plan", type=Path, required=True)
    parser.add_argument("--exit-confirmations", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--captured-at", required=True)
    parser.add_argument("--source", default="用户确认的同花顺/券商CSV导出")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(
        args.entry_report, args.exit_orders, args.exit_fills, args.exit_trade_date,
        args.exit_plan, args.exit_confirmations, args.sessions, args.captured_at,
        args.source, args.output,
    ), ensure_ascii=False, indent=2))
