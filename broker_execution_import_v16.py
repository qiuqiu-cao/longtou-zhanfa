#!/usr/bin/env python3
"""Import broker/Tonghuashun entry orders and fills against one live decision.

The adapter records actual user execution without retroactively promoting a
WATCH/CANCEL/CASH decision to ENTER.  A fill is a strategy sample only when the
pre-existing decision explicitly authorised entry and every execution boundary
is satisfied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "v16" / "broker_execution"
VERSION = "v16-broker-execution-import-2"
SHANGHAI = ZoneInfo("Asia/Shanghai")

ORDER_ALIASES = {
    "order_id": ("order_id", "委托编号", "合同编号"),
    "code": ("code", "股票代码", "证券代码", "代码"),
    "name": ("name", "股票简称", "证券名称", "名称"),
    "side": ("side", "买卖方向", "委托方向", "操作"),
    "order_time": ("order_time", "委托时间", "申报时间"),
    "order_price": ("order_price", "委托价格", "申报价格"),
    "order_quantity": ("order_quantity", "委托数量", "申报数量"),
    "order_status": ("order_status", "委托状态", "状态"),
}
FILL_ALIASES = {
    "fill_id": ("fill_id", "成交编号", "成交序号"),
    "order_id": ("order_id", "委托编号", "合同编号"),
    "code": ("code", "股票代码", "证券代码", "代码"),
    "name": ("name", "股票简称", "证券名称", "名称"),
    "side": ("side", "买卖方向", "成交方向", "操作"),
    "fill_time": ("fill_time", "成交时间"),
    "fill_price": ("fill_price", "成交价格", "成交均价"),
    "fill_quantity": ("fill_quantity", "成交数量"),
    "commission": ("commission", "佣金"),
    "stamp_tax": ("stamp_tax", "印花税"),
    "transfer_fee": ("transfer_fee", "过户费"),
    "other_fee": ("other_fee", "其他费用", "规费"),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code6(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if "." in text and len(text.split(".")[0]) == 6:
        text = text.split(".")[0]
    return text.zfill(6)


def _column(frame: pd.DataFrame, aliases: dict[str, tuple[str, ...]], field: str, required: bool = True) -> str | None:
    hits = [name for name in aliases[field] if name in frame.columns]
    if len(hits) > 1:
        raise ValueError(f"multiple columns map to {field}: {hits}")
    if required and not hits:
        raise ValueError(f"export missing required field: {field}")
    return hits[0] if hits else None


def _timestamp(value: Any, trade_date: str) -> pd.Timestamp:
    text = str(value).strip()
    stamp = pd.to_datetime(f"{trade_date} {text}" if len(text) <= 12 and ":" in text else text, errors="coerce")
    if pd.isna(stamp):
        raise ValueError(f"invalid execution timestamp: {value}")
    stamp = pd.Timestamp(stamp)
    return stamp.tz_localize(SHANGHAI) if stamp.tzinfo is None else stamp.tz_convert(SHANGHAI)


def _side(value: Any) -> str:
    text = str(value).strip().upper()
    mapping = {"买入": "BUY", "证券买入": "BUY", "B": "BUY", "BUY": "BUY",
               "卖出": "SELL", "证券卖出": "SELL", "S": "SELL", "SELL": "SELL"}
    if text not in mapping:
        raise ValueError(f"unsupported side: {value}")
    return mapping[text]


def _number(value: Any, *, positive: bool = False) -> float:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number) or not math.isfinite(float(number)) or float(number) < 0 or (positive and float(number) == 0):
        raise ValueError(f"invalid numeric execution value: {value}")
    return float(number)


def normalize_orders(frame: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    columns = {field: _column(frame, ORDER_ALIASES, field, field != "name") for field in ORDER_ALIASES}
    rows = []
    for _, source in frame.iterrows():
        quantity = _number(source[columns["order_quantity"]], positive=True)
        if not quantity.is_integer():
            raise ValueError("order quantity must be an integer")
        side = _side(source[columns["side"]])
        if side == "BUY" and int(quantity) % 100:
            raise ValueError("A-share buy order quantity must be a multiple of 100")
        stamp = _timestamp(source[columns["order_time"]], trade_date)
        if stamp.date().isoformat() != trade_date:
            raise ValueError("order timestamp differs from decision trade_date")
        rows.append({
            "order_id": str(source[columns["order_id"]]).strip(), "code": code6(source[columns["code"]]),
            "name": str(source[columns["name"]]).strip() if columns["name"] else "",
            "side": side, "order_time": stamp.isoformat(),
            "order_price": _number(source[columns["order_price"]], positive=True),
            "order_quantity": int(quantity), "order_status": str(source[columns["order_status"]]).strip(),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=["order_id", "code", "name", "side", "order_time", "order_price", "order_quantity", "order_status"])
    if result["order_id"].eq("").any() or result["order_id"].duplicated().any():
        raise ValueError("order ids must be nonempty and unique")
    return result.sort_values(["order_time", "order_id"]).reset_index(drop=True)


def normalize_fills(frame: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    required = {"fill_id", "order_id", "code", "side", "fill_time", "fill_price", "fill_quantity"}
    columns = {field: _column(frame, FILL_ALIASES, field, field in required) for field in FILL_ALIASES}
    rows = []
    for _, source in frame.iterrows():
        quantity = _number(source[columns["fill_quantity"]], positive=True)
        if not quantity.is_integer():
            raise ValueError("fill quantity must be an integer")
        stamp = _timestamp(source[columns["fill_time"]], trade_date)
        if stamp.date().isoformat() != trade_date:
            raise ValueError("fill timestamp differs from decision trade_date")
        fee_values, fee_complete = [], True
        for field in ("commission", "stamp_tax", "transfer_fee", "other_fee"):
            if columns[field] is None or pd.isna(source[columns[field]]) or str(source[columns[field]]).strip() == "":
                fee_complete = False
            else:
                fee_values.append(_number(source[columns[field]]))
        rows.append({
            "fill_id": str(source[columns["fill_id"]]).strip(),
            "order_id": str(source[columns["order_id"]]).strip(),
            "code": code6(source[columns["code"]]),
            "name": str(source[columns["name"]]).strip() if columns["name"] else "",
            "side": _side(source[columns["side"]]), "fill_time": stamp.isoformat(),
            "fill_price": _number(source[columns["fill_price"]], positive=True),
            "fill_quantity": int(quantity), "fee_cash": sum(fee_values) if fee_complete else None,
            "fee_data_complete": fee_complete,
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=[
            "fill_id", "order_id", "code", "name", "side", "fill_time", "fill_price",
            "fill_quantity", "fee_cash", "fee_data_complete",
        ])
    if result["fill_id"].eq("").any() or result["fill_id"].duplicated().any():
        raise ValueError("fill ids must be nonempty and unique")
    return result.sort_values(["fill_time", "fill_id"]).reset_index(drop=True)


def _primary(decision: dict[str, Any]) -> dict[str, Any] | None:
    value = decision.get("primary_choice")
    if isinstance(value, dict) and value.get("code"):
        return {**value, "code": code6(value["code"])}
    return None


def reconcile(decision: dict[str, Any], orders: pd.DataFrame, fills: pd.DataFrame, captured_at: Any) -> dict[str, Any]:
    trade_date = pd.Timestamp(decision.get("trade_date")).strftime("%Y-%m-%d")
    captured = _timestamp(captured_at, trade_date)
    decision_time = _timestamp(decision.get("asof"), trade_date)
    if decision.get("order_submitted") is not False:
        raise ValueError("live decision must not claim an already submitted order")
    if any(pd.Timestamp(value) > captured for value in list(orders.get("order_time", [])) + list(fills.get("fill_time", []))):
        raise ValueError("execution export contains records after captured_at")
    if not fills.empty and not set(fills["order_id"]).issubset(set(orders["order_id"])):
        raise ValueError("fill references an order absent from the order export")
    primary = _primary(decision)
    maximum_price = None if primary is None else primary.get("maximum_acceptable_price")
    try:
        maximum_price = float(maximum_price) if maximum_price is not None else None
    except (TypeError, ValueError):
        maximum_price = None
    plan_authorized = bool(
        decision.get("status") == "ENTER" and decision.get("direct_entry_authorized") is True
        and primary is not None and maximum_price is not None and maximum_price > 0
    )
    records = []
    for order in orders.to_dict("records"):
        matched = fills[fills["order_id"].eq(order["order_id"])] if not fills.empty else fills
        if not matched.empty and ((matched["code"] != order["code"]).any() or (matched["side"] != order["side"]).any()):
            raise ValueError("fill code/side differs from referenced order")
        filled_quantity = int(matched["fill_quantity"].sum()) if not matched.empty else 0
        if filled_quantity > int(order["order_quantity"]):
            raise ValueError("filled quantity exceeds order quantity")
        weighted_price = (
            float((matched["fill_price"] * matched["fill_quantity"]).sum() / filled_quantity)
            if filled_quantity else None
        )
        timing_ok = pd.Timestamp(order["order_time"]) >= decision_time and (
            matched.empty or pd.to_datetime(matched["fill_time"]).min() >= decision_time
        )
        code_ok = primary is not None and order["code"] == primary["code"]
        price_ok = weighted_price is not None and maximum_price is not None and weighted_price <= maximum_price + 1e-9
        lot_ok = order["side"] == "BUY" and order["order_quantity"] % 100 == 0
        compliant = bool(plan_authorized and timing_ok and code_ok and price_ok and lot_ok and filled_quantity > 0)
        if filled_quantity and compliant:
            classification = "STRATEGY_COMPLIANT_FULL_FILL" if filled_quantity == order["order_quantity"] else "STRATEGY_COMPLIANT_PARTIAL_FILL"
        elif filled_quantity:
            classification = "USER_EXECUTION_OFF_PLAN_NOT_STRATEGY_SAMPLE"
        elif str(order["order_status"]).strip().upper() in {"已撤", "废单", "拒单", "CANCELLED", "REJECTED"}:
            classification = "NO_FILL_FINAL"
        else:
            classification = "NO_FILL_OR_PENDING_UNRESOLVED"
        reasons = []
        if not plan_authorized:
            reasons.append("decision did not pre-authorize ENTER with a price ceiling")
        if not timing_ok:
            reasons.append("order/fill occurred before the recorded decision point")
        if not code_ok:
            reasons.append("order code is not the frozen primary choice")
        if filled_quantity and not price_ok:
            reasons.append("average fill price exceeds or lacks the frozen maximum price")
        records.append({
            **order, "filled_quantity": filled_quantity, "average_fill_price": weighted_price,
            "unfilled_quantity": int(order["order_quantity"]) - filled_quantity,
            "fee_data_complete": bool(not matched.empty and matched["fee_data_complete"].all()),
            "classification": classification, "strategy_sample_eligible": compliant,
            "reasons": reasons,
        })
    strategy_fills = sum(row["strategy_sample_eligible"] for row in records)
    actual_filled_orders = sum(row["filled_quantity"] > 0 for row in records)
    return {
        "version": VERSION, "trade_date": trade_date, "decision_asof": decision.get("asof"),
        "decision_status": decision.get("status"), "primary_choice": primary,
        "maximum_acceptable_price": maximum_price, "captured_at": captured.isoformat(),
        "orders": records, "order_count": len(records), "actual_filled_orders": actual_filled_orders,
        "partial_fill_orders": sum(0 < row["filled_quantity"] < row["order_quantity"] for row in records),
        "no_fill_orders": sum(row["filled_quantity"] == 0 for row in records),
        "strategy_eligible_filled_orders": strategy_fills,
        "off_plan_filled_orders": actual_filled_orders - strategy_fills,
        "complete_trade_created": False,
        "complete_trade_reason": "entry evidence alone is not a completed trade; verified exit is still required",
        "automatic_order_submission": False,
    }


def run(
    decision_path: Path, orders_path: Path, fills_path: Path, captured_at: Any,
    source: str, output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    decision_path, orders_path, fills_path = map(Path, (decision_path, orders_path, fills_path))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    trade_date = pd.Timestamp(decision.get("trade_date")).strftime("%Y-%m-%d")
    orders = normalize_orders(pd.read_csv(orders_path, dtype="string"), trade_date)
    fills = normalize_fills(pd.read_csv(fills_path, dtype="string"), trade_date)
    result = reconcile(decision, orders, fills, captured_at)
    result.update({
        "source": source, "data_status": "USER_CONFIRMED_TERMINAL_EXPORT",
        "source_limit": "CSV provenance is user-confirmed unless obtained from an authenticated broker API",
        "source_hashes": {
            str(decision_path.resolve()): sha256_file(decision_path),
            str(orders_path.resolve()): sha256_file(orders_path),
            str(fills_path.resolve()): sha256_file(fills_path),
            str(Path(__file__).resolve()): sha256_file(Path(__file__)),
        },
    })
    identity = hashlib.sha256(json.dumps(result["source_hashes"], sort_keys=True).encode()).hexdigest()[:20]
    target = Path(output) / trade_date / identity
    if target.exists():
        raise FileExistsError("immutable broker execution import already exists")
    target.mkdir(parents=True, exist_ok=False)
    normalized_orders_path = target / "normalized_orders.csv"
    normalized_fills_path = target / "normalized_fills.csv"
    orders.to_csv(normalized_orders_path, index=False, encoding="utf-8-sig")
    fills.to_csv(normalized_fills_path, index=False, encoding="utf-8-sig")
    result["normalized_files"] = {
        "orders": {"path": str(normalized_orders_path.resolve()), "sha256": sha256_file(normalized_orders_path)},
        "fills": {"path": str(normalized_fills_path.resolve()), "sha256": sha256_file(normalized_fills_path)},
    }
    result["import_id"] = identity
    report_path = target / "report.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**result, "report_path": str(report_path.resolve()), "report_sha256": sha256_file(report_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bind broker/Tonghuashun orders and fills to one live decision")
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--orders", type=Path, required=True)
    parser.add_argument("--fills", type=Path, required=True)
    parser.add_argument("--captured-at", required=True)
    parser.add_argument("--source", default="用户确认的同花顺/券商CSV导出")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.decision, args.orders, args.fills, args.captured_at, args.source, args.output), ensure_ascii=False, indent=2))
