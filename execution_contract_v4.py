#!/usr/bin/env python3
"""Immutable paper-execution contracts for confirmed A-share board relays.

The contract is intentionally conservative: A-share T+1 means an entry cannot
be stopped on the entry day, so size is derived from an overnight stress loss,
not from a fictitious same-day stop.  Nothing in this module sends an order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

import open_confirmation_v4 as opening
import session_forward_v4 as sessions
from leader_system_v3 import zfill_code


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "v4" / "execution_contracts"
DEFAULT_CANDIDATES = ROOT / "data" / "v3" / "leader_candidates.csv"
LEDGER_NAME = "execution_contract_ledger.jsonl"
VERSION = "v4-execution-contract-2"
SHANGHAI = ZoneInfo("Asia/Shanghai")

RISK_BUDGET_PCT = 0.005
STRESS_LOSS_PCT = 0.12
TAIL_SCENARIO_LOSS_PCT = 0.20
MAX_GROSS_RELAY_EXPOSURE_PCT = 0.30
MAX_THEME_EXPOSURE_PCT = 0.15
MAX_OPEN_RELAY_POSITIONS = 2
BOARD_CAPS = {1: 0.05, 2: 0.05, 3: 0.05, 4: 0.04, 5: 0.03}
EMOTION_CAPS = {"启动": 0.05, "升温": 0.05, "加速": 0.04, "一致": 0.03, "修复": 0.025}
REVIEW_CLASSES = {"MODEL_ERROR", "EXECUTION_ERROR", "MARKET_ERROR", "RISK_ERROR", "INFORMATION_ERROR", "NO_ERROR"}


def json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"cannot serialize {type(value)!r}")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=json_default)


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now(SHANGHAI)
    return now.astimezone(SHANGHAI) if now.tzinfo else now.replace(tzinfo=SHANGHAI)


def ledger_path(output: Path) -> Path:
    return output / LEDGER_NAME


def read_events(output: Path) -> list[dict[str, Any]]:
    path = ledger_path(output)
    if not path.exists():
        return []
    events = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid execution ledger JSON at line {line_no}: {exc}") from exc
    return events


def _artifact_path(output: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else output / path


def verify_ledger(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    errors = []
    try:
        events = read_events(output)
    except RuntimeError as exc:
        return {"status": "FAIL", "events": 0, "errors": [str(exc)]}
    previous = "GENESIS"
    contracts: dict[str, dict[str, Any]] = {}
    fills: dict[str, dict[str, Any]] = {}
    exits: dict[str, list[dict[str, Any]]] = {}
    contract_keys = set()
    for index, event in enumerate(events, 1):
        claimed = event.get("event_hash")
        body = {key: value for key, value in event.items() if key != "event_hash"}
        if event.get("sequence") != index:
            errors.append(f"event {index}: sequence mismatch")
        if event.get("previous_event_hash") != previous:
            errors.append(f"event {index}: previous hash mismatch")
        if claimed != sha256_value(body):
            errors.append(f"event {index}: event hash mismatch")
        previous = str(claimed)
        event_type = event.get("event_type")
        contract_id = str(event.get("contract_id", ""))
        if event_type == "CONTRACT":
            key = (event.get("session_id"), event.get("code"), event.get("asof"))
            if key in contract_keys:
                errors.append(f"event {index}: duplicate contract key {key}")
            contract_keys.add(key)
            if contract_id in contracts:
                errors.append(f"event {index}: duplicate contract id")
            contracts[contract_id] = event
            path = _artifact_path(output, str(event.get("contract_file", "")))
            if not path.exists():
                errors.append(f"event {index}: contract file missing")
            elif sha256_file(path) != event.get("contract_sha256"):
                errors.append(f"event {index}: contract file hash mismatch")
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("version") != VERSION:
                    errors.append(
                        f"event {index}: unsupported contract version "
                        f"{payload.get('version')!r}; expected {VERSION!r}"
                    )
                basis = {key: value for key, value in payload.items() if key not in {"contract_id", "generated_at"}}
                if payload.get("contract_id") != sha256_value(basis)[:20]:
                    errors.append(f"event {index}: contract id does not match content")
                if payload.get("contract_id") != contract_id:
                    errors.append(f"event {index}: event/file contract id mismatch")
        elif event_type == "FILL":
            if contract_id not in contracts:
                errors.append(f"event {index}: fill without contract")
            if contract_id in fills:
                errors.append(f"event {index}: duplicate fill")
            fills[contract_id] = event
            if event.get("execution_status") == "VERIFIED_PAPER_FILL" and contracts.get(contract_id, {}).get("contract_status") != "PAPER_ENTRY_ALLOWED":
                errors.append(f"event {index}: fill attached to non-entry contract")
            if event.get("execution_status") == "VERIFIED_PAPER_FILL" and contract_id in contracts:
                try:
                    contract_path = _artifact_path(output, contracts[contract_id]["contract_file"])
                    payload = json.loads(contract_path.read_text(encoding="utf-8"))
                    price = float(event.get("fill_price"))
                    quantity = int(event.get("fill_quantity"))
                    if quantity <= 0 or quantity % 100:
                        errors.append(f"event {index}: invalid A-share board lot")
                    notional = price * quantity
                    equity = float(payload["portfolio_state"]["account_equity"])
                    if abs(notional - float(event.get("fill_notional"))) > 1e-8:
                        errors.append(f"event {index}: fill notional mismatch")
                    if abs(notional / equity - float(event.get("fill_weight_pct"))) > 1e-12:
                        errors.append(f"event {index}: fill weight mismatch")
                    if notional > float(payload["sizing"]["maximum_notional"]) + 1e-8:
                        errors.append(f"event {index}: fill exceeds frozen maximum")
                    recorded = pd.Timestamp(event.get("recorded_at"))
                    if recorded.tzinfo is None:
                        recorded = recorded.tz_localize(SHANGHAI)
                    else:
                        recorded = recorded.tz_convert(SHANGHAI)
                    if recorded.date().isoformat() != payload["trade_date"]:
                        errors.append(f"event {index}: verified fill not recorded on trade_date")
                except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                    errors.append(f"event {index}: invalid verified fill payload ({exc})")
        elif event_type == "EXIT":
            if contract_id not in fills or fills[contract_id].get("execution_status") != "VERIFIED_PAPER_FILL":
                errors.append(f"event {index}: exit without verified fill")
            prior_exits = exits.setdefault(contract_id, [])
            if prior_exits and not prior_exits[-1].get("sell_blocked"):
                errors.append(f"event {index}: exit after resolved exit")
            try:
                session_offset = int(event.get("session_offset", 1))
                if session_offset < 1 or isinstance(event.get("session_offset"), bool):
                    raise ValueError
                if prior_exits and session_offset <= int(prior_exits[-1].get("session_offset", 1)):
                    errors.append(f"event {index}: exit session offset did not advance")
                if not prior_exits and session_offset > 3:
                    errors.append(f"event {index}: post-B+3 exit lacks prior blockage")
            except (TypeError, ValueError):
                errors.append(f"event {index}: invalid exit session offset")
            prior_exits.append(event)
            if event.get("review_classification") not in REVIEW_CLASSES:
                errors.append(f"event {index}: invalid review classification")
            if event.get("sell_blocked") and event.get("net_return") is not None:
                errors.append(f"event {index}: blocked exit cannot have realized net return")
            if contract_id in fills:
                try:
                    entry = float(fills[contract_id]["fill_price"])
                    close = float(event["t1_close"])
                    cost = float(event["round_trip_cost"])
                    mark = close / entry - 1 - cost
                    if abs(mark - float(event["mark_to_market_return"])) > 1e-12:
                        errors.append(f"event {index}: mark-to-market return mismatch")
                    if bool(event.get("large_loss")) != bool(mark <= -0.07):
                        errors.append(f"event {index}: large-loss label mismatch")
                    if not event.get("sell_blocked"):
                        net = float(event["actual_exit_price"]) / entry - 1 - cost
                        if abs(net - float(event["net_return"])) > 1e-12:
                            errors.append(f"event {index}: net return mismatch")
                except (ValueError, TypeError, KeyError) as exc:
                    errors.append(f"event {index}: invalid exit payload ({exc})")
        else:
            errors.append(f"event {index}: unknown event type {event_type}")
    return {
        "status": "PASS" if not errors else "FAIL",
        "version": VERSION,
        "events": len(events),
        "contracts": len(contracts),
        "fills": len(fills),
        "exits": sum(len(rows) for rows in exits.values()),
        "resolved_exit_contracts": sum(bool(rows) and not rows[-1].get("sell_blocked") for rows in exits.values()),
        "sell_blocked_contracts": sum(bool(rows) and bool(rows[-1].get("sell_blocked")) for rows in exits.values()),
        "head_event_hash": previous,
        "errors": errors,
        "assurance": "LOCAL_TAMPER_EVIDENT_ONLY",
    }


def append_event(output: Path, payload: dict[str, Any]) -> dict[str, Any]:
    verification = verify_ledger(output)
    if verification["status"] != "PASS":
        raise RuntimeError("execution ledger is invalid: " + "; ".join(verification["errors"]))
    output.mkdir(parents=True, exist_ok=True)
    event = {
        **payload,
        "sequence": verification["events"] + 1,
        "previous_event_hash": verification["head_event_hash"],
    }
    event["event_hash"] = sha256_value(event)
    with ledger_path(output).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, default=json_default) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return event


def validate_portfolio_state(frame: pd.DataFrame, decision_time: datetime) -> pd.Series:
    required = {
        "account_equity", "current_gross_exposure_pct", "current_theme_exposure_pct",
        "open_relay_positions", "source", "source_timestamp", "data_status",
    }
    missing = sorted(required - set(frame.columns))
    if missing or len(frame) != 1:
        raise ValueError(f"portfolio state must contain one row and fields {missing}")
    row = frame.iloc[0].copy()
    numeric = ["account_equity", "current_gross_exposure_pct", "current_theme_exposure_pct", "open_relay_positions"]
    for field in numeric:
        row[field] = pd.to_numeric(row[field], errors="coerce")
    if row[numeric].isna().any():
        raise ValueError("portfolio state contains invalid numeric values")
    if row["account_equity"] <= 0:
        raise ValueError("account_equity must be positive")
    if not 0 <= row["current_gross_exposure_pct"] <= 1 or not 0 <= row["current_theme_exposure_pct"] <= 1:
        raise ValueError("portfolio exposure must be within [0,1]")
    if row["open_relay_positions"] < 0 or int(row["open_relay_positions"]) != row["open_relay_positions"]:
        raise ValueError("open_relay_positions must be a nonnegative integer")
    timestamp = pd.Timestamp(row["source_timestamp"])
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(SHANGHAI)
    else:
        timestamp = timestamp.tz_convert(SHANGHAI)
    if timestamp > pd.Timestamp(decision_time):
        raise ValueError("portfolio source_timestamp is later than contract decision")
    if str(row["data_status"]).lower() not in {"verified", "verified_import", "verified_broker_export"}:
        raise ValueError("portfolio state is not verified")
    return row


def size_position(candidate: pd.Series, snapshot: pd.Series, portfolio: pd.Series) -> dict[str, Any]:
    boards = int(float(candidate["boards"]))
    emotion = str(candidate.get("emotion_state", ""))
    gap = float(pd.to_numeric(snapshot.get("auction_gap"), errors="coerce"))
    risk_cap = RISK_BUDGET_PCT / STRESS_LOSS_PCT
    # Do not use the board number as a mechanical veto. Very high boards get
    # progressively smaller exposure because T+1 prevents an entry-day exit.
    board_cap = BOARD_CAPS.get(boards, 0.02 if boards <= 7 else 0.01)
    emotion_cap = EMOTION_CAPS.get(emotion, 0.0)
    gap_cap = 0.02 if gap > 0.07 else 0.03 if gap > 0.05 else 0.05
    gross_remaining = max(0.0, MAX_GROSS_RELAY_EXPOSURE_PCT - float(portfolio["current_gross_exposure_pct"]))
    theme_remaining = max(0.0, MAX_THEME_EXPOSURE_PCT - float(portfolio["current_theme_exposure_pct"]))
    position_room = 0.0 if int(portfolio["open_relay_positions"]) >= MAX_OPEN_RELAY_POSITIONS else 1.0
    maximum = min(risk_cap, board_cap, emotion_cap, gap_cap, gross_remaining, theme_remaining, position_room)
    maximum = max(0.0, maximum)
    return {
        "risk_budget_pct": RISK_BUDGET_PCT,
        "stress_loss_pct": STRESS_LOSS_PCT,
        "tail_scenario_loss_pct": TAIL_SCENARIO_LOSS_PCT,
        "risk_implied_weight_pct": risk_cap,
        "board_cap_pct": board_cap,
        "emotion_cap_pct": emotion_cap,
        "gap_cap_pct": gap_cap,
        "gross_remaining_pct": gross_remaining,
        "theme_remaining_pct": theme_remaining,
        "position_slot_available": bool(position_room),
        "maximum_position_pct": maximum,
        "maximum_notional": float(portfolio["account_equity"]) * maximum,
        "stress_equity_loss_pct": maximum * STRESS_LOSS_PCT,
        "tail_equity_loss_pct": maximum * TAIL_SCENARIO_LOSS_PCT,
    }


def build_contract(
    session_id: str,
    candidate: pd.Series,
    snapshot: pd.Series,
    capture_event: dict[str, Any],
    portfolio: pd.Series,
    generated_at: datetime,
) -> dict[str, Any]:
    asof = opening.normalize_asof(snapshot["asof"])
    confirmation = opening.evaluate_row(candidate, snapshot, asof)
    deadline = pd.Timestamp.combine(generated_at.date(), datetime.strptime(asof, "%H:%M").time()) + pd.Timedelta(minutes=5)
    generated_late = generated_at.replace(tzinfo=None) > deadline.to_pydatetime()
    sizing = size_position(candidate, snapshot, portfolio)
    allowed = (
        confirmation["confirmation_status"] == "CONFIRM_RESEAL"
        and bool(capture_event.get("capture_complete"))
        and not bool(capture_event.get("late_capture"))
        and not generated_late
        and sizing["maximum_position_pct"] > 0
    )
    reasons = []
    if confirmation["confirmation_status"] != "CONFIRM_RESEAL":
        reasons.append(f"开盘确认状态为{confirmation['confirmation_status']}")
    if not capture_event.get("capture_complete"):
        reasons.append("预注册候选覆盖不完整")
    if capture_event.get("late_capture") or generated_late:
        reasons.append("快照或契约生成迟到")
    if sizing["maximum_position_pct"] <= 0:
        reasons.append("组合仓位、题材暴露或持仓数量已无空间")
    basis = {
        "version": VERSION,
        "session_id": session_id,
        "signal_date": str(candidate["date"]),
        "trade_date": str(snapshot["trade_date"]),
        "asof": asof,
        "code": zfill_code(candidate["code"]),
        "name": str(candidate["name"]),
        "boards": int(float(candidate["boards"])),
        "emotion_state": str(candidate.get("emotion_state", "")),
        "theme_group_id": str(snapshot.get("theme_group_id", "")),
        "confirmation_status": confirmation["confirmation_status"],
        "confirmation_reasons": confirmation["reasons"],
        "capture_event_hash": str(capture_event.get("event_hash", "")),
        "capture_raw_sha256": str(capture_event.get("raw_sha256", "")),
        "portfolio_source": str(portfolio["source"]),
        "portfolio_source_timestamp": str(portfolio["source_timestamp"]),
        "portfolio_state": {
            "account_equity": float(portfolio["account_equity"]),
            "current_gross_exposure_pct": float(portfolio["current_gross_exposure_pct"]),
            "current_theme_exposure_pct": float(portfolio["current_theme_exposure_pct"]),
            "open_relay_positions": int(portfolio["open_relay_positions"]),
        },
        "sizing": sizing,
        "generated_late": generated_late,
        "contract_status": "PAPER_ENTRY_ALLOWED" if allowed else "NO_TRADE",
        "no_trade_reasons": reasons,
        "setup_type": ("主板首板题材先锋/换手核心确认"
                       if int(float(candidate["boards"])) == 1
                       else "主板连续板真实题材换手回封"),
        "entry_trigger": "仅当当前CONFIRM_RESEAL仍有效、涨停价真实可成交且无新硬风险时记录纸面成交",
        "entry_cancel": "两次冲板不封；放量跌破VWAP；题材同步<35%；板块核心破板；公告硬风险；封板不可成交",
        "same_day_rule": "A股T+1：买入当日不能卖出，不设置虚假的当日止损；禁止补仓摊薄",
        "evaluation_exit": "分别记录B+1、B+2、B+3扣费后收盘估值；估值不冒充卖出成交",
        "exit_monitoring_contract": {
            "default_monitoring_mode": "UNMONITORED_UNLESS_TIMESTAMPED_LIVE_EVIDENCE_EXISTS",
            "pre_open_exit": "如选择B+1集合竞价退出，计划必须在正式开盘结果产生前冻结；只回填真实成交",
            "same_open_prohibition": "看到正式开盘强弱后不得仍按同一开盘价记卖出；只能使用确认后的新成交价",
            "intraday_trigger": "VWAP、量能、板块破位等条件只有在实时证据与后续真实成交均存在时才可记为执行",
            "offline_rule": "未打开电脑且无预先委托时，盘中条件只能事后复盘，不得冒充已执行止损",
        },
        "operational_exit": {
            "t1_one_word_down": "标记SELL_BLOCKED，净收益保持未解决，等待首次可成交价格",
            "t1_gap_down": "低开超过5%只触发退出意图；看到开盘后必须按9:30以后真实可成交价记录，不得回填开盘价",
            "t1_high_open_fade": "高开后跌破VWAP且卖量扩大只在实时监控并取得后续成交时记录退出",
            "t1_failed_new_high": "09:35后不能创新高且跌破VWAP只在时间戳证据完整时记录退出",
            "sector_break": "同题材核心破板或梯队批量转弱时触发退出意图；成交仍需真实回填",
            "time_stop": "最迟B+3收盘退出；仅卖出受阻可延后到首次可成交时点",
        },
        "holding_period": "entry day through B+1/B+2/B+3 scenario review; delayed beyond B+3 only by sell blockage",
        "execution_mode": "PAPER_ONLY_NO_AUTOMATIC_ORDER",
        "model_status": "DIAGNOSTIC_ONLY",
    }
    return {**basis, "contract_id": sha256_value(basis)[:20], "generated_at": generated_at.isoformat(timespec="seconds")}


def _find_capture(session_output: Path, session_id: str, asof: str) -> tuple[dict[str, Any], pd.DataFrame]:
    verification = sessions.verify_ledger(session_output)
    if verification["status"] != "PASS":
        raise RuntimeError("session ledger is invalid")
    matches = [
        event for event in sessions.read_events(session_output)
        if event.get("event_type") == "CAPTURE" and event.get("session_id") == session_id
        and event.get("asof") == asof
    ]
    if len(matches) != 1:
        raise RuntimeError("exactly one registered capture is required")
    event = matches[0]
    raw_path = Path(event["raw_file"])
    if not raw_path.is_absolute():
        raw_path = session_output / raw_path
    return event, pd.read_csv(raw_path, dtype={"code": "string"})


def generate_contract(
    session_id: str,
    asof: str,
    code: str,
    portfolio_path: Path,
    output: Path = DEFAULT_OUTPUT,
    session_output: Path = sessions.DEFAULT_OUTPUT,
    candidates_path: Path = DEFAULT_CANDIDATES,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = _now(now)
    point = opening.normalize_asof(asof)
    if opening.clock_seconds(point) < opening.clock_seconds("09:35"):
        raise RuntimeError("execution contract requires 09:35 or later confirmation")
    capture_event, raw = _find_capture(session_output, session_id, point)
    code = zfill_code(code)
    snapshots = raw[raw["code"].map(zfill_code).eq(code)]
    if len(snapshots) != 1:
        raise RuntimeError("registered capture must contain exactly one row for contract code")
    snapshot = snapshots.iloc[0]
    candidates = pd.read_csv(candidates_path, dtype={"code": "string"}, low_memory=False)
    candidates = candidates[
        candidates["date"].astype(str).eq(str(snapshot["signal_date"]))
        & candidates["code"].map(zfill_code).eq(code)
    ]
    if len(candidates) != 1:
        raise RuntimeError("candidate row is missing or duplicated")
    portfolio = validate_portfolio_state(pd.read_csv(portfolio_path), current)
    contract = build_contract(session_id, candidates.iloc[0], snapshot, capture_event, portfolio, current)
    if any(event.get("event_type") == "CONTRACT" and event.get("session_id") == session_id
           and event.get("code") == code and event.get("asof") == point for event in read_events(output)):
        raise RuntimeError("execution contract already exists for session/code/asof")
    contract_dir = output / session_id
    contract_dir.mkdir(parents=True, exist_ok=True)
    path = contract_dir / f"contract_{code}_{point.replace(':', '')}.json"
    if path.exists():
        raise RuntimeError("immutable contract file already exists")
    path.write_text(json.dumps(contract, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    event = append_event(output, {
        "event_type": "CONTRACT",
        "contract_id": contract["contract_id"],
        "session_id": session_id,
        "signal_date": contract["signal_date"],
        "trade_date": contract["trade_date"],
        "asof": point,
        "code": code,
        "contract_status": contract["contract_status"],
        "maximum_position_pct": contract["sizing"]["maximum_position_pct"],
        "generated_at": contract["generated_at"],
        "contract_file": str(path.resolve().relative_to(output.resolve())),
        "contract_sha256": sha256_file(path),
    })
    return {"contract": contract, "event": event, "verification": verify_ledger(output)}


def record_fill(
    contract_id: str,
    execution_status: str,
    fill_time: str | None,
    fill_price: float | None,
    fill_quantity: int | None,
    source: str,
    output: Path = DEFAULT_OUTPUT,
    now: datetime | None = None,
) -> dict[str, Any]:
    statuses = {"VERIFIED_PAPER_FILL", "NO_FILL", "UNBUYABLE_ONE_WORD_UP", "CONDITION_CANCELLED"}
    if execution_status not in statuses:
        raise ValueError(f"execution_status must be one of {sorted(statuses)}")
    contracts = {event["contract_id"]: event for event in read_events(output) if event.get("event_type") == "CONTRACT"}
    if contract_id not in contracts:
        raise RuntimeError("unknown contract_id")
    if any(event.get("event_type") == "FILL" and event.get("contract_id") == contract_id for event in read_events(output)):
        raise RuntimeError("fill already recorded")
    contract = contracts[contract_id]
    contract_path = _artifact_path(output, contract["contract_file"])
    contract_payload = json.loads(contract_path.read_text(encoding="utf-8"))
    fill_weight_pct = None
    fill_notional = None
    if execution_status == "VERIFIED_PAPER_FILL":
        if contract["contract_status"] != "PAPER_ENTRY_ALLOWED":
            raise RuntimeError("cannot fill a NO_TRADE contract")
        if fill_price is None or fill_price <= 0 or fill_quantity is None or fill_quantity <= 0:
            raise ValueError("verified fill requires positive price and quantity")
        if int(fill_quantity) != fill_quantity or int(fill_quantity) % 100 != 0:
            raise ValueError("A-share fill quantity must be a positive multiple of 100 shares")
        fill_quantity = int(fill_quantity)
        fill_notional = float(fill_price) * fill_quantity
        equity = float(contract_payload["portfolio_state"]["account_equity"])
        fill_weight_pct = fill_notional / equity
        if fill_notional > float(contract_payload["sizing"]["maximum_notional"]) + 1e-8:
            raise ValueError("fill notional exceeds frozen contract maximum")
        if not fill_time:
            raise ValueError("verified fill requires fill_time")
        current = _now(now)
        if current.date().isoformat() != contract_payload["trade_date"]:
            raise ValueError("verified fill must be recorded on the contract trade_date")
    else:
        current = _now(now)
    event = append_event(output, {
        "event_type": "FILL",
        "contract_id": contract_id,
        "execution_status": execution_status,
        "fill_time": fill_time,
        "fill_price": fill_price,
        "fill_quantity": fill_quantity,
        "fill_notional": fill_notional,
        "fill_weight_pct": fill_weight_pct,
        "source": source,
        "recorded_at": current.isoformat(timespec="seconds"),
    })
    return {"event": event, "verification": verify_ledger(output)}


def record_exit(
    contract_id: str,
    t1_close: float,
    actual_exit_price: float | None,
    round_trip_cost: float,
    sell_blocked: bool,
    review_classification: str,
    review_note: str,
    output: Path = DEFAULT_OUTPUT,
    now: datetime | None = None,
    session_offset: int = 1,
) -> dict[str, Any]:
    if review_classification not in REVIEW_CLASSES:
        raise ValueError(f"review_classification must be one of {sorted(REVIEW_CLASSES)}")
    events = read_events(output)
    fill = next((event for event in events if event.get("event_type") == "FILL" and event.get("contract_id") == contract_id), None)
    if fill is None or fill.get("execution_status") != "VERIFIED_PAPER_FILL":
        raise RuntimeError("verified fill is required before exit")
    prior_exits = [event for event in events if event.get("event_type") == "EXIT"
                   and event.get("contract_id") == contract_id]
    if any(not event.get("sell_blocked") for event in prior_exits):
        raise RuntimeError("resolved exit already recorded")
    if not isinstance(session_offset, int) or isinstance(session_offset, bool) or session_offset < 1:
        raise ValueError("session_offset must be a positive integer")
    if prior_exits:
        previous_offset = int(prior_exits[-1].get("session_offset", 1))
        if session_offset <= previous_offset:
            raise ValueError("a later blocked-exit follow-up requires a later session_offset")
    elif session_offset > 3:
        raise ValueError("an exit after B+3 requires a prior sell-blocked attempt")
    contract_event = next(event for event in events if event.get("event_type") == "CONTRACT" and event.get("contract_id") == contract_id)
    contract_payload = json.loads(_artifact_path(output, contract_event["contract_file"]).read_text(encoding="utf-8"))
    current = _now(now)
    if current.date().isoformat() <= contract_payload["trade_date"]:
        raise ValueError("exit with T+1 close must be recorded after the entry trade_date")
    if current.timetz().replace(tzinfo=None) < time(15, 0):
        raise ValueError("T+1 close outcome can be recorded only after 15:00")
    entry = float(fill["fill_price"])
    if t1_close <= 0 or round_trip_cost < 0:
        raise ValueError("t1_close must be positive and costs nonnegative")
    mark = t1_close / entry - 1 - round_trip_cost
    if sell_blocked:
        if actual_exit_price is not None:
            raise ValueError("blocked unresolved exit must not have actual_exit_price")
        net = None
    else:
        if actual_exit_price is None or actual_exit_price <= 0:
            raise ValueError("unblocked exit requires actual_exit_price")
        net = actual_exit_price / entry - 1 - round_trip_cost
    event = append_event(output, {
        "event_type": "EXIT",
        "contract_id": contract_id,
        "session_offset": session_offset,
        "entry_price": entry,
        "t1_close": t1_close,
        "actual_exit_price": actual_exit_price,
        "round_trip_cost": round_trip_cost,
        "mark_to_market_return": mark,
        "net_return": net,
        "large_loss": bool(mark <= -0.07),
        "sell_blocked": bool(sell_blocked),
        "review_classification": review_classification,
        "review_note": review_note,
        "recorded_at": current.isoformat(timespec="seconds"),
    })
    return {"event": event, "verification": verify_ledger(output)}


def monitor(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    verification = verify_ledger(output)
    events = read_events(output) if verification["status"] == "PASS" else []
    contracts = [event for event in events if event.get("event_type") == "CONTRACT"]
    fills = {event["contract_id"]: event for event in events if event.get("event_type") == "FILL"}
    exit_attempts = [event for event in events if event.get("event_type") == "EXIT"]
    exits = {}
    for event in exit_attempts:
        exits[event["contract_id"]] = event
    return {
        "status": verification["status"],
        "contracts": len(contracts),
        "entry_allowed": sum(event.get("contract_status") == "PAPER_ENTRY_ALLOWED" for event in contracts),
        "verified_fills": sum(event.get("execution_status") == "VERIFIED_PAPER_FILL" for event in fills.values()),
        "resolved_exits": sum(not event.get("sell_blocked") for event in exits.values()),
        "sell_blocked_unresolved": sum(bool(event.get("sell_blocked")) for event in exits.values()),
        "exit_attempts": len(exit_attempts),
        "large_losses": sum(bool(event.get("large_loss")) for event in exits.values()),
        "verification": verification,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate immutable paper-only relay execution contracts")
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--session-id", required=True)
    generate.add_argument("--asof", required=True, choices=("09:35", "10:00"))
    generate.add_argument("--code", required=True)
    generate.add_argument("--portfolio", required=True, type=Path)
    generate.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    generate.add_argument("--session-output", type=Path, default=sessions.DEFAULT_OUTPUT)
    fill = sub.add_parser("fill")
    fill.add_argument("--contract-id", required=True)
    fill.add_argument("--status", required=True)
    fill.add_argument("--fill-time")
    fill.add_argument("--fill-price", type=float)
    fill.add_argument("--fill-quantity", type=int)
    fill.add_argument("--source", required=True)
    fill.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    exit_cmd = sub.add_parser("exit")
    exit_cmd.add_argument("--contract-id", required=True)
    exit_cmd.add_argument("--t1-close", type=float, required=True)
    exit_cmd.add_argument("--actual-exit-price", type=float)
    exit_cmd.add_argument("--round-trip-cost", type=float, default=0.003)
    exit_cmd.add_argument("--sell-blocked", action="store_true")
    exit_cmd.add_argument("--session-offset", type=int, default=1)
    exit_cmd.add_argument("--review-classification", required=True)
    exit_cmd.add_argument("--review-note", default="")
    exit_cmd.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    verify = sub.add_parser("verify")
    verify.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    status = sub.add_parser("monitor")
    status.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "generate":
        result = generate_contract(args.session_id, args.asof, args.code, args.portfolio, args.output, args.session_output)
    elif args.command == "fill":
        result = record_fill(args.contract_id, args.status, args.fill_time, args.fill_price, args.fill_quantity, args.source, args.output)
    elif args.command == "exit":
        result = record_exit(
            args.contract_id, args.t1_close, args.actual_exit_price, args.round_trip_cost,
            args.sell_blocked, args.review_classification, args.review_note, args.output,
            session_offset=args.session_offset,
        )
    elif args.command == "verify":
        result = verify_ledger(args.output)
    else:
        result = monitor(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
