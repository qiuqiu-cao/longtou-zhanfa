#!/usr/bin/env python3
"""Tamper-evident V10 relay shadow plan, confirmation and T+1 ledger."""

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

import numpy as np
import pandas as pd

import open_confirmation_v4 as opening
import relay_shadow_model_v10 as shadow_model
from leader_system_v3 import zfill_code


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "v10" / "intraday_forward"
SNAPSHOTS = ROOT / "data" / "v9" / "derived_open_snapshots.csv"
MARKET_DATES = ROOT / "data" / "v7" / "point_in_time_rebuild" / "market_emotion.csv"
LEDGER_NAME = "intraday_forward_ledger.jsonl"
MONITOR_NAME = "intraday_forward_monitor.json"
VERSION = "v10-intraday-forward-1"
SHANGHAI = ZoneInfo("Asia/Shanghai")
CONFIRM_TIMES = ("09:25", "09:35", "10:00")
VERIFIED_OUTCOME_STATUSES = {
    "verified_broker_export", "verified_terminal_export", "verified_official_export",
}
OUTCOME_FIELDS = {
    "signal_date", "trade_date", "exit_date", "code", "source", "source_timestamp",
    "data_status", "trade_close", "trade_limit_up_price", "exit_open", "exit_high",
    "exit_low", "exit_close", "exit_limit_down_price", "buy_cost_bps",
    "sell_cost_bps", "slippage_bps",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def hash_value(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return shadow_model.sha256_file(path)


def _now(value: datetime | None = None) -> datetime:
    if value is None:
        return datetime.now(SHANGHAI)
    return value.astimezone(SHANGHAI) if value.tzinfo else value.replace(tzinfo=SHANGHAI)


def ledger_path(output: Path = OUTPUT) -> Path:
    return output / LEDGER_NAME


def read_events(output: Path = OUTPUT) -> list[dict[str, Any]]:
    path = ledger_path(output)
    if not path.exists():
        return []
    events = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid V10 ledger line {number}: {exc}") from exc
    return events


def _resolve(output: Path, reference: str) -> Path:
    path = Path(reference)
    return path if path.is_absolute() else output / path


def _relative(output: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(output.resolve()))
    except ValueError:
        return str(path.resolve())


def verify(output: Path = OUTPUT) -> dict[str, Any]:
    lock_check = shadow_model.verify()
    errors = [] if lock_check["status"] == "PASS" else ["V10 model lock invalid"]
    try:
        events = read_events(output)
    except RuntimeError as exc:
        return {"status": "FAIL", "errors": [str(exc)]}
    previous = "GENESIS"
    plans: dict[str, dict[str, Any]] = {}
    plan_payloads: dict[str, dict[str, Any]] = {}
    plan_scores: dict[str, pd.DataFrame] = {}
    confirmation_events: dict[str, list[dict[str, Any]]] = {}
    date_pairs: set[tuple[str, str]] = set()
    confirmations: set[tuple[str, str]] = set()
    settlements: set[str] = set()
    script_hash = sha256_file(Path(__file__))
    development_end = pd.Timestamp(lock_check.get("development_data_end"))
    for number, event in enumerate(events, 1):
        body = {key: value for key, value in event.items() if key != "event_hash"}
        if event.get("previous_event_hash") != previous:
            errors.append(f"event {number}: previous hash mismatch")
        if event.get("event_hash") != hash_value(body):
            errors.append(f"event {number}: event hash mismatch")
        previous = str(event.get("event_hash", ""))
        if event.get("model_lock_id") != lock_check.get("lock_id"):
            errors.append(f"event {number}: wrong model lock")
        if event.get("forward_script_sha256") != script_hash:
            errors.append(f"event {number}: forward script hash mismatch")
        session_id = str(event.get("session_id", ""))
        kind = event.get("event_type")
        if kind == "PLAN":
            pair = (str(event.get("signal_date")), str(event.get("trade_date")))
            if session_id in plans or pair in date_pairs:
                errors.append(f"event {number}: duplicate plan")
            signal = pd.to_datetime(event.get("signal_date"), errors="coerce")
            if pd.isna(signal) or signal <= development_end:
                errors.append(f"event {number}: plan is not post-lock")
            plan_file = _resolve(output, str(event.get("plan_file", "")))
            score_file = _resolve(output, str(event.get("score_file", "")))
            if not plan_file.exists() or sha256_file(plan_file) != event.get("plan_sha256"):
                errors.append(f"event {number}: plan file hash mismatch")
            if not score_file.exists() or sha256_file(score_file) != event.get("score_sha256"):
                errors.append(f"event {number}: score file hash mismatch")
            if plan_file.exists() and score_file.exists():
                try:
                    plan = json.loads(plan_file.read_text(encoding="utf-8"))
                    scores = pd.read_csv(score_file, dtype={"code": "string"})
                    planned_at = pd.Timestamp(plan["planned_at"])
                    if planned_at.tzinfo is None:
                        planned_at = planned_at.tz_localize(SHANGHAI)
                    else:
                        planned_at = planned_at.tz_convert(SHANGHAI)
                    validate_plan_time(plan["signal_date"], plan["trade_date"], planned_at.to_pydatetime())
                    basis = {
                        "version": plan["version"], "model_lock_id": plan["model_lock_id"],
                        "signal_date": plan["signal_date"], "trade_date": plan["trade_date"],
                        "emotion_lock_id": plan["emotion_lock_id"],
                        "emotion_event_hash": plan["emotion_event_hash"],
                        "candidate_code": plan["candidate_code"], "candidate_name": plan["candidate_name"],
                        "candidate_boards": plan["candidate_boards"], "joint_score": plan["joint_score"],
                        "input_sha256": plan["input_sha256"],
                    }
                    if hash_value(basis)[:20] != session_id:
                        errors.append(f"event {number}: session id does not match frozen plan")
                    selected = scores[
                        scores["shadow_observation"].astype(str).str.lower().isin({"true", "1", "1.0"})
                    ]
                    expected_count = 0 if plan["candidate_code"] is None else 1
                    if len(selected) != expected_count:
                        errors.append(f"event {number}: frozen score selection count mismatch")
                    elif expected_count and zfill_code(selected.iloc[0]["code"]) != plan["candidate_code"]:
                        errors.append(f"event {number}: frozen score candidate mismatch")
                    for field in ("model_lock_id", "signal_date", "trade_date", "candidate_code", "candidate_boards"):
                        if event.get(field) != plan.get(field):
                            errors.append(f"event {number}: plan event {field} mismatch")
                    plan_payloads[session_id] = plan
                    plan_scores[session_id] = scores
                except (ValueError, KeyError, TypeError, json.JSONDecodeError, RuntimeError) as exc:
                    errors.append(f"event {number}: invalid frozen plan ({exc})")
            plans[session_id] = event
            date_pairs.add(pair)
        elif kind == "CONFIRMATION":
            key = (session_id, str(event.get("asof")))
            if session_id not in plans:
                errors.append(f"event {number}: confirmation without plan")
            if key in confirmations:
                errors.append(f"event {number}: duplicate confirmation")
            snapshot_file = _resolve(output, str(event.get("snapshot_file", "")))
            if not snapshot_file.exists() or sha256_file(snapshot_file) != event.get("snapshot_sha256"):
                errors.append(f"event {number}: snapshot file hash mismatch")
            if session_id in plan_payloads and snapshot_file.exists():
                try:
                    plan = plan_payloads[session_id]
                    point = opening.normalize_asof(event["asof"])
                    if point not in CONFIRM_TIMES:
                        raise RuntimeError("invalid confirmation asof")
                    recorded = pd.Timestamp(event["recorded_at"])
                    if recorded.tzinfo is None:
                        recorded = recorded.tz_localize(SHANGHAI)
                    else:
                        recorded = recorded.tz_convert(SHANGHAI)
                    if recorded.date().isoformat() != plan["trade_date"]:
                        raise RuntimeError("confirmation recorded outside trade date")
                    scheduled = datetime.strptime(point, "%H:%M:%S" if point.count(":") == 2 else "%H:%M").time()
                    if recorded.timetz().replace(tzinfo=None) < scheduled:
                        raise RuntimeError("confirmation recorded before asof")
                    grace = 2 if point == "09:25" else 5
                    latest = (pd.Timestamp.combine(recorded.date(), scheduled) + pd.Timedelta(minutes=grace)).time()
                    expected_late = recorded.timetz().replace(tzinfo=None) > latest
                    if bool(event.get("late")) != expected_late:
                        errors.append(f"event {number}: late flag mismatch")
                    if bool(event.get("training_eligible")) != (not expected_late):
                        errors.append(f"event {number}: training eligibility mismatch")
                    snapshot = opening.validate_snapshots(pd.read_csv(snapshot_file, dtype={"code": "string"}))
                    if len(snapshot) != 1:
                        raise RuntimeError("confirmation snapshot is not one row")
                    row = snapshot.iloc[0]
                    if not (
                        row["signal_date"] == plan["signal_date"] and row["trade_date"] == plan["trade_date"]
                        and row["asof"] == point and row["code"] == plan["candidate_code"]
                    ):
                        raise RuntimeError("confirmation snapshot key mismatch")
                    score = plan_scores[session_id]
                    candidate = score[score["code"].map(zfill_code).eq(plan["candidate_code"])]
                    evaluated = opening.evaluate_row(
                        candidate.iloc[0], row, point, allow_height_confirmed_research=True,
                    )
                    if event.get("confirmation_status") != evaluated["confirmation_status"]:
                        errors.append(f"event {number}: confirmation status mismatch")
                    if event.get("reasons") != evaluated["reasons"]:
                        errors.append(f"event {number}: confirmation reasons mismatch")
                except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                    errors.append(f"event {number}: invalid confirmation ({exc})")
            confirmations.add(key)
            confirmation_events.setdefault(session_id, []).append(event)
        elif kind == "SETTLEMENT":
            if session_id not in plans:
                errors.append(f"event {number}: settlement without plan")
            if session_id in settlements:
                errors.append(f"event {number}: duplicate settlement")
            outcome_file = _resolve(output, str(event.get("outcome_file", "")))
            if not outcome_file.exists() or sha256_file(outcome_file) != event.get("outcome_sha256"):
                errors.append(f"event {number}: outcome file hash mismatch")
            if session_id in plan_payloads and outcome_file.exists():
                try:
                    recorded = pd.Timestamp(event["recorded_at"])
                    if recorded.tzinfo is None:
                        recorded = recorded.tz_localize(SHANGHAI)
                    else:
                        recorded = recorded.tz_convert(SHANGHAI)
                    raw = pd.read_csv(outcome_file, dtype={"code": "string"})
                    if len(raw) != 1:
                        raise RuntimeError("settlement source is not one row")
                    row = validate_outcome(raw.iloc[0], plan_payloads[session_id], recorded.to_pydatetime())
                    eligible_confirmations = [
                        item for item in confirmation_events.get(session_id, [])
                        if item.get("asof") in {"09:35", "10:00"} and not item.get("late")
                    ]
                    final = sorted(eligible_confirmations, key=lambda item: opening.clock_seconds(item["asof"]))[-1] if eligible_confirmations else None
                    executed_proxy = bool(final and final.get("confirmation_status") == "CONFIRM_RESEAL")
                    derived = derive_outcome(row, executed_proxy)
                    for field, value in derived.items():
                        if event.get(field) != value:
                            errors.append(f"event {number}: settlement {field} mismatch")
                    expected_asof = None if final is None else final["asof"]
                    expected_status = None if final is None else final["confirmation_status"]
                    if event.get("final_confirmation_asof") != expected_asof:
                        errors.append(f"event {number}: final confirmation asof mismatch")
                    if event.get("final_confirmation_status") != expected_status:
                        errors.append(f"event {number}: final confirmation status mismatch")
                except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                    errors.append(f"event {number}: invalid settlement ({exc})")
            settlements.add(session_id)
        else:
            errors.append(f"event {number}: unknown event type")
    return {
        "status": "PASS" if not errors else "FAIL", "version": VERSION,
        "model_lock_id": lock_check.get("lock_id"), "events": len(events),
        "plans": len(plans), "confirmations": len(confirmations),
        "settlements": len(settlements), "pending": len(plans) - len(settlements),
        "head_event_hash": previous, "errors": errors,
        "assurance": "LOCAL_TAMPER_EVIDENT_ONLY",
    }


def append_event(payload: dict[str, Any], output: Path = OUTPUT) -> dict[str, Any]:
    check = verify(output)
    if check["status"] != "PASS":
        raise RuntimeError("cannot append to invalid V10 ledger: " + "; ".join(check["errors"]))
    output.mkdir(parents=True, exist_ok=True)
    event = {
        **payload, "sequence": check["events"] + 1,
        "previous_event_hash": check["head_event_hash"],
        "forward_script_sha256": sha256_file(Path(__file__)),
    }
    event["event_hash"] = hash_value(event)
    with ledger_path(output).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return event


def validate_plan_time(signal_date: str, trade_date: str, current: datetime) -> None:
    signal = pd.Timestamp(signal_date).date()
    trade = pd.Timestamp(trade_date).date()
    if trade <= signal:
        raise RuntimeError("trade_date must be after signal_date")
    local_time = current.timetz().replace(tzinfo=None)
    if current.date() < signal or (current.date() == signal and local_time < time(15, 5)):
        raise RuntimeError("V10 plan is allowed only after signal-day 15:05")
    if current.date() > trade or (current.date() == trade and local_time >= time(9, 15)):
        raise RuntimeError("V10 plan must be frozen before trade-day 09:15")


def prepare_plan(
    input_path: Path,
    signal_date: str,
    trade_date: str,
    output: Path = OUTPUT,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = _now(now)
    signal = pd.Timestamp(signal_date).strftime("%Y-%m-%d")
    trade = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
    validate_plan_time(signal, trade, current)
    if any(event.get("event_type") == "PLAN" and event.get("signal_date") == signal
           for event in read_events(output)):
        raise RuntimeError("V10 plan already exists for signal date")
    emotion_event = shadow_model.v7_forecast(signal)
    scores, metadata = shadow_model.build_scores(input_path, signal, emotion_event)
    selected = scores[scores["shadow_observation"]]
    if len(selected) > 1:
        raise RuntimeError("V10 frozen model selected more than one daily shadow candidate")
    candidate = None if selected.empty else selected.iloc[0]
    basis = {
        "version": VERSION, "model_lock_id": metadata["model_lock_id"],
        "signal_date": signal, "trade_date": trade,
        "emotion_lock_id": metadata["emotion_lock_id"],
        "emotion_event_hash": metadata["emotion_event_hash"],
        "candidate_code": None if candidate is None else zfill_code(candidate["code"]),
        "candidate_name": None if candidate is None else str(candidate["name"]),
        "candidate_boards": None if candidate is None else int(candidate["boards"]),
        "joint_score": None if candidate is None else float(candidate["joint_score"]),
        "input_sha256": metadata["input_sha256"],
    }
    session_id = hash_value(basis)[:20]
    session_dir = output / session_id
    if session_dir.exists():
        raise RuntimeError("immutable V10 session directory already exists")
    session_dir.mkdir(parents=True)
    score_path = session_dir / "frozen_scores.csv"
    scores.to_csv(score_path, index=False, encoding="utf-8-sig")
    plan = {
        **basis, "session_id": session_id,
        "planned_at": current.isoformat(timespec="seconds"),
        "decision": "NO_SHADOW_CANDIDATE" if candidate is None else "SHADOW_ONLY_NEVER_ENTER",
        "large_loss_probability": None,
        "large_loss_probability_status": "UNAVAILABLE_REJECTED_V8_MODEL",
        "capture_times": list(CONFIRM_TIMES),
        "execution": "PAPER_PROXY_ONLY_NO_LEVEL2_QUEUE",
    }
    plan_path = session_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    event = append_event({
        "event_type": "PLAN", "session_id": session_id,
        "model_lock_id": metadata["model_lock_id"], "signal_date": signal,
        "trade_date": trade, "planned_at": plan["planned_at"],
        "candidate_code": basis["candidate_code"], "candidate_boards": basis["candidate_boards"],
        "plan_file": _relative(output, plan_path), "plan_sha256": sha256_file(plan_path),
        "score_file": _relative(output, score_path), "score_sha256": sha256_file(score_path),
    }, output)
    return {"plan": plan, "event": event, "verification": verify(output)}


def _plan(output: Path, session_id: str) -> tuple[dict[str, Any], pd.DataFrame]:
    event = next((item for item in read_events(output) if item.get("event_type") == "PLAN"
                  and item.get("session_id") == session_id), None)
    if event is None:
        raise RuntimeError("unknown V10 session")
    plan = json.loads(_resolve(output, event["plan_file"]).read_text(encoding="utf-8"))
    scores = pd.read_csv(_resolve(output, event["score_file"]), dtype={"code": "string"})
    return plan, scores


def record_confirmation(
    session_id: str,
    asof: str,
    snapshots_path: Path = SNAPSHOTS,
    output: Path = OUTPUT,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = _now(now)
    point = opening.normalize_asof(asof)
    if point not in CONFIRM_TIMES:
        raise RuntimeError(f"V10 confirmation asof must be one of {CONFIRM_TIMES}")
    plan, scores = _plan(output, session_id)
    if plan["candidate_code"] is None:
        raise RuntimeError("session has no shadow candidate")
    if current.date().isoformat() != plan["trade_date"]:
        raise RuntimeError("confirmation must be recorded on trade_date")
    scheduled = datetime.strptime(point, "%H:%M:%S" if point.count(":") == 2 else "%H:%M").time()
    local_time = current.timetz().replace(tzinfo=None)
    if local_time < scheduled:
        raise RuntimeError("confirmation cannot be recorded before asof")
    grace = 2 if point == "09:25" else 5
    latest = (pd.Timestamp.combine(current.date(), scheduled) + pd.Timedelta(minutes=grace)).time()
    late = local_time > latest
    if any(event.get("event_type") == "CONFIRMATION" and event.get("session_id") == session_id
           and event.get("asof") == point for event in read_events(output)):
        raise RuntimeError("confirmation already registered for session/asof")
    if not snapshots_path.exists() or not snapshots_path.stat().st_size:
        raise RuntimeError("verified V9 snapshot archive is unavailable")
    archive = opening.validate_snapshots(pd.read_csv(snapshots_path, dtype={"code": "string"}))
    selected = archive[
        archive["signal_date"].eq(plan["signal_date"])
        & archive["trade_date"].eq(plan["trade_date"])
        & archive["asof"].eq(point)
        & archive["code"].eq(plan["candidate_code"])
    ]
    if len(selected) != 1:
        raise RuntimeError("exactly one bound V9 candidate snapshot is required")
    candidate = scores[scores["code"].map(zfill_code).eq(plan["candidate_code"])]
    if len(candidate) != 1:
        raise RuntimeError("frozen score snapshot has no unique candidate row")
    evaluated = opening.evaluate_row(
        candidate.iloc[0], selected.iloc[0], point, allow_height_confirmed_research=True,
    )
    session_dir = output / session_id
    snapshot_file = session_dir / f"snapshot_{point.replace(':', '')}.csv"
    selected.to_csv(snapshot_file, index=False, encoding="utf-8-sig")
    event = append_event({
        "event_type": "CONFIRMATION", "session_id": session_id,
        "model_lock_id": plan["model_lock_id"], "signal_date": plan["signal_date"],
        "trade_date": plan["trade_date"], "asof": point,
        "recorded_at": current.isoformat(timespec="seconds"),
        "late": late, "training_eligible": not late,
        "confirmation_status": evaluated["confirmation_status"],
        "reasons": evaluated["reasons"],
        "snapshot_file": _relative(output, snapshot_file),
        "snapshot_sha256": sha256_file(snapshot_file),
        "snapshot_archive_sha256": sha256_file(snapshots_path),
    }, output)
    return {"event": event, "verification": verify(output)}


def _next_market_date(trade_date: str, market_dates_path: Path = MARKET_DATES) -> str:
    dates = pd.to_datetime(pd.read_csv(market_dates_path, usecols=["date"])["date"], errors="coerce").dropna()
    later = sorted(date for date in dates.dt.normalize().unique() if date > pd.Timestamp(trade_date))
    if not later:
        raise RuntimeError("next market date is not available in the market archive")
    return pd.Timestamp(later[0]).strftime("%Y-%m-%d")


def validate_outcome(row: pd.Series, plan: dict[str, Any], now: datetime) -> pd.Series:
    missing = sorted(OUTCOME_FIELDS - set(row.index))
    if missing:
        raise RuntimeError(f"V10 outcome missing fields: {missing}")
    checked = row.copy()
    for field in ("signal_date", "trade_date", "exit_date"):
        checked[field] = pd.Timestamp(checked[field]).strftime("%Y-%m-%d")
    if checked["signal_date"] != plan["signal_date"] or checked["trade_date"] != plan["trade_date"]:
        raise RuntimeError("outcome dates do not match frozen plan")
    if zfill_code(checked["code"]) != plan["candidate_code"]:
        raise RuntimeError("outcome code does not match frozen plan")
    if checked["exit_date"] != _next_market_date(plan["trade_date"]):
        raise RuntimeError("outcome exit_date is not the next archived market date")
    if now.date().isoformat() < checked["exit_date"] or (
        now.date().isoformat() == checked["exit_date"] and now.timetz().replace(tzinfo=None) < time(15, 5)
    ):
        raise RuntimeError("settlement is allowed only after exit-day 15:05")
    status = str(checked["data_status"]).strip().lower()
    if status not in VERIFIED_OUTCOME_STATUSES:
        raise RuntimeError("outcome requires a verified raw export status")
    stamp = pd.to_datetime(checked["source_timestamp"], errors="coerce")
    if pd.isna(stamp):
        raise RuntimeError("outcome source_timestamp is invalid")
    stamp = pd.Timestamp(stamp)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("Asia/Shanghai").tz_localize(None)
    if stamp.date().isoformat() != checked["exit_date"] or stamp.time() < time(15, 0) or stamp > pd.Timestamp(now.replace(tzinfo=None)):
        raise RuntimeError("outcome timestamp is outside the verified exit close boundary")
    numeric = [
        "trade_close", "trade_limit_up_price", "exit_open", "exit_high", "exit_low",
        "exit_close", "exit_limit_down_price", "buy_cost_bps", "sell_cost_bps", "slippage_bps",
    ]
    for field in numeric:
        checked[field] = pd.to_numeric(checked[field], errors="coerce")
        if not np.isfinite(checked[field]):
            raise RuntimeError(f"outcome {field} is unavailable")
    if min(checked[field] for field in numeric[:7]) <= 0:
        raise RuntimeError("outcome prices must be positive")
    if checked["exit_high"] < max(checked["exit_open"], checked["exit_close"], checked["exit_low"]):
        raise RuntimeError("outcome exit OHLC is invalid")
    if checked["exit_low"] > min(checked["exit_open"], checked["exit_close"], checked["exit_high"]):
        raise RuntimeError("outcome exit OHLC is invalid")
    return checked


def derive_outcome(row: pd.Series, executed_proxy: bool) -> dict[str, Any]:
    promotion = bool(row["trade_close"] >= row["trade_limit_up_price"] * (1 - 1e-6))
    one_word_down = bool(
        abs(row["exit_open"] - row["exit_limit_down_price"]) <= 1e-6
        and abs(row["exit_high"] - row["exit_limit_down_price"]) <= 1e-6
        and abs(row["exit_low"] - row["exit_limit_down_price"]) <= 1e-6
        and abs(row["exit_close"] - row["exit_limit_down_price"]) <= 1e-6
    )
    if not executed_proxy:
        return {
            "executed_proxy": False, "fill_status": "NO_CONFIRMED_RESEAL_NO_TRADE",
            "promotion_target": promotion, "entry_price": None,
            "mark_to_market_return": None, "net_return": None,
            "large_loss": None, "sell_blocked": False,
        }
    entry = float(row["trade_limit_up_price"])
    buy_cost = (float(row["buy_cost_bps"]) + float(row["slippage_bps"])) / 10000
    sell_cost = float(row["sell_cost_bps"]) / 10000
    mark_return = float((row["exit_close"] * (1 - sell_cost) - entry * (1 + buy_cost)) / (entry * (1 + buy_cost)))
    return {
        "executed_proxy": True, "fill_status": "INFERRED_MINUTE_RESEAL_FILL",
        "promotion_target": promotion, "entry_price": entry,
        "mark_to_market_return": mark_return,
        "net_return": None if one_word_down else mark_return,
        "large_loss": bool(mark_return <= -0.07), "sell_blocked": one_word_down,
    }


def settle(
    session_id: str,
    outcome_path: Path,
    output: Path = OUTPUT,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = _now(now)
    plan, _ = _plan(output, session_id)
    if any(event.get("event_type") == "SETTLEMENT" and event.get("session_id") == session_id
           for event in read_events(output)):
        raise RuntimeError("V10 session is already settled")
    raw = pd.read_csv(outcome_path, dtype={"code": "string"})
    if len(raw) != 1:
        raise RuntimeError("outcome input must contain exactly one row")
    row = validate_outcome(raw.iloc[0], plan, current)
    confirmations = [
        event for event in read_events(output)
        if event.get("event_type") == "CONFIRMATION" and event.get("session_id") == session_id
        and event.get("asof") in {"09:35", "10:00"} and not event.get("late")
    ]
    final = sorted(confirmations, key=lambda item: opening.clock_seconds(item["asof"]))[-1] if confirmations else None
    executed_proxy = bool(final and final.get("confirmation_status") == "CONFIRM_RESEAL")
    derived = derive_outcome(row, executed_proxy)
    session_dir = output / session_id
    stored = session_dir / "settlement_source.csv"
    raw.to_csv(stored, index=False, encoding="utf-8-sig")
    event = append_event({
        "event_type": "SETTLEMENT", "session_id": session_id,
        "model_lock_id": plan["model_lock_id"], "signal_date": plan["signal_date"],
        "trade_date": plan["trade_date"], "exit_date": row["exit_date"],
        "recorded_at": current.isoformat(timespec="seconds"),
        "final_confirmation_asof": None if final is None else final["asof"],
        "final_confirmation_status": None if final is None else final["confirmation_status"],
        **derived,
        "outcome_file": _relative(output, stored), "outcome_sha256": sha256_file(stored),
        "outcome_input_sha256": sha256_file(outcome_path),
    }, output)
    return {"event": event, "verification": verify(output)}


def _profit_factor(values: pd.Series) -> float | None:
    wins = values[values > 0].sum()
    losses = -values[values < 0].sum()
    return float(wins / losses) if losses > 0 else None


def monitor(output: Path = OUTPUT) -> dict[str, Any]:
    check = verify(output)
    events = read_events(output) if check["status"] == "PASS" else []
    plans = [event for event in events if event.get("event_type") == "PLAN"]
    confirmations = [event for event in events if event.get("event_type") == "CONFIRMATION"]
    settlements = [event for event in events if event.get("event_type") == "SETTLEMENT"]
    executed = [event for event in settlements if event.get("executed_proxy")]
    returns = pd.Series([event["net_return"] for event in executed if event.get("net_return") is not None], dtype=float)
    promotions = [event for event in executed if event.get("promotion_target") is not None]
    losses = [event for event in executed if event.get("large_loss") is not None]
    report = {
        "version": VERSION, "status": "PASS" if check["status"] == "PASS" else "FAIL",
        "model_lock_id": check.get("model_lock_id"), "registered_plans": len(plans),
        "registered_candidates": sum(event.get("candidate_code") is not None for event in plans),
        "confirmations": len(confirmations),
        "on_time_confirmations": sum(not event.get("late") for event in confirmations),
        "shadow_reseal_confirmations": sum(
            not event.get("late") and event.get("confirmation_status") == "CONFIRM_RESEAL"
            for event in confirmations
        ),
        "settlements": len(settlements), "executed_proxy": len(executed),
        "executed_promotion_precision": (
            sum(bool(event["promotion_target"]) for event in promotions) / len(promotions) if promotions else None
        ),
        "large_loss_rate": sum(bool(event["large_loss"]) for event in losses) / len(losses) if losses else None,
        "resolved_return_count": int(len(returns)),
        "mean_net_return": float(returns.mean()) if len(returns) else None,
        "profit_factor": _profit_factor(returns) if len(returns) else None,
        "sell_blocked": sum(bool(event.get("sell_blocked")) for event in executed),
        "deployment_allowed": False,
        "deployment_reason": "forward evidence remains shadow-only until the frozen deployment gates pass",
        "verification": check,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / MONITOR_NAME).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V10 tamper-evident intraday shadow forward ledger")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--signal-date", required=True)
    prepare.add_argument("--trade-date", required=True)
    confirm = sub.add_parser("confirm")
    confirm.add_argument("--session-id", required=True)
    confirm.add_argument("--asof", required=True, choices=CONFIRM_TIMES)
    confirm.add_argument("--snapshots", type=Path, default=SNAPSHOTS)
    finish = sub.add_parser("settle")
    finish.add_argument("--session-id", required=True)
    finish.add_argument("--outcome", type=Path, required=True)
    sub.add_parser("verify")
    sub.add_parser("monitor")
    for child in (prepare, confirm, finish):
        child.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "prepare":
        result = prepare_plan(args.input, args.signal_date, args.trade_date, args.output)
    elif args.command == "confirm":
        result = record_confirmation(args.session_id, args.asof, args.snapshots, args.output)
    elif args.command == "settle":
        result = settle(args.session_id, args.outcome, args.output)
    elif args.command == "verify":
        result = verify()
    else:
        result = monitor()
    print(json.dumps(result, ensure_ascii=False, indent=2))
