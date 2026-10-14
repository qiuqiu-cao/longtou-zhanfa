#!/usr/bin/env python3
"""Pre-registered intraday evidence ledger for the V4 board-relay workflow.

Plans are frozen after the signal-day close and before the next session.  Each
capture time can be appended only once and is bound to its raw CSV hash.  A late,
partial, or missing capture remains visible and cannot silently become a clean
training sample after the market outcome is known.
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
from leader_system_v3 import zfill_code


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "v4" / "session_forward"
DEFAULT_CANDIDATES = ROOT / "data" / "v3" / "leader_candidates.csv"
DEFAULT_FORWARD_SIGNALS = ROOT / "data" / "v3" / "forward_signals.csv"
DEFAULT_THEME_SYNC = ROOT / "data" / "v4" / "theme_open_sync.csv"
SHANGHAI = ZoneInfo("Asia/Shanghai")
LEDGER_NAME = "session_forward_ledger.jsonl"
CAPTURE_TIMES = opening.SNAPSHOT_TIMES
VERSION = "v4-session-forward-1"
PLAN_BASIS_FIELDS = (
    "version", "signal_date", "trade_date", "freeze_id", "signal_event_hash",
    "signal_snapshot_sha256", "candidates_sha256", "forward_signals_sha256",
    "capture_times", "required_codes", "shadow_codes", "candidates",
)


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
            raise RuntimeError(f"invalid session ledger JSON at line {line_no}: {exc}") from exc
    return events


def _referenced_path(output: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else output / path


def verify_ledger(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    errors: list[str] = []
    try:
        events = read_events(output)
    except RuntimeError as exc:
        return {"status": "FAIL", "events": 0, "errors": [str(exc)]}
    previous = "GENESIS"
    plans: dict[str, dict[str, Any]] = {}
    captures: set[tuple[str, str]] = set()
    assessments: set[str] = set()
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
        session_id = str(event.get("session_id", ""))
        if event_type == "SESSION_PLAN":
            if session_id in plans:
                errors.append(f"event {index}: duplicate session plan {session_id}")
            plans[session_id] = event
            plan_path = _referenced_path(output, str(event.get("plan_file", "")))
            if not plan_path.exists():
                errors.append(f"event {index}: plan file missing")
            elif sha256_file(plan_path) != event.get("plan_sha256"):
                errors.append(f"event {index}: plan file hash mismatch")
            else:
                try:
                    plan = json.loads(plan_path.read_text(encoding="utf-8"))
                    basis = {field: plan.get(field) for field in PLAN_BASIS_FIELDS}
                    if plan.get("session_id") != sha256_value(basis)[:20]:
                        errors.append(f"event {index}: session_id does not match frozen plan basis")
                    for field in ("session_id", "signal_date", "trade_date", "freeze_id", "signal_event_hash"):
                        if str(plan.get(field)) != str(event.get(field)):
                            errors.append(f"event {index}: plan/event {field} mismatch")
                    planned_at = pd.Timestamp(plan.get("planned_at"))
                    if planned_at.tzinfo is None:
                        planned_at = planned_at.tz_localize(SHANGHAI)
                    else:
                        planned_at = planned_at.tz_convert(SHANGHAI)
                    signal_day = pd.Timestamp(plan["signal_date"]).date()
                    trade_day = pd.Timestamp(plan["trade_date"]).date()
                    local_clock = planned_at.timetz().replace(tzinfo=None)
                    if planned_at.date() < signal_day or (
                        planned_at.date() == signal_day and local_clock < time(15, 5)
                    ):
                        errors.append(f"event {index}: plan was registered before signal-day cutoff")
                    if planned_at.date() > trade_day or (
                        planned_at.date() == trade_day and local_clock >= time(9, 15)
                    ):
                        errors.append(f"event {index}: plan was registered after trade-day cutoff")
                except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    errors.append(f"event {index}: invalid plan payload ({exc})")
        elif event_type == "CAPTURE":
            key = (session_id, str(event.get("asof")))
            if session_id not in plans:
                errors.append(f"event {index}: capture without plan")
            if key in captures:
                errors.append(f"event {index}: duplicate capture {key}")
            captures.add(key)
            raw_path = _referenced_path(output, str(event.get("raw_file", "")))
            if not raw_path.exists():
                errors.append(f"event {index}: raw capture missing")
            elif sha256_file(raw_path) != event.get("raw_sha256"):
                errors.append(f"event {index}: raw capture hash mismatch")
            else:
                try:
                    plan = json.loads(_referenced_path(output, plans[session_id]["plan_file"]).read_text(encoding="utf-8"))
                    raw = pd.read_csv(raw_path, dtype={"code": "string"})
                    actual_codes = sorted(raw["code"].map(zfill_code).unique())
                    if actual_codes != sorted(event.get("observed_codes", [])):
                        errors.append(f"event {index}: observed codes differ from raw capture")
                    if not raw["signal_date"].astype(str).eq(plan["signal_date"]).all():
                        errors.append(f"event {index}: raw signal_date differs from plan")
                    if not raw["trade_date"].astype(str).eq(plan["trade_date"]).all():
                        errors.append(f"event {index}: raw trade_date differs from plan")
                    if not raw["asof"].astype(str).eq(str(event.get("asof"))).all():
                        errors.append(f"event {index}: raw asof differs from capture event")
                    missing_required = sorted(set(plan["required_codes"]) - set(actual_codes))
                    if missing_required != sorted(event.get("missing_required_codes", [])):
                        errors.append(f"event {index}: missing required codes mismatch")
                    recorded_at = pd.Timestamp(event.get("recorded_at"))
                    if recorded_at.tzinfo is None:
                        recorded_at = recorded_at.tz_localize(SHANGHAI)
                    else:
                        recorded_at = recorded_at.tz_convert(SHANGHAI)
                    scheduled = _clock(str(event.get("asof")))
                    if recorded_at.date().isoformat() != plan["trade_date"] or recorded_at.timetz().replace(tzinfo=None) < scheduled:
                        errors.append(f"event {index}: capture recorded outside its trade-date/asof boundary")
                    grace = 2 if event.get("asof") in {"09:20", "09:24:50", "09:25", "09:30"} else 5
                    latest = (pd.Timestamp.combine(recorded_at.date(), scheduled) + pd.Timedelta(minutes=grace)).time()
                    expected_late = recorded_at.timetz().replace(tzinfo=None) > latest
                    if bool(event.get("late_capture")) != expected_late:
                        errors.append(f"event {index}: late_capture flag mismatch")
                    expected_complete = not expected_late and not missing_required
                    if bool(event.get("capture_complete")) != expected_complete:
                        errors.append(f"event {index}: capture_complete flag mismatch")
                except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    errors.append(f"event {index}: invalid capture payload ({exc})")
        elif event_type == "ASSESSMENT":
            if session_id not in plans:
                errors.append(f"event {index}: assessment without plan")
            if session_id in assessments:
                errors.append(f"event {index}: duplicate assessment")
            assessments.add(session_id)
            assessment_path = _referenced_path(output, str(event.get("assessment_file", "")))
            if not assessment_path.exists():
                errors.append(f"event {index}: assessment file missing")
            elif sha256_file(assessment_path) != event.get("assessment_sha256"):
                errors.append(f"event {index}: assessment file hash mismatch")
            else:
                try:
                    plan = json.loads(_referenced_path(output, plans[session_id]["plan_file"]).read_text(encoding="utf-8"))
                    recorded_at = pd.Timestamp(event.get("recorded_at"))
                    if recorded_at.tzinfo is None:
                        recorded_at = recorded_at.tz_localize(SHANGHAI)
                    else:
                        recorded_at = recorded_at.tz_convert(SHANGHAI)
                    if recorded_at.date().isoformat() < plan["trade_date"] or (
                        recorded_at.date().isoformat() == plan["trade_date"]
                        and recorded_at.timetz().replace(tzinfo=None) < time(10, 0)
                    ):
                        errors.append(f"event {index}: assessment recorded before 10:00 boundary")
                    expected_training = bool(event.get("evidence_complete") and plan["required_codes"])
                    if bool(event.get("training_eligible")) != expected_training:
                        errors.append(f"event {index}: training eligibility mismatch")
                except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    errors.append(f"event {index}: invalid assessment payload ({exc})")
        else:
            errors.append(f"event {index}: unknown event_type {event_type}")
    return {
        "status": "PASS" if not errors else "FAIL",
        "version": VERSION,
        "events": len(events),
        "plans": len(plans),
        "captures": len(captures),
        "assessments": len(assessments),
        "head_event_hash": previous,
        "errors": errors,
        "assurance": "LOCAL_TAMPER_EVIDENT_ONLY",
    }


def append_event(output: Path, payload: dict[str, Any]) -> dict[str, Any]:
    verification = verify_ledger(output)
    if verification["status"] != "PASS":
        raise RuntimeError("session ledger is invalid: " + "; ".join(verification["errors"]))
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


def _local_now(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now(SHANGHAI)
    return now.astimezone(SHANGHAI) if now.tzinfo else now.replace(tzinfo=SHANGHAI)


def _clock(value: str) -> time:
    parsed = datetime.strptime(value, "%H:%M:%S" if value.count(":") == 2 else "%H:%M")
    return parsed.time()


def _relative(output: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(output.resolve()))
    except ValueError:
        return str(path.resolve())


def _latest_forward_signal(signals: pd.DataFrame, signal_date: str) -> pd.DataFrame:
    selected = signals[signals["signal_date"].astype(str).eq(signal_date)].copy()
    if selected.empty:
        raise RuntimeError(f"no frozen forward signal snapshot for {signal_date}")
    hashes = selected["snapshot_sha256"].dropna().astype(str).unique()
    events = selected["signal_event_hash"].dropna().astype(str).unique()
    freezes = selected["freeze_id"].dropna().astype(str).unique()
    if len(hashes) != 1 or len(events) != 1 or len(freezes) != 1:
        raise RuntimeError("forward signal rows do not share one frozen snapshot/event/freeze")
    return selected


def build_plan(
    signal_date: str,
    trade_date: str,
    candidates_path: Path = DEFAULT_CANDIDATES,
    signals_path: Path = DEFAULT_FORWARD_SIGNALS,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = _local_now(now)
    signal = pd.Timestamp(signal_date).date()
    trade = pd.Timestamp(trade_date).date()
    if trade <= signal:
        raise RuntimeError("trade_date must be later than signal_date")
    if current.date() < signal or (current.date() == signal and current.timetz().replace(tzinfo=None) < time(15, 5)):
        raise RuntimeError("session plan can be registered only after signal-day 15:05")
    if current.date() > trade or (current.date() == trade and current.timetz().replace(tzinfo=None) >= time(9, 15)):
        raise RuntimeError("session plan must be frozen before trade-day 09:15")
    candidates = pd.read_csv(candidates_path, dtype={"code": "string"}, low_memory=False)
    candidates = candidates[candidates["date"].astype(str).eq(signal.isoformat())].copy()
    if candidates.empty:
        raise RuntimeError(f"no candidates for signal date {signal}")
    signals = _latest_forward_signal(
        pd.read_csv(signals_path, dtype={"code": "string"}, low_memory=False), signal.isoformat()
    )
    decision_map = signals.assign(code=signals["code"].map(zfill_code)).set_index("code")["decision"].to_dict()
    rows = []
    for _, candidate in candidates.iterrows():
        code = zfill_code(candidate["code"])
        preopen_ok, preopen_reasons = opening.preopen_eligibility(candidate)
        decision = str(decision_map.get(code, "UNREGISTERED"))
        execution_watch = bool(preopen_ok and decision in {"ENTER", "WATCH"})
        rows.append({
            "code": code,
            "name": str(candidate.get("name", "")),
            "boards": int(float(candidate.get("boards"))),
            "decision": decision,
            "emotion_state": str(candidate.get("emotion_state", "")),
            "valid_identity": None if pd.isna(candidate.get("valid_identity")) else int(float(candidate.get("valid_identity"))),
            "one_word_proxy": None if pd.isna(candidate.get("one_word_proxy")) else int(float(candidate.get("one_word_proxy"))),
            "execution_watch": execution_watch,
            "preopen_reasons": preopen_reasons,
        })
    required_codes = sorted(row["code"] for row in rows if row["execution_watch"])
    shadow_codes = sorted(row["code"] for row in rows)
    basis = {
        "version": VERSION,
        "signal_date": signal.isoformat(),
        "trade_date": trade.isoformat(),
        "freeze_id": str(signals.iloc[0]["freeze_id"]),
        "signal_event_hash": str(signals.iloc[0]["signal_event_hash"]),
        "signal_snapshot_sha256": str(signals.iloc[0]["snapshot_sha256"]),
        "candidates_sha256": sha256_file(candidates_path),
        "forward_signals_sha256": sha256_file(signals_path),
        "capture_times": list(CAPTURE_TIMES),
        "required_codes": required_codes,
        "shadow_codes": shadow_codes,
        "candidates": sorted(rows, key=lambda row: row["code"]),
    }
    return {
        **basis,
        "session_id": sha256_value(basis)[:20],
        "planned_at": current.isoformat(timespec="seconds"),
        "required_fields_by_asof": {point: list(opening.REQUIRED_BY_ASOF[point]) for point in CAPTURE_TIMES},
        "capture_policy": (
            "one immutable capture per time; required codes are frozen before trade; "
            "late or partial captures never qualify as complete forward evidence"
        ),
    }


def register_plan(
    signal_date: str,
    trade_date: str,
    output: Path = DEFAULT_OUTPUT,
    candidates_path: Path = DEFAULT_CANDIDATES,
    signals_path: Path = DEFAULT_FORWARD_SIGNALS,
    now: datetime | None = None,
) -> dict[str, Any]:
    plan = build_plan(signal_date, trade_date, candidates_path, signals_path, now)
    existing = read_events(output)
    if any(event.get("event_type") == "SESSION_PLAN" and event.get("signal_date") == plan["signal_date"]
           and event.get("trade_date") == plan["trade_date"] for event in existing):
        raise RuntimeError("session plan already registered for signal/trade date")
    session_dir = output / plan["session_id"]
    session_dir.mkdir(parents=True, exist_ok=False)
    plan_path = session_dir / "session_plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    event = append_event(output, {
        "event_type": "SESSION_PLAN",
        "session_id": plan["session_id"],
        "signal_date": plan["signal_date"],
        "trade_date": plan["trade_date"],
        "planned_at": plan["planned_at"],
        "freeze_id": plan["freeze_id"],
        "signal_event_hash": plan["signal_event_hash"],
        "required_code_count": len(plan["required_codes"]),
        "shadow_code_count": len(plan["shadow_codes"]),
        "plan_file": _relative(output, plan_path),
        "plan_sha256": sha256_file(plan_path),
    })
    return {"plan": plan, "event": event, "verification": verify_ledger(output)}


def _plan_event(output: Path, session_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    for event in read_events(output):
        if event.get("event_type") == "SESSION_PLAN" and event.get("session_id") == session_id:
            plan_path = _referenced_path(output, event["plan_file"])
            return event, json.loads(plan_path.read_text(encoding="utf-8"))
    raise RuntimeError(f"unknown session_id {session_id}")


def record_capture(
    session_id: str,
    asof: str,
    source: Path,
    output: Path = DEFAULT_OUTPUT,
    theme_sync_registry: Path | pd.DataFrame | None = DEFAULT_THEME_SYNC,
    now: datetime | None = None,
) -> dict[str, Any]:
    verification = verify_ledger(output)
    if verification["status"] != "PASS":
        raise RuntimeError("cannot record into invalid ledger")
    point = opening.normalize_asof(asof)
    _, plan = _plan_event(output, session_id)
    current = _local_now(now)
    if current.date().isoformat() != plan["trade_date"]:
        raise RuntimeError("capture must be recorded on the registered trade_date")
    scheduled = _clock(point)
    current_clock = current.timetz().replace(tzinfo=None)
    grace_minutes = 2 if point in {"09:20", "09:24:50", "09:25", "09:30"} else 5
    latest = (pd.Timestamp.combine(current.date(), scheduled) + pd.Timedelta(minutes=grace_minutes)).time()
    if current_clock < scheduled:
        raise RuntimeError("capture cannot be registered before its asof")
    late = current_clock > latest
    if any(event.get("event_type") == "CAPTURE" and event.get("session_id") == session_id
           and event.get("asof") == point for event in read_events(output)):
        raise RuntimeError("capture already registered for this session/asof")
    checked = opening.validate_snapshots(
        pd.read_csv(source, dtype={"code": "string"}), theme_sync_registry
    )
    checked = checked[
        checked["signal_date"].eq(plan["signal_date"])
        & checked["trade_date"].eq(plan["trade_date"])
        & checked["asof"].eq(point)
    ].copy()
    if checked.empty:
        raise RuntimeError("source contains no rows matching registered session/asof")
    allowed = set(plan["shadow_codes"])
    unknown = sorted(set(checked["code"]) - allowed)
    if unknown:
        raise RuntimeError(f"capture contains codes outside frozen candidate list: {unknown[:5]}")
    required = set(plan["required_codes"])
    observed = set(checked["code"])
    missing_required = sorted(required - observed)
    complete = not late and not missing_required
    session_dir = output / session_id
    raw_path = session_dir / f"capture_{point.replace(':', '')}.csv"
    if raw_path.exists():
        raise RuntimeError("immutable raw capture file already exists")
    checked.sort_values("code").to_csv(raw_path, index=False, encoding="utf-8-sig")
    event = append_event(output, {
        "event_type": "CAPTURE",
        "session_id": session_id,
        "signal_date": plan["signal_date"],
        "trade_date": plan["trade_date"],
        "asof": point,
        "recorded_at": current.isoformat(timespec="seconds"),
        "late_capture": late,
        "capture_complete": complete,
        "required_codes": sorted(required),
        "observed_codes": sorted(observed),
        "missing_required_codes": missing_required,
        "row_count": int(len(checked)),
        "raw_file": _relative(output, raw_path),
        "raw_sha256": sha256_file(raw_path),
        "source_input_sha256": sha256_file(source),
    })
    return {"event": event, "verification": verify_ledger(output)}


def assess_session(
    session_id: str,
    output: Path = DEFAULT_OUTPUT,
    candidates_path: Path = DEFAULT_CANDIDATES,
    now: datetime | None = None,
) -> dict[str, Any]:
    verification = verify_ledger(output)
    if verification["status"] != "PASS":
        raise RuntimeError("cannot assess invalid ledger")
    _, plan = _plan_event(output, session_id)
    current = _local_now(now)
    if current.date().isoformat() < plan["trade_date"] or (
        current.date().isoformat() == plan["trade_date"] and current.timetz().replace(tzinfo=None) < time(10, 0)
    ):
        raise RuntimeError("assessment is allowed only after trade-day 10:00")
    if any(event.get("event_type") == "ASSESSMENT" and event.get("session_id") == session_id
           for event in read_events(output)):
        raise RuntimeError("session assessment already exists")
    capture_events = [
        event for event in read_events(output)
        if event.get("event_type") == "CAPTURE" and event.get("session_id") == session_id
    ]
    captures = []
    for event in capture_events:
        frame = pd.read_csv(_referenced_path(output, event["raw_file"]), dtype={"code": "string"})
        frame["_asof_seconds"] = opening.clock_seconds(event["asof"])
        captures.append(frame)
    latest = pd.concat(captures, ignore_index=True).sort_values("_asof_seconds").drop_duplicates("code", keep="last") if captures else pd.DataFrame()
    snapshot_map = latest.set_index("code").to_dict("index") if not latest.empty else {}
    candidates = pd.read_csv(candidates_path, dtype={"code": "string"}, low_memory=False)
    candidates = candidates[candidates["date"].astype(str).eq(plan["signal_date"])].copy()
    rows = []
    for _, candidate in candidates.iterrows():
        code = zfill_code(candidate["code"])
        snapshot = pd.Series(snapshot_map[code]) if code in snapshot_map else None
        latest_asof = "10:00" if snapshot is not None and opening.clock_seconds(str(snapshot["asof"])) >= opening.clock_seconds("10:00") else "09:35"
        rows.append(opening.evaluate_row(candidate, snapshot, latest_asof))
    result = pd.DataFrame(rows)
    required_capture_times = {"09:25", "09:35"}
    captured_on_time = {
        event["asof"] for event in capture_events
        if event.get("capture_complete") and not event.get("late_capture")
    }
    evidence_complete = required_capture_times.issubset(captured_on_time) if plan["required_codes"] else True
    session_dir = output / session_id
    assessment_path = session_dir / "assessment.csv"
    result.to_csv(assessment_path, index=False, encoding="utf-8-sig")
    counts = result["confirmation_status"].value_counts().to_dict()
    event = append_event(output, {
        "event_type": "ASSESSMENT",
        "session_id": session_id,
        "signal_date": plan["signal_date"],
        "trade_date": plan["trade_date"],
        "recorded_at": current.isoformat(timespec="seconds"),
        "evidence_complete": evidence_complete,
        "required_capture_times": sorted(required_capture_times),
        "captured_on_time": sorted(captured_on_time),
        "status_counts": {str(key): int(value) for key, value in counts.items()},
        "confirmed_count": int((result["confirmation_status"] == "CONFIRM_RESEAL").sum()),
        "assessment_file": _relative(output, assessment_path),
        "assessment_sha256": sha256_file(assessment_path),
        "training_eligible": bool(evidence_complete and plan["required_codes"]),
    })
    return {"event": event, "verification": verify_ledger(output)}


def monitor(output: Path = DEFAULT_OUTPUT, now: datetime | None = None) -> dict[str, Any]:
    current = _local_now(now)
    verification = verify_ledger(output)
    events = read_events(output) if verification["status"] == "PASS" else []
    sessions = []
    for plan_event in [event for event in events if event.get("event_type") == "SESSION_PLAN"]:
        _, plan = _plan_event(output, plan_event["session_id"])
        captures = [
            event for event in events
            if event.get("event_type") == "CAPTURE" and event.get("session_id") == plan["session_id"]
        ]
        captured = {event["asof"] for event in captures}
        due = []
        if current.date().isoformat() > plan["trade_date"]:
            due = list(CAPTURE_TIMES)
        elif current.date().isoformat() == plan["trade_date"]:
            due = [point for point in CAPTURE_TIMES if current.timetz().replace(tzinfo=None) >= _clock(point)]
        missing_due = sorted(set(due) - captured, key=opening.clock_seconds)
        assessment = next((
            event for event in events
            if event.get("event_type") == "ASSESSMENT" and event.get("session_id") == plan["session_id"]
        ), None)
        if not plan["required_codes"]:
            status = "NO_EXECUTION_CANDIDATES"
        elif missing_due:
            status = "CAPTURE_MISSING"
        elif assessment is None:
            status = "AWAITING_ASSESSMENT"
        elif assessment.get("training_eligible"):
            status = "COMPLETE_FORWARD_EVIDENCE"
        else:
            status = "INCOMPLETE_FORWARD_EVIDENCE"
        sessions.append({
            "session_id": plan["session_id"],
            "signal_date": plan["signal_date"],
            "trade_date": plan["trade_date"],
            "required_code_count": len(plan["required_codes"]),
            "captured_times": sorted(captured, key=opening.clock_seconds),
            "missing_due_times": missing_due,
            "assessment_registered": assessment is not None,
            "status": status,
        })
    return {
        "status": "PASS" if verification["status"] == "PASS" else "FAIL",
        "asof": current.isoformat(timespec="seconds"),
        "sessions": sessions,
        "verification": verification,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-register and audit V4 intraday relay evidence")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--signal-date", required=True)
    prepare.add_argument("--trade-date", required=True)
    prepare.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    record = sub.add_parser("record")
    record.add_argument("--session-id", required=True)
    record.add_argument("--asof", required=True, choices=CAPTURE_TIMES)
    record.add_argument("--file", required=True, type=Path)
    record.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    record.add_argument("--theme-sync", type=Path, default=DEFAULT_THEME_SYNC)
    assess = sub.add_parser("assess")
    assess.add_argument("--session-id", required=True)
    assess.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    verify = sub.add_parser("verify")
    verify.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    status = sub.add_parser("monitor")
    status.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        result = register_plan(args.signal_date, args.trade_date, args.output)
    elif args.command == "record":
        result = record_capture(args.session_id, args.asof, args.file, args.output, args.theme_sync)
    elif args.command == "assess":
        result = assess_session(args.session_id, args.output)
    elif args.command == "verify":
        result = verify_ledger(args.output)
    else:
        result = monitor(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
