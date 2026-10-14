#!/usr/bin/env python3
"""Tamper-evident forward evidence for the locked V7 adverse-state candidate."""

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

import emotion_model_lab_v7 as lab
from emotion_challenger_v4 import ACTIVE_STATES, add_targets


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "v7" / "point_in_time_rebuild" / "market_emotion.csv"
OUTPUT = ROOT / "results" / "v7" / "emotion_model_lab"
LOCK_NAME = "research_candidate_lock.json"
LEDGER_NAME = "research_forward_ledger.jsonl"
MONITOR_NAME = "research_forward_monitor.json"
SHANGHAI = ZoneInfo("Asia/Shanghai")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def event_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_lock(output: Path = OUTPUT) -> dict[str, Any]:
    path = output / LOCK_NAME
    verification = lab.verify_lock(path)
    if verification["status"] != "PASS":
        raise RuntimeError("V7 research lock verification failed: " + "; ".join(verification["errors"]))
    return json.loads(path.read_text(encoding="utf-8"))


def read_events(output: Path = OUTPUT) -> list[dict[str, Any]]:
    path = output / LEDGER_NAME
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid V7 forward ledger line {number}: {exc}") from exc
    return rows


def append_event(payload: dict[str, Any], output: Path = OUTPUT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    events = read_events(output)
    event = {**payload, "previous_event_hash": events[-1]["event_hash"] if events else "GENESIS"}
    event["event_hash"] = event_hash(event)
    with (output / LEDGER_NAME).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return event


def validate_input(path: Path, features: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "date", "emotion_state", "emotion_adverse_markov_probability",
        "source", "source_timestamp", "data_status", "emotion_data_completeness", *features,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"V7 market input missing fields: {missing}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise RuntimeError("V7 market input has invalid or duplicate dates")
    return frame.sort_values("date").reset_index(drop=True)


def validate_row_timestamp(row: pd.Series, now: datetime) -> None:
    timestamp = pd.to_datetime(row["source_timestamp"], errors="coerce")
    if pd.isna(timestamp):
        raise RuntimeError("latest V7 market row has no usable timestamp")
    value = timestamp.to_pydatetime()
    if value.tzinfo is None:
        value = value.replace(tzinfo=SHANGHAI)
    if value > now:
        raise RuntimeError("latest V7 source timestamp is in the future")
    completeness = pd.to_numeric(row["emotion_data_completeness"], errors="coerce")
    if not np.isfinite(completeness) or float(completeness) < 0.75:
        raise RuntimeError("latest V7 emotion data completeness is below 0.75")
    if str(row["data_status"]).lower() in {"", "failed", "unavailable"}:
        raise RuntimeError("latest V7 market data status is unusable")


def build_forecast(path: Path = INPUT, output: Path = OUTPUT, now: datetime | None = None) -> dict[str, Any]:
    current = now.astimezone(SHANGHAI) if now else datetime.now(SHANGHAI)
    lock = load_lock(output)
    features = list(lock["features"])
    raw = validate_input(path, features)
    latest = raw.iloc[-1]
    signal_date = latest["date"].date()
    development_end = pd.Timestamp(lock["development_data_end"]).date()
    if signal_date <= development_end:
        raise RuntimeError("forecast signal_date must be after the locked development period")
    if signal_date != current.date():
        raise RuntimeError(f"forecast must use today's latest row: signal={signal_date}, today={current.date()}")
    if current.timetz().replace(tzinfo=None) < time(15, 5):
        raise RuntimeError("V7 forecast registration is allowed only after 15:05 Asia/Shanghai")
    validate_row_timestamp(latest, current)
    key = (lock["lock_id"], signal_date.isoformat())
    if any(event.get("event_type") == "FORECAST" and (event.get("lock_id"), event.get("signal_date")) == key
           for event in read_events(output)):
        raise RuntimeError(f"forecast already registered for {key}")
    frame = add_targets(raw.assign(date=raw["date"].dt.strftime("%Y-%m-%d")))
    p = lab.prequential_predictions(frame, "adverse_state_next", features, "logistic")
    probability = float(p[-1])
    baseline = float(pd.to_numeric(latest["emotion_adverse_markov_probability"], errors="coerce"))
    if not np.isfinite(probability) or not np.isfinite(baseline):
        raise RuntimeError("V7 locked probability or baseline is unavailable")
    snapshot = {name: None if pd.isna(latest[name]) else float(latest[name]) for name in features}
    return {
        "event_type": "FORECAST", "lock_id": lock["lock_id"],
        "signal_date": signal_date.isoformat(), "recorded_at": current.isoformat(timespec="seconds"),
        "emotion_state": str(latest["emotion_state"]),
        "active_state": bool(str(latest["emotion_state"]) in ACTIVE_STATES),
        "probability": probability, "markov_baseline": baseline,
        "feature_snapshot": snapshot, "input_sha256": lab.sha256_file(path),
        "source": str(latest["source"]), "source_timestamp": str(latest["source_timestamp"]),
        "data_status": str(latest["data_status"]),
        "emotion_data_completeness": float(latest["emotion_data_completeness"]),
        "forward_script_sha256": lab.sha256_file(Path(__file__)),
    }


def record(path: Path = INPUT, output: Path = OUTPUT) -> dict[str, Any]:
    return append_event(build_forecast(path, output), output)


def verify(output: Path = OUTPUT) -> dict[str, Any]:
    errors: list[str] = []
    try:
        lock = load_lock(output)
        events = read_events(output)
    except RuntimeError as exc:
        return {"status": "FAIL", "errors": [str(exc)]}
    previous = "GENESIS"
    forecasts: dict[tuple[str, str], dict[str, Any]] = {}
    settlements: set[tuple[str, str]] = set()
    development_end = pd.Timestamp(lock["development_data_end"])
    script_hash = lab.sha256_file(Path(__file__))
    for number, event in enumerate(events, 1):
        body = {key: value for key, value in event.items() if key != "event_hash"}
        if event.get("previous_event_hash") != previous:
            errors.append(f"event {number}: previous hash mismatch")
        if event.get("event_hash") != event_hash(body):
            errors.append(f"event {number}: event hash mismatch")
        previous = str(event.get("event_hash"))
        if event.get("lock_id") != lock["lock_id"]:
            errors.append(f"event {number}: wrong lock_id")
        signal = pd.to_datetime(event.get("signal_date"), errors="coerce")
        if pd.isna(signal) or signal <= development_end:
            errors.append(f"event {number}: signal date is not post-lock")
        key = (str(event.get("lock_id")), str(event.get("signal_date")))
        if event.get("event_type") == "FORECAST":
            if key in forecasts:
                errors.append(f"event {number}: duplicate forecast")
            if event.get("forward_script_sha256") != script_hash:
                errors.append(f"event {number}: forward script hash mismatch")
            forecasts[key] = event
        elif event.get("event_type") == "SETTLEMENT":
            if key not in forecasts:
                errors.append(f"event {number}: settlement without forecast")
            if key in settlements:
                errors.append(f"event {number}: duplicate settlement")
            settlements.add(key)
        else:
            errors.append(f"event {number}: invalid event type")
    return {
        "status": "PASS" if not errors else "FAIL", "events": len(events),
        "forecasts": len(forecasts), "settlements": len(settlements),
        "pending": len(forecasts) - len(settlements), "head_event_hash": previous,
        "lock_id": lock["lock_id"], "errors": errors,
        "assurance": "LOCAL_TAMPER_EVIDENT_ONLY",
    }


def settle(path: Path = INPUT, output: Path = OUTPUT, now: datetime | None = None) -> list[dict[str, Any]]:
    current = now.astimezone(SHANGHAI) if now else datetime.now(SHANGHAI)
    verification = verify(output)
    if verification["status"] != "PASS":
        raise RuntimeError("cannot settle invalid V7 ledger: " + "; ".join(verification["errors"]))
    lock = load_lock(output)
    raw = validate_input(path, list(lock["features"]))
    events = read_events(output)
    forecasts = {(e["lock_id"], e["signal_date"]): e for e in events if e["event_type"] == "FORECAST"}
    settled = {(e["lock_id"], e["signal_date"]) for e in events if e["event_type"] == "SETTLEMENT"}
    results = []
    for key, forecast in forecasts.items():
        if key in settled:
            continue
        signal = pd.Timestamp(forecast["signal_date"])
        later = raw[raw["date"] > signal]
        if later.empty:
            continue
        outcome = later.iloc[0]
        if outcome["date"].date() > current.date():
            continue
        validate_row_timestamp(outcome, current)
        payload = {
            "event_type": "SETTLEMENT", "lock_id": forecast["lock_id"],
            "signal_date": forecast["signal_date"],
            "outcome_date": outcome["date"].date().isoformat(),
            "settled_at": current.isoformat(timespec="seconds"),
            "label": float(str(outcome["emotion_state"]) in {"大分歧", "退潮"}),
            "outcome_emotion_state": str(outcome["emotion_state"]),
            "outcome_source": str(outcome["source"]),
            "outcome_source_timestamp": str(outcome["source_timestamp"]),
            "input_sha256": lab.sha256_file(path),
        }
        results.append(append_event(payload, output))
    return results


def monitor(output: Path = OUTPUT) -> dict[str, Any]:
    verification = verify(output)
    if verification["status"] != "PASS":
        return {"status": "INVALID_LEDGER", "verification": verification}
    events = read_events(output)
    forecasts = {(e["lock_id"], e["signal_date"]): e for e in events if e["event_type"] == "FORECAST"}
    rows = []
    for event in events:
        if event["event_type"] != "SETTLEMENT":
            continue
        forecast = forecasts[(event["lock_id"], event["signal_date"])]
        rows.append({
            "label": float(event["label"]), "probability": float(forecast["probability"]),
            "baseline": float(forecast["markov_baseline"]), "active": bool(forecast["active_state"]),
        })
    frame = pd.DataFrame(rows)
    active = frame[frame["active"]] if not frame.empty else frame.copy()

    def block(part: pd.DataFrame) -> dict[str, Any]:
        if part.empty:
            return {"n": 0}
        return lab.metrics(
            part["label"].to_numpy(float), part["probability"].to_numpy(float), part["baseline"].to_numpy(float),
        )

    all_metrics, active_metrics = block(frame), block(active)
    enough = len(frame) >= 60 and len(active) >= 30
    stable = enough and all(
        metrics.get("auc", 0) > 0.5
        and metrics.get("brier_skill_vs_baseline", -math.inf) > 0
        and metrics.get("log_loss_improvement_vs_baseline", -math.inf) > 0
        for metrics in (all_metrics, active_metrics)
    )
    report = {
        "status": "FORWARD_CANDIDATE" if stable else ("FORWARD_REJECTED" if enough else "INSUFFICIENT_FORWARD_EVIDENCE"),
        "lock_id": verification["lock_id"], "registered": verification["forecasts"],
        "resolved": len(frame), "active_resolved": len(active), "pending": verification["pending"],
        "all_dates": all_metrics, "active_states": active_metrics,
        "forward_gate": {
            "passed": stable,
            "rule": ">=60 resolved and >=30 active; both views need AUC>0.5, positive Brier skill and lower log loss",
            "does_not_imply_trading_deployment": True,
        },
        "verification": verification,
    }
    (output / MONITOR_NAME).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward evidence for V7 locked adverse-state candidate")
    parser.add_argument("command", choices=("record", "settle", "verify", "monitor"))
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.command == "record":
        result: Any = record(args.input, args.output)
    elif args.command == "settle":
        result = {"settled": settle(args.input, args.output)}
    elif args.command == "verify":
        result = verify(args.output)
    else:
        result = monitor(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
