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

import emotion_challenger_v4 as challenger


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "v3" / "market_emotion.csv"
DEFAULT_OUTPUT = ROOT / "results" / "v4"
LOCK_FILE = "emotion_challenger_lock.json"
LEDGER_FILE = "emotion_forward_ledger.jsonl"
SHANGHAI = ZoneInfo("Asia/Shanghai")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        default=challenger.json_default,
    )


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_lock(output_dir: Path) -> dict[str, Any]:
    path = output_dir / LOCK_FILE
    if not path.exists():
        raise RuntimeError(f"V4 lock missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_challenger_lock(output_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    try:
        locked = load_lock(output_dir)
    except (RuntimeError, json.JSONDecodeError) as exc:
        return {"status": "FAIL", "errors": [str(exc)]}
    script_path = Path(challenger.__file__).resolve()
    current_script_hash = challenger.file_sha256(script_path)
    if current_script_hash != locked.get("script_sha256"):
        errors.append("locked challenger script hash mismatch")
    if canonical_json(challenger.FEATURES) != canonical_json(locked.get("features")):
        errors.append("locked features mismatch")
    if canonical_json(challenger.TARGETS) != canonical_json(locked.get("targets")):
        errors.append("locked target definitions mismatch")
    if canonical_json(challenger.LOCKED_PARAMETERS) != canonical_json(locked.get("locked_parameters")):
        errors.append("locked parameters mismatch")
    payload = {key: value for key, value in locked.items() if key not in {"lock_id", "created_at"}}
    expected_lock_id = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]
    if locked.get("lock_id") != expected_lock_id:
        errors.append("lock_id mismatch")
    return {
        "status": "PASS" if not errors else "FAIL",
        "lock_id": locked.get("lock_id"),
        "development_data_end": locked.get("development_data_end"),
        "script_sha256": current_script_hash,
        "errors": errors,
        "assurance": "LOCAL_TAMPER_EVIDENT_ONLY",
    }


def ledger_path(output_dir: Path) -> Path:
    return output_dir / LEDGER_FILE


def read_events(output_dir: Path) -> list[dict[str, Any]]:
    path = ledger_path(output_dir)
    if not path.exists():
        return []
    events = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid ledger JSON at line {line_number}: {exc}") from exc
    return events


def append_event(output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    events = read_events(output_dir)
    previous_hash = events[-1]["event_hash"] if events else "GENESIS"
    event = {**payload, "previous_event_hash": previous_hash}
    event["event_hash"] = sha256_value(event)
    path = ledger_path(output_dir)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, default=challenger.json_default) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return event


def verify_ledger(output_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    try:
        events = read_events(output_dir)
    except RuntimeError as exc:
        return {"status": "FAIL", "events": 0, "errors": [str(exc)]}
    previous_hash = "GENESIS"
    forecasts: dict[tuple[str, str], dict[str, Any]] = {}
    settlements: set[tuple[str, str]] = set()
    lock_verification = verify_challenger_lock(output_dir)
    active_lock_id = lock_verification.get("lock_id")
    development_end = pd.to_datetime(lock_verification.get("development_data_end"), errors="coerce")
    for index, event in enumerate(events, 1):
        claimed_hash = event.get("event_hash")
        body = {key: value for key, value in event.items() if key != "event_hash"}
        if event.get("previous_event_hash") != previous_hash:
            errors.append(f"event {index}: previous hash mismatch")
        if claimed_hash != sha256_value(body):
            errors.append(f"event {index}: event hash mismatch")
        previous_hash = str(claimed_hash)
        key = (str(event.get("lock_id")), str(event.get("signal_date")))
        if event.get("lock_id") != active_lock_id:
            errors.append(f"event {index}: event lock_id does not match active lock")
        signal_date = pd.to_datetime(event.get("signal_date"), errors="coerce")
        if pd.isna(signal_date):
            errors.append(f"event {index}: invalid signal_date")
        elif not pd.isna(development_end) and signal_date <= development_end:
            errors.append(f"event {index}: signal_date is not after development end")
        event_type = event.get("event_type")
        if event_type == "FORECAST":
            if key in forecasts:
                errors.append(f"event {index}: duplicate forecast {key}")
            forecasts[key] = event
        elif event_type == "SETTLEMENT":
            if key not in forecasts:
                errors.append(f"event {index}: settlement without forecast {key}")
            if key in settlements:
                errors.append(f"event {index}: duplicate settlement {key}")
            settlements.add(key)
        else:
            errors.append(f"event {index}: unknown event_type {event_type}")
    if lock_verification.get("status") != "PASS":
        errors.extend(f"lock: {item}" for item in lock_verification.get("errors", []))
    return {
        "status": "PASS" if not errors else "FAIL",
        "events": len(events),
        "forecasts": len(forecasts),
        "settlements": len(settlements),
        "pending": len(forecasts) - len(settlements),
        "head_event_hash": previous_hash,
        "errors": errors,
        "lock_verification": lock_verification,
        "assurance": "LOCAL_TAMPER_EVIDENT_ONLY",
    }


def validate_market_input(input_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(input_path)
    required = {
        "date", "emotion_state", "emotion_adverse_markov_probability",
        "loss_effect_health", "promotion_mid", "down_limit_count",
        "source", "source_timestamp", "data_status", "emotion_data_completeness",
        *challenger.FEATURES,
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise RuntimeError(f"market emotion input missing columns: {missing}")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    if raw["date"].isna().any():
        raise RuntimeError("market emotion input contains invalid dates")
    if raw["date"].duplicated().any():
        raise RuntimeError("market emotion input contains duplicate dates")
    return raw.sort_values("date").reset_index(drop=True)


def validate_latest_timestamp(row: pd.Series, now: datetime) -> None:
    timestamp = pd.to_datetime(row.get("source_timestamp"), errors="coerce")
    if pd.isna(timestamp):
        raise RuntimeError("latest market row has no valid source_timestamp")
    timestamp_value = timestamp.to_pydatetime()
    if timestamp_value.tzinfo is None:
        timestamp_value = timestamp_value.replace(tzinfo=SHANGHAI)
    if timestamp_value > now:
        raise RuntimeError("latest market row source_timestamp is in the future")
    completeness = float(pd.to_numeric(row.get("emotion_data_completeness"), errors="coerce"))
    if not np.isfinite(completeness) or completeness < 0.75:
        raise RuntimeError("latest emotion data completeness is below 0.75")
    if str(row.get("data_status", "")).lower() in {"unavailable", "failed", ""}:
        raise RuntimeError("latest market row data_status is not usable")


def build_forecast_payload(
    input_path: Path,
    output_dir: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = now.astimezone(SHANGHAI) if now is not None else datetime.now(SHANGHAI)
    lock_check = verify_challenger_lock(output_dir)
    if lock_check["status"] != "PASS":
        raise RuntimeError("V4 lock verification failed: " + "; ".join(lock_check["errors"]))
    locked = load_lock(output_dir)
    raw = validate_market_input(input_path)
    latest = raw.iloc[-1]
    signal_date = latest["date"].date()
    development_end = pd.Timestamp(locked["development_data_end"]).date()
    if signal_date <= development_end:
        raise RuntimeError(f"forecast date {signal_date} is not later than locked development end {development_end}")
    if signal_date != current.date():
        raise RuntimeError(f"forecast must use today's latest market row; signal={signal_date}, today={current.date()}")
    if current.timetz().replace(tzinfo=None) < time(15, 5):
        raise RuntimeError("forecast registration is allowed only after 15:05 Asia/Shanghai")
    validate_latest_timestamp(latest, current)
    existing = read_events(output_dir)
    key = (locked["lock_id"], signal_date.isoformat())
    if any((event.get("lock_id"), event.get("signal_date")) == key
           and event.get("event_type") == "FORECAST" for event in existing):
        raise RuntimeError(f"forecast already registered for {key}")

    frame = challenger.add_targets(raw.assign(date=raw["date"].dt.strftime("%Y-%m-%d")))
    forecasts: dict[str, Any] = {}
    model_hashes: dict[str, str] = {}
    for target in challenger.TARGETS:
        probability, causal_base, latest_model = challenger.causal_predictions(frame, target)
        value = float(probability[-1])
        if not np.isfinite(value) or latest_model is None:
            raise RuntimeError(f"no finite locked forecast for {target}")
        forecasts[target] = {
            "probability": value,
            "causal_base": float(causal_base[-1]),
        }
        model_hashes[target] = sha256_value(latest_model)
    forecasts["adverse_state_next"]["comparison_baseline"] = float(
        pd.to_numeric(latest["emotion_adverse_markov_probability"], errors="coerce")
    )
    feature_snapshot = {
        feature: (None if pd.isna(latest.get(feature)) else float(latest.get(feature)))
        for feature in challenger.FEATURES
    }
    return {
        "event_type": "FORECAST",
        "lock_id": locked["lock_id"],
        "signal_date": signal_date.isoformat(),
        "recorded_at": current.isoformat(timespec="seconds"),
        "emotion_state": str(latest["emotion_state"]),
        "active_state": bool(str(latest["emotion_state"]) in challenger.ACTIVE_STATES),
        "forecasts": forecasts,
        "feature_snapshot": feature_snapshot,
        "model_hashes": model_hashes,
        "source": str(latest["source"]),
        "source_timestamp": str(latest["source_timestamp"]),
        "data_status": str(latest["data_status"]),
        "emotion_data_completeness": float(latest["emotion_data_completeness"]),
        "input_sha256": challenger.file_sha256(input_path),
        "challenger_script_sha256": lock_check["script_sha256"],
    }


def record_forecast(input_path: Path, output_dir: Path, now: datetime | None = None) -> dict[str, Any]:
    payload = build_forecast_payload(input_path, output_dir, now=now)
    return append_event(output_dir, payload)


def damage_label(row: pd.Series) -> float | None:
    loss = pd.to_numeric(row.get("loss_effect_health"), errors="coerce")
    mid = pd.to_numeric(row.get("promotion_mid"), errors="coerce")
    down = pd.to_numeric(row.get("down_limit_count"), errors="coerce")
    loss_known = bool(pd.notna(loss))
    ladder_known = bool(pd.notna(mid) and pd.notna(down))
    if (loss_known and float(loss) < 0.35) or (
        ladder_known and float(mid) < 0.20 and float(down) >= 5
    ):
        return 1.0
    if loss_known and ladder_known:
        return 0.0
    return None


def settle_forecasts(
    input_path: Path,
    output_dir: Path,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    current = now.astimezone(SHANGHAI) if now is not None else datetime.now(SHANGHAI)
    verification = verify_ledger(output_dir)
    if verification["status"] != "PASS":
        raise RuntimeError("cannot settle an invalid ledger: " + "; ".join(verification["errors"]))
    raw = validate_market_input(input_path)
    events = read_events(output_dir)
    forecasts = {
        (event["lock_id"], event["signal_date"]): event
        for event in events if event["event_type"] == "FORECAST"
    }
    settled = {
        (event["lock_id"], event["signal_date"])
        for event in events if event["event_type"] == "SETTLEMENT"
    }
    dates = raw["date"].dt.date.tolist()
    results = []
    for key, forecast in forecasts.items():
        if key in settled:
            continue
        signal = pd.Timestamp(forecast["signal_date"]).date()
        later_positions = [index for index, date_value in enumerate(dates) if date_value > signal]
        if not later_positions:
            continue
        outcome = raw.iloc[later_positions[0]]
        if outcome["date"].date() > current.date():
            continue
        validate_latest_timestamp(outcome, current)
        damage = damage_label(outcome)
        if damage is None:
            continue
        payload = {
            "event_type": "SETTLEMENT",
            "lock_id": forecast["lock_id"],
            "signal_date": forecast["signal_date"],
            "outcome_date": outcome["date"].date().isoformat(),
            "settled_at": current.isoformat(timespec="seconds"),
            "labels": {
                "adverse_state_next": float(str(outcome["emotion_state"]) in {"大分歧", "退潮"}),
                "relay_damage_next": damage,
            },
            "outcome_emotion_state": str(outcome["emotion_state"]),
            "outcome_source": str(outcome["source"]),
            "outcome_source_timestamp": str(outcome["source_timestamp"]),
            "outcome_data_status": str(outcome["data_status"]),
            "input_sha256": challenger.file_sha256(input_path),
        }
        results.append(append_event(output_dir, payload))
    return results


def metric_block(rows: pd.DataFrame, target: str, baseline_column: str) -> dict[str, Any]:
    if rows.empty:
        return {"n": 0}
    return challenger.binary_metrics(
        rows[f"{target}_label"].to_numpy(float),
        rows[f"{target}_probability"].to_numpy(float),
        rows[baseline_column].to_numpy(float),
    )


def forward_monitor(output_dir: Path) -> dict[str, Any]:
    verification = verify_ledger(output_dir)
    if verification["status"] != "PASS":
        return {"status": "INVALID_LEDGER", "verification": verification}
    events = read_events(output_dir)
    forecasts = {
        (event["lock_id"], event["signal_date"]): event
        for event in events if event["event_type"] == "FORECAST"
    }
    rows = []
    for event in events:
        if event["event_type"] != "SETTLEMENT":
            continue
        forecast = forecasts[(event["lock_id"], event["signal_date"])]
        rows.append({
            "signal_date": event["signal_date"],
            "active_state": bool(forecast["active_state"]),
            "adverse_state_next_label": float(event["labels"]["adverse_state_next"]),
            "adverse_state_next_probability": float(forecast["forecasts"]["adverse_state_next"]["probability"]),
            "adverse_markov_baseline": float(forecast["forecasts"]["adverse_state_next"]["comparison_baseline"]),
            "relay_damage_next_label": float(event["labels"]["relay_damage_next"]),
            "relay_damage_next_probability": float(forecast["forecasts"]["relay_damage_next"]["probability"]),
            "relay_damage_base": float(forecast["forecasts"]["relay_damage_next"]["causal_base"]),
        })
    frame = pd.DataFrame(rows)
    active = frame[frame["active_state"]].copy() if not frame.empty else frame.copy()
    all_metrics = {
        "adverse_state_next": metric_block(frame, "adverse_state_next", "adverse_markov_baseline"),
        "relay_damage_next": metric_block(frame, "relay_damage_next", "relay_damage_base"),
    }
    active_metrics = {
        "adverse_state_next": metric_block(active, "adverse_state_next", "adverse_markov_baseline"),
        "relay_damage_next": metric_block(active, "relay_damage_next", "relay_damage_base"),
    }
    enough = len(frame) >= 60 and len(active) >= 30
    stable = enough and all(
        metrics.get("brier_skill_vs_baseline", -math.inf) > 0
        and metrics.get("log_loss_improvement_vs_baseline", -math.inf) > 0
        and (metrics.get("auc") or 0) > 0.5
        for group in (all_metrics, active_metrics) for metrics in group.values()
    )
    status = "FORWARD_CANDIDATE" if stable else (
        "FORWARD_REJECTED" if enough else "INSUFFICIENT_FORWARD_EVIDENCE"
    )
    report = {
        "status": status,
        "lock_id": verification["lock_verification"].get("lock_id"),
        "registered_forecasts": len(forecasts),
        "resolved_forecasts": len(frame),
        "active_resolved_forecasts": len(active),
        "pending_forecasts": len(forecasts) - len(frame),
        "all_dates": all_metrics,
        "active_states": active_metrics,
        "forward_gate": {
            "passed": stable,
            "rule": ">=60 resolved, >=30 active-state resolved; both targets in both views require positive Brier skill, lower log loss and AUC>0.5",
            "does_not_imply_trading_deployment": True,
        },
        "verification": verification,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "emotion_forward_monitor.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=challenger.json_default), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Tamper-evident forward evidence for locked V4 emotion forecasts")
    parser.add_argument("command", choices=("record", "settle", "verify", "monitor"))
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        if args.command == "record":
            result = record_forecast(args.input, args.output)
        elif args.command == "settle":
            result = {"settled": settle_forecasts(args.input, args.output)}
        elif args.command == "verify":
            result = verify_ledger(args.output)
        else:
            result = forward_monitor(args.output)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2, default=challenger.json_default))


if __name__ == "__main__":
    main()
