"""Freeze one next-session emotion forecast before its outcome is known."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(report_path: Path, output: Path, target_date: str) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    forecast = report["latest_prediction"]
    signal_date = forecast["signal_date"]
    if target_date <= signal_date:
        raise ValueError("target_date must be after signal_date")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    target = date.fromisoformat(target_date)
    if now.date() > target or (now.date() == target and now.time().isoformat() >= "09:15:00"):
        raise ValueError("refusing to freeze after target-session pre-open boundary")
    if output.exists():
        raise FileExistsError("refusing to overwrite frozen forecast")
    payload = {
        "version": "v83-emotion-forecast-freeze-1",
        "status": "FROZEN_FORWARD_DIAGNOSTIC",
        "signal_date": signal_date,
        "target_date": target_date,
        "frozen_at": now.isoformat(),
        "current_state": forecast["current_state"],
        "probabilities": forecast["probabilities"],
        "adverse_probability": forecast["adverse_probability"],
        "supportive_probability": forecast["supportive_probability"],
        "direction_call": forecast["direction_call"],
        "decision_use": {
            "high_board_relay": "DISABLED" if forecast["adverse_probability"] >= .5 else "CONDITIONAL",
            "middle_board_relay": "DISABLED" if forecast["adverse_probability"] >= .5 else "CONDITIONAL",
            "first_board": "WATCH_ONLY" if forecast["adverse_probability"] >= .5 else "CONDITIONAL",
            "default_position": 0,
        },
        "model_status": report["status"],
        "model_method": report["method"],
        "historical_evaluation": report["evaluation"],
        "source_report": str(report_path.resolve()),
        "source_report_sha256": sha256(report_path),
        "outcome": None,
        "outcome_status": "UNAVAILABLE_BEFORE_TARGET_CLOSE",
        "guarantee": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = freeze(Path(args.report), Path(args.output), args.target_date)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
