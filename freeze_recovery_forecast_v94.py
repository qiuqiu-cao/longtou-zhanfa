"""Freeze a risk-off recovery research forecast before target pre-open."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(report_path: Path, target_date: str, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("refusing to overwrite frozen forecast")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    latest = report["latest_research_prediction"]
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    target = date.fromisoformat(target_date)
    if target_date <= latest["signal_date"]:
        raise ValueError("target_date must be after signal_date")
    if now.date() > target or (now.date() == target and now.time().isoformat() >= "09:15:00"):
        raise ValueError("refusing to freeze after target-session pre-open boundary")
    result = {
        "version": "v94-riskoff-recovery-freeze-1",
        "status": "FROZEN_FORWARD_RESEARCH_ONLY",
        "model_version": report["version"],
        "historical_gate_passed": bool(report["promotion_gate_passed"]),
        "signal_date": latest["signal_date"], "target_date": target_date,
        "frozen_at": now.isoformat(), "current_state": latest["current_state"],
        "recovery_probability": latest["recovery_probability"],
        "continued_risk_off_probability": latest["continued_risk_off_probability"],
        "top_class": latest["top_class"],
        "decision_authorized": False, "position": 0,
        "outcome": None, "outcome_status": "UNAVAILABLE_BEFORE_TARGET_CLOSE",
        "source_report": str(report_path.resolve()), "source_report_sha256": sha256(report_path),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(Path(args.report), args.target_date, Path(args.output)),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
