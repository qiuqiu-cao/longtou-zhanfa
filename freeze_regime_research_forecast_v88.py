"""Freeze a three-regime research forecast before the target session opens."""
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
        raise FileExistsError("refusing to overwrite frozen research forecast")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    latest = report["latest_research_prediction"]
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    target = date.fromisoformat(target_date)
    if target_date <= latest["signal_date"]:
        raise ValueError("target_date must be after signal_date")
    if now.date() > target or (now.date() == target and now.time().isoformat() >= "09:15:00"):
        raise ValueError("refusing to freeze after target-session pre-open boundary")
    probabilities = latest["probabilities"]
    if set(probabilities) != {"RISK_ON", "RISK_OFF", "REPAIR"}:
        raise ValueError("three regime probabilities are incomplete")
    if abs(sum(float(value) for value in probabilities.values()) - 1) > 1e-9:
        raise ValueError("probabilities do not sum to one")
    payload = {
        "version": "v88-regime-research-forecast-freeze-1",
        "status": "FROZEN_FORWARD_RESEARCH_ONLY",
        "model_version": report["version"],
        "model_promotion_gate_passed": bool(report["promotion_gate_passed"]),
        "signal_date": latest["signal_date"], "target_date": target_date,
        "frozen_at": now.isoformat(), "probabilities": probabilities,
        "top_class": latest["top_class"],
        "decision_authorized": False, "position": 0,
        "outcome": None, "outcome_status": "UNAVAILABLE_BEFORE_TARGET_CLOSE",
        "source_report": str(report_path.resolve()),
        "source_report_sha256": sha256(report_path),
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
    result = freeze(Path(args.report), args.target_date, Path(args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
