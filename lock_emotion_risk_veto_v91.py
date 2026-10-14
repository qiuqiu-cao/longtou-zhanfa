"""Freeze a paper-only emotion risk-veto policy after nested blind validation."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lock(economic_path: Path, ensemble_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable policy lock")
    economic = json.loads(economic_path.read_text(encoding="utf-8"))
    ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
    if not economic.get("economic_gate_passed"):
        raise ValueError("economic validation gate did not pass")
    threshold = float(economic["fixed_high_risk_cutoff"])
    latest = ensemble["latest_research_prediction"]
    probability = float(latest["probabilities"]["RISK_OFF"])
    result = {
        "version": "v91-emotion-risk-veto-paper-lock-1",
        "status": "FROZEN_PAPER_VALIDATION_ONLY",
        "locked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "risk_off_threshold": threshold,
        "model": {
            "ensemble_version": ensemble["version"],
            "challenger_weight": economic["selected_challenger_weight"],
            "markov_weight": 1 - economic["selected_challenger_weight"],
        },
        "paper_rule": {
            "trigger": "RISK_OFF probability >= fixed threshold",
            "effect": {
                "high_board_relay": "VETO",
                "middle_board_relay": "VETO",
                "first_board": "WATCH_ONLY_UNLESS_SEPARATE_RULE_VERSION_IS_VALIDATED",
            },
            "never_creates_buy_signal": True,
            "missing_probability_action": "VETO_FAIL_CLOSED",
        },
        "forward_gate": {
            "minimum_predictions": 60,
            "required_high_risk_events": 15,
            "required_direction_replication": ["failed_rate_higher", "tail_loss_lower", "down_limit_count_higher"],
            "status": "NOT_STARTED",
        },
        "current_shadow_application": {
            "signal_date": latest["signal_date"], "risk_off_probability": probability,
            "triggered": probability >= threshold,
            "status": "SHADOW_ONLY_NO_TRADE_AUTHORITY",
        },
        "evidence": {
            "economic_audit": {"path": str(economic_path.resolve()), "sha256": sha256(economic_path)},
            "ensemble_report": {"path": str(ensemble_path.resolve()), "sha256": sha256(ensemble_path)},
            "outer_blind_n": economic["nested_time_integrity"]["outer_blind_n"],
            "high_risk_precision": economic["outer_blind_risk_off_detection"]["precision"],
        },
        "deployment_allowed": False,
        "guarantee": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--economic-audit", required=True)
    parser.add_argument("--ensemble-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = lock(Path(args.economic_audit), Path(args.ensemble_report), Path(args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
