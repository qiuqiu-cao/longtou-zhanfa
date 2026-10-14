"""Combine point-estimate and uncertainty gates into an immutable model verdict."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def govern(model_path: Path, uncertainty_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable governance verdict")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    uncertainty = json.loads(uncertainty_path.read_text(encoding="utf-8"))
    point_pass = bool(model["promotion_gate_passed"])
    stable_pass = bool(uncertainty["stable_gate_passed"])
    approved = point_pass and stable_pass
    latest = model["latest_research_prediction"]
    result = {
        "version": "v96-recovery-model-governance-1",
        "status": "HISTORICAL_GATE_APPROVED_FORWARD_REQUIRED" if approved
                  else "REJECTED_UNCERTAINTY_GATE",
        "governed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "point_estimate_gate_passed": point_pass,
        "uncertainty_stability_gate_passed": stable_pass,
        "effective_historical_gate_passed": approved,
        "current_prediction_retained_for_audit": latest,
        "decision_authorized": False,
        "deployment_allowed": False,
        "forward_requirements": {
            "minimum_applicable_predictions": 60,
            "minimum_recovery_outcomes": 20,
            "required_brier_gain_lower_ci": 0.0,
            "required_log_loss_gain_lower_ci": 0.0,
            "required_auc_lower_ci": 0.5,
        },
        "sources": {
            "model": {"path": str(model_path.resolve()), "sha256": sha256(model_path)},
            "uncertainty": {"path": str(uncertainty_path.resolve()), "sha256": sha256(uncertainty_path)},
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--uncertainty", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(govern(Path(args.model), Path(args.uncertainty), Path(args.output)),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
