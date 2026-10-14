"""Calibrate a next-session defensive exposure gate on chronological data."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


RISK_OFF_STATES = ("大分歧", "退潮")
RISK_ON_STATES = ("启动", "升温", "加速", "一致")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evaluate(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    # Missing a risk-off day is twice as costly as becoming defensive unnecessarily.
    weighted_error = (2 * fn + fp) / len(y) if len(y) else None
    return {
        "n": int(len(y)), "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "false_positive_rate": fpr,
        "weighted_error_2fn_plus_fp_per_observation": weighted_error,
        "defensive_rate": float(pred.mean()) if len(pred) else None,
    }


def calibrate(predictions: pd.DataFrame, development_fraction: float = .70) -> dict:
    required = {"signal_date", "target_date", "target_state", "p_大分歧", "p_退潮"}
    if required - set(predictions):
        raise ValueError("walk-forward prediction fields missing")
    frame = predictions.sort_values("signal_date").reset_index(drop=True).copy()
    if len(frame) < 100:
        raise ValueError("insufficient walk-forward observations")
    split = int(len(frame) * development_fraction)
    if split <= 0 or split >= len(frame):
        raise ValueError("invalid chronological split")
    frame["risk_off_probability"] = frame["p_大分歧"] + frame["p_退潮"]
    frame["risk_off_actual"] = frame.target_state.isin(RISK_OFF_STATES)
    development, validation = frame.iloc[:split], frame.iloc[split:]
    candidates = []
    for threshold in np.arange(.30, .701, .01):
        metrics = _evaluate(
            development.risk_off_actual.to_numpy(bool),
            development.risk_off_probability.to_numpy(float), float(round(threshold, 2)))
        # Avoid a trivial always-defensive policy.
        if metrics["defensive_rate"] <= .80:
            candidates.append(metrics)
    if not candidates:
        raise ValueError("no non-trivial threshold candidate")
    selected = min(candidates, key=lambda item: (
        item["weighted_error_2fn_plus_fp_per_observation"],
        -item["recall"], item["defensive_rate"], -item["threshold"]))
    threshold = selected["threshold"]
    validation_metrics = _evaluate(
        validation.risk_off_actual.to_numpy(bool),
        validation.risk_off_probability.to_numpy(float), threshold)
    always_metrics = _evaluate(
        validation.risk_off_actual.to_numpy(bool),
        np.ones(len(validation)), .5)
    advantage = (always_metrics["weighted_error_2fn_plus_fp_per_observation"]
                 - validation_metrics["weighted_error_2fn_plus_fp_per_observation"])
    accepted = bool(advantage > 0 and validation_metrics["false_positive_rate"] <= .50)
    return {
        "split": {
            "method": "chronological_70_30",
            "development_n": int(len(development)), "validation_n": int(len(validation)),
            "development_end_target_date": str(development.iloc[-1].target_date),
            "validation_start_target_date": str(validation.iloc[0].target_date),
        },
        "loss_function": "2*false_negative_risk_off + 1*false_positive_defensive",
        "selected_threshold": threshold,
        "development": selected,
        "validation": validation_metrics,
        "always_defensive_validation": always_metrics,
        "validation_weighted_error_advantage_vs_always_defensive": advantage,
        "gate_accepted": accepted,
        "gate_status": "ACCEPTED_VALIDATION_ADVANTAGE" if accepted else "REJECTED_NO_VALIDATION_ADVANTAGE",
    }


def apply_latest(calibration: dict, forecast_report: dict) -> dict:
    latest = forecast_report["latest_prediction"]
    threshold = float(calibration["selected_threshold"])
    probability = float(latest["group_probabilities"]["RISK_OFF"])
    defensive_lean = probability >= threshold
    accepted = bool(calibration.get("gate_accepted"))
    return {
        "signal_date": latest["signal_date"],
        "risk_off_probability": probability,
        "defensive_threshold": threshold,
        "gate_status": calibration.get("gate_status"),
        "exposure_regime": ("DEFENSIVE" if defensive_lean else "NOT_DEFENSIVE")
        if accepted else "DIAGNOSTIC_LEAN_ONLY",
        "diagnostic_lean": "DEFENSIVE" if defensive_lean else "NOT_DEFENSIVE",
        "high_board_relay": "DISABLED" if accepted and defensive_lean else "UNDECIDED_BY_MODEL",
        "middle_board_relay": "DISABLED" if accepted and defensive_lean else "UNDECIDED_BY_MODEL",
        "first_board": "WATCH_ONLY" if accepted and defensive_lean else "UNDECIDED_BY_MODEL",
        "default_position": 0,
        "direct_buy_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--forecast-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    predictions_path = Path(args.predictions)
    forecast_path = Path(args.forecast_report)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable calibration directory")
    calibration = calibrate(pd.read_csv(predictions_path))
    forecast_report = json.loads(forecast_path.read_text(encoding="utf-8"))
    result = {
        "version": "v85-emotion-exposure-calibration-2",
        "status": "DIAGNOSTIC_ONLY",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "calibration": calibration,
        "latest_decision": apply_latest(calibration, forecast_report),
        "limitations": [
            "the threshold is selected once on development data and evaluated once on later validation data",
            "the gate predicts aggregate market risk, not an individual stock return",
            "no position is authorised without a separately frozen, executable stock plan",
        ],
        "sources": {
            "predictions": {"path": str(predictions_path.resolve()), "sha256": sha256(predictions_path)},
            "forecast_report": {"path": str(forecast_path.resolve()), "sha256": sha256(forecast_path)},
        },
    }
    output.mkdir(parents=True)
    report = output / "report.json"
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
