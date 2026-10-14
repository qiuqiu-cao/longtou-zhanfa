"""Validate an emotion risk cutoff chosen only inside the training period."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from emotion_economic_value_audit_v89 import OUTCOMES, block_bootstrap_difference
from emotion_regime_ensemble_v87 import choose_weight
from emotion_three_regime_challenger_v86 import (
    FEATURES, make_model, markov_probabilities, ordered_probabilities, prepare,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(frame: pd.DataFrame, outer_fraction: float = .70, inner_fraction: float = .70) -> tuple[dict, pd.DataFrame]:
    data = prepare(frame)
    for column in OUTCOMES:
        # Economic consequences belong to the target session, never the signal row.
        data[f"target_{column}"] = pd.to_numeric(data[column], errors="coerce").shift(-1)
    labeled = data.iloc[:-1].copy()
    outer_split = int(len(labeled) * outer_fraction)
    outer_train, blind = labeled.iloc[:outer_split], labeled.iloc[outer_split:]
    inner_split = int(len(outer_train) * inner_fraction)
    inner_train, inner_validation = outer_train.iloc[:inner_split], outer_train.iloc[inner_split:]

    inner_model = make_model()
    inner_model.fit(inner_train[FEATURES], inner_train.target_class)
    inner_challenger = ordered_probabilities(inner_model, inner_validation[FEATURES])
    inner_markov = markov_probabilities(inner_validation)
    weight, _ = choose_weight(inner_validation.target_class, inner_challenger, inner_markov)
    inner_risk = weight * inner_challenger[:, 1] + (1 - weight) * inner_markov[:, 1]
    low_cutoff, high_cutoff = float(np.quantile(inner_risk, .20)), float(np.quantile(inner_risk, .80))

    outer_model = make_model()
    outer_model.fit(outer_train[FEATURES], outer_train.target_class)
    blind_challenger = ordered_probabilities(outer_model, blind[FEATURES])
    blind_markov = markov_probabilities(blind)
    blind_risk = weight * blind_challenger[:, 1] + (1 - weight) * blind_markov[:, 1]
    low_mask, high_mask = blind_risk <= low_cutoff, blind_risk >= high_cutoff
    outcomes, direction_passes, strong_passes = {}, 0, 0
    for column, expected_sign in OUTCOMES.items():
        values = pd.to_numeric(blind[f"target_{column}"], errors="coerce").to_numpy(float)
        high, low = values[high_mask & np.isfinite(values)], values[low_mask & np.isfinite(values)]
        interval = block_bootstrap_difference(high, low)
        difference = interval["difference_high_minus_low"]
        direction = difference is not None and np.sign(difference) == expected_sign
        ci = interval["ci95"]
        strong = direction and ci[0] is not None and (ci[0] > 0 if expected_sign > 0 else ci[1] < 0)
        direction_passes += int(direction)
        strong_passes += int(strong)
        outcomes[column] = {
            "expected_high_risk_direction": "HIGHER" if expected_sign > 0 else "LOWER",
            "high_risk_n": int(len(high)), "low_risk_n": int(len(low)),
            "high_risk_mean": float(high.mean()) if len(high) else None,
            "low_risk_mean": float(low.mean()) if len(low) else None,
            **interval, "direction_pass": bool(direction),
            "ci_excludes_zero_in_expected_direction": bool(strong),
        }
    actual_risk_off = blind.target_class.eq("RISK_OFF").to_numpy(bool)
    tp = int((high_mask & actual_risk_off).sum())
    fp = int((high_mask & ~actual_risk_off).sum())
    fn = int((~high_mask & actual_risk_off).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    gate = bool(direction_passes >= 4 and strong_passes >= 2 and len(blind_risk[high_mask]) >= 20)
    evidence = pd.DataFrame({
        "signal_date": blind.date.dt.strftime("%Y-%m-%d"),
        "target_date": blind.target_date.dt.strftime("%Y-%m-%d"),
        "target_class": blind.target_class,
        "risk_off_probability": blind_risk,
        "fixed_high_risk": high_mask,
        "fixed_low_risk": low_mask,
    })
    for column in OUTCOMES:
        evidence[column] = blind[f"target_{column}"].to_numpy()
    report = {
        "version": "v90-emotion-fixed-cutoff-audit-1", "status": "DIAGNOSTIC_ONLY",
        "nested_time_integrity": {
            "inner_train_n": int(len(inner_train)), "inner_validation_n": int(len(inner_validation)),
            "outer_blind_n": int(len(blind)),
            "cutoffs_selected_from": "inner_validation_predictions_only",
            "outer_blind_used_for_selection": False,
        },
        "selected_challenger_weight": weight,
        "fixed_low_risk_cutoff": low_cutoff, "fixed_high_risk_cutoff": high_cutoff,
        "outer_blind_group_counts": {"high": int(high_mask.sum()), "low": int(low_mask.sum())},
        "outer_blind_risk_off_detection": {"tp": tp, "fp": fp, "fn": fn,
                                            "precision": precision, "recall": recall},
        "outcomes": outcomes, "direction_pass_count": direction_passes,
        "strong_ci_pass_count": strong_passes, "economic_gate_passed": gate,
        "gate_rule": "fixed inner cutoff; expected direction >=4/5, CI support >=2/5, high-risk n>=20",
        "allowed_use_if_passed": "RISK_VETO_RESEARCH_CANDIDATE_ONLY",
        "prohibited_use": ["positive buy signal", "individual-stock ranking", "claim of guaranteed premium"],
    }
    return report, evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/v3/market_emotion.csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.input), Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable audit directory")
    report, evidence = run(pd.read_csv(source))
    output.mkdir(parents=True)
    evidence_path = output / "outer_blind_fixed_groups.csv"
    evidence.to_csv(evidence_path, index=False, encoding="utf-8-sig")
    report["generated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    report["source"] = {"path": str(source.resolve()), "sha256": sha256(source)}
    report["evidence"] = {"path": str(evidence_path.resolve()), "sha256": sha256(evidence_path)}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
