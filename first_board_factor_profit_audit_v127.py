"""Descriptive profit audit for predeclared first-board speed/control factors."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


FACTORS = ["fast_seal", "thin_control_risk", "validated_fast_seal", "one_price_proxy"]


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n == 0:
        return None
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [center - radius, center + radius]


def describe(part: pd.DataFrame) -> dict:
    b1 = part.net_return_B_plus_1.dropna()
    big = part.big_loss_B_plus_1.dropna()
    wins = int(b1.gt(0).sum())
    return {
        "n_B_plus_1": int(len(b1)),
        "mean_net_return_B_plus_1": float(b1.mean()) if len(b1) else None,
        "median_net_return_B_plus_1": float(b1.median()) if len(b1) else None,
        "positive_rate_B_plus_1": float(wins / len(b1)) if len(b1) else None,
        "positive_rate_wilson95": wilson(wins, len(b1)),
        "big_loss_rate_B_plus_1": float(big.mean()) if len(big) else None,
        "mean_net_return_B_plus_2": float(part.net_return_B_plus_2.dropna().mean())
        if part.net_return_B_plus_2.notna().any() else None,
        "mean_net_return_B_plus_3": float(part.net_return_B_plus_3.dropna().mean())
        if part.net_return_B_plus_3.notna().any() else None,
    }


def audit(features: pd.DataFrame, outcomes: pd.DataFrame) -> dict:
    outcome_fields = [c for c in outcomes.columns if c not in {"name", "promotion_label"}]
    joined = features.merge(outcomes[outcome_fields], on=["signal_date", "code"],
                            how="left", validate="one_to_one")
    filled = joined[joined.fill_status.eq("FILLED_OPEN_PROXY")].copy()
    groups = {}
    for factor in FACTORS:
        groups[factor] = {
            "factor_0": describe(filled[filled[factor].eq(0)]),
            "factor_1": describe(filled[filled[factor].eq(1)]),
        }
    return {"filled_rows": int(len(filled)), "groups": groups}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite immutable v127 output")
    features = pd.read_csv(args.features, dtype={"code": str})
    outcomes = pd.read_csv(args.outcomes, dtype={"code": str})
    result = audit(features, outcomes)
    report = {
        "version": "v127-first-board-factor-profit-audit-1",
        "status": "DIAGNOSTIC_ONLY",
        **result,
        "interpretation_guardrail": (
            "These are descriptive development-sample groups, not causal effects or deployment thresholds."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
