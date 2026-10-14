"""Audit whether frozen blind emotion probabilities predict tradable next-day damage."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


OUTCOMES = {
    "promotion_mid": -1,
    "yesterday_premium": -1,
    "failed_rate": 1,
    "previous_limit_tail_loss": -1,
    "down_limit_count": 1,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def block_bootstrap_difference(high: np.ndarray, low: np.ndarray, repetitions: int = 1000,
                               block_length: int = 5, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    high, low = high[np.isfinite(high)], low[np.isfinite(low)]
    if len(high) < 10 or len(low) < 10:
        return {"difference_high_minus_low": None, "ci95": [None, None], "status": "INSUFFICIENT"}

    def sample_blocks(values: np.ndarray) -> np.ndarray:
        starts = rng.integers(0, len(values), size=int(np.ceil(len(values) / block_length)))
        sampled = np.concatenate([
            values[(start + np.arange(block_length)) % len(values)] for start in starts
        ])
        return sampled[:len(values)]

    differences = np.array([
        sample_blocks(high).mean() - sample_blocks(low).mean() for _ in range(repetitions)
    ])
    return {
        "difference_high_minus_low": float(high.mean() - low.mean()),
        "ci95": [float(np.quantile(differences, .025)), float(np.quantile(differences, .975))],
        "status": "BLOCK_BOOTSTRAP_DESCRIPTIVE",
    }


def audit(predictions: pd.DataFrame, market: pd.DataFrame) -> dict:
    required_predictions = {"target_date", "ensemble_p_RISK_OFF"}
    if required_predictions - set(predictions):
        raise ValueError("ensemble blind prediction fields missing")
    required_market = {"date", *OUTCOMES}
    if required_market - set(market):
        raise ValueError("market outcome fields missing")
    pred = predictions.copy()
    pred["target_date"] = pd.to_datetime(pred.target_date, errors="coerce").dt.strftime("%Y-%m-%d")
    actual = market.copy()
    actual["target_date"] = pd.to_datetime(actual.date, errors="coerce").dt.strftime("%Y-%m-%d")
    for column in OUTCOMES:
        actual[column] = pd.to_numeric(actual[column], errors="coerce")
    joined = pred.merge(actual[["target_date", *OUTCOMES]], on="target_date", how="left", validate="one_to_one")
    if joined[list(OUTCOMES)].notna().any(axis=1).mean() < .95:
        raise ValueError("market outcome coverage below 95%")
    probability = pd.to_numeric(joined.ensemble_p_RISK_OFF, errors="coerce")
    low_cut, high_cut = probability.quantile(.20), probability.quantile(.80)
    low_mask, high_mask = probability.le(low_cut), probability.ge(high_cut)
    outcomes = {}
    sign_passes = 0
    strong_passes = 0
    for column, expected_sign in OUTCOMES.items():
        valid = joined[column].notna() & probability.notna()
        high = joined.loc[high_mask & valid, column].to_numpy(float)
        low = joined.loc[low_mask & valid, column].to_numpy(float)
        difference = block_bootstrap_difference(high, low)
        observed = difference["difference_high_minus_low"]
        sign_pass = observed is not None and np.sign(observed) == expected_sign
        ci = difference["ci95"]
        strong = sign_pass and ci[0] is not None and (ci[0] > 0 if expected_sign > 0 else ci[1] < 0)
        sign_passes += int(sign_pass)
        strong_passes += int(strong)
        outcomes[column] = {
            "expected_high_risk_direction": "HIGHER" if expected_sign > 0 else "LOWER",
            "high_risk_n": int(len(high)), "low_risk_n": int(len(low)),
            "high_risk_mean": float(np.mean(high)) if len(high) else None,
            "low_risk_mean": float(np.mean(low)) if len(low) else None,
            **difference,
            "spearman_probability_vs_outcome": float(joined.loc[valid, ["ensemble_p_RISK_OFF", column]].corr(method="spearman").iloc[0, 1]),
            "direction_pass": bool(sign_pass), "ci_excludes_zero_in_expected_direction": bool(strong),
        }
    return {
        "joined_n": int(len(joined)), "outcome_coverage": float(joined[list(OUTCOMES)].notna().any(axis=1).mean()),
        "low_risk_cutoff_20pct": float(low_cut), "high_risk_cutoff_80pct": float(high_cut),
        "outcomes": outcomes, "direction_pass_count": sign_passes,
        "strong_ci_pass_count": strong_passes,
        "economic_gate_passed": bool(sign_passes >= 4 and strong_passes >= 2),
        "gate_rule": "expected direction in >=4/5 outcomes and block-bootstrap CI excludes zero in >=2/5",
        "joined": joined,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--market", default="data/v3/market_emotion.csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pred_path, market_path, output = Path(args.predictions), Path(args.market), Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable audit directory")
    result = audit(pd.read_csv(pred_path), pd.read_csv(market_path))
    joined = result.pop("joined")
    output.mkdir(parents=True)
    joined_path = output / "joined_blind_outcomes.csv"
    joined.to_csv(joined_path, index=False, encoding="utf-8-sig")
    report = {
        "version": "v89-emotion-economic-value-audit-1",
        "status": "DIAGNOSTIC_ONLY",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        **result,
        "limitations": [
            "outcomes are aggregate daily market measures, not executable trade returns",
            "block-bootstrap intervals are descriptive and do not correct for all model-selection effects",
            "a passed economic gate would still require forward prediction and complete-trade validation",
        ],
        "sources": {
            "predictions": {"path": str(pred_path.resolve()), "sha256": sha256(pred_path)},
            "market": {"path": str(market_path.resolve()), "sha256": sha256(market_path)},
            "joined": {"path": str(joined_path.resolve()), "sha256": sha256(joined_path)},
        },
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
