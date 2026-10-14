"""Paired block-bootstrap uncertainty audit for the recovery challenger."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _draw_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    return np.concatenate([(start + np.arange(block)) % n for start in starts])[:n]


def audit(frame: pd.DataFrame, repetitions: int = 2000, block_length: int = 5,
          seed: int = 0) -> dict:
    required = {"recovery_actual", "challenger_recovery_probability", "prior_recovery_probability"}
    if required - set(frame):
        raise ValueError("recovery blind prediction fields missing")
    y = frame.recovery_actual.astype(str).str.lower().isin({"true", "1", "1.0"}).to_numpy(bool)
    challenger = pd.to_numeric(frame.challenger_recovery_probability, errors="coerce").to_numpy(float)
    baseline = pd.to_numeric(frame.prior_recovery_probability, errors="coerce").to_numpy(float)
    valid = np.isfinite(challenger) & np.isfinite(baseline)
    y, challenger, baseline = y[valid], challenger[valid], baseline[valid]
    if len(y) < 50 or len(set(y)) < 2:
        raise ValueError("insufficient valid binary blind observations")

    def summary(indices: np.ndarray) -> tuple[float, float, float]:
        yy, cp, bp = y[indices], challenger[indices], baseline[indices]
        brier_gain = np.mean((bp - yy) ** 2) - np.mean((cp - yy) ** 2)
        c_log = -np.mean(yy * np.log(np.clip(cp, 1e-12, 1)) + (~yy) * np.log(np.clip(1 - cp, 1e-12, 1)))
        b_log = -np.mean(yy * np.log(np.clip(bp, 1e-12, 1)) + (~yy) * np.log(np.clip(1 - bp, 1e-12, 1)))
        auc = roc_auc_score(yy, cp) if len(set(yy)) == 2 else np.nan
        return float(brier_gain), float(b_log - c_log), float(auc)

    point = summary(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    draws = np.array([summary(_draw_indices(len(y), min(tar, len(y)), rng))
                      for _ in range(repetitions) for tar in [block_length]])
    valid_auc = draws[:, 2][np.isfinite(draws[:, 2])]
    brier_ci = np.quantile(draws[:, 0], [.025, .975])
    log_ci = np.quantile(draws[:, 1], [.025, .975])
    auc_ci = np.quantile(valid_auc, [.025, .975])
    stable = bool(brier_ci[0] > 0 and log_ci[0] > 0 and auc_ci[0] > .5)
    return {
        "n": int(len(y)), "repetitions": repetitions, "block_length": min(block_length, len(y)),
        "point_estimates": {"brier_improvement_vs_prior": point[0],
                            "log_loss_improvement_vs_prior": point[1], "auc": point[2]},
        "ci95": {"brier_improvement_vs_prior": [float(x) for x in brier_ci],
                 "log_loss_improvement_vs_prior": [float(x) for x in log_ci],
                 "auc": [float(x) for x in auc_ci]},
        "bootstrap_probability_brier_improves": float((draws[:, 0] > 0).mean()),
        "bootstrap_probability_log_loss_impro": float((draws[:, 1] > 0).mean()),
        "stable_gate_passed": stable,
        "stable_gate_rule": "lower 95% bounds: Brier gain>0, log-loss gain>0, AUC>0.5",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.predictions), Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable uncertainty audit")
    report = {
        "version": "v95-recovery-uncertainty-audit-1", "status": "DIAGNOSTIC_ONLY",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        **audit(pd.read_csv(source)),
        "source": {"path": str(source.resolve()), "sha256": sha256(source)},
        "limitations": ["block bootstrap is descriptive for serial dependence",
                        "forward validation remains required even if the historical stability gate passes"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
