"""Paired proper-score audit of saved repair predictions; no parameter tuning."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from emotion_repair_quality_v118 import CLASSES


def evaluate(rows, repeats=2000):
    rows = rows.sort_values("signal_date").reset_index(drop=True)
    if rows.empty or rows.signal_date.duplicated().any():
        raise ValueError("nonempty unique predictions required")
    if not (pd.to_datetime(rows.target_date) > pd.to_datetime(rows.signal_date)).all():
        raise ValueError("invalid forecast time ordering")
    if not rows.target_class.isin(CLASSES).all():
        raise ValueError("unknown outcome class")
    truth = np.eye(len(CLASSES))[rows.target_class.map(CLASSES.index)]
    losses = {}
    for prefix in ("challenger", "prior"):
        p = rows[[f"{prefix}_p_{c}" for c in CLASSES]].to_numpy(float)
        if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1):
            raise ValueError("invalid probabilities")
        losses[prefix] = np.sum((p - truth)**2, axis=1)
    delta = losses["challenger"] - losses["prior"]
    rng = np.random.default_rng(137)
    intervals = {}
    for block in (5, 20):
        samples = []
        for _ in range(repeats):
            starts = rng.integers(0, len(rows), size=int(np.ceil(len(rows)/block)))
            indexes = ((starts[:, None]+np.arange(block)) % len(rows)).ravel()[:len(rows)]
            samples.append(float(delta[indexes].mean()))
        intervals[str(block)] = np.quantile(samples, [.025, .975]).tolist()
    return dict(status="DIAGNOSTIC_ONLY", sessions=len(rows),
                challenger_brier=float(losses["challenger"].mean()),
                prior_brier=float(losses["prior"].mean()),
                challenger_minus_prior_brier=float(delta.mean()),
                lower_is_better=True, paired_block_bootstrap_95=intervals,
                block_unit="consecutive recorded risk-off observations, not consecutive calendar sessions",
                deployment_authorized=False, complete_trade_count=0,
                limitations=["Previously inspected research holdout, not a new blind test",
                             "Outcome is next-close market repair, not intraday sequence or trade profit"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(pd.read_csv(args.input))
    result["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))
