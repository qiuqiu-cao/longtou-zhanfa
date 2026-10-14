"""Untuned paired comparison against persistence using saved OOS predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from emotion_multiclass_audit_v82 import STATES


def evaluate(rows, repetitions=2000):
    rows = rows.sort_values("signal_date").reset_index(drop=True)
    if rows.empty or rows.signal_date.duplicated().any():
        raise ValueError("require nonempty unique daily predictions")
    if not (pd.to_datetime(rows.target_date) > pd.to_datetime(rows.signal_date)).all():
        raise ValueError("target must follow signal")
    if not rows.current_state.isin(STATES).all() or not rows.target_state.isin(STATES).all():
        raise ValueError("unknown states")
    scores = rows[["p_" + s for s in STATES]].to_numpy(float)
    if not np.isfinite(scores).all() or (scores < 0).any() or not np.allclose(scores.sum(axis=1), 1):
        raise ValueError("invalid probability vector")
    predicted = np.array(STATES)[scores.argmax(axis=1)]
    marginal = rows[["marginal_p_" + s for s in STATES]].to_numpy(float)
    if not np.isfinite(marginal).all() or (marginal < 0).any() or not np.allclose(marginal.sum(axis=1), 1):
        raise ValueError("invalid marginal probability vector")
    marginal_correct = np.array(STATES)[marginal.argmax(axis=1)] == rows.target_state.to_numpy()
    model = predicted == rows.target_state.to_numpy()
    persistence = rows.current_state.eq(rows.target_state).to_numpy()
    delta = model.astype(float) - persistence.astype(float)
    rng = np.random.default_rng(136)
    intervals = {}
    for block in (5, 20):
        means = []
        for _ in range(repetitions):
            starts = rng.integers(0, len(rows), size=int(np.ceil(len(rows)/block)))
            indexes = ((starts[:, None] + np.arange(block)) % len(rows)).ravel()[:len(rows)]
            means.append(float(delta[indexes].mean()))
        intervals[str(block)] = np.quantile(means, [.025, .975]).tolist()
    return dict(status="DIAGNOSTIC_ONLY", sessions=len(rows), model_accuracy=float(model.mean()),
                expanding_marginal_accuracy=float(marginal_correct.mean()),
                model_minus_marginal_accuracy=float(model.mean()-marginal_correct.mean()),
                persistence_accuracy=float(persistence.mean()), accuracy_difference=float(delta.mean()),
                paired_circular_block_bootstrap_95=intervals,
                model_only_correct=int((model & ~persistence).sum()),
                persistence_only_correct=int((persistence & ~model).sum()),
                zero_recall_states=[s for s in STATES if rows.target_state.eq(s).any()
                                    and not (model & rows.target_state.eq(s).to_numpy()).any()],
                state_prediction_not_intraday_path=True, deployment_authorized=False,
                limitation="Saved research predictions, not untouched blind test; no trade outcomes or causal proof.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(pd.read_csv(args.predictions))
    result["input_sha256"] = hashlib.sha256(args.predictions.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))
