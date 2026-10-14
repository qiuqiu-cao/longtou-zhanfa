"""Link frozen prequential ranks to label-only execution outcomes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METHODS = {
    "earliest_seal": ("first_seal_minute", True),
    "speed_only": ("speed_probability", False),
    "base_quality": ("base_quality_probability", False),
    "emotion_challenger": ("emotion_quality_probability", False),
}


def method_metrics(frame: pd.DataFrame) -> dict:
    missing = frame.fill_status.isin(["MISSING_B_BAR", "INVALID_B_PRICE"])
    observed = frame[~missing]
    filled = observed[observed.fill_status.eq("FILLED_OPEN_PROXY")]
    result = {
        "selected_sessions": int(frame.signal_date.nunique()),
        "execution_observed_sessions": int(observed.signal_date.nunique()),
        "missing_execution_evidence_sessions": int(missing.sum()),
        "unverifiable_limit_queue_sessions": int(
            observed.fill_status.eq("UNVERIFIABLE_LIMIT_QUEUE").sum()),
        "filled_trades": int(len(filled)),
        "fill_rate": float(len(filled) / len(observed)) if len(observed) else None,
        "promotion_precision_all_selections": float(frame.promotion_label.mean()) if len(frame) else None,
        "promotion_precision_filled": float(filled.promotion_label.mean()) if len(filled) else None,
    }
    for horizon in (1, 2, 3):
        values = filled[f"net_return_B_plus_{horizon}"].dropna()
        result[f"B_plus_{horizon}"] = {
            "n": int(len(values)),
            "mean_net_return": float(values.mean()) if len(values) else None,
            "positive_rate": float(values.gt(0).mean()) if len(values) else None,
        }
    big = filled.big_loss_B_plus_1.dropna()
    result["big_loss_B_plus_1_rate"] = float(big.mean()) if len(big) else None
    return result


def audit(predictions: pd.DataFrame, outcomes: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    outcome_fields = [column for column in outcomes.columns
                      if column not in {"name", "promotion_label"}]
    joined = predictions.merge(outcomes[outcome_fields], on=["signal_date", "code"],
                               how="left", validate="one_to_one")
    selections, metrics = [], {}
    for method, (score, ascending) in METHODS.items():
        selected = (joined.sort_values(["signal_date", score, "code"],
                                       ascending=[True, ascending, True], na_position="last")
                    .groupby("signal_date", as_index=False).first())
        selected.insert(0, "selection_method", method)
        selections.append(selected)
        metrics[method] = method_metrics(selected)
    return pd.concat(selections, ignore_index=True), metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite immutable v126 output")
    predictions = pd.read_csv(args.predictions, dtype={"code": str})
    outcomes = pd.read_csv(args.outcomes, dtype={"code": str})
    selections, metrics = audit(predictions, outcomes)
    args.output.mkdir(parents=True, exist_ok=False)
    selections.to_csv(args.output / "daily_top1_execution_audit.csv", index=False,
                      encoding="utf-8-sig")
    report = {
        "version": "v126-first-board-profit-link-2",
        "status": "DIAGNOSTIC_ONLY",
        "metrics": metrics,
        "governance": {
            "one_selection_per_day": True,
            "unfilled_top1_not_replaced": True,
            "test_results_not_used_to_retune": True,
            "missing_bar_is_unfilled_trade": False,
            "deployment_authorized": False,
        },
        "conclusion_rule": (
            "A ranker is not promoted unless it improves mean net return and positive rate, "
            "does not worsen big-loss rate, and is confirmed on a larger frozen forward sample."
        ),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
