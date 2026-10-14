"""Nested chronological ensemble of Markov and repair-aware challenger."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from emotion_three_regime_challenger_v86 import (
    CLASSES, FEATURES, make_model, markov_probabilities, metrics,
    ordered_probabilities, prepare,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose_weight(y: pd.Series, challenger: np.ndarray, markov: np.ndarray) -> tuple[float, list[dict]]:
    rows = []
    for weight in np.arange(0, 1.01, .1):
        probability = weight * challenger + (1 - weight) * markov
        result = metrics(y, probability)
        rows.append({"challenger_weight": float(round(weight, 1)),
                     "multiclass_brier": result["multiclass_brier"],
                     "log_loss": result["log_loss"],
                     "balanced_accuracy": result["balanced_accuracy"],
                     "repair_recall": result["by_class"]["REPAIR"]["recall"]})
    selected = min(rows, key=lambda row: (row["multiclass_brier"], row["log_loss"],
                                          -row["balanced_accuracy"], row["challenger_weight"]))
    return selected["challenger_weight"], rows


def run(frame: pd.DataFrame, outer_fraction: float = .70, inner_fraction: float = .70) -> tuple[dict, pd.DataFrame]:
    data = prepare(frame)
    labeled = data.iloc[:-1].copy()
    outer_split = int(len(labeled) * outer_fraction)
    outer_train, outer_blind = labeled.iloc[:outer_split], labeled.iloc[outer_split:]
    inner_split = int(len(outer_train) * inner_fraction)
    inner_train, inner_validation = outer_train.iloc[:inner_split], outer_train.iloc[inner_split:]
    if len(inner_train) < 200 or len(inner_validation) < 50 or len(outer_blind) < 100:
        raise ValueError("insufficient nested chronological samples")

    inner_model = make_model()
    inner_model.fit(inner_train[FEATURES], inner_train.target_class)
    inner_challenger = ordered_probabilities(inner_model, inner_validation[FEATURES])
    inner_markov = markov_probabilities(inner_validation)
    weight, search = choose_weight(inner_validation.target_class, inner_challenger, inner_markov)

    outer_model = make_model()
    outer_model.fit(outer_train[FEATURES], outer_train.target_class)
    blind_challenger = ordered_probabilities(outer_model, outer_blind[FEATURES])
    blind_markov = markov_probabilities(outer_blind)
    blind_ensemble = weight * blind_challenger + (1 - weight) * blind_markov
    ensemble_metrics = metrics(outer_blind.target_class, blind_ensemble)
    markov_metrics = metrics(outer_blind.target_class, blind_markov)
    challenger_metrics = metrics(outer_blind.target_class, blind_challenger)
    repair_recall = ensemble_metrics["by_class"]["REPAIR"]["recall"] or 0.0
    passed = bool(
        ensemble_metrics["multiclass_brier"] < markov_metrics["multiclass_brier"]
        and ensemble_metrics["log_loss"] < markov_metrics["log_loss"]
        and ensemble_metrics["balanced_accuracy"] > markov_metrics["balanced_accuracy"]
        and repair_recall >= .15
    )

    final_model = make_model()
    final_model.fit(labeled[FEATURES], labeled.target_class)
    latest_challenger = ordered_probabilities(final_model, data.tail(1)[FEATURES])[0]
    latest_markov = markov_probabilities(data.tail(1))[0]
    latest_ensemble = weight * latest_challenger + (1 - weight) * latest_markov
    predictions = pd.DataFrame({
        "signal_date": outer_blind.date.dt.strftime("%Y-%m-%d"),
        "target_date": outer_blind.target_date.dt.strftime("%Y-%m-%d"),
        "target_class": outer_blind.target_class,
    })
    for index, name in enumerate(CLASSES):
        predictions[f"ensemble_p_{name}"] = blind_ensemble[:, index]
        predictions[f"markov_p_{name}"] = blind_markov[:, index]
        predictions[f"challenger_p_{name}"] = blind_challenger[:, index]
    report = {
        "version": "v87-emotion-regime-ensemble-1",
        "status": "RESEARCH_ONLY",
        "nested_split": {
            "outer_train_n": int(len(outer_train)), "outer_blind_n": int(len(outer_blind)),
            "inner_train_n": int(len(inner_train)), "inner_validation_n": int(len(inner_validation)),
            "outer_train_end": outer_train.iloc[-1].date.strftime("%Y-%m-%d"),
            "outer_blind_start": outer_blind.iloc[0].date.strftime("%Y-%m-%d"),
        },
        "weight_selection": {
            "scope": "inner_validation_only", "selected_challenger_weight": weight,
            "selected_markov_weight": 1 - weight, "grid": search,
        },
        "outer_blind": {
            "ensemble": ensemble_metrics, "markov": markov_metrics,
            "challenger": challenger_metrics,
        },
        "promotion_gate_passed": passed,
        "promotion_rule": "outer blind ensemble improves Markov Brier, log loss and balanced accuracy; repair recall>=15%",
        "latest_research_prediction": {
            "signal_date": data.iloc[-1].date.strftime("%Y-%m-%d"),
            "probabilities": {name: float(latest_ensemble[i]) for i, name in enumerate(CLASSES)},
            "markov_probabilities": {name: float(latest_markov[i]) for i, name in enumerate(CLASSES)},
            "challenger_probabilities": {name: float(latest_challenger[i]) for i, name in enumerate(CLASSES)},
            "top_class": CLASSES[int(latest_ensemble.argmax())],
            "status": "RESEARCH_ONLY_NOT_FORWARD_VALIDATED",
        },
    }
    return report, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/v3/market_emotion.csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.input), Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable experiment directory")
    report, predictions = run(pd.read_csv(source))
    output.mkdir(parents=True)
    prediction_path = output / "outer_blind_predictions.csv"
    predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    report["generated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    report["source"] = {"path": str(source.resolve()), "sha256": sha256(source)}
    report["predictions"] = {"path": str(prediction_path.resolve()), "sha256": sha256(prediction_path)}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
