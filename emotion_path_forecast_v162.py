"""Time-isolated three-session emotion-path experiment.

The one-session risk warning must not be presented as a two-to-three-session
cycle forecast.  This experiment freezes four economically interpretable paths,
uses a three-session purge at each split boundary, selects the model/blend only
on the validation segment, and evaluates once on a later segment.  Failure of
the gate explicitly disables multi-session probability claims.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from emotion_three_regime_challenger_v86 import FEATURES, STATE_TO_CLASS, prepare


VERSION = "v162-emotion-three-session-path-1"
BASE_CLASSES = ("RISK_ON", "RISK_OFF", "REPAIR")
PATH_CLASSES = (
    "RISK_OFF_DOMINANT",
    "RISK_ON_DOMINANT",
    "REPAIR_TO_RISK_ON",
    "MIXED_ROTATION",
)


def path_class(sequence) -> str:
    values = list(sequence)
    if len(values) != 3 or any(value not in BASE_CLASSES for value in values):
        raise ValueError("exactly three known base classes required")
    risk_off, risk_on = values.count("RISK_OFF"), values.count("RISK_ON")
    if risk_off >= 2:
        return "RISK_OFF_DOMINANT"
    if risk_on >= 2 and risk_off == 0:
        return "RISK_ON_DOMINANT"
    if "REPAIR" in values and values[-1] == "RISK_ON" and risk_off <= 1:
        return "REPAIR_TO_RISK_ON"
    return "MIXED_ROTATION"


def build_targets(frame: pd.DataFrame) -> pd.DataFrame:
    data = prepare(frame)
    base = data.emotion_state.map(STATE_TO_CLASS)
    targets = []
    for index in range(len(data)):
        if index + 3 >= len(data):
            targets.append(None)
        else:
            targets.append(path_class(base.iloc[index + 1:index + 4]))
    data["current_class"] = base
    data["path_target"] = targets
    data["path_end_date"] = data.date.shift(-3)
    return data


def split_rows(data: pd.DataFrame, train_fraction=.60, validation_fraction=.20,
               purge_sessions=3):
    labeled = data.dropna(subset=["path_target"]).reset_index(drop=True)
    if len(labeled) < 400:
        raise ValueError("at least 400 fully observed path labels required")
    train_boundary = int(len(labeled) * train_fraction)
    validation_boundary = int(len(labeled) * (train_fraction + validation_fraction))
    if not 0 < train_boundary < validation_boundary < len(labeled):
        raise ValueError("invalid chronological split fractions")
    train = labeled.iloc[:train_boundary - purge_sessions].copy()
    validation = labeled.iloc[train_boundary:validation_boundary - purge_sessions].copy()
    blind = labeled.iloc[validation_boundary:].copy()
    if min(len(train), len(validation), len(blind)) < 50:
        raise ValueError("split segments too small")
    if not (train.path_end_date.max() < validation.date.min()
            and validation.path_end_date.max() < blind.date.min()):
        raise ValueError("future path labels cross a split boundary")
    return labeled, train, validation, blind, train_boundary, validation_boundary


def _model(c=.5, class_weight="balanced") -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            C=c, class_weight=class_weight, max_iter=4000, random_state=0,
        )),
    ])


def _ordered_probability(model: Pipeline, rows: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(rows[FEATURES])
    classes = list(model.named_steps["model"].classes_)
    missing = set(PATH_CLASSES) - set(classes)
    if missing:
        raise ValueError(f"training segment lacks path classes: {sorted(missing)}")
    return np.column_stack([raw[:, classes.index(name)] for name in PATH_CLASSES])


def transition_matrix(base_states: pd.Series, smoothing=1.0) -> np.ndarray:
    values = list(base_states)
    counts = np.full((len(BASE_CLASSES), len(BASE_CLASSES)), float(smoothing))
    for first, second in zip(values[:-1], values[1:]):
        if first in BASE_CLASSES and second in BASE_CLASSES:
            counts[BASE_CLASSES.index(first), BASE_CLASSES.index(second)] += 1
    return counts / counts.sum(axis=1, keepdims=True)


def markov_path_probabilities(current_states: pd.Series, matrix: np.ndarray) -> np.ndarray:
    if matrix.shape != (3, 3) or not np.allclose(matrix.sum(axis=1), 1):
        raise ValueError("valid three-state transition matrix required")
    output = []
    for current in current_states:
        if current not in BASE_CLASSES:
            raise ValueError("unknown current class")
        totals = {name: 0.0 for name in PATH_CLASSES}
        start = BASE_CLASSES.index(current)
        for sequence in itertools.product(BASE_CLASSES, repeat=3):
            indexes = [BASE_CLASSES.index(value) for value in sequence]
            probability = (matrix[start, indexes[0]] * matrix[indexes[0], indexes[1]]
                           * matrix[indexes[1], indexes[2]])
            totals[path_class(sequence)] += float(probability)
        output.append([totals[name] for name in PATH_CLASSES])
    result = np.asarray(output, dtype=float)
    if not np.allclose(result.sum(axis=1), 1):
        raise AssertionError("Markov path probabilities do not sum to one")
    return result


def prior_probabilities(target: pd.Series, n: int, smoothing=1.0) -> np.ndarray:
    counts = target.value_counts().reindex(PATH_CLASSES, fill_value=0).astype(float) + smoothing
    probabilities = (counts / counts.sum()).to_numpy(float)
    return np.tile(probabilities, (n, 1))


def metrics(target: pd.Series, probabilities: np.ndarray) -> dict:
    labels = target.reset_index(drop=True).astype(str)
    indexes = labels.map(PATH_CLASSES.index).to_numpy(int)
    onehot = np.eye(len(PATH_CLASSES))[indexes]
    predicted_indexes = probabilities.argmax(axis=1)
    predicted = pd.Series([PATH_CLASSES[index] for index in predicted_indexes])
    recalls, f1s, by_class = [], [], {}
    for name in PATH_CLASSES:
        truth, called = labels.eq(name), predicted.eq(name)
        tp = int((truth & called).sum())
        precision = tp / int(called.sum()) if called.any() else 0.0
        recall = tp / int(truth.sum()) if truth.any() else None
        f1 = (2 * precision * recall / (precision + recall)
              if recall is not None and precision + recall else 0.0)
        if recall is not None:
            recalls.append(recall)
            f1s.append(f1)
        by_class[name] = {
            "support": int(truth.sum()), "predicted": int(called.sum()),
            "precision": precision, "recall": recall, "f1": f1,
        }
    selected = probabilities[np.arange(len(indexes)), indexes]
    return {
        "n": int(len(labels)),
        "accuracy": float((predicted_indexes == indexes).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "multiclass_brier": float(np.mean(np.sum((probabilities - onehot) ** 2, axis=1))),
        "log_loss": float(-np.mean(np.log(np.clip(selected, 1e-12, 1)))),
        "by_class": by_class,
    }


def _loss_rows(target: pd.Series, probabilities: np.ndarray):
    indexes = target.reset_index(drop=True).astype(str).map(PATH_CLASSES.index).to_numpy(int)
    onehot = np.eye(len(PATH_CLASSES))[indexes]
    return (
        np.sum((probabilities - onehot) ** 2, axis=1),
        -np.log(np.clip(probabilities[np.arange(len(indexes)), indexes], 1e-12, 1)),
    )


def _block_difference_ci(challenger, baseline, block=10, repetitions=3000, seed=162):
    challenger, baseline = np.asarray(challenger), np.asarray(baseline)
    if len(challenger) != len(baseline) or not len(challenger):
        raise ValueError("aligned non-empty losses required")
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        starts = rng.integers(0, len(challenger), size=int(np.ceil(len(challenger) / block)))
        indexes = ((starts[:, None] + np.arange(block)) % len(challenger)).ravel()[:len(challenger)]
        differences.append(float((challenger[indexes] - baseline[indexes]).mean()))
    return [float(value) for value in np.quantile(differences, [.025, .975])]


def _fit_probabilities(train, evaluate, c, class_weight):
    estimator = _model(c, class_weight)
    estimator.fit(train[FEATURES], train.path_target)
    return estimator, _ordered_probability(estimator, evaluate)


def run(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    data = build_targets(frame)
    labeled, train, validation, blind, train_boundary, validation_boundary = split_rows(data)
    train_matrix = transition_matrix(data.iloc[:train_boundary].current_class)
    validation_markov = markov_path_probabilities(validation.current_class, train_matrix)
    searches = []
    candidates = {}
    for c in (.1, .5, 1.0):
        for class_weight in (None, "balanced"):
            _, model_probability = _fit_probabilities(train, validation, c, class_weight)
            for model_weight in (0.0, .5, 1.0):
                probability = model_weight * model_probability + (1 - model_weight) * validation_markov
                result = metrics(validation.path_target, probability)
                key = (c, class_weight, model_weight)
                candidates[key] = result
                searches.append({
                    "C": c, "class_weight": class_weight or "none",
                    "model_weight": model_weight, "markov_weight": 1 - model_weight,
                    "multiclass_brier": result["multiclass_brier"],
                    "log_loss": result["log_loss"], "macro_f1": result["macro_f1"],
                })
    chosen = min(searches, key=lambda row: (
        row["multiclass_brier"], row["log_loss"], -row["macro_f1"],
        row["C"], row["class_weight"], row["model_weight"],
    ))

    # After validation-only selection, refit on all labels whose complete
    # three-session outcome predates the blind signal boundary.
    development = labeled.iloc[:validation_boundary - 3].copy()
    development_states = data.loc[data.date < blind.date.min(), "current_class"]
    matrix = transition_matrix(development_states)
    class_weight = None if chosen["class_weight"] == "none" else chosen["class_weight"]
    estimator, model_probability = _fit_probabilities(
        development, blind, chosen["C"], class_weight)
    markov_probability = markov_path_probabilities(blind.current_class, matrix)
    probability = (chosen["model_weight"] * model_probability
                   + chosen["markov_weight"] * markov_probability)
    prior_probability = prior_probabilities(development.path_target, len(blind))
    challenger_metrics = metrics(blind.path_target, probability)
    markov_metrics = metrics(blind.path_target, markov_probability)
    prior_metrics = metrics(blind.path_target, prior_probability)

    challenger_brier, challenger_log = _loss_rows(blind.path_target, probability)
    baselines = {
        "markov": (markov_metrics, *_loss_rows(blind.path_target, markov_probability)),
        "development_prior": (prior_metrics, *_loss_rows(blind.path_target, prior_probability)),
    }
    comparisons = {}
    for name, (baseline_metrics, baseline_brier, baseline_log) in baselines.items():
        comparisons[name] = {
            "brier_difference": float(challenger_brier.mean() - baseline_brier.mean()),
            "brier_difference_block_ci95": _block_difference_ci(challenger_brier, baseline_brier),
            "log_loss_difference": float(challenger_log.mean() - baseline_log.mean()),
            "log_loss_difference_block_ci95": _block_difference_ci(challenger_log, baseline_log),
            "macro_f1_difference": float(
                challenger_metrics["macro_f1"] - baseline_metrics["macro_f1"]),
        }
    support_ok = min(item["support"] for item in challenger_metrics["by_class"].values()) >= 10
    gate = bool(support_ok and all(
        comparison["brier_difference"] < 0
        and comparison["log_loss_difference"] < 0
        and comparison["macro_f1_difference"] >= 0
        and comparison["brier_difference_block_ci95"][1] < 0
        and comparison["log_loss_difference_block_ci95"][1] < 0
        for comparison in comparisons.values()
    ))

    predictions = pd.DataFrame({
        "signal_date": blind.date.dt.strftime("%Y-%m-%d"),
        "path_end_date": blind.path_end_date.dt.strftime("%Y-%m-%d"),
        "current_state": blind.emotion_state,
        "path_actual": blind.path_target,
    })
    for index, name in enumerate(PATH_CLASSES):
        predictions[f"path_probability_{name}"] = probability[:, index]
        predictions[f"markov_probability_{name}"] = markov_probability[:, index]
    predictions["predicted_path"] = [PATH_CLASSES[index] for index in probability.argmax(axis=1)]

    all_labeled = data.dropna(subset=["path_target"])
    final_model = _model(chosen["C"], class_weight)
    final_model.fit(all_labeled[FEATURES], all_labeled.path_target)
    latest = data.tail(1)
    latest_model = _ordered_probability(final_model, latest)
    final_matrix = transition_matrix(data.current_class)
    latest_markov = markov_path_probabilities(latest.current_class, final_matrix)
    latest_probability = (chosen["model_weight"] * latest_model
                          + chosen["markov_weight"] * latest_markov)[0]
    report = {
        "version": VERSION,
        "status": "RESEARCH_GATE_PASSED_NOT_FORWARD_VALIDATED" if gate else "DIAGNOSTIC_ONLY",
        "target_contract": {
            "horizon": "next three trading sessions",
            "classes": list(PATH_CLASSES),
            "definitions": {
                "RISK_OFF_DOMINANT": "at least two of next three sessions are RISK_OFF",
                "RISK_ON_DOMINANT": "at least two are RISK_ON and none is RISK_OFF",
                "REPAIR_TO_RISK_ON": "contains REPAIR, ends RISK_ON and has at most one RISK_OFF",
                "MIXED_ROTATION": "all other three-session paths",
            },
            "overlapping_future_labels": True,
            "purge_sessions_at_each_split": 3,
        },
        "split": {
            "train_n": int(len(train)), "validation_n": int(len(validation)),
            "development_refit_n": int(len(development)), "blind_n": int(len(blind)),
            "train_end": train.date.max().strftime("%Y-%m-%d"),
            "validation_start": validation.date.min().strftime("%Y-%m-%d"),
            "validation_end": validation.date.max().strftime("%Y-%m-%d"),
            "blind_start": blind.date.min().strftime("%Y-%m-%d"),
            "blind_end": blind.date.max().strftime("%Y-%m-%d"),
            "blind_reused_by_prior_project_research": True,
        },
        "validation_selection": {"chosen": chosen, "search": searches,
                                 "blind_labels_used_for_selection": False},
        "blind": {
            "challenger": challenger_metrics,
            "markov_baseline": markov_metrics,
            "development_prior_baseline": prior_metrics,
            "comparisons": comparisons,
            "minimum_class_support_at_least_10": support_ok,
            "promotion_gate_passed": gate,
        },
        "latest_research_scenario": {
            "signal_date": latest.iloc[0].date.strftime("%Y-%m-%d"),
            "probabilities": {name: float(latest_probability[index])
                              for index, name in enumerate(PATH_CLASSES)},
            "top_path": PATH_CLASSES[int(latest_probability.argmax())],
            "status": ("RESEARCH_ONLY_NOT_FORWARD_VALIDATED" if gate
                       else "UNAUTHORIZED_PROBABILITY_DIAGNOSTIC_ONLY"),
        },
        "policy": {
            "reliable_two_to_three_day_path_claim_allowed": gate,
            "stock_reranking_allowed": False,
            "positive_buy_bonus_allowed": False,
            "allowed_if_failed": "qualitative main/alternative/risk scenarios with explicit invalidation only",
        },
        "limitations": [
            "the outer segment was previously visible to project research and is not a pristine blind set",
            "state labels are derived market summaries, not executable trade outcomes",
            "overlapping three-session targets require block uncertainty and split purging",
        ],
    }
    return report, predictions


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/v3/market_emotion.csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, target = Path(args.input), Path(args.output)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable path experiment")
    report, predictions = run(pd.read_csv(source))
    target.mkdir(parents=True)
    predictions_path = target / "blind_predictions.csv"
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    report["generated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    report["source"] = {"path": str(source.resolve()), "sha256": _sha256(source)}
    report["predictions"] = {
        "path": str(predictions_path.resolve()), "sha256": _sha256(predictions_path),
    }
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
