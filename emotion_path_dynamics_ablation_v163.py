"""Causal dynamic-feature ablation for three-session emotion paths.

V162 showed that a same-day cross-sectional snapshot did not beat a simple
Markov baseline.  This experiment adds only predeclared backward-looking
changes, deviations, volatility and state duration.  It uses expanding
walk-forward folds with a three-session purge and never selects features from
the test folds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from emotion_path_forecast_v162 import (
    FEATURES, PATH_CLASSES, _block_difference_ci, _model, _loss_rows,
    build_targets, markov_path_probabilities, metrics, transition_matrix,
)


VERSION = "v163-emotion-path-dynamics-ablation-1"
DYNAMIC_SOURCES = (
    "failed_rate", "down_limit_count", "promotion_mid", "yesterday_premium",
    "previous_limit_tail_loss", "max_board", "limit_count", "promotion_overall",
    "middle_density",
)
OPTIONAL_CAUSAL_FIELDS = (
    "emotion_transition_entropy", "emotion_state_age", "space_feedback_3d",
)


def add_dynamic_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    result = data.sort_values("date").reset_index(drop=True).copy()
    created = []
    for source in DYNAMIC_SOURCES:
        values = pd.to_numeric(result[source], errors="coerce")
        for lag in (1, 3):
            name = f"dyn_{source}_delta_{lag}"
            result[name] = values - values.shift(lag)
            created.append(name)
        prior = values.shift(1).rolling(5, min_periods=3)
        deviation = f"dyn_{source}_vs_prior5_mean"
        volatility = f"dyn_{source}_prior5_std"
        result[deviation] = values - prior.mean()
        result[volatility] = prior.std(ddof=0)
        created.extend([deviation, volatility])

    # Causal run length: the current row may use its state, but never a later row.
    groups = result.emotion_state.ne(result.emotion_state.shift()).cumsum()
    result["dyn_state_age"] = result.groupby(groups).cumcount() + 1
    created.append("dyn_state_age")
    for source in OPTIONAL_CAUSAL_FIELDS:
        name = f"dyn_raw_{source}"
        result[name] = pd.to_numeric(result[source], errors="coerce") if source in result else np.nan
        created.append(name)
    return result, created


def walk_forward_slices(n: int, initial_train=300, test_size=60,
                        purge_sessions=3) -> list[tuple[slice, slice]]:
    if n < initial_train + test_size or purge_sessions < 3:
        raise ValueError("insufficient labels or purge")
    folds = []
    start = initial_train
    while start + test_size <= n:
        train_end = start - purge_sessions
        if train_end <= 0:
            raise ValueError("invalid train boundary")
        folds.append((slice(0, train_end), slice(start, start + test_size)))
        start += test_size
    return folds


def _probability(model, rows, columns):
    raw = model.predict_proba(rows[columns])
    classes = list(model.named_steps["model"].classes_)
    missing = set(PATH_CLASSES) - set(classes)
    if missing:
        raise ValueError(f"training fold lacks path classes: {sorted(missing)}")
    return np.column_stack([raw[:, classes.index(name)] for name in PATH_CLASSES])


def _fit(train, test, columns):
    estimator = _model(c=.1, class_weight=None)
    estimator.fit(train[columns], train.path_target)
    return estimator, _probability(estimator, test, columns)


def _feature_importance(model, columns):
    coefficients = np.abs(model.named_steps["model"].coef_).mean(axis=0)
    # SimpleImputer adds indicators only for columns missing during fit.  The
    # original coefficient prefix always preserves the supplied column order.
    return {column: float(coefficients[index]) for index, column in enumerate(columns)}


def run(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    target_data = build_targets(frame)
    data, dynamic = add_dynamic_features(target_data)
    labeled = data.dropna(subset=["path_target"]).reset_index(drop=True)
    folds = walk_forward_slices(len(labeled))
    dynamic_columns = [*FEATURES, *dynamic]
    prediction_frames, fold_reports, importances = [], [], []
    for fold_number, (train_slice, test_slice) in enumerate(folds, 1):
        train, test = labeled.iloc[train_slice].copy(), labeled.iloc[test_slice].copy()
        if train.path_end_date.max() >= test.date.min():
            raise ValueError("path target crosses walk-forward boundary")
        base_model, base_probability = _fit(train, test, FEATURES)
        dynamic_model, dynamic_probability = _fit(train, test, dynamic_columns)
        historical_states = data.loc[data.date < test.date.min(), "current_class"]
        matrix = transition_matrix(historical_states)
        markov_probability = markov_path_probabilities(test.current_class, matrix)
        fold_metrics = {
            "base_snapshot": metrics(test.path_target, base_probability),
            "dynamic": metrics(test.path_target, dynamic_probability),
            "markov": metrics(test.path_target, markov_probability),
        }
        fold_reports.append({
            "fold": fold_number, "train_n": int(len(train)), "test_n": int(len(test)),
            "train_end": train.date.max().strftime("%Y-%m-%d"),
            "train_path_end": train.path_end_date.max().strftime("%Y-%m-%d"),
            "test_start": test.date.min().strftime("%Y-%m-%d"),
            "test_end": test.date.max().strftime("%Y-%m-%d"),
            "metrics": fold_metrics,
        })
        importances.append(_feature_importance(dynamic_model, dynamic_columns))
        payload = pd.DataFrame({
            "fold": fold_number, "signal_date": test.date.dt.strftime("%Y-%m-%d"),
            "path_end_date": test.path_end_date.dt.strftime("%Y-%m-%d"),
            "path_actual": test.path_target,
        })
        for index, name in enumerate(PATH_CLASSES):
            payload[f"dynamic_p_{name}"] = dynamic_probability[:, index]
            payload[f"base_p_{name}"] = base_probability[:, index]
            payload[f"markov_p_{name}"] = markov_probability[:, index]
        prediction_frames.append(payload)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    def probability(prefix):
        return predictions[[f"{prefix}_p_{name}" for name in PATH_CLASSES]].to_numpy(float)

    dynamic_probability = probability("dynamic")
    base_probability = probability("base")
    markov_probability = probability("markov")
    overall = {
        "dynamic": metrics(predictions.path_actual, dynamic_probability),
        "base_snapshot": metrics(predictions.path_actual, base_probability),
        "markov": metrics(predictions.path_actual, markov_probability),
    }
    dynamic_brier, dynamic_log = _loss_rows(predictions.path_actual, dynamic_probability)
    comparisons = {}
    fold_wins = {}
    for name, baseline_probability in (("base_snapshot", base_probability),
                                       ("markov", markov_probability)):
        baseline_brier, baseline_log = _loss_rows(predictions.path_actual, baseline_probability)
        comparisons[name] = {
            "brier_difference": float(dynamic_brier.mean() - baseline_brier.mean()),
            "brier_difference_block_ci95": _block_difference_ci(dynamic_brier, baseline_brier),
            "log_loss_difference": float(dynamic_log.mean() - baseline_log.mean()),
            "log_loss_difference_block_ci95": _block_difference_ci(dynamic_log, baseline_log),
            "macro_f1_difference": float(
                overall["dynamic"]["macro_f1"] - overall[name]["macro_f1"]),
        }
        fold_wins[name] = sum(
            fold["metrics"]["dynamic"]["multiclass_brier"]
            < fold["metrics"][name]["multiclass_brier"] for fold in fold_reports
        )
    required_wins = int(np.ceil(len(folds) * .67))
    support_ok = min(row["support"] for row in overall["dynamic"]["by_class"].values()) >= 20
    gate = bool(support_ok and all(
        comparison["brier_difference"] < 0
        and comparison["log_loss_difference"] < 0
        and comparison["brier_difference_block_ci95"][1] < 0
        and comparison["log_loss_difference_block_ci95"][1] < 0
        and fold_wins[name] >= required_wins
        for name, comparison in comparisons.items()
    ))
    mean_importance = {
        column: float(np.mean([fold[column] for fold in importances]))
        for column in dynamic_columns
    }
    dynamic_importance = [
        {"feature": name, "mean_absolute_standardized_coefficient": value}
        for name, value in sorted(mean_importance.items(), key=lambda item: item[1], reverse=True)
        if name in dynamic
    ][:15]
    report = {
        "version": VERSION,
        "status": "DYNAMIC_FACTORS_DIAGNOSTICALLY_PROMOTED" if gate else "DIAGNOSTIC_ONLY",
        "target": "V162 four-class next-three-session emotion path",
        "design": {
            "method": "expanding_walk_forward_fixed_specification",
            "fold_count": len(folds), "initial_train": 300, "test_size": 60,
            "purge_sessions": 3, "test_rows": int(len(predictions)),
            "model": "multinomial_logistic_C_0.1_unweighted",
            "dynamic_sources": list(DYNAMIC_SOURCES),
            "dynamic_feature_count": len(dynamic),
            "test_folds_used_for_feature_selection": False,
        },
        "overall": overall,
        "comparisons": comparisons,
        "fold_brier_wins": fold_wins,
        "required_fold_wins": required_wins,
        "minimum_class_support_at_least_20": support_ok,
        "promotion_gate_passed": gate,
        "top_dynamic_coefficients_diagnostic_only": dynamic_importance,
        "folds": fold_reports,
        "policy": {
            "dynamic_factors_allowed_in_daily_probability": gate,
            "two_to_three_day_probability_claim_allowed": False,
            "stock_reranking_allowed": False,
            "failed_features_treated_as_zero": False,
        },
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
        raise FileExistsError("refusing to overwrite immutable ablation")
    report, predictions = run(pd.read_csv(source))
    target.mkdir(parents=True)
    evidence = target / "walk_forward_predictions.csv"
    predictions.to_csv(evidence, index=False, encoding="utf-8-sig")
    report["generated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    report["source"] = {"path": str(source.resolve()), "sha256": _sha256(source)}
    report["predictions"] = {"path": str(evidence.resolve()), "sha256": _sha256(evidence)}
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "overall": report["overall"],
        "comparisons": report["comparisons"], "fold_brier_wins": report["fold_brier_wins"],
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
