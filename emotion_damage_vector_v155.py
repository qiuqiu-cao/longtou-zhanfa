"""Forecast measurable next-session short-term damage instead of a vague label.

Each component target is defined from the development period only and tested on
a later chronological block.  Components that fail to beat a development-prior
baseline are excluded from the aggregate forecast rather than being averaged in.
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from emotion_three_regime_challenger_v86 import FEATURES, prepare


COMPONENTS = {
    "failed_rate_damage": ("failed_rate", "HIGH", .70),
    "previous_limit_tail_damage": ("previous_limit_tail_loss", "LOW", .30),
    "down_limit_spread": ("down_limit_count", "HIGH", .70),
    "premium_loss": ("yesterday_premium", "LOW", .30),
    "middle_promotion_damage": ("promotion_mid", "LOW", .30),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=.5, max_iter=3000, random_state=155)),
    ])


def _probability(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    classes = list(model.named_steps["model"].classes_)
    return model.predict_proba(features)[:, classes.index(True)]


def _metrics(y: pd.Series, probability: np.ndarray) -> dict:
    actual = y.astype(bool).to_numpy()
    probability = np.asarray(probability, dtype=float)
    prior = np.full(len(actual), float(actual.mean()))
    return {
        "n": int(len(actual)),
        "positive_rate": float(actual.mean()),
        "auc": float(roc_auc_score(actual, probability)) if len(np.unique(actual)) == 2 else None,
        "brier": float(np.mean((probability - actual.astype(float)) ** 2)),
        "log_loss": float(log_loss(actual, probability, labels=[False, True])),
        "evaluation_prevalence_baseline_brier": float(
            np.mean((prior - actual.astype(float)) ** 2)),
    }


def _development_prior_metrics(y: pd.Series, train_prior: float) -> dict:
    actual = y.astype(bool).to_numpy()
    probability = np.full(len(actual), train_prior)
    return {
        "brier": float(np.mean((probability - actual.astype(float)) ** 2)),
        "log_loss": float(log_loss(actual, probability, labels=[False, True])),
        "probability": float(train_prior),
    }


def _labels(data: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    labeled = data.copy()
    for name, (column, direction, _) in COMPONENTS.items():
        next_value = pd.to_numeric(labeled[column], errors="coerce").shift(-1)
        labeled["next_" + column] = next_value
        if direction == "HIGH":
            damage = (next_value >= thresholds[name]).astype("boolean")
        else:
            damage = (next_value <= thresholds[name]).astype("boolean")
        labeled[name] = damage.mask(next_value.isna(), pd.NA)
    return labeled


def _thresholds(rows: pd.DataFrame) -> dict:
    result = {}
    for name, (column, _, quantile) in COMPONENTS.items():
        values = pd.to_numeric(rows[column], errors="coerce").dropna()
        if len(values) < 200:
            raise ValueError(f"insufficient development values for {column}")
        result[name] = float(values.quantile(quantile))
    return result


def run(frame: pd.DataFrame, split_fraction: float = .70) -> tuple[dict, pd.DataFrame]:
    missing = {value[0] for value in COMPONENTS.values()} - set(frame)
    if missing:
        raise ValueError(f"damage outcome fields missing: {sorted(missing)}")
    data = prepare(frame)
    labeled_count = len(data) - 1
    split = int(labeled_count * split_fraction)
    if split < 350 or labeled_count - split < 150:
        raise ValueError("insufficient chronological development/evaluation samples")
    development_rows = data.iloc[:split]
    thresholds = _thresholds(development_rows)
    labeled = _labels(data, thresholds).iloc[:-1].copy()
    development, evaluation = labeled.iloc[:split], labeled.iloc[split:]

    component_reports = {}
    prediction_rows = pd.DataFrame({
        "signal_date": evaluation.date.dt.strftime("%Y-%m-%d"),
        "target_date": evaluation.target_date.dt.strftime("%Y-%m-%d"),
    })
    validated_probabilities = []
    latest_components = {}
    for name, (column, direction, quantile) in COMPONENTS.items():
        train = development.dropna(subset=[name])
        blind = evaluation.dropna(subset=[name])
        if train[name].nunique() < 2 or blind[name].nunique() < 2:
            raise ValueError(f"component lacks two classes: {name}")
        estimator = _model()
        estimator.fit(train[FEATURES], train[name].astype(bool))
        probability = _probability(estimator, blind[FEATURES])
        metrics = _metrics(blind[name], probability)
        prior = _development_prior_metrics(blind[name], float(train[name].astype(bool).mean()))
        passed = bool(
            metrics["auc"] is not None and metrics["auc"] >= .55
            and metrics["brier"] < prior["brier"]
            and metrics["log_loss"] < prior["log_loss"]
        )
        aligned = pd.Series(probability, index=blind.index).reindex(evaluation.index)
        prediction_rows[name + "_actual"] = evaluation[name].astype("boolean").to_numpy()
        prediction_rows[name + "_probability"] = aligned.to_numpy()
        if passed:
            validated_probabilities.append(aligned.to_numpy(float))

        final_thresholds = _thresholds(data.iloc[:-1])
        final_labeled = _labels(data, final_thresholds).iloc[:-1].dropna(subset=[name])
        final_model = _model()
        final_model.fit(final_labeled[FEATURES], final_labeled[name].astype(bool))
        latest_probability = float(_probability(final_model, data.tail(1)[FEATURES])[0])
        latest_components[name] = {
            "probability": latest_probability,
            "validated_for_aggregate": passed,
            "outcome": column,
            "damage_direction": direction,
            "current_historical_threshold": final_thresholds[name],
        }
        component_reports[name] = {
            "outcome": column,
            "damage_direction": direction,
            "development_quantile": quantile,
            "frozen_development_threshold": thresholds[name],
            "development_n": int(len(train)),
            "evaluation": metrics,
            "development_prior_baseline": prior,
            "promotion_gate_passed": passed,
        }

    validated_count = len(validated_probabilities)
    if validated_probabilities:
        aggregate = np.nanmean(np.vstack(validated_probabilities), axis=0)
        prediction_rows["validated_damage_probability"] = aggregate
        latest_probability = float(np.mean([
            row["probability"] for row in latest_components.values()
            if row["validated_for_aggregate"]
        ]))
    else:
        prediction_rows["validated_damage_probability"] = np.nan
        latest_probability = None
    aggregate_ready = validated_count >= 3
    if not aggregate_ready or latest_probability is None:
        band = "UNAVAILABLE"
    elif latest_probability >= .60:
        band = "HIGH_DAMAGE_RISK"
    elif latest_probability <= .40:
        band = "LOW_DAMAGE_RISK"
    else:
        band = "MIXED_DAMAGE_RISK"

    report = {
        "version": "v155-emotion-damage-vector-1",
        "status": "DIAGNOSTIC_ONLY",
        "target": "next-session measurable short-term damage vector",
        "split": {
            "method": "chronological_70_30_no_outer_tuning",
            "development_n": int(len(development)),
            "evaluation_n": int(len(evaluation)),
            "development_end": development.iloc[-1].date.strftime("%Y-%m-%d"),
            "evaluation_start": evaluation.iloc[0].date.strftime("%Y-%m-%d"),
            "evaluation_reused_by_prior_project_research": True,
        },
        "components": component_reports,
        "validated_component_count": validated_count,
        "aggregate_gate": {
            "required_validated_components": 3,
            "passed": aggregate_ready,
            "stock_reranking_allowed": False,
            "allowed_effect": "corroborate or weaken the V144 aggregate market scenario",
        },
        "latest_research_prediction": {
            "signal_date": data.iloc[-1].date.strftime("%Y-%m-%d"),
            "components": latest_components,
            "validated_damage_probability": latest_probability if aggregate_ready else None,
            "damage_band": band,
            "status": "RESEARCH_ONLY_NOT_FORWARD_VALIDATED",
        },
        "limitations": [
            "component thresholds describe historical tail conditions rather than fixed market laws",
            "outer dates were already inspected by earlier project research",
            "aggregate market damage cannot by itself rank an individual stock",
            "specific-theme and candidate execution evidence remain separate requirements",
        ],
    }
    return report, prediction_rows


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
    prediction_path = output / "outer_predictions.csv"
    predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    report["generated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    report["source"] = {"path": str(source.resolve()), "sha256": _sha256(source)}
    report["predictions"] = {
        "path": str(prediction_path.resolve()), "sha256": _sha256(prediction_path)}
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
