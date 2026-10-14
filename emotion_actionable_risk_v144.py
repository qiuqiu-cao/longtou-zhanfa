"""Actionable next-session risk forecast with an explicit three-session falsification.

The useful target is whether the next session is 大分歧/退潮.  A second target
tests whether at least two of the next three sessions are risk-off; it is kept
as a falsification rather than silently omitted when it has no predictive value.
All model/ensemble choices and probability cutoffs are selected on the inner
validation window.  The outer window is evaluation-only, but has been used by
earlier project research and is therefore diagnostic rather than untouched.
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

from emotion_three_regime_challenger_v86 import FEATURES, markov_probabilities, prepare


RISK_OFF = {"大分歧", "退潮"}
ECONOMIC_OUTCOMES = {
    "failed_rate": "HIGHER",
    "previous_limit_tail_loss": "LOWER",
    "down_limit_count": "HIGHER",
    "yesterday_premium": "LOWER",
    "promotion_mid": "LOWER",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model(class_weight=None) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=.5, class_weight=class_weight, max_iter=3000,
                                     random_state=0)),
    ])


def _probability(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    classes = list(model.named_steps["model"].classes_)
    return model.predict_proba(x)[:, classes.index(True)]


def _metrics(y, p, high_cut=None, low_cut=None) -> dict:
    y = np.asarray(y, dtype=bool)
    p = np.asarray(p, dtype=float)
    if len(y) == 0 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("invalid binary predictions")
    result = {
        "n": int(len(y)), "positive_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "brier": float(np.mean((p - y.astype(float)) ** 2)),
        "log_loss": float(log_loss(y, p, labels=[False, True])),
    }
    if high_cut is not None:
        called = p >= high_cut
        tp = int((called & y).sum())
        result["high_risk"] = {
            "cutoff": float(high_cut), "called": int(called.sum()),
            "coverage": float(called.mean()),
            "precision": tp / int(called.sum()) if called.any() else None,
            "recall": tp / int(y.sum()) if y.any() else None,
        }
    if low_cut is not None:
        called = p <= low_cut
        negatives = ~y
        tn = int((called & negatives).sum())
        result["low_risk"] = {
            "cutoff": float(low_cut), "called": int(called.sum()),
            "coverage": float(called.mean()),
            "negative_precision": tn / int(called.sum()) if called.any() else None,
            "negative_recall": tn / int(negatives.sum()) if negatives.any() else None,
        }
    return result


def _targets(frame: pd.DataFrame) -> pd.DataFrame:
    data = prepare(frame)
    required = set(ECONOMIC_OUTCOMES)
    missing = required - set(data)
    if missing:
        raise ValueError(f"economic outcome columns missing: {sorted(missing)}")
    data["risk_off_next"] = data.target_state.isin(RISK_OFF)
    next_risk = data["risk_off_next"].astype(float)
    data["risk_off_majority_next3"] = (
        pd.concat([next_risk, next_risk.shift(-1), next_risk.shift(-2)], axis=1).sum(axis=1) >= 2
    )
    for column in ECONOMIC_OUTCOMES:
        data["next_" + column] = pd.to_numeric(data[column], errors="coerce").shift(-1)
    return data


def _choose_inner(inner_train, inner_validation):
    markov = markov_probabilities(inner_validation)[:, 1]
    searches = []
    candidates = {}
    for name, class_weight in (("unweighted", None), ("balanced", "balanced")):
        estimator = _model(class_weight)
        estimator.fit(inner_train[FEATURES], inner_train.risk_off_next)
        candidate = _probability(estimator, inner_validation[FEATURES])
        candidates[name] = (class_weight, candidate)
        for weight in np.arange(0, 1.01, .1):
            combined = weight * candidate + (1 - weight) * markov
            metric = _metrics(inner_validation.risk_off_next, combined)
            searches.append({
                "specification": name, "logistic_weight": float(round(weight, 1)),
                "markov_weight": float(round(1 - weight, 1)),
                "brier": metric["brier"], "log_loss": metric["log_loss"],
                "auc": metric["auc"],
            })
    chosen = min(searches, key=lambda row: (row["brier"], row["log_loss"],
                                            -(row["auc"] or 0), row["specification"]))
    class_weight, candidate = candidates[chosen["specification"]]
    probability = chosen["logistic_weight"] * candidate + chosen["markov_weight"] * markov
    low_cut, high_cut = np.quantile(probability, [.20, .80]).tolist()
    return chosen, searches, class_weight, float(low_cut), float(high_cut)


def _difference_interval(values, high, low, repetitions=2000, block=10, seed=144):
    values = np.asarray(values, dtype=float)
    high, low = np.asarray(high, dtype=bool), np.asarray(low, dtype=bool)
    if not len(values) or len(values) != len(high) or len(values) != len(low):
        return None
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        starts = rng.integers(0, len(values), size=int(np.ceil(len(values) / block)))
        indexes = ((starts[:, None] + np.arange(block)) % len(values)).ravel()[:len(values)]
        sampled_values, sampled_high, sampled_low = values[indexes], high[indexes], low[indexes]
        high_values = sampled_values[sampled_high & np.isfinite(sampled_values)]
        low_values = sampled_values[sampled_low & np.isfinite(sampled_values)]
        if len(high_values) and len(low_values):
            differences.append(float(high_values.mean() - low_values.mean()))
    return np.quantile(differences, [.025, .975]).tolist() if differences else None


def _economic_separation(rows: pd.DataFrame, p: np.ndarray, low_cut: float, high_cut: float):
    high, low = p >= high_cut, p <= low_cut
    results = {}
    direction_passes = strong_passes = 0
    for column, expected in ECONOMIC_OUTCOMES.items():
        values = pd.to_numeric(rows["next_" + column], errors="coerce").to_numpy(float)
        high_values = values[high & np.isfinite(values)]
        low_values = values[low & np.isfinite(values)]
        difference = float(high_values.mean() - low_values.mean()) if len(high_values) and len(low_values) else None
        interval = _difference_interval(values, high, low)
        direction = (difference is not None and
                     ((expected == "HIGHER" and difference > 0)
                      or (expected == "LOWER" and difference < 0)))
        strong = bool(interval and ((expected == "HIGHER" and interval[0] > 0)
                                    or (expected == "LOWER" and interval[1] < 0)))
        direction_passes += int(direction)
        strong_passes += int(strong)
        results[column] = {
            "expected_high_minus_low": expected, "high_n": int(len(high_values)),
            "low_n": int(len(low_values)),
            "high_mean": float(high_values.mean()) if len(high_values) else None,
            "low_mean": float(low_values.mean()) if len(low_values) else None,
            "difference_high_minus_low": difference, "circular_block_difference_ci95": interval,
            "direction_pass": direction, "strong_ci_pass": strong,
        }
    return results, direction_passes, strong_passes


def run(frame: pd.DataFrame, outer_fraction=.70, inner_fraction=.70):
    data = _targets(frame)
    one_day = data.iloc[:-1].copy()
    outer_split = int(len(one_day) * outer_fraction)
    outer_train, outer_blind = one_day.iloc[:outer_split], one_day.iloc[outer_split:]
    inner_split = int(len(outer_train) * inner_fraction)
    inner_train, inner_validation = outer_train.iloc[:inner_split], outer_train.iloc[inner_split:]
    if len(inner_train) < 250 or len(inner_validation) < 100 or len(outer_blind) < 150:
        raise ValueError("insufficient nested chronological samples")
    chosen, search, class_weight, low_cut, high_cut = _choose_inner(inner_train, inner_validation)

    estimator = _model(class_weight)
    estimator.fit(outer_train[FEATURES], outer_train.risk_off_next)
    logistic = _probability(estimator, outer_blind[FEATURES])
    markov = markov_probabilities(outer_blind)[:, 1]
    probability = chosen["logistic_weight"] * logistic + chosen["markov_weight"] * markov
    prior = np.full(len(outer_blind), float(outer_train.risk_off_next.mean()))
    economic, direction_passes, strong_passes = _economic_separation(
        outer_blind, probability, low_cut, high_cut)
    one_day_metrics = _metrics(outer_blind.risk_off_next, probability, high_cut, low_cut)
    markov_metrics = _metrics(outer_blind.risk_off_next, markov)
    prior_metrics = _metrics(outer_blind.risk_off_next, prior)
    one_day_pass = bool(
        one_day_metrics["brier"] < markov_metrics["brier"]
        and one_day_metrics["log_loss"] < markov_metrics["log_loss"]
        and (one_day_metrics["auc"] or 0) >= .58
        and direction_passes >= 4 and strong_passes >= 2
    )

    three_day = data.iloc[:-3].copy()
    three_split = int(len(three_day) * outer_fraction)
    three_train, three_blind = three_day.iloc[:three_split], three_day.iloc[three_split:]
    three_estimator = _model(class_weight)
    three_estimator.fit(three_train[FEATURES], three_train.risk_off_majority_next3)
    three_probability = _probability(three_estimator, three_blind[FEATURES])
    three_prior = np.full(len(three_blind), float(three_train.risk_off_majority_next3.mean()))
    three_metrics = _metrics(three_blind.risk_off_majority_next3, three_probability)
    three_prior_metrics = _metrics(three_blind.risk_off_majority_next3, three_prior)
    three_pass = bool(
        three_metrics["brier"] < three_prior_metrics["brier"]
        and three_metrics["log_loss"] < three_prior_metrics["log_loss"]
        and (three_metrics["auc"] or 0) >= .58
    )

    final = _model(class_weight)
    final.fit(one_day[FEATURES], one_day.risk_off_next)
    latest_logistic = float(_probability(final, data.tail(1)[FEATURES])[0])
    latest_markov = float(markov_probabilities(data.tail(1))[0, 1])
    latest = chosen["logistic_weight"] * latest_logistic + chosen["markov_weight"] * latest_markov
    if latest >= high_cut:
        main_path = "RISK_RELEASE_OR_CONTINUED_DIVERGENCE"
    elif latest <= low_cut:
        main_path = "NON_RISK_OFF_BUT_NOT_AUTOMATIC_BUY"
    else:
        main_path = "MIXED_ROTATION_OR_DIVERGENCE"

    predictions = pd.DataFrame({
        "signal_date": outer_blind.date.dt.strftime("%Y-%m-%d"),
        "target_date": outer_blind.target_date.dt.strftime("%Y-%m-%d"),
        "current_state": outer_blind.emotion_state,
        "target_state": outer_blind.target_state,
        "risk_off_actual": outer_blind.risk_off_next,
        "risk_off_probability": probability,
        "logistic_probability": logistic, "markov_probability": markov,
        "risk_band": np.where(probability >= high_cut, "HIGH",
                              np.where(probability <= low_cut, "LOW", "MIXED")),
    })
    report = {
        "version": "v144-actionable-emotion-risk-1", "status": "DIAGNOSTIC_ONLY",
        "target": "next session emotion state is 大分歧 or 退潮",
        "features": FEATURES,
        "nested_time_split": {
            "inner_train_n": int(len(inner_train)),
            "inner_validation_n": int(len(inner_validation)),
            "outer_blind_n": int(len(outer_blind)),
            "inner_train_end": inner_train.iloc[-1].date.strftime("%Y-%m-%d"),
            "inner_validation_end": inner_validation.iloc[-1].date.strftime("%Y-%m-%d"),
            "outer_blind_start": outer_blind.iloc[0].date.strftime("%Y-%m-%d"),
            "outer_blind_reused_by_prior_project_research": True,
        },
        "inner_selection": {"chosen": chosen, "low_risk_cutoff": low_cut,
                            "high_risk_cutoff": high_cut, "search": search,
                            "outer_labels_used_for_selection": False},
        "outer_evaluation": {
            "actionable_model": one_day_metrics, "markov_baseline": markov_metrics,
            "expanding_train_prior_baseline": prior_metrics,
            "economic_separation": economic,
            "economic_direction_pass_count": direction_passes,
            "economic_strong_ci_pass_count": strong_passes,
            "risk_warning_gate_passed": one_day_pass,
        },
        "three_session_falsification": {
            "target": "at least two of next three sessions are 大分歧/退潮",
            "model": three_metrics, "train_prior_baseline": three_prior_metrics,
            "promotion_gate_passed": three_pass,
            "allowed_to_claim_reliable_two_to_three_day_regime_forecast": three_pass,
        },
        "latest_research_prediction": {
            "signal_date": data.iloc[-1].date.strftime("%Y-%m-%d"),
            "current_state": data.iloc[-1].emotion_state,
            "next_session_risk_off_probability": float(latest),
            "risk_band": "HIGH" if latest >= high_cut else "LOW" if latest <= low_cut else "MIXED",
            "main_path": main_path,
            "status": "RESEARCH_ONLY_NOT_FORWARD_VALIDATED",
        },
        "policy": {
            "aggregate_risk_warning_allowed": one_day_pass,
            "positive_buy_bonus_allowed": False,
            "individual_stock_ranking_use_allowed": False,
            "three_session_probability_claim_allowed": three_pass,
            "allowed_effect": "raise confirmation strictness or resolve to cash",
        },
        "limitations": [
            "outer dates have been inspected by prior project research and are not untouched blind",
            "emotion-state target is a market regime label, not an executable trade outcome",
            "specific-theme and stock-level competition still require separate point-in-time evidence",
        ],
    }
    return report, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/v3/market_emotion.csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.input), Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable emotion experiment")
    report, predictions = run(pd.read_csv(source))
    output.mkdir(parents=True)
    prediction_path = output / "outer_predictions.csv"
    predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    report["generated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    report["source"] = {"path": str(source.resolve()), "sha256": _sha256(source)}
    report["predictions"] = {"path": str(prediction_path.resolve()),
                             "sha256": _sha256(prediction_path)}
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
