"""Predeclared market-direction and numeric-band extensions to V271.

Direction, coarse regime, binary risk and index direction are different targets.
None may substitute for another's reported accuracy. Interval coverage is not
classification accuracy. All historical evidence remains diagnostic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from next_session_forecast_lab_v271 import DATA, ROOT, FOLDS, prepare_market, evaluate_channel, split, sha

VERSION = "v272-market-direction-bands-1"
QUALITY_NEUTRAL_WIDTH = .05  # Five quality-score points; frozen descriptive label, not a fitted threshold.
INDEX_NEUTRAL_PCT = .3
INDEX_PATH = ROOT / "results/v203/index_emotion_divergence_20260917_v3/index_emotion_calendar_rows.csv"
QUANTITIES = ["yesterday_premium", "promotion_mid", "failed_rate", "down_limit_count", "emotion_quality_score"]


def direction(value, width):
    return pd.Series(np.select([value > width, value < -width], ["IMPROVE", "WORSEN"], default="CONTINUE"), index=value.index).where(value.notna())


def prepare():
    market, features = prepare_market(DATA / "market_emotion.csv")
    index = pd.read_csv(INDEX_PATH)[["date", "sh_a_return_pct", "sz_a_return_pct"]]
    market = market.merge(index, on="date", how="left", validate="one_to_one").sort_values("date").reset_index(drop=True)
    for field in ["sh_a_return_pct", "sz_a_return_pct"]:
        features.append(field)
        for lag in [1, 3]:
            name = field+f"_lag_{lag}"
            market[name] = market[field].shift(lag)
            features.append(name)
    return market, features


def numeric_bands(frame, features):
    reports, predictions = {}, []
    for fold in FOLDS:
        fold_reports = {}
        for field in QUANTITIES:
            f = frame.copy()
            f["target"] = f[field].shift(-1)
            f["label_valid"] &= f.target.notna()
            train, validation, test = split(f, fold)
            estimators = []
            for quantile in [.2, .5, .8]:
                estimator = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                    HistGradientBoostingRegressor(loss="quantile", quantile=quantile, max_iter=100,
                        max_leaf_nodes=7, min_samples_leaf=25, learning_rate=.04, l2_regularization=4,
                        early_stopping=False, random_state=272))
                estimator.fit(train[features], train.target)
                estimators.append(estimator)
            validation_p = np.sort(np.column_stack([e.predict(validation[features]) for e in estimators]), axis=1)
            residual = np.maximum(validation_p[:, 0]-validation.target, validation.target-validation_p[:, 2]).to_numpy()
            rank = min(len(residual)-1, int(np.ceil((len(residual)+1)*.8))-1)
            pad = max(0., float(np.sort(residual)[rank]))
            raw = np.sort(np.column_stack([e.predict(test[features]) for e in estimators]), axis=1)
            pred = test[["date", "target_date", "target", "label_valid"]].copy()
            pred["metric"] = field
            pred["low"], pred["median"], pred["high"] = raw[:, 0]-pad, raw[:, 1], raw[:, 2]+pad
            if field in {"failed_rate", "promotion_mid", "emotion_quality_score"}:
                pred[["low", "median", "high"]] = pred[["low", "median", "high"]].clip(0, 1)
            if field == "down_limit_count":
                pred[["low", "median", "high"]] = pred[["low", "median", "high"]].clip(lower=0)
            pred["fold"] = fold["name"]
            known = pred[pred.label_valid & pred.target.notna()]
            in_band = known.target.between(known.low, known.high)
            fold_reports[field] = {"n": len(known), "coverage_80_band": float(in_band.mean()),
                "mean_width": float((known.high-known.low).mean()),
                "median_prediction_mae": float((known.target-known['median']).abs().mean()),
                "previous_value_baseline_mae": float((known.target-test.loc[known.index, field]).abs().mean()),
                "validation_conformal_padding": pad, "classification_accuracy_claim_allowed": False,
                "comparison_direction": "LOWER_BETTER" if field in {"failed_rate", "down_limit_count"} else "HIGHER_BETTER"}
            fold_reports[field]["outside_band"] = {"below": int((known.target < known.low).sum()),
                                                  "above": int((known.target > known.high).sum())}
            predictions.append(pred)
        reports[fold["name"]] = fold_reports
    return reports, pd.concat(predictions, ignore_index=True)


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("immutable research output exists")
    market, features = prepare()
    tasks, predictions = {}, {}
    # Keep the requested movement forecast alongside coarse market-regime forecasts.
    f = market.copy()
    f["target"] = direction(f.emotion_quality_score.shift(-1)-f.emotion_quality_score, QUALITY_NEUTRAL_WIDTH)
    f["current_class"] = direction(f.emotion_quality_score-f.emotion_quality_score.shift(1), QUALITY_NEUTRAL_WIDTH)
    f.loc[f.current_class.isna(), "current_class"] = "CONTINUE"
    tasks["short_ecology_direction"], predictions["short_ecology_direction"] = evaluate_channel(f, features, "market")
    f = market.copy()
    f["target"] = f.target.map({"RISK_OFF": "ADVERSE", "REPAIR": "NOT_ADVERSE", "RISK_ON": "NOT_ADVERSE"})
    f["current_class"] = f.current_class.map({"RISK_OFF": "ADVERSE", "REPAIR": "NOT_ADVERSE", "RISK_ON": "NOT_ADVERSE"})
    tasks["binary_adverse_secondary"], predictions["binary_adverse_secondary"] = evaluate_channel(f, features, "market")
    for index_field in ["sh_a_return_pct", "sz_a_return_pct"]:
        f = market.copy()
        f["target"] = direction(f[index_field].shift(-1), INDEX_NEUTRAL_PCT)
        f["current_class"] = direction(f[index_field], INDEX_NEUTRAL_PCT).fillna("CONTINUE")
        f["label_valid"] &= f.target.notna()
        tasks[index_field], predictions[index_field] = evaluate_channel(f, features, "market")
    bands, band_predictions = numeric_bands(market, features)
    report = {"version": VERSION, "status": "DIAGNOSTIC_ONLY", "goal_verified": False,
        "definitions": {"short_ecology_direction": "next quality-score change > +0.05 improve, < -0.05 worsen, otherwise continue",
                        "binary_adverse_secondary": "next coarse regime RISK_OFF or not; cannot substitute for three-class forecast",
                        "index_direction": "next close-to-close return >+0.3% up, <-0.3% down, otherwise flat; both indices separately",
                        "numeric_bands": "80% validation-conformal descriptive prediction bands, not 80% scenario accuracy"},
        "tasks": tasks, "numeric_bands": bands, "fields": features,
        "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in [DATA / "market_emotion.csv", INDEX_PATH, Path(__file__), ROOT / "next_session_forecast_lab_v271.py"]},
        "source_limitations": ["previously inspected archive; no pristine future test",
                               "broad A-share indices are sh000002/sz399107, not Shanghai Composite/ChiNext",
                               "historical exchange calendar and actual live fills are not verified",
                               "current returned market data ends 2026-08-31; no current-session forecast is generated"],
        "live_accuracy_claim_allowed": False}
    output.mkdir(parents=True)
    for name, pred in predictions.items():
        pred.to_csv(output / (name+"_LABEL_ONLY.csv"), index=False, encoding="utf-8-sig")
    band_predictions.to_csv(output / "numeric_bands_LABEL_ONLY.csv", index=False, encoding="utf-8-sig")
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    r = run(args.output)
    print(json.dumps({k: {fold: {"accuracy": v['all_predictions']['accuracy'], "n": v['all_predictions']['n']}
        for fold, v in task.items()} for k, task in r['tasks'].items()}, ensure_ascii=True))
