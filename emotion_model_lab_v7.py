#!/usr/bin/env python3
"""Time-causal emotion factor ablation and calibrated boosting laboratory.

This is a research-only challenger.  It uses expanding prequential forecasts,
one-row train/calibration embargo, fixed model settings and three chronological
evaluation folds.  Reused historical dates can reject ideas but cannot promote
the production model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from emotion_challenger_v4 import ACTIVE_STATES, add_targets


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "v7" / "point_in_time_rebuild" / "market_emotion.csv"
OUTPUT = ROOT / "results" / "v7" / "emotion_model_lab"
VERSION = "v7-emotion-model-lab-1"
LOCK_PATH = OUTPUT / "research_candidate_lock.json"
CANDIDATE_NAME = "adverse_state_next__plus_dimensions__logistic"
INITIAL_HISTORY = 160
RETRAIN_EVERY = 20

STATE = [
    "state_start", "state_warming", "state_acceleration", "state_consensus",
    "state_divergence", "state_retreat", "state_repair",
    "emotion_state_age_scaled", "emotion_transition_entropy",
    "emotion_transition_evidence_scaled", "emotion_adverse_markov_probability",
]
HEIGHT_BREADTH = [
    "limit_count", "limit_change", "seal_rate", "failed_rate", "max_board",
    "height_change", "middle_density", "ladder_gap_count", "unique_space",
    "one_word_share", "breadth_height_divergence",
]
DIMENSIONS = [
    "breadth_health", "relay_appetite", "ladder_health", "loss_effect_health",
    "crowding_risk", "regime_fragility", "false_acceleration",
    "emotion_quality_score", "emotion_data_completeness",
]
RELAY_LOSS = [
    "promotion_overall", "promotion_mid", "promotion_1to2", "promotion_2to3",
    "promotion_3to4", "promotion_4to5", "space_feedback", "space_feedback_3d",
    "yesterday_premium", "previous_limit_tail_loss", "down_limit_count",
    "failed_rate_percentile_60d", "down_limit_percentile_60d",
    "promotion_mid_percentile_60d", "yesterday_premium_percentile_60d",
    "tail_loss_percentile_60d",
]
DYNAMIC_SOURCES = [
    "breadth_health", "relay_appetite", "ladder_health", "loss_effect_health",
    "crowding_risk", "regime_fragility", "promotion_mid", "failed_rate",
    "down_limit_count", "yesterday_premium", "previous_limit_tail_loss", "max_board",
]
FEATURE_STAGES = {
    "height_state_baseline": STATE + HEIGHT_BREADTH,
    "plus_dimensions": STATE + HEIGHT_BREADTH + DIMENSIONS,
    "plus_relay_loss": STATE + HEIGHT_BREADTH + DIMENSIONS + RELAY_LOSS,
}
TARGETS = ("adverse_state_next", "relay_damage_next")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def add_causal_dynamics(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    ordered = frame.sort_values("date").reset_index(drop=True).copy()
    names: list[str] = []
    for source in DYNAMIC_SOURCES:
        values = pd.to_numeric(ordered[source], errors="coerce")
        for lag in (1, 3):
            name = f"{source}_delta_{lag}d"
            ordered[name] = values - values.shift(lag)
            names.append(name)
        mean_name = f"{source}_mean_3d"
        ordered[mean_name] = values.rolling(3, min_periods=2).mean()
        names.append(mean_name)
    return ordered, names


def relay_causal_baseline(y: pd.Series) -> np.ndarray:
    known = y.notna().astype(int).shift(1, fill_value=0).cumsum().to_numpy(float)
    positive = y.fillna(0).shift(1, fill_value=0).cumsum().to_numpy(float)
    return (positive + 1.0) / (known + 2.0)


def make_model(kind: str) -> Pipeline:
    if kind == "logistic":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.25, max_iter=2000, class_weight=None, random_state=7)),
        ])
    if kind == "hist_gb":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.035, max_iter=120, max_leaf_nodes=7,
                min_samples_leaf=28, l2_regularization=2.0,
                early_stopping=False, random_state=7,
            )),
        ])
    raise ValueError(f"unknown model kind {kind}")


def platt_calibrate(raw_cal: np.ndarray, y_cal: np.ndarray, raw_test: np.ndarray) -> np.ndarray:
    if len(y_cal) < 30 or np.unique(y_cal).size < 2:
        return raw_test
    eps = 1e-6
    x_cal = np.log(np.clip(raw_cal, eps, 1 - eps) / np.clip(1 - raw_cal, eps, 1 - eps)).reshape(-1, 1)
    x_test = np.log(np.clip(raw_test, eps, 1 - eps) / np.clip(1 - raw_test, eps, 1 - eps)).reshape(-1, 1)
    calibrator = LogisticRegression(C=0.5, max_iter=1000, random_state=11)
    calibrator.fit(x_cal, y_cal)
    return calibrator.predict_proba(x_test)[:, 1]


def prequential_predictions(
    frame: pd.DataFrame,
    target: str,
    features: list[str],
    kind: str,
    initial_history: int = INITIAL_HISTORY,
    retrain_every: int = RETRAIN_EVERY,
) -> np.ndarray:
    probabilities = np.full(len(frame), np.nan)
    for position in range(initial_history, len(frame), retrain_every):
        available = frame.iloc[:position].dropna(subset=[target])
        if len(available) < initial_history or available[target].nunique() < 2:
            continue
        stop = min(len(frame), position + retrain_every)
        if kind == "hist_gb":
            calibration_n = max(30, int(len(available) * 0.20))
            core_end = len(available) - calibration_n - 1
            if core_end < 80:
                continue
            core_train = available.iloc[:core_end]
            calibration = available.iloc[core_end + 1:]
            model = make_model(kind)
            model.fit(core_train[features], core_train[target].astype(int))
            raw_cal = model.predict_proba(calibration[features])[:, 1]
            raw_test = model.predict_proba(frame.iloc[position:stop][features])[:, 1]
            probabilities[position:stop] = platt_calibrate(
                raw_cal, calibration[target].astype(int).to_numpy(), raw_test,
            )
        else:
            model = make_model(kind)
            model.fit(available[features], available[target].astype(int))
            probabilities[position:stop] = model.predict_proba(frame.iloc[position:stop][features])[:, 1]
    return probabilities


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if np.unique(y).size == 2 else math.nan


def metrics(y: np.ndarray, p: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(y) & np.isfinite(p) & np.isfinite(baseline)
    y, p, baseline = y[mask], np.clip(p[mask], 1e-6, 1 - 1e-6), np.clip(baseline[mask], 1e-6, 1 - 1e-6)
    if not len(y):
        return {"n": 0}
    brier = float(np.mean((p - y) ** 2))
    baseline_brier = float(np.mean((baseline - y) ** 2))
    cutoff = float(np.quantile(p, 0.80))
    selected = p >= cutoff
    return {
        "n": int(len(y)), "positive_rate": float(y.mean()), "auc": safe_auc(y, p),
        "brier": brier, "baseline_brier": baseline_brier,
        "brier_skill_vs_baseline": 1 - brier / baseline_brier if baseline_brier else math.nan,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "baseline_log_loss": float(log_loss(y, baseline, labels=[0, 1])),
        "log_loss_improvement_vs_baseline": float(log_loss(y, baseline, labels=[0, 1]) - log_loss(y, p, labels=[0, 1])),
        "top_quintile_n": int(selected.sum()),
        "top_quintile_precision": float(y[selected].mean()) if selected.any() else math.nan,
        "top_quintile_lift": float(y[selected].mean() / y.mean()) if selected.any() and y.mean() else math.nan,
    }


def chronological_folds(indices: np.ndarray, folds: int = 3) -> list[np.ndarray]:
    return [part for part in np.array_split(indices, folds) if len(part)]


def evaluate(frame: pd.DataFrame, target: str, p: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    y = frame[target].to_numpy(float)
    valid = np.flatnonzero(np.isfinite(y) & np.isfinite(p) & np.isfinite(baseline))
    all_folds = []
    for number, indices in enumerate(chronological_folds(valid), 1):
        all_folds.append({"fold": number, **metrics(y[indices], p[indices], baseline[indices])})
    active = frame["emotion_state"].isin(ACTIVE_STATES).to_numpy(bool)
    active_indices = valid[active[valid]]
    active_folds = []
    for number, indices in enumerate(chronological_folds(active_indices), 1):
        active_folds.append({"fold": number, **metrics(y[indices], p[indices], baseline[indices])})

    def gate(folds: list[dict[str, Any]]) -> bool:
        return len(folds) == 3 and all(
            item.get("auc", math.nan) > 0.5
            and item.get("brier_skill_vs_baseline", math.nan) > 0
            and item.get("log_loss_improvement_vs_baseline", math.nan) > 0
            for item in folds
        )

    return {
        "overall": metrics(y[valid], p[valid], baseline[valid]),
        "folds": all_folds, "all_date_gate": gate(all_folds),
        "active_overall": metrics(y[active_indices], p[active_indices], baseline[active_indices]),
        "active_folds": active_folds, "active_gate": gate(active_folds),
        "joint_gate": gate(all_folds) and gate(active_folds),
    }


def run(input_path: Path = INPUT, output: Path = OUTPUT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(input_path)
    frame = add_targets(raw)
    frame, dynamic_features = add_causal_dynamics(frame)
    stages = {**FEATURE_STAGES, "plus_causal_dynamics": FEATURE_STAGES["plus_relay_loss"] + dynamic_features}
    missing = sorted({name for values in stages.values() for name in values}.difference(frame.columns))
    if missing:
        raise ValueError(f"emotion lab missing fields: {missing}")
    baselines = {
        "adverse_state_next": pd.to_numeric(frame["emotion_adverse_markov_probability"], errors="coerce").to_numpy(float),
        "relay_damage_next": relay_causal_baseline(frame["relay_damage_next"]),
    }
    prediction_table = frame[["date", "emotion_state", *TARGETS]].copy()
    experiments: list[dict[str, Any]] = []
    for target in TARGETS:
        for stage, features in stages.items():
            kinds = ("logistic", "hist_gb") if stage == "plus_causal_dynamics" else ("logistic",)
            for kind in kinds:
                name = f"{target}__{stage}__{kind}"
                p = prequential_predictions(frame, target, features, kind)
                prediction_table[name] = p
                experiments.append({
                    "name": name, "target": target, "feature_stage": stage,
                    "model": kind, "feature_count": len(features),
                    **evaluate(frame, target, p, baselines[target]),
                })
    prediction_path = output / "prequential_predictions.csv"
    prediction_table.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    stable = [item for item in experiments if item["joint_gate"]]
    result = {
        "version": VERSION, "status": "RESEARCH_ONLY",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_start": str(frame["date"].min()), "data_end": str(frame["date"].max()),
        "rows": int(len(frame)), "initial_history": INITIAL_HISTORY,
        "retrain_every": RETRAIN_EVERY, "one_row_calibration_embargo": True,
        "feature_stages": {key: value for key, value in stages.items()},
        "experiments": experiments, "stable_joint_gate_experiments": [item["name"] for item in stable],
        "promotion_allowed": False,
        "promotion_reason": (
            "reused historical development dates can reject factors but cannot promote a model; "
            "a stable experiment must be frozen and tested on new forward dates"
        ),
        "input": {"path": str(input_path.resolve()), "sha256": sha256_file(input_path)},
        "predictions": {"path": str(prediction_path.resolve()), "sha256": sha256_file(prediction_path)},
    }
    card = output / "emotion_model_lab_card.json"
    card.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["card"] = str(card.resolve())
    return result


def lock_research_candidate(
    card_path: Path = OUTPUT / "emotion_model_lab_card.json",
    input_path: Path = INPUT,
    lock_path: Path = LOCK_PATH,
) -> dict[str, Any]:
    card = json.loads(card_path.read_text(encoding="utf-8"))
    matches = [item for item in card["experiments"] if item["name"] == CANDIDATE_NAME]
    if len(matches) != 1 or not matches[0]["joint_gate"]:
        raise ValueError("research candidate is absent or has not passed the historical joint gate")
    experiment = matches[0]
    payload = {
        "version": "v7-emotion-research-lock-1",
        "candidate_name": CANDIDATE_NAME,
        "target": "adverse_state_next",
        "deployment_status": "FORWARD_RESEARCH_ONLY",
        "development_data_end": card["data_end"],
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "features": FEATURE_STAGES["plus_dimensions"],
        "model": {
            "kind": "logistic", "C": 0.25, "max_iter": 2000,
            "imputer": "median_with_missing_indicators", "scaler": "standard",
            "initial_history": INITIAL_HISTORY, "retrain_every": RETRAIN_EVERY,
        },
        "historical_evidence": {
            "overall": experiment["overall"], "folds": experiment["folds"],
            "active_overall": experiment["active_overall"], "active_folds": experiment["active_folds"],
            "joint_gate": experiment["joint_gate"],
        },
        "integrity": {
            "input_path": str(input_path.resolve()), "input_sha256": sha256_file(input_path),
            "card_path": str(card_path.resolve()), "card_sha256": sha256_file(card_path),
            "script_path": str(Path(__file__).resolve()), "script_sha256": sha256_file(Path(__file__)),
        },
        "promotion_rule": (
            "historical dates are reused development evidence; require at least 60 settled post-lock forecasts, "
            "including 30 active-state forecasts, with positive Brier skill, lower log loss and AUC above 0.5 "
            "in both all-date and active-state views"
        ),
    }
    payload["lock_id"] = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if canonical_json(existing) != canonical_json(payload):
            raise ValueError("research candidate lock is immutable and already exists")
    else:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def verify_lock(lock_path: Path = LOCK_PATH) -> dict[str, Any]:
    errors: list[str] = []
    if not lock_path.exists():
        return {"status": "FAIL", "errors": ["lock missing"]}
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    claimed = lock.get("lock_id", "")
    payload = {key: value for key, value in lock.items() if key != "lock_id"}
    expected = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]
    if claimed != expected:
        errors.append("lock_id mismatch")
    for key in ("input", "card", "script"):
        path = Path(lock["integrity"][f"{key}_path"])
        if not path.exists():
            errors.append(f"{key} file missing")
        elif sha256_file(path) != lock["integrity"][f"{key}_sha256"]:
            errors.append(f"{key} sha256 mismatch")
    return {
        "status": "PASS" if not errors else "FAIL", "lock_id": claimed,
        "candidate_name": lock.get("candidate_name"), "development_data_end": lock.get("development_data_end"),
        "errors": errors, "assurance": "LOCAL_TAMPER_EVIDENT_ONLY",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run time-causal V7 emotion factor lab")
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--lock", action="store_true")
    parser.add_argument("--verify-lock", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.verify_lock:
        summary = verify_lock(args.output / LOCK_PATH.name)
    elif args.lock:
        summary = lock_research_candidate(
            args.output / "emotion_model_lab_card.json", args.input, args.output / LOCK_PATH.name,
        )
    else:
        result = run(args.input, args.output)
        summary = {
            "status": result["status"], "rows": result["rows"],
            "stable_joint_gate_experiments": result["stable_joint_gate_experiments"],
            "promotion_allowed": result["promotion_allowed"], "card": result["card"],
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
