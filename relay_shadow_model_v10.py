#!/usr/bin/env python3
"""Freeze and score the V10 research-only relay shadow model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import emotion_forward_v7
import relay_model_lab_v8 as lab
from emotion_challenger_v4 import ACTIVE_STATES


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "v7" / "point_in_time_rebuild" / "leader_candidates.csv"
EMOTION = ROOT / "results" / "v7" / "emotion_model_lab" / "prequential_predictions.csv"
V8_CARD = ROOT / "results" / "v8" / "relay_model_lab" / "relay_model_lab_card.json"
OUTPUT = ROOT / "results" / "v10" / "relay_shadow_model"
ARTIFACT_NAME = "relay_shadow_bundle.joblib"
LOCK_NAME = "relay_shadow_lock.json"
VERSION = "v10-relay-shadow-model-1"
TARGETS = ("promotion", "fill", "two_more")
FEATURES = list(lab.FEATURE_STAGES["plus_stock_memory"])
MODEL_KIND = "hist_gb"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: Path) -> str:
    return lab.sha256_file(path)


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def fit_bundle(frame: pd.DataFrame, development_end: pd.Timestamp) -> dict[str, Any]:
    bundle: dict[str, Any] = {
        "version": VERSION, "development_data_end": development_end.strftime("%Y-%m-%d"),
        "features": FEATURES, "models": {},
    }
    for target in TARGETS:
        spec = lab.TARGETS[target]
        label, maturity = spec["label"], spec["maturity"]
        train = frame[
            (frame["date"] <= development_end) & frame[label].notna()
            & frame[maturity].notna() & (frame[maturity] <= development_end)
        ].copy()
        dates = np.array(sorted(train["date"].unique()))
        if len(train) < 500 or len(dates) < 160 or train[label].nunique() < 2:
            raise RuntimeError(f"insufficient frozen training data for {target}")
        split = max(1, int(len(dates) * 0.80))
        core_dates = dates[:split]
        embargo = dates[split:split + 1]
        calibration_dates = dates[split + 1:]
        core = train[train["date"].isin(core_dates)]
        calibration = train[train["date"].isin(calibration_dates)]
        if len(core) < 300 or len(calibration) < 60 or calibration[label].nunique() < 2:
            raise RuntimeError(f"insufficient frozen calibration data for {target}")
        model = lab.make_model(MODEL_KIND)
        model.fit(core[FEATURES], core[label].astype(int))
        raw_calibration = model.predict_proba(calibration[FEATURES])[:, 1]
        calibrator = LogisticRegression(C=0.5, max_iter=1000, random_state=19)
        calibrator.fit(_logit(raw_calibration), calibration[label].astype(int))
        bundle["models"][target] = {
            "model": model, "calibrator": calibrator,
            "core_rows": int(len(core)), "calibration_rows": int(len(calibration)),
            "core_end": pd.Timestamp(core_dates[-1]).strftime("%Y-%m-%d"),
            "embargo_date": pd.Timestamp(embargo[0]).strftime("%Y-%m-%d"),
            "calibration_start": pd.Timestamp(calibration_dates[0]).strftime("%Y-%m-%d"),
            "calibration_end": pd.Timestamp(calibration_dates[-1]).strftime("%Y-%m-%d"),
            "positive_rate": float(train[label].mean()),
        }
    return bundle


def predict_bundle(bundle: dict[str, Any], frame: pd.DataFrame) -> dict[str, np.ndarray]:
    result = {}
    for target in TARGETS:
        item = bundle["models"][target]
        raw = item["model"].predict_proba(frame[FEATURES])[:, 1]
        result[target] = item["calibrator"].predict_proba(_logit(raw))[:, 1]
    return result


def freeze(
    input_path: Path = INPUT,
    emotion_path: Path = EMOTION,
    output: Path = OUTPUT,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / LOCK_NAME
    artifact_path = output / ARTIFACT_NAME
    if lock_path.exists() or artifact_path.exists():
        raise RuntimeError("V10 shadow freeze already exists; use verify")
    frame = lab.prepare_frame(input_path, emotion_path)
    development_end = frame["date"].max()
    bundle = fit_bundle(frame, development_end)
    joblib.dump(bundle, artifact_path, compress=3)
    v8_card = json.loads(V8_CARD.read_text(encoding="utf-8"))
    selected = {}
    for target in TARGETS:
        name = f"{target}__plus_stock_memory__hist_gb"
        experiment = next(item for item in v8_card["experiments"] if item["name"] == name)
        if not experiment["stability_gate"]:
            raise RuntimeError(f"selected V8 component is not stable: {name}")
        selected[target] = {
            "experiment": name, "historical_overall": experiment["overall"],
            "historical_folds": experiment["folds"], "active": experiment["active"],
        }
    payload = {
        "version": VERSION, "deployment_status": "FORWARD_SHADOW_ONLY",
        "development_data_end": development_end.strftime("%Y-%m-%d"),
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_components": selected,
        "joint_score": "0.65*p_two_more + 0.25*p_promotion + 0.10*p_fill_proxy",
        "large_loss_model": "REJECTED_NOT_USED",
        "scope": "active emotion, mainboard 3--5 boards, prior day buyable, daily top one",
        "features": FEATURES,
        "integrity": {
            "input_path": str(input_path.resolve()), "input_sha256": sha256_file(input_path),
            "emotion_path": str(emotion_path.resolve()), "emotion_sha256": sha256_file(emotion_path),
            "v8_card_path": str(V8_CARD.resolve()), "v8_card_sha256": sha256_file(V8_CARD),
            "v8_script_path": str(Path(lab.__file__).resolve()), "v8_script_sha256": sha256_file(Path(lab.__file__)),
            "artifact_path": str(artifact_path.resolve()), "artifact_sha256": sha256_file(artifact_path),
            "script_path": str(Path(__file__).resolve()), "script_sha256": sha256_file(Path(__file__)),
        },
        "promotion_rule": (
            "never emits ENTER; require preregistered post-lock shadow evidence and the original 70% promotion / "
            "10% large-loss / minimum-100 executed-signal deployment gate"
        ),
    }
    payload["lock_id"] = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]
    lock_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "FROZEN", "lock_id": payload["lock_id"], "lock": str(lock_path.resolve())}


def verify(output: Path = OUTPUT) -> dict[str, Any]:
    lock_path = output / LOCK_NAME
    if not lock_path.exists():
        return {"status": "FAIL", "errors": ["lock missing"]}
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    claimed = lock.get("lock_id", "")
    payload = {key: value for key, value in lock.items() if key != "lock_id"}
    expected = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]
    errors = []
    if claimed != expected:
        errors.append("lock_id mismatch")
    integrity = lock.get("integrity", {})
    for key in ("input", "emotion", "v8_card", "v8_script", "artifact", "script"):
        path = Path(integrity.get(f"{key}_path", ""))
        if not path.exists():
            errors.append(f"{key} missing")
        elif sha256_file(path) != integrity.get(f"{key}_sha256"):
            errors.append(f"{key} sha256 mismatch")
    return {
        "status": "PASS" if not errors else "FAIL", "lock_id": claimed,
        "development_data_end": lock.get("development_data_end"), "errors": errors,
        "assurance": "LOCAL_TAMPER_EVIDENT_ONLY",
    }


def load_lock_and_bundle(output: Path = OUTPUT) -> tuple[dict[str, Any], dict[str, Any]]:
    verification = verify(output)
    if verification["status"] != "PASS":
        raise RuntimeError("V10 shadow lock verification failed: " + "; ".join(verification["errors"]))
    lock = json.loads((output / LOCK_NAME).read_text(encoding="utf-8"))
    bundle = joblib.load(Path(lock["integrity"]["artifact_path"]))
    return lock, bundle


def v7_forecast(signal_date: str) -> dict[str, Any]:
    verification = emotion_forward_v7.verify()
    if verification["status"] != "PASS":
        raise RuntimeError("V7 emotion forward ledger is invalid")
    events = [
        event for event in emotion_forward_v7.read_events()
        if event.get("event_type") == "FORECAST" and event.get("signal_date") == signal_date
    ]
    if len(events) != 1:
        raise RuntimeError("exactly one locked V7 emotion forecast is required for signal date")
    return events[0]


def build_scores(
    input_path: Path,
    signal_date: str,
    emotion_event: dict[str, Any],
    output: Path = OUTPUT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    lock, bundle = load_lock_and_bundle(output)
    signal = pd.Timestamp(signal_date).normalize()
    if signal <= pd.Timestamp(lock["development_data_end"]):
        raise RuntimeError("V10 forward score must be after the locked development period")
    frame = lab.prepare_frame(input_path, Path(lock["integrity"]["emotion_path"]))
    if frame["date"].max() != signal:
        raise RuntimeError("signal date must be the latest candidate date")
    current = frame[frame["date"].eq(signal)].copy()
    if current.empty:
        raise RuntimeError("no 2--5 board candidates for signal date")
    if str(emotion_event.get("signal_date")) != signal.strftime("%Y-%m-%d"):
        raise RuntimeError("V7 emotion event date does not match relay signal date")
    current["emotion_v7_adverse_probability"] = float(emotion_event["probability"])
    probabilities = predict_bundle(bundle, current)
    result = current[[
        "date", "code", "name", "boards", "emotion_state", "valid_identity",
        "buyable_eod", "one_word_proxy", "announcement_risk", "source", "source_timestamp",
    ]].copy()
    for target, values in probabilities.items():
        result[f"p_{target}"] = values
    result["joint_score"] = (
        0.65 * result["p_two_more"] + 0.25 * result["p_promotion"] + 0.10 * result["p_fill"]
    )
    result["shadow_scope"] = (
        result["boards"].between(3, 5)
        & result["emotion_state"].isin(ACTIVE_STATES)
        & pd.to_numeric(result["buyable_eod"], errors="coerce").eq(1)
        & pd.to_numeric(result["one_word_proxy"], errors="coerce").eq(0)
    )
    result["shadow_observation"] = False
    eligible = result[result["shadow_scope"]]
    if not eligible.empty:
        top = eligible.sort_values(["joint_score", "boards", "code"], ascending=[False, False, True]).index[0]
        result.loc[top, "shadow_observation"] = True
    result["model_lock_id"] = lock["lock_id"]
    result["emotion_lock_id"] = emotion_event["lock_id"]
    result["emotion_event_hash"] = emotion_event["event_hash"]
    metadata = {
        "model_lock_id": lock["lock_id"], "development_data_end": lock["development_data_end"],
        "emotion_lock_id": emotion_event["lock_id"], "emotion_event_hash": emotion_event["event_hash"],
        "input_path": str(input_path.resolve()), "input_sha256": sha256_file(input_path),
        "deployment_allowed": False, "decision": "SHADOW_ONLY_NEVER_ENTER",
    }
    return result.sort_values(["shadow_observation", "joint_score"], ascending=[False, False]), metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze or verify V10 relay shadow model")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    sub.add_parser("verify")
    score = sub.add_parser("score")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--signal-date", required=True)
    score.add_argument("--output-file", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "freeze":
        summary = freeze()
    elif args.command == "verify":
        summary = verify()
    else:
        event = v7_forecast(pd.Timestamp(args.signal_date).strftime("%Y-%m-%d"))
        scores, metadata = build_scores(args.input, args.signal_date, event)
        if args.output_file.exists():
            raise RuntimeError("immutable V10 score file already exists")
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(args.output_file, index=False, encoding="utf-8-sig")
        summary = {**metadata, "rows": int(len(scores)), "scores": str(args.output_file.resolve())}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
