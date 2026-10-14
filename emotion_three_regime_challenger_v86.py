"""Causal three-regime challenger focused on repair recognition.

This is one predeclared balanced multinomial logistic model.  It is evaluated on
a chronological frozen 70/30 split and compared with the existing expanding
Markov probabilities on exactly the same target dates.
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CLASSES = ["RISK_ON", "RISK_OFF", "REPAIR"]
STATE_TO_CLASS = {
    **{state: "RISK_ON" for state in ("启动", "升温", "加速", "一致")},
    **{state: "RISK_OFF" for state in ("大分歧", "退潮")},
    "修复": "REPAIR",
}
FEATURES = [
    "limit_count", "limit_change", "seal_rate", "failed_rate", "max_board",
    "height_change", "middle_density", "ladder_gap_count", "unique_space",
    "one_word_share", "breadth_height_divergence", "promotion_overall",
    "promotion_mid", "yesterday_premium", "previous_limit_tail_loss",
    "down_limit_count", "breadth_health", "relay_appetite", "ladder_health",
    "loss_effect_health", "regime_fragility", "emotion_quality_score",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=.5, class_weight="balanced", max_iter=3000,
                                     random_state=0)),
    ])


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    missing = {"date", "emotion_state", *FEATURES} - set(frame)
    if missing:
        raise ValueError(f"market fields missing: {sorted(missing)}")
    data = frame.copy()
    data["date"] = pd.to_datetime(data.date, errors="coerce")
    data = data.dropna(subset=["date", "emotion_state"]).sort_values("date").drop_duplicates("date", keep="last")
    unknown = set(data.emotion_state) - set(STATE_TO_CLASS)
    if unknown:
        raise ValueError(f"unknown emotion states: {sorted(unknown)}")
    for column in FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["target_state"] = data.emotion_state.shift(-1)
    data["target_class"] = data.target_state.map(STATE_TO_CLASS)
    data["target_date"] = data.date.shift(-1)
    return data.reset_index(drop=True)


def metrics(y: pd.Series, probabilities: np.ndarray) -> dict:
    y = y.reset_index(drop=True)
    index = y.map(CLASSES.index).to_numpy()
    onehot = np.eye(len(CLASSES))[index]
    predicted_index = probabilities.argmax(axis=1)
    predicted = pd.Series([CLASSES[i] for i in predicted_index])
    confidence = probabilities.max(axis=1)
    correct = predicted.eq(y).to_numpy(float)
    recalls, f1s, by_class = [], [], {}
    for class_name in CLASSES:
        truth = y.eq(class_name)
        called = predicted.eq(class_name)
        tp = int((truth & called).sum())
        precision = tp / int(called.sum()) if called.any() else 0.0
        recall = tp / int(truth.sum()) if truth.any() else None
        f1 = 2 * precision * recall / (precision + recall) if recall is not None and precision + recall else 0.0
        if recall is not None:
            recalls.append(recall)
            f1s.append(f1)
        by_class[class_name] = {"support": int(truth.sum()), "predicted": int(called.sum()),
                                "precision": precision, "recall": recall, "f1": f1}
    ece = 0.0
    for lower in np.linspace(0, .9, 10):
        mask = (confidence >= lower) & (confidence < lower + .1 if lower < .9 else confidence <= 1)
        if mask.any():
            ece += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return {
        "n": int(len(y)), "accuracy": float(correct.mean()),
        "balanced_accuracy": float(np.mean(recalls)), "macro_f1": float(np.mean(f1s)),
        "multiclass_brier": float(np.mean(np.sum((probabilities - onehot) ** 2, axis=1))),
        "log_loss": float(-np.mean(np.log(np.clip(probabilities[np.arange(len(index)), index], 1e-12, 1)))),
        "top_class_ece_10bin": float(ece), "by_class": by_class,
    }


def ordered_probabilities(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(x)
    classes = list(model.named_steps["model"].classes_)
    return np.column_stack([raw[:, classes.index(name)] for name in CLASSES])


def markov_probabilities(rows: pd.DataFrame) -> np.ndarray:
    required = {
        "emotion_next_start_probability", "emotion_next_warming_probability",
        "emotion_next_acceleration_probability", "emotion_next_consensus_probability",
        "emotion_next_divergence_probability", "emotion_next_retreat_probability",
        "emotion_next_repair_probability",
    }
    if required - set(rows):
        raise ValueError("existing Markov probability fields missing")
    risk_on = rows[["emotion_next_start_probability", "emotion_next_warming_probability",
                    "emotion_next_acceleration_probability", "emotion_next_consensus_probability"]].sum(axis=1)
    risk_off = rows[["emotion_next_divergence_probability", "emotion_next_retreat_probability"]].sum(axis=1)
    repair = rows["emotion_next_repair_probability"]
    return np.column_stack([risk_on, risk_off, repair]).astype(float)


def run(frame: pd.DataFrame, split_fraction: float = .70) -> tuple[dict, pd.DataFrame]:
    data = prepare(frame)
    labeled = data.iloc[:-1].copy()
    if len(labeled) < 300:
        raise ValueError("insufficient labeled sessions")
    split = int(len(labeled) * split_fraction)
    train, blind = labeled.iloc[:split], labeled.iloc[split:]
    model = make_model()
    model.fit(train[FEATURES], train.target_class)
    challenger_p = ordered_probabilities(model, blind[FEATURES])
    markov_p = markov_probabilities(blind)
    challenger_metrics = metrics(blind.target_class, challenger_p)
    markov_metrics = metrics(blind.target_class, markov_p)
    repair_recall = challenger_metrics["by_class"]["REPAIR"]["recall"] or 0.0
    promote = bool(
        challenger_metrics["multiclass_brier"] < markov_metrics["multiclass_brier"]
        and challenger_metrics["log_loss"] < markov_metrics["log_loss"]
        and challenger_metrics["balanced_accuracy"] > markov_metrics["balanced_accuracy"]
        and repair_recall >= .15
    )
    final_model = make_model()
    final_model.fit(labeled[FEATURES], labeled.target_class)
    latest_p = ordered_probabilities(final_model, data.tail(1)[FEATURES])[0]
    predictions = pd.DataFrame({
        "signal_date": blind.date.dt.strftime("%Y-%m-%d"),
        "target_date": blind.target_date.dt.strftime("%Y-%m-%d"),
        "target_class": blind.target_class,
    })
    for index, name in enumerate(CLASSES):
        predictions[f"challenger_p_{name}"] = challenger_p[:, index]
        predictions[f"markov_p_{name}"] = markov_p[:, index]
    report = {
        "version": "v86-emotion-three-regime-challenger-1",
        "status": "RESEARCH_ONLY",
        "features": FEATURES,
        "model": "median_imputer_training_only + scaler + balanced_multinomial_logistic_C_0.5",
        "split": {"method": "frozen_chronological_70_30", "train_n": int(len(train)),
                  "blind_n": int(len(blind)), "train_end_signal_date": train.iloc[-1].date.strftime("%Y-%m-%d"),
                  "blind_start_signal_date": blind.iloc[0].date.strftime("%Y-%m-%d")},
        "blind": {"challenger": challenger_metrics, "markov_baseline": markov_metrics},
        "promotion_gate_passed": promote,
        "promotion_rule": "challenger must improve Brier, log loss, balanced accuracy and repair recall>=15%",
        "latest_research_prediction": {
            "signal_date": data.iloc[-1].date.strftime("%Y-%m-%d"),
            "probabilities": {name: float(latest_p[i]) for i, name in enumerate(CLASSES)},
            "top_class": CLASSES[int(latest_p.argmax())],
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
    prediction_path = output / "blind_predictions.csv"
    predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    report["generated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    report["source"] = {"path": str(source.resolve()), "sha256": sha256(source)}
    report["blind_predictions"] = {"path": str(prediction_path.resolve()), "sha256": sha256(prediction_path)}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
