"""Separate effective repair, weak/narrow repair, and continued risk-off.

The model is restricted to signal days already classified as 大分歧 or 退潮.
Targets use next-session close evidence; features use signal-day close evidence.
It is research-only and cannot authorize a trade.
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

from emotion_three_regime_challenger_v86 import FEATURES


CLASSES = ["CONTINUED_RISK_OFF", "WEAK_OR_NARROW_REPAIR", "EFFECTIVE_REPAIR"]
RISK_OFF_STATES = {"大分歧", "退潮"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            C=.5, class_weight="balanced", max_iter=3000, random_state=0,
        )),
    ])


def prepare(frame: pd.DataFrame, effective_score_required: int = 3) -> pd.DataFrame:
    required = {
        "date", "emotion_state", "previous_limit_median_return", "promotion_mid",
        "failed_rate", "down_limit_count", *FEATURES,
    }
    if required - set(frame):
        raise ValueError(f"market fields missing: {sorted(required - set(frame))}")
    if effective_score_required not in {2, 3, 4}:
        raise ValueError("effective repair score must be 2, 3, or 4")
    data = frame.copy()
    data["date"] = pd.to_datetime(data.date, errors="coerce")
    data = data.dropna(subset=["date", "emotion_state"]).sort_values("date")
    data = data.drop_duplicates("date", keep="last").reset_index(drop=True)
    numeric = set(FEATURES) | {
        "previous_limit_median_return", "promotion_mid", "failed_rate", "down_limit_count",
    }
    for field in numeric:
        data[field] = pd.to_numeric(data[field], errors="coerce")
    next_state = data.emotion_state.shift(-1)
    next_premium = data.previous_limit_median_return.shift(-1)
    next_promotion = data.promotion_mid.shift(-1)
    next_failed = data.failed_rate.shift(-1)
    next_down = data.down_limit_count.shift(-1)
    support = ~next_state.isin(RISK_OFF_STATES)
    available = pd.concat(
        [next_state, next_premium, next_promotion, next_failed, next_down], axis=1
    ).notna().all(axis=1)
    quality_score = (
        next_premium.gt(0).astype(int)
        + next_promotion.ge(.25).astype(int)
        + next_failed.le(.30).astype(int)
        + next_down.le(5).astype(int)
    )
    label = pd.Series("CONTINUED_RISK_OFF", index=data.index, dtype=object)
    label.loc[support] = "WEAK_OR_NARROW_REPAIR"
    label.loc[support & quality_score.ge(effective_score_required)] = "EFFECTIVE_REPAIR"
    data["target_date"] = data.date.shift(-1)
    data["target_state"] = next_state
    data["target_quality_score"] = quality_score
    data["target_class"] = label.where(available)
    return data


def ordered_probabilities(estimator: Pipeline, x: pd.DataFrame) -> np.ndarray:
    raw = estimator.predict_proba(x)
    names = list(estimator.named_steps["model"].classes_)
    return np.column_stack([raw[:, names.index(name)] for name in CLASSES])


def metrics(y: pd.Series, probabilities: np.ndarray) -> dict:
    truth = y.reset_index(drop=True)
    truth_index = truth.map(CLASSES.index).to_numpy()
    onehot = np.eye(len(CLASSES))[truth_index]
    predicted = pd.Series([CLASSES[index] for index in probabilities.argmax(axis=1)])
    recalls, f1s, by_class = [], [], {}
    for name in CLASSES:
        actual, called = truth.eq(name), predicted.eq(name)
        tp = int((actual & called).sum())
        precision = tp / int(called.sum()) if called.any() else 0.0
        recall = tp / int(actual.sum()) if actual.any() else None
        f1 = (2 * precision * recall / (precision + recall)
              if recall is not None and precision + recall else 0.0)
        if recall is not None:
            recalls.append(recall)
            f1s.append(f1)
        by_class[name] = {
            "support": int(actual.sum()), "predicted": int(called.sum()),
            "precision": precision, "recall": recall, "f1": f1,
        }
    return {
        "n": int(len(truth)),
        "accuracy": float(predicted.eq(truth).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "multiclass_brier": float(np.mean(np.sum((probabilities - onehot) ** 2, axis=1))),
        "log_loss": float(-np.mean(np.log(np.clip(
            probabilities[np.arange(len(truth_index)), truth_index], 1e-12, 1,
        )))),
        "by_class": by_class,
    }


def run(frame: pd.DataFrame, split_fraction: float = .70) -> tuple[dict, pd.DataFrame]:
    data = prepare(frame)
    labeled = data.loc[
        data.emotion_state.isin(RISK_OFF_STATES) & data.target_class.notna()
    ].copy()
    if len(labeled) < 180 or set(labeled.target_class) != set(CLASSES):
        raise ValueError("insufficient risk-off cases for all repair-quality classes")
    split = int(len(labeled) * split_fraction)
    raw_train, blind = labeled.iloc[:split].copy(), labeled.iloc[split:].copy()
    if blind.empty:
        raise ValueError("empty chronological blind segment")
    blind_start = blind.iloc[0].date
    train = raw_train[raw_train.target_date.lt(blind_start)].copy()
    embargoed = int(len(raw_train) - len(train))
    if train.empty or train.target_date.max() >= blind_start:
        raise ValueError("one-session label embargo failed")
    estimator = make_model()
    estimator.fit(train[FEATURES], train.target_class)
    challenger = ordered_probabilities(estimator, blind[FEATURES])
    priors = train.target_class.value_counts(normalize=True)
    baseline = np.tile([float(priors.get(name, 0)) for name in CLASSES], (len(blind), 1))
    challenger_metrics, baseline_metrics = metrics(blind.target_class, challenger), metrics(blind.target_class, baseline)
    effective_recall = challenger_metrics["by_class"]["EFFECTIVE_REPAIR"]["recall"] or 0.0
    all_blind_classes = all(
        challenger_metrics["by_class"][name]["support"] >= 10 for name in CLASSES
    )
    promoted = bool(
        all_blind_classes
        and challenger_metrics["multiclass_brier"] < baseline_metrics["multiclass_brier"]
        and challenger_metrics["log_loss"] < baseline_metrics["log_loss"]
        and challenger_metrics["balanced_accuracy"] > baseline_metrics["balanced_accuracy"]
        and challenger_metrics["macro_f1"] > baseline_metrics["macro_f1"]
        and effective_recall >= .15
    )
    final = make_model()
    final.fit(labeled[FEATURES], labeled.target_class)
    latest = data.iloc[-1]
    if latest.emotion_state in RISK_OFF_STATES:
        latest_p = ordered_probabilities(final, data.tail(1)[FEATURES])[0]
        latest_prediction = {
            "signal_date": latest.date.strftime("%Y-%m-%d"),
            "target_date": None,
            "current_state": latest.emotion_state,
            "probabilities": {name: float(latest_p[index]) for index, name in enumerate(CLASSES)},
            "top_class": CLASSES[int(latest_p.argmax())],
            "status": "RESEARCH_ONLY_NOT_FORWARD_VALIDATED",
        }
    else:
        latest_prediction = {
            "signal_date": latest.date.strftime("%Y-%m-%d"),
            "current_state": latest.emotion_state,
            "status": "NOT_APPLICABLE_CURRENT_STATE",
        }
    predictions = pd.DataFrame({
        "signal_date": blind.date.dt.strftime("%Y-%m-%d"),
        "target_date": blind.target_date.dt.strftime("%Y-%m-%d"),
        "target_state": blind.target_state,
        "target_quality_score": blind.target_quality_score,
        "target_class": blind.target_class,
    })
    for index, name in enumerate(CLASSES):
        predictions[f"challenger_p_{name}"] = challenger[:, index]
        predictions[f"prior_p_{name}"] = baseline[:, index]
    sensitivity = {}
    for required_score in (2, 3, 4):
        alternative = prepare(frame, required_score)
        counts = alternative.loc[
            alternative.emotion_state.isin(RISK_OFF_STATES) & alternative.target_class.notna(),
            "target_class",
        ].value_counts()
        sensitivity[str(required_score)] = {
            name: int(counts.get(name, 0)) for name in CLASSES
        }
    report = {
        "version": "v118-emotion-repair-quality-1",
        "status": "RESEARCH_ONLY",
        "scope": "only signal dates classified as 大分歧 or 退潮",
        "target_definition": {
            "CONTINUED_RISK_OFF": "next state remains 大分歧 or 退潮",
            "WEAK_OR_NARROW_REPAIR": "next state exits risk-off but fewer than 3 of 4 quality checks pass",
            "EFFECTIVE_REPAIR": "next state exits risk-off and at least 3 of 4 quality checks pass",
            "quality_checks": [
                "next-day previous-limit median return > 0",
                "next-day middle-board promotion >= 25%",
                "next-day failed-board rate <= 30%",
                "next-day down-limit count <= 5",
            ],
            "threshold_selection": "predeclared economic interpretation; not optimized on blind results",
        },
        "features": FEATURES,
        "split": {
            "method": "chronological_70_30_with_one_session_label_embargo",
            "train_n": int(len(train)), "embargoed_n": embargoed, "blind_n": int(len(blind)),
            "train_last_target_date": train.target_date.max().strftime("%Y-%m-%d"),
            "blind_start_signal_date": blind_start.strftime("%Y-%m-%d"),
        },
        "class_count_sensitivity_by_required_quality_checks": sensitivity,
        "blind": {"challenger": challenger_metrics, "train_prior_baseline": baseline_metrics},
        "promotion_gate_passed": promoted,
        "promotion_rule": (
            "each blind class support>=10; challenger improves Brier, log loss, balanced accuracy, "
            "and macro F1 versus train-prior baseline; effective-repair recall>=15%"
        ),
        "latest_research_prediction": latest_prediction,
        "decision_authorized": False,
        "position": 0,
        "limitations": [
            "repair quality is known only at the target close and is not an intraday buy signal",
            "threshold sensitivity is disclosed and the blind period is not used to choose a threshold",
            "forward frozen validation is required even if the historical gate passes",
        ],
    }
    return report, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="explicit immutable point-in-time market-emotion table")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.input), Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable experiment directory")
    report, predictions = run(pd.read_csv(source))
    output.mkdir(parents=True)
    predictions_path = output / "blind_predictions.csv"
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    report["generated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    report["source"] = {"path": str(source.resolve()), "sha256": sha256(source)}
    report["blind_predictions"] = {
        "path": str(predictions_path.resolve()), "sha256": sha256(predictions_path),
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
