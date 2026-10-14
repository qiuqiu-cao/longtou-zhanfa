"""Predict recovery versus continued risk-off after divergence/retreat days."""
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
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from emotion_three_regime_challenger_v86 import FEATURES, prepare


CURRENT_RISK_OFF = {"大分歧", "退潮"}
RECOVERY_NEXT = {"启动", "升温", "加速", "一致", "修复"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=.5, class_weight="balanced", max_iter=3000,
                                     random_state=0)),
    ])


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    predicted = p >= .5
    tp = int((predicted & y).sum())
    fp = int((predicted & ~y).sum())
    fn = int((~predicted & y).sum())
    tn = int((~predicted & ~y).sum())
    return {
        "n": int(len(y)), "positive_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
        "brier": float(np.mean((p - y.astype(float)) ** 2)),
        "log_loss": float(-np.mean(y * np.log(np.clip(p, 1e-12, 1))
                                   + (~y) * np.log(np.clip(1 - p, 1e-12, 1)))),
        "accuracy_at_0_5": float((predicted == y).mean()),
        "precision_at_0_5": tp / (tp + fp) if tp + fp else None,
        "recall_at_0_5": tp / (tp + fn) if tp + fn else None,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def run(frame: pd.DataFrame, split_fraction: float = .70) -> tuple[dict, pd.DataFrame]:
    data = prepare(frame)
    cases = data.iloc[:-1].loc[data.iloc[:-1].emotion_state.isin(CURRENT_RISK_OFF)].copy()
    cases["recovery_next"] = cases.target_state.isin(RECOVERY_NEXT)
    if len(cases) < 120 or cases.recovery_next.nunique() < 2:
        raise ValueError("insufficient risk-off transition cases")
    split = int(len(cases) * split_fraction)
    train, blind = cases.iloc[:split], cases.iloc[split:]
    estimator = model()
    estimator.fit(train[FEATURES], train.recovery_next)
    p = estimator.predict_proba(blind[FEATURES])[:, list(estimator.classes_).index(True)]
    prior = float(train.recovery_next.mean())
    baseline_p = np.full(len(blind), prior)
    challenger_metrics = binary_metrics(blind.recovery_next.to_numpy(bool), p)
    baseline_metrics = binary_metrics(blind.recovery_next.to_numpy(bool), baseline_p)
    passed = bool(
        challenger_metrics["brier"] < baseline_metrics["brier"]
        and challenger_metrics["log_loss"] < baseline_metrics["log_loss"]
        and (challenger_metrics["auc"] or 0) >= .55
    )
    final = model()
    final.fit(cases[FEATURES], cases.recovery_next)
    latest = data.iloc[-1]
    if latest.emotion_state in CURRENT_RISK_OFF:
        latest_p = float(final.predict_proba(data.tail(1)[FEATURES])[:, list(final.classes_).index(True)][0])
        latest_prediction = {
            "signal_date": latest.date.strftime("%Y-%m-%d"),
            "current_state": latest.emotion_state,
            "recovery_probability": latest_p,
            "continued_risk_off_probability": 1 - latest_p,
            "top_class": "RECOVERY" if latest_p >= .5 else "CONTINUED_RISK_OFF",
            "status": "RESEARCH_ONLY_NOT_FORWARD_VALIDATED",
        }
    else:
        latest_prediction = {"signal_date": latest.date.strftime("%Y-%m-%d"),
                             "current_state": latest.emotion_state,
                             "status": "NOT_APPLICABLE_CURRENT_STATE"}
    predictions = pd.DataFrame({
        "signal_date": blind.date.dt.strftime("%Y-%m-%d"),
        "target_date": blind.target_date.dt.strftime("%Y-%m-%d"),
        "current_state": blind.emotion_state,
        "target_state": blind.target_state,
        "recovery_actual": blind.recovery_next,
        "challenger_recovery_probability": p,
        "prior_recovery_probability": baseline_p,
    })
    report = {
        "version": "v93-riskoff-recovery-challenger-1", "status": "RESEARCH_ONLY",
        "scope": "only signal dates whose state is 大分歧 or 退潮",
        "target": "next state is 启动/升温/加速/一致/修复 versus 大分歧/退潮",
        "features": FEATURES,
        "split": {"method": "chronological_70_30", "train_n": int(len(train)),
                  "blind_n": int(len(blind)), "train_end": train.iloc[-1].date.strftime("%Y-%m-%d"),
                  "blind_start": blind.iloc[0].date.strftime("%Y-%m-%d")},
        "blind": {"challenger": challenger_metrics, "train_prior_baseline": baseline_metrics},
        "promotion_gate_passed": passed,
        "promotion_rule": "blind Brier and log loss improve train-prior baseline; AUC>=0.55",
        "latest_research_prediction": latest_prediction,
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
    report["predictions"] = {"path": str(prediction_path.resolve()), "sha256": sha256(prediction_path)}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
