"""Train two independent, time-isolated first-board targets.

Target A estimates an executable B+2 net-positive outcome.  Target B estimates
an actually filled B+1 loss of at least 7%.  They have different outcome
availability dates, models and metrics.  The five requested structural factors
form an enhanced challenger only when their point-in-time evidence is complete;
missing factor values are never converted to zero.  Outputs are diagnostic and
do not authorize trades.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from first_board_profit_ranker_v196 import (
    EMOTION_FEATURES, build_dataset, _eligible_profit_train, _eligible_risk_train,
)


VERSION = "v206-first-board-dual-target-1"
KEYS = ["signal_date", "code"]
FIVE_FACTORS = [
    "turnover_absorption_quality", "theme_purity", "sector_leadership",
    "competitor_elimination", "announcement_soft_risk",
]
STATUS_FIELDS = [name + "_evidence_status" for name in FIVE_FACTORS]
ENHANCED_FEATURES = EMOTION_FEATURES + FIVE_FACTORS + ["same_height_peer_count"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model(train: pd.DataFrame, score: pd.DataFrame, features: list[str], target: str,
           balanced: bool = False) -> np.ndarray:
    y = pd.to_numeric(train[target], errors="raise").astype(int)
    if y.nunique() < 2:
        return np.repeat(float(y.mean()), len(score))
    pipe = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=.5, class_weight="balanced" if balanced else None,
                           max_iter=3000, random_state=206),
    )
    pipe.fit(train[features], y)
    return pipe.predict_proba(score[features])[:, 1]


def attach_targets(features: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    required = set(KEYS + ["name", "same_height_peer_count"] + EMOTION_FEATURES
                   + FIVE_FACTORS + STATUS_FIELDS + ["evidence_available_at"])
    missing = required - set(features)
    if missing:
        raise ValueError(f"dual-target feature fields missing: {sorted(missing)}")
    rows = build_dataset(features, outcomes)
    status_ok = pd.Series(True, index=rows.index)
    for value in STATUS_FIELDS:
        status_ok &= rows[value].fillna("UNAVAILABLE").astype(str).str.upper().isin(
            {"VERIFIED", "POINT_IN_TIME_VERIFIED"})
    numeric_ok = rows[FIVE_FACTORS].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    stamps = pd.to_datetime(rows.evidence_available_at, errors="coerce", utc=True)
    if "feature_cutoff_at" in rows:
        cutoff = pd.to_datetime(rows.feature_cutoff_at, errors="coerce", utc=True)
    else:
        cutoff = pd.to_datetime(rows.signal_date.astype(str) + "T15:30:00+08:00", utc=True)
    rows["five_factor_point_in_time_complete"] = status_ok & numeric_ok & stamps.notna() & cutoff.notna() & stamps.le(cutoff)
    rows["missing_five_factor_is_zero"] = False
    return rows


def _eligible_enhanced(train: pd.DataFrame) -> pd.DataFrame:
    return train[train.five_factor_point_in_time_complete].copy()


def prequential(rows: pd.DataFrame, min_train_sessions: int = 20) -> pd.DataFrame:
    outputs = []
    for score_date in sorted(rows.signal_date.astype(str).unique()):
        current = rows[rows.signal_date.astype(str).eq(score_date)].copy()
        profit = _eligible_profit_train(rows, score_date)
        risk = _eligible_risk_train(rows, score_date)
        if min(profit.signal_date.nunique(), risk.signal_date.nunique()) < min_train_sessions:
            continue
        current["b2_positive_probability_base"] = _model(
            profit, current, EMOTION_FEATURES, "executable_positive_B_plus_2")
        current["b1_big_loss_probability_base"] = _model(
            risk, current, EMOTION_FEATURES, "big_loss_target", balanced=True)
        ep, er = _eligible_enhanced(profit), _eligible_enhanced(risk)
        eligible_current = current.five_factor_point_in_time_complete
        current["b2_positive_probability_enhanced"] = np.nan
        current["b1_big_loss_probability_enhanced"] = np.nan
        if (ep.signal_date.nunique() >= min_train_sessions
                and er.signal_date.nunique() >= min_train_sessions
                and eligible_current.any()):
            current.loc[eligible_current, "b2_positive_probability_enhanced"] = _model(
                ep, current.loc[eligible_current], ENHANCED_FEATURES,
                "executable_positive_B_plus_2")
            current.loc[eligible_current, "b1_big_loss_probability_enhanced"] = _model(
                er, current.loc[eligible_current], ENHANCED_FEATURES,
                "big_loss_target", balanced=True)
        current["b2_profit_train_sessions"] = int(profit.signal_date.nunique())
        current["b1_risk_train_sessions"] = int(risk.signal_date.nunique())
        current["enhanced_profit_train_sessions"] = int(ep.signal_date.nunique())
        current["enhanced_risk_train_sessions"] = int(er.signal_date.nunique())
        current["dual_target_status"] = "TIME_ISOLATED_DIAGNOSTIC"
        current["entry_authorized"] = False
        outputs.append(current)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def _metrics(rows: pd.DataFrame, target: str, prediction: str) -> dict:
    y = pd.to_numeric(rows[target], errors="coerce")
    p = pd.to_numeric(rows[prediction], errors="coerce")
    valid = y.notna() & p.notna()
    y, p = y[valid].astype(int), p[valid]
    return {
        "n": int(len(y)), "positive_rate": float(y.mean()) if len(y) else None,
        "auc": float(roc_auc_score(y, p)) if y.nunique() == 2 else None,
        "brier": float(brier_score_loss(y, p)) if len(y) else None,
        "average_precision": float(average_precision_score(y, p)) if y.nunique() == 2 else None,
    }


def audit(rows: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    if predictions.empty:
        return {
            "version": VERSION, "status": "INSUFFICIENT_TIME_ISOLATED_SAMPLE",
            "targets": ["B+2_executable_net_positive", "B+1_big_loss_at_least_7pct"],
            "entry_authorized": False,
        }
    return {
        "version": VERSION,
        "status": "DIAGNOSTIC_ONLY",
        "sessions": int(predictions.signal_date.nunique()),
        "candidate_rows": int(len(predictions)),
        "five_factor_complete_rows": int(predictions.five_factor_point_in_time_complete.sum()),
        "targets": {
            "B_plus_2_positive": {
                "label": "executable_positive_B_plus_2",
                "outcome_availability": "exit_date_B_plus_2",
                "base": _metrics(predictions, "executable_positive_B_plus_2",
                                 "b2_positive_probability_base"),
                "enhanced": _metrics(predictions, "executable_positive_B_plus_2",
                                     "b2_positive_probability_enhanced"),
            },
            "B_plus_1_big_loss": {
                "label": "big_loss_target",
                "outcome_availability": "exit_date_B_plus_1",
                "base": _metrics(predictions, "big_loss_target",
                                 "b1_big_loss_probability_base"),
                "enhanced": _metrics(predictions, "big_loss_target",
                                     "b1_big_loss_probability_enhanced"),
            },
        },
        "governance": {
            "two_targets_trained_separately": True,
            "B_plus_1_big_loss_threshold": -0.07,
            "outcome_availability_time_isolated": True,
            "five_factor_missing_filled_as_zero": False,
            "next_day_promotion_used_as_profit_target": False,
            "probabilities_deployment_calibrated": False,
            "entry_authorized": False,
        },
    }


def run(features_path, outcomes_path, output_dir, min_train_sessions=20):
    sources = [Path(features_path), Path(outcomes_path)]
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable dual-target audit")
    features = pd.read_csv(sources[0], dtype={"code": str})
    outcomes = pd.read_csv(sources[1], dtype={"code": str})
    rows = attach_targets(features, outcomes)
    predictions = prequential(rows, min_train_sessions)
    report = audit(rows, predictions)
    target.mkdir(parents=True)
    rows_path = target / "dual_target_rows_LABEL_ONLY.csv"
    predictions_path = target / "prequential_dual_target_predictions.csv"
    rows.to_csv(rows_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    report["source_hashes"] = {str(path.resolve()): _sha256(path) for path in sources}
    report["artifact_hashes"] = {
        rows_path.name: _sha256(rows_path), predictions_path.name: _sha256(predictions_path)}
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--outcomes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-train-sessions", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.outcomes, args.output,
                         args.min_train_sessions), ensure_ascii=False, indent=2))
