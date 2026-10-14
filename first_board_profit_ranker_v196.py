"""Build a time-isolated first-board ranker for executable B+2 profit.

The target is deliberately different from next-session promotion.  A candidate
is positive only when the opening-entry proxy is observable, executable and has
a positive net return at B+2.  An unverifiable limit-up queue is a known zero
for this *executable opportunity* target.  Missing B bars and right-censored
B+2 exits remain unavailable and are never converted to losses.
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
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


KEYS = ["signal_date", "code"]
BASE_FEATURES = [
    "first_seal_minute", "last_minus_first_minute", "failed_seal_count",
    "turnover", "amount_to_float_cap", "seal_fund_to_float_cap",
    "one_price_proxy", "ultra_fast", "fast_seal", "thin_control_risk",
    "validated_fast_seal", "industry_first_board_count",
    "industry_all_board_count", "industry_higher_board_count",
    "market_first_board_count", "market_relay_count",
]
EMOTION_FEATURES = BASE_FEATURES + [
    "emotion_score", "failed_rate", "down_limit_count", "promotion_1to2",
    "relay_appetite", "crowding_risk", "regime_fragility",
    "emotion_adverse_next_probability", "emotion_supportive_regime",
]
MISSING_EXECUTION = {"MISSING_B_BAR", "INVALID_B_PRICE"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fit_predict(train: pd.DataFrame, score: pd.DataFrame, features: list[str],
                 target: str) -> np.ndarray:
    y = pd.to_numeric(train[target], errors="raise").astype(int)
    if y.nunique() < 2:
        return np.repeat(float(y.mean()), len(score))
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=.5, max_iter=3000, random_state=196),
    )
    model.fit(train[features], y)
    return model.predict_proba(score[features])[:, 1]


def build_dataset(features: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    required_features = set(KEYS + ["name"] + EMOTION_FEATURES)
    required_outcomes = set(KEYS + [
        "fill_status", "exit_date_B_plus_1", "exit_date_B_plus_2",
        "net_return_B_plus_1", "net_return_B_plus_2", "big_loss_B_plus_1",
    ])
    if required_features - set(features):
        raise ValueError(f"feature fields missing: {sorted(required_features - set(features))}")
    if required_outcomes - set(outcomes):
        raise ValueError(f"outcome fields missing: {sorted(required_outcomes - set(outcomes))}")
    if features.duplicated(KEYS).any() or outcomes.duplicated(KEYS).any():
        raise ValueError("candidate keys must be unique")
    outcome_columns = list(required_outcomes)
    rows = features.merge(outcomes[outcome_columns], on=KEYS, how="left", validate="one_to_one")
    filled = rows.fill_status.eq("FILLED_OPEN_PROXY")
    queue = rows.fill_status.eq("UNVERIFIABLE_LIMIT_QUEUE")
    b2 = pd.to_numeric(rows.net_return_B_plus_2, errors="coerce")
    missing_execution = rows.fill_status.isin(MISSING_EXECUTION) | rows.fill_status.isna()
    rows["executable_positive_B_plus_2"] = np.nan
    rows.loc[queue, "executable_positive_B_plus_2"] = 0.0
    rows.loc[filled & b2.notna(), "executable_positive_B_plus_2"] = (
        b2[filled & b2.notna()].gt(0).astype(float))
    rows["profit_target_available"] = rows.executable_positive_B_plus_2.notna().astype(int)
    rows["execution_evidence_missing"] = missing_execution.astype(int)
    big = pd.to_numeric(rows.big_loss_B_plus_1, errors="coerce")
    rows["big_loss_target"] = np.where(filled & big.notna(), big, np.nan)
    rows["big_loss_target_available"] = pd.Series(rows.big_loss_target).notna().astype(int)
    return rows


def _eligible_profit_train(rows: pd.DataFrame, score_date: str) -> pd.DataFrame:
    exit_date = rows.exit_date_B_plus_2.fillna("").astype(str)
    return rows[
        rows.profit_target_available.eq(1)
        & rows.signal_date.astype(str).lt(score_date)
        & exit_date.ne("")
        & exit_date.le(score_date)
    ]


def _eligible_risk_train(rows: pd.DataFrame, score_date: str) -> pd.DataFrame:
    exit_date = rows.exit_date_B_plus_1.fillna("").astype(str)
    return rows[
        rows.big_loss_target_available.eq(1)
        & rows.signal_date.astype(str).lt(score_date)
        & exit_date.ne("")
        & exit_date.le(score_date)
    ]


def prequential_rank(rows: pd.DataFrame, min_train_sessions: int = 10) -> pd.DataFrame:
    scored = []
    for score_date in sorted(rows.signal_date.astype(str).unique()):
        current = rows[rows.signal_date.astype(str).eq(score_date)].copy()
        profit_train = _eligible_profit_train(rows, score_date)
        if profit_train.signal_date.nunique() < min_train_sessions:
            continue
        current["profit_base_probability"] = _fit_predict(
            profit_train, current, BASE_FEATURES, "executable_positive_B_plus_2")
        current["profit_emotion_probability"] = _fit_predict(
            profit_train, current, EMOTION_FEATURES, "executable_positive_B_plus_2")
        risk_train = _eligible_risk_train(rows, score_date)
        if risk_train.empty:
            current["big_loss_probability"] = np.nan
            current["profit_risk_adjusted_score"] = current.profit_emotion_probability
        else:
            current["big_loss_probability"] = _fit_predict(
                risk_train, current, EMOTION_FEATURES, "big_loss_target")
            current["profit_risk_adjusted_score"] = (
                current.profit_emotion_probability * (1 - current.big_loss_probability))
        current["profit_train_sessions"] = int(profit_train.signal_date.nunique())
        current["profit_train_rows"] = int(len(profit_train))
        current["risk_train_sessions"] = int(risk_train.signal_date.nunique())
        current["risk_train_rows"] = int(len(risk_train))
        scored.append(current)
    return pd.concat(scored, ignore_index=True) if scored else pd.DataFrame()


def _candidate_metrics(rows: pd.DataFrame, score: str) -> dict:
    known = rows[rows.profit_target_available.eq(1)].copy()
    y = pd.to_numeric(known.executable_positive_B_plus_2, errors="coerce")
    p = pd.to_numeric(known[score], errors="coerce")
    valid = y.notna() & p.notna()
    y, p = y[valid], p[valid]
    return {
        "known_rows": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else None,
        "auc": float(roc_auc_score(y, p)) if y.nunique() == 2 else None,
        "brier": float(brier_score_loss(y, p)) if len(y) else None,
    }


def daily_top1_metrics(rows: pd.DataFrame, score: str) -> tuple[pd.DataFrame, dict]:
    selected = (rows.sort_values(
        ["signal_date", score, "code"], ascending=[True, False, True], na_position="last")
        .groupby("signal_date", as_index=False, group_keys=False).head(1).reset_index(drop=True))
    known = selected[selected.profit_target_available.eq(1)]
    filled = selected[selected.fill_status.eq("FILLED_OPEN_PROXY")]
    b1 = pd.to_numeric(filled.net_return_B_plus_1, errors="coerce").dropna()
    b2 = pd.to_numeric(filled.net_return_B_plus_2, errors="coerce").dropna()
    big = pd.to_numeric(filled.big_loss_B_plus_1, errors="coerce").dropna()
    return selected, {
        "selected_sessions": int(len(selected)),
        "target_observed_sessions": int(len(known)),
        "target_unavailable_sessions": int(len(selected) - len(known)),
        "executable_positive_B_plus_2_count": int(
            pd.to_numeric(known.executable_positive_B_plus_2, errors="coerce").sum()),
        "executable_positive_B_plus_2_precision": (
            float(pd.to_numeric(known.executable_positive_B_plus_2, errors="coerce").mean())
            if len(known) else None),
        "unverifiable_limit_queue_sessions": int(
            selected.fill_status.eq("UNVERIFIABLE_LIMIT_QUEUE").sum()),
        "filled_trades": int(len(filled)),
        "B_plus_1_mean_net_return": float(b1.mean()) if len(b1) else None,
        "B_plus_1_positive_rate": float(b1.gt(0).mean()) if len(b1) else None,
        "B_plus_2_mean_net_return": float(b2.mean()) if len(b2) else None,
        "B_plus_2_positive_rate_filled": float(b2.gt(0).mean()) if len(b2) else None,
        "B_plus_1_big_loss_rate": float(big.mean()) if len(big) else None,
    }


def _score_current(rows: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    if current.empty:
        return current.copy()
    score_date = str(current.signal_date.astype(str).max())
    profit_train = _eligible_profit_train(rows, score_date)
    risk_train = _eligible_risk_train(rows, score_date)
    result = current.copy()
    result["profit_base_probability"] = _fit_predict(
        profit_train, result, BASE_FEATURES, "executable_positive_B_plus_2")
    result["profit_emotion_probability"] = _fit_predict(
        profit_train, result, EMOTION_FEATURES, "executable_positive_B_plus_2")
    result["big_loss_probability"] = _fit_predict(
        risk_train, result, EMOTION_FEATURES, "big_loss_target")
    result["profit_risk_adjusted_score"] = (
        result.profit_emotion_probability * (1 - result.big_loss_probability))
    result["execution_gate"] = "RESEARCH_ONLY_NO_ENTRY_AUTHORIZATION"
    return result.sort_values(
        ["profit_risk_adjusted_score", "code"], ascending=[False, True])


def audit(rows: pd.DataFrame, promotion_predictions: pd.DataFrame | None = None,
          min_train_sessions: int = 10) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    predictions = prequential_rank(rows, min_train_sessions=min_train_sessions)
    if predictions.empty:
        raise ValueError("insufficient time-isolated sessions for profit ranker")
    right_censored = (
        predictions.fill_status.eq("FILLED_OPEN_PROXY")
        & pd.to_numeric(predictions.net_return_B_plus_2, errors="coerce").isna())
    complete_dates = set(
        predictions.assign(_right_censored=right_censored)
        .groupby("signal_date")._right_censored.any().loc[lambda x: ~x].index.astype(str))
    score_names = [
        "profit_base_probability", "profit_emotion_probability",
        "profit_risk_adjusted_score",
    ]
    if promotion_predictions is not None and not promotion_predictions.empty:
        comparison = promotion_predictions[KEYS + ["base_quality_probability"]]
        predictions = predictions.merge(comparison, on=KEYS, how="left", validate="one_to_one")
        if predictions.base_quality_probability.notna().any():
            score_names.append("base_quality_probability")
    evaluation = predictions[predictions.signal_date.astype(str).isin(complete_dates)].copy()
    candidate = {name: _candidate_metrics(evaluation, name) for name in score_names}
    selected_frames, top1 = [], {}
    for name in score_names:
        eligible = evaluation[evaluation[name].notna()].copy()
        selected, metrics = daily_top1_metrics(eligible, name)
        selected["ranker"] = name
        selected_frames.append(selected)
        top1[name] = metrics
    selected_all = pd.concat(selected_frames, ignore_index=True)
    report = {
        "version": "v196-first-board-profit-ranker-1",
        "status": "DIAGNOSTIC_ONLY",
        "target": "observable_and_executable_open_proxy_with_positive_net_return_at_B_plus_2",
        "scored_sessions": int(predictions.signal_date.nunique()),
        "evaluation_sessions": int(evaluation.signal_date.nunique()),
        "right_censored_sessions_excluded": int(
            predictions.signal_date.nunique() - evaluation.signal_date.nunique()),
        "scored_candidate_rows": int(len(predictions)),
        "evaluation_candidate_rows": int(len(evaluation)),
        "candidate_metrics": candidate,
        "daily_top1": top1,
        "governance": {
            "time_isolated_by_outcome_availability": True,
            "unverifiable_limit_queue_is_negative_for_executable_target": True,
            "missing_B_bar_is_negative": False,
            "right_censored_B_plus_2_is_negative": False,
            "unfilled_top1_replaced": False,
            "emotion_factor_promoted": False,
            "risk_adjusted_ranker_promoted": False,
            "threshold_selected": False,
            "stock_entry_authorized": False,
        },
        "limitations": [
            "the evaluation window is short and has been inspected",
            "daily open is an execution proxy, not verified auction or intraday fill",
            "industry fields are proxies and do not yet encode verified specific themes",
            "cross-theme leader competition is not yet represented in this isolated model",
            "reported probabilities are diagnostic and not deployment-calibrated",
        ],
    }
    return predictions, selected_all, report


def run(features_path, outcomes_path, promotion_path, current_path, output_dir,
        min_train_sessions: int = 10) -> dict:
    paths = [Path(features_path), Path(outcomes_path), Path(promotion_path), Path(current_path)]
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable profit-ranker audit")
    features = pd.read_csv(paths[0], dtype={"code": str})
    outcomes = pd.read_csv(paths[1], dtype={"code": str})
    promotion = pd.read_csv(paths[2], dtype={"code": str})
    current = pd.read_csv(paths[3], dtype={"code": str})
    rows = build_dataset(features, outcomes)
    predictions, selected, report = audit(
        rows, promotion_predictions=promotion, min_train_sessions=min_train_sessions)
    current_scores = _score_current(rows, current)
    target.mkdir(parents=True)
    rows_path = target / "profit_target_rows_LABEL_ONLY.csv"
    predictions_path = target / "prequential_profit_predictions.csv"
    selected_path = target / "daily_top1_no_replacement.csv"
    current_scores_path = target / "current_profit_scores_RESEARCH_ONLY.csv"
    rows.to_csv(rows_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    selected.to_csv(selected_path, index=False, encoding="utf-8-sig")
    current_scores.to_csv(current_scores_path, index=False, encoding="utf-8-sig")
    report["current_signal_date"] = (
        str(current_scores.signal_date.max()) if not current_scores.empty else None)
    report["current_top10_research_only"] = (
        current_scores[[
            "code", "name", "signal_date", "profit_base_probability",
            "profit_emotion_probability", "big_loss_probability",
            "profit_risk_adjusted_score", "execution_gate",
        ]].head(10).to_dict("records") if not current_scores.empty else [])
    report["source_hashes"] = {str(path.resolve()): _sha256(path) for path in paths}
    report["artifact_hashes"] = {
        path.name: _sha256(path) for path in (
            rows_path, predictions_path, selected_path, current_scores_path)}
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--outcomes", required=True)
    parser.add_argument("--promotion", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-train-sessions", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(run(
        args.features, args.outcomes, args.promotion, args.current,
        args.output, args.min_train_sessions), ensure_ascii=False, indent=2))
