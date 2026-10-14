"""Build a time-isolated B+1 large-loss veto diagnostic for first boards.

The profit ranker and the tail-risk model have different jobs.  This module
keeps the previously frozen daily profit top-one, predicts whether an actually
filled opening-entry proxy loses at least 7% by B+1 close, and audits fixed
abstention thresholds.  A veto creates cash; it never replaces the frozen
selection with a hindsight second choice.
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
RISK_MODELS = {
    "risk_base_probability": (BASE_FEATURES, None),
    "risk_emotion_probability": (EMOTION_FEATURES, None),
    "risk_emotion_balanced_probability": (EMOTION_FEATURES, "balanced"),
}
FIXED_RESEARCH_THRESHOLDS = (0.05, 0.10, 0.20, 0.30, 0.40)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fit_predict(train: pd.DataFrame, score: pd.DataFrame, features: list[str],
                 class_weight: str | None) -> np.ndarray:
    y = pd.to_numeric(train.big_loss_target, errors="raise").astype(int)
    if y.nunique() < 2:
        return np.repeat(float(y.mean()), len(score))
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(
            C=.5, class_weight=class_weight, max_iter=3000, random_state=199),
    )
    model.fit(train[features], y)
    return model.predict_proba(score[features])[:, 1]


def eligible_risk_train(rows: pd.DataFrame, score_date: str) -> pd.DataFrame:
    """Return only B+1 outcomes observable by the signal-date close."""
    exit_date = rows.exit_date_B_plus_1.fillna("").astype(str)
    return rows[
        rows.big_loss_target_available.eq(1)
        & rows.signal_date.astype(str).lt(score_date)
        & exit_date.ne("")
        & exit_date.le(score_date)
    ].copy()


def prequential_risk(rows: pd.DataFrame, min_train_sessions: int = 10) -> pd.DataFrame:
    required = set(KEYS + EMOTION_FEATURES + [
        "feature_available_at", "big_loss_target", "big_loss_target_available",
        "exit_date_B_plus_1", "fill_status", "net_return_B_plus_1",
        "net_return_B_plus_2", "big_loss_B_plus_1",
    ])
    if required - set(rows):
        raise ValueError(f"risk fields missing: {sorted(required - set(rows))}")
    if rows.duplicated(KEYS).any():
        raise ValueError("candidate keys must be unique")
    scored = []
    for score_date in sorted(rows.signal_date.astype(str).unique()):
        train = eligible_risk_train(rows, score_date)
        if train.signal_date.nunique() < min_train_sessions:
            continue
        current = rows[rows.signal_date.astype(str).eq(score_date)].copy()
        y = pd.to_numeric(train.big_loss_target, errors="raise").astype(int)
        current["historical_big_loss_prior"] = float(y.mean())
        for name, (features, class_weight) in RISK_MODELS.items():
            current[name] = _fit_predict(train, current, features, class_weight)
        current["risk_train_sessions"] = int(train.signal_date.nunique())
        current["risk_train_rows"] = int(len(train))
        current["risk_train_positive_rows"] = int(y.sum())
        scored.append(current)
    return pd.concat(scored, ignore_index=True) if scored else pd.DataFrame()


def risk_metrics(rows: pd.DataFrame, probability: str) -> dict:
    known = rows[rows.big_loss_target_available.eq(1)].copy()
    y = pd.to_numeric(known.big_loss_target, errors="coerce")
    p = pd.to_numeric(known[probability], errors="coerce")
    valid = y.notna() & p.notna()
    y, p = y[valid].astype(int), p[valid]
    if not len(y):
        return {"known_filled_rows": 0}
    order = p.sort_values(ascending=False).index
    top_n = max(1, int(np.ceil(len(order) * .20)))
    top_y = y.loc[order[:top_n]]
    return {
        "known_filled_rows": int(len(y)),
        "big_loss_rows": int(y.sum()),
        "observed_big_loss_rate": float(y.mean()),
        "mean_predicted_probability": float(p.mean()),
        "auc": float(roc_auc_score(y, p)) if y.nunique() == 2 else None,
        "average_precision": (
            float(average_precision_score(y, p)) if y.nunique() == 2 else None),
        "brier": float(brier_score_loss(y, p)),
        "descriptive_top_20pct_risk_rows": int(top_n),
        "descriptive_top_20pct_observed_big_loss_rate": float(top_y.mean()),
    }


def _profit_factor(returns: pd.Series) -> float | None:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    gross_profit = float(values[values.gt(0)].sum())
    gross_loss = float(-values[values.le(0)].sum())
    return float(gross_profit / gross_loss) if gross_loss > 0 else None


def fixed_veto_metrics(frozen: pd.DataFrame, probability: str,
                       thresholds=FIXED_RESEARCH_THRESHOLDS) -> list[dict]:
    """Audit abstention only; a vetoed top-one is never replaced."""
    results = []
    p = pd.to_numeric(frozen[probability], errors="coerce")
    for threshold in thresholds:
        retained = frozen[p.notna() & p.le(threshold)].copy()
        filled = retained[retained.fill_status.eq("FILLED_OPEN_PROXY")].copy()
        big = pd.to_numeric(filled.big_loss_B_plus_1, errors="coerce").dropna()
        b2 = pd.to_numeric(filled.net_return_B_plus_2, errors="coerce").dropna()
        results.append({
            "threshold": float(threshold),
            "frozen_decisions": int(len(frozen)),
            "risk_score_missing_decisions": int(p.isna().sum()),
            "retained_without_replacement": int(len(retained)),
            "abstained_to_cash": int(len(frozen) - len(retained)),
            "filled_trades": int(len(filled)),
            "big_loss_known_trades": int(len(big)),
            "B_plus_1_big_loss_count": int(big.sum()) if len(big) else 0,
            "B_plus_1_big_loss_rate": float(big.mean()) if len(big) else None,
            "B_plus_2_known_trades": int(len(b2)),
            "B_plus_2_win_rate": float(b2.gt(0).mean()) if len(b2) else None,
            "B_plus_2_mean_net_return": float(b2.mean()) if len(b2) else None,
            "B_plus_2_profit_factor": _profit_factor(b2),
        })
    return results


def audit(predictions: pd.DataFrame, frozen_top1: pd.DataFrame,
          profit_report: dict | None = None) -> tuple[pd.DataFrame, dict]:
    if predictions.empty:
        raise ValueError("no prequential risk predictions")
    if predictions.duplicated(KEYS).any():
        raise ValueError("prequential risk keys must be unique")
    frozen = frozen_top1[
        frozen_top1.ranker.eq("profit_emotion_probability")].copy()
    if frozen.empty or frozen.duplicated("signal_date").any():
        raise ValueError("frozen emotion top-one must have one row per session")
    risk_columns = list(RISK_MODELS)
    joined = frozen.merge(
        predictions[KEYS + ["historical_big_loss_prior"] + risk_columns],
        on=KEYS, how="left", validate="one_to_one")
    candidate_metrics = {
        "historical_big_loss_prior": risk_metrics(
            predictions, "historical_big_loss_prior")}
    candidate_metrics.update({
        name: risk_metrics(predictions, name) for name in risk_columns})
    gates = {name: fixed_veto_metrics(joined, name) for name in risk_columns}

    profit_comparison = {}
    if profit_report:
        top = profit_report.get("daily_top1", {})
        emotion = top.get("profit_emotion_probability", {})
        adjusted = top.get("profit_risk_adjusted_score", {})
        if emotion and adjusted:
            profit_comparison = {
                "emotion_B_plus_2_mean": emotion.get("B_plus_2_mean_net_return"),
                "multiplied_risk_adjusted_B_plus_2_mean": adjusted.get(
                    "B_plus_2_mean_net_return"),
                "emotion_big_loss_rate": emotion.get("B_plus_1_big_loss_rate"),
                "multiplied_risk_adjusted_big_loss_rate": adjusted.get(
                    "B_plus_1_big_loss_rate"),
                "simple_probability_multiplication_improved_both": bool(
                    adjusted.get("B_plus_2_mean_net_return") is not None
                    and emotion.get("B_plus_2_mean_net_return") is not None
                    and adjusted.get("B_plus_1_big_loss_rate") is not None
                    and emotion.get("B_plus_1_big_loss_rate") is not None
                    and adjusted["B_plus_2_mean_net_return"]
                    > emotion["B_plus_2_mean_net_return"]
                    and adjusted["B_plus_1_big_loss_rate"]
                    < emotion["B_plus_1_big_loss_rate"]),
            }

    timestamp_dates = pd.to_datetime(
        predictions.feature_available_at, errors="coerce", utc=True
    ).dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m-%d")
    same_day = timestamp_dates.eq(predictions.signal_date.astype(str))
    best_auc = max(
        (values.get("auc") for name, values in candidate_metrics.items()
         if name != "historical_big_loss_prior" and values.get("auc") is not None),
        default=None,
    )
    report = {
        "version": "v199-first-board-big-loss-veto-1",
        "status": "DIAGNOSTIC_ONLY",
        "target": "filled_open_proxy_net_return_at_B_plus_1_close_le_minus_7pct",
        "scored_sessions": int(predictions.signal_date.nunique()),
        "scored_candidate_rows": int(len(predictions)),
        "known_filled_risk_rows": int(predictions.big_loss_target_available.eq(1).sum()),
        "frozen_top1_sessions": int(len(joined)),
        "candidate_metrics": candidate_metrics,
        "fixed_threshold_abstention_audit": gates,
        "profit_reranking_comparison": profit_comparison,
        "diagnosis": {
            "best_candidate_auc": best_auc,
            "tail_model_discriminates_better_than_chance": bool(
                best_auc is not None and best_auc > .5),
            "validated_10pct_big_loss_ceiling_demonstrated": False,
            "positive_expectancy_after_veto_demonstrated": False,
            "simple_probability_multiplication_supported": bool(
                profit_comparison.get(
                    "simple_probability_multiplication_improved_both", False)),
        },
        "governance": {
            "point_in_time_feature_timestamp_same_signal_date": bool(same_day.all()),
            "outcome_available_before_training_score_date": True,
            "frozen_profit_top1_used": True,
            "vetoed_selection_replaced": False,
            "fixed_thresholds_are_descriptive_not_selected": True,
            "risk_probability_is_position_size": False,
            "risk_model_promoted": False,
            "veto_policy_promoted": False,
            "entry_authorized": False,
        },
        "limitations": [
            "the evaluated dates have already been inspected and are not a sealed blind test",
            "the opening price is an execution proxy rather than a verified auction or intraday fill",
            "fixed thresholds are sensitivity checks and no threshold is selected on this sample",
            "specific-theme identity and cross-theme competition remain unavailable to this model",
            "a 10% large-loss ceiling requires a later frozen policy and substantially more signals",
        ],
    }
    return joined, report


def run(rows_path, top1_path, profit_report_path, output_dir,
        min_train_sessions: int = 10) -> dict:
    paths = [Path(rows_path), Path(top1_path), Path(profit_report_path)]
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable big-loss veto audit")
    rows = pd.read_csv(paths[0], dtype={"code": str})
    top1 = pd.read_csv(paths[1], dtype={"code": str})
    profit_report = json.loads(paths[2].read_text(encoding="utf-8"))
    predictions = prequential_risk(rows, min_train_sessions=min_train_sessions)
    joined, report = audit(predictions, top1, profit_report)
    target.mkdir(parents=True)
    predictions_path = target / "prequential_big_loss_predictions.csv"
    joined_path = target / "frozen_emotion_top1_veto_audit.csv"
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    joined.to_csv(joined_path, index=False, encoding="utf-8-sig")
    report["source_hashes"] = {str(path.resolve()): _sha256(path) for path in paths}
    report["artifact_hashes"] = {
        predictions_path.name: _sha256(predictions_path),
        joined_path.name: _sha256(joined_path),
    }
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--top1", required=True)
    parser.add_argument("--profit-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-train-sessions", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(run(
        args.rows, args.top1, args.profit_report, args.output,
        args.min_train_sessions), ensure_ascii=False, indent=2))
