"""Re-audit highest-board buyability with corrected no-touch negatives.

The target is a daily proxy, not a complete trade: next session must touch the
limit, must not be a one-price limit, and the B+2 net close mark must be
positive.  No-touch and next-session one-price rows are explicit negatives.
Rows with a touchable path but unavailable B+2 mark remain unknown.  Features
are signal-close fields only.  The historical universe is still incomplete and
no actual fill exists, so the result cannot authorize ranking or entry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


VERSION = "v181-height-anchor-buyability-1"
BASE_FEATURES = ["boards"]
CHALLENGER_FEATURES = [
    "boards", "sole_known_top_proxy", "open_gap_pct", "amplitude_pct",
    "amount_to_float_cap", "one_price_day_proxy", "turnover",
    "market_failed_rate", "market_seal_rate", "market_ladder_gap_count",
    "market_one_word_share", "market_previous_limit_tail_loss",
    "market_promotion_mid", "market_space_feedback", "market_crowding_risk",
    "market_regime_fragility", "market_false_acceleration",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")
    values = series.astype(str).str.strip().str.lower()
    mapped = values.map({"true": True, "1": True, "false": False, "0": False,
                         "nan": pd.NA, "none": pd.NA, "": pd.NA})
    return mapped.astype("boolean")


def _model(features: list[str]) -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    return Pipeline([
        ("prepare", ColumnTransformer([("numeric", numeric, features)])),
        ("model", LogisticRegression(C=.5, max_iter=3000, random_state=181)),
    ])


def _metrics(actual: pd.Series, probability: np.ndarray) -> dict:
    y = actual.astype(int).to_numpy()
    probability = np.asarray(probability, dtype=float)
    return {
        "rows": int(len(y)),
        "positive_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
    }


def _cluster_metric_differences(validation: pd.DataFrame,
                                baseline_probability: np.ndarray,
                                challenger_probability: np.ndarray,
                                draws: int, seed: int) -> dict:
    work = validation[["signal_date", "target"]].copy().reset_index(drop=True)
    work["baseline_probability"] = baseline_probability
    work["challenger_probability"] = challenger_probability
    dates = work.signal_date.drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    samples = {"auc_gain": [], "brier_gain": [], "log_loss_gain": []}
    for _ in range(draws):
        chosen = rng.choice(dates, len(dates), replace=True)
        parts = [work[work.signal_date.eq(date)] for date in chosen]
        sample = pd.concat(parts, ignore_index=True)
        y = sample.target.astype(int).to_numpy()
        base = sample.baseline_probability.to_numpy()
        challenger = sample.challenger_probability.to_numpy()
        if len(np.unique(y)) == 2:
            samples["auc_gain"].append(
                roc_auc_score(y, challenger) - roc_auc_score(y, base))
        samples["brier_gain"].append(
            brier_score_loss(y, base) - brier_score_loss(y, challenger))
        samples["log_loss_gain"].append(
            log_loss(y, base, labels=[0, 1])
            - log_loss(y, challenger, labels=[0, 1]))
    result = {}
    for name, values in samples.items():
        result[name] = {
            "estimate": None if not values else float(np.mean(values)),
            "ci95": [None, None] if not values else [
                float(np.quantile(values, .025)), float(np.quantile(values, .975))],
        }
    return result


def _daily_top(frame: pd.DataFrame, probability_column: str) -> pd.DataFrame:
    ordered = frame.sort_values(
        ["signal_date", probability_column, "boards", "code"],
        ascending=[True, False, False, True])
    return ordered.groupby("signal_date", as_index=False).first()


def _paired_daily_difference(baseline: pd.DataFrame, challenger: pd.DataFrame,
                             draws: int, seed: int) -> dict:
    paired = baseline[["signal_date", "target"]].merge(
        challenger[["signal_date", "target"]], on="signal_date",
        suffixes=("_baseline", "_challenger"), validate="one_to_one")
    differences = (paired.target_challenger.astype(float)
                   - paired.target_baseline.astype(float)).to_numpy()
    rng = np.random.default_rng(seed)
    simulations = [
        float(rng.choice(differences, len(differences), replace=True).mean())
        for _ in range(draws)
    ]
    return {
        "paired_sessions": int(len(paired)),
        "baseline_success_rate": float(paired.target_baseline.mean()),
        "challenger_success_rate": float(paired.target_challenger.mean()),
        "difference": float(differences.mean()),
        "bootstrap95": [float(np.quantile(simulations, .025)),
                        float(np.quantile(simulations, .975))],
    }


def build(features: pd.DataFrame, outcomes: pd.DataFrame,
          cutoff_exclusive: str = "2025-12-26", split_fraction: float = .70,
          bootstrap_draws: int = 2000) -> tuple[pd.DataFrame, dict]:
    feature_required = {
        "code", "signal_date", "board_group", "height_gap_to_known_top",
        "feature_available_not_before", *CHALLENGER_FEATURES,
    }
    outcome_required = {
        "code", "signal_date", "next_session_touched_limit_LABEL_ONLY",
        "next_session_one_price_limit_LABEL_ONLY", "b2_net_close_mark_LABEL_ONLY",
        "actual_entry_verified", "complete_trade_profit_LABEL_ONLY",
    }
    if feature_required - set(features):
        raise ValueError("feature fields missing")
    if outcome_required - set(outcomes):
        raise ValueError("outcome fields missing")
    left, right = features.copy(), outcomes.copy()
    for frame in (left, right):
        frame["code"] = frame.code.astype(str).str.zfill(6)
        frame["signal_date"] = pd.to_datetime(
            frame.signal_date, errors="coerce").dt.strftime("%Y-%m-%d")
        if frame[["code", "signal_date"]].isna().any().any():
            raise ValueError("invalid candidate key")
        if frame.duplicated(["code", "signal_date"]).any():
            raise ValueError("candidate keys must be unique")
    merged = left.merge(right, on=["code", "signal_date"], how="inner",
                        validate="one_to_one")
    if merged.empty:
        raise ValueError("no merged corrected candidates")
    available = pd.to_datetime(merged.feature_available_not_before, errors="coerce")
    signal_close = pd.to_datetime(merged.signal_date) + pd.Timedelta(hours=15, minutes=5)
    if available.isna().any() or (available.dt.tz_localize(None) > signal_close).any():
        raise ValueError("features are not available by signal close")

    merged = merged[
        merged.board_group.eq("CONSECUTIVE_BOARD")
        & pd.to_numeric(merged.height_gap_to_known_top, errors="coerce").eq(0)
        & merged.signal_date.lt(pd.Timestamp(cutoff_exclusive).strftime("%Y-%m-%d"))
    ].copy()
    touch = _boolean(merged.next_session_touched_limit_LABEL_ONLY)
    one_price = _boolean(merged.next_session_one_price_limit_LABEL_ONLY)
    b2 = pd.to_numeric(merged.b2_net_close_mark_LABEL_ONLY, errors="coerce")
    unknown = touch.eq(True) & one_price.eq(False) & b2.isna()
    merged["target_known"] = ~unknown
    merged["target"] = (
        touch.eq(True) & one_price.eq(False) & b2.gt(0)
    ).astype(bool)
    merged = merged[merged.target_known].copy()
    if _boolean(merged.actual_entry_verified).fillna(False).any():
        raise ValueError("historical proxy unexpectedly claims verified entries")
    if pd.to_numeric(merged.complete_trade_profit_LABEL_ONLY,
                     errors="coerce").notna().any():
        raise ValueError("historical proxy unexpectedly contains complete trades")
    for column in CHALLENGER_FEATURES:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    dates = merged.signal_date.drop_duplicates().sort_values().tolist()
    if len(dates) < 200:
        raise ValueError("insufficient height-anchor sessions")
    split_index = max(1, min(len(dates) - 1, int(len(dates) * split_fraction)))
    split_date = dates[split_index]
    development = merged[merged.signal_date.lt(split_date)].copy()
    validation = merged[merged.signal_date.ge(split_date)].copy()
    if development.target.nunique() < 2 or validation.target.nunique() < 2:
        raise ValueError("height-anchor split lacks two target classes")

    baseline_model = _model(BASE_FEATURES)
    challenger_model = _model(CHALLENGER_FEATURES)
    baseline_model.fit(development[BASE_FEATURES], development.target)
    challenger_model.fit(development[CHALLENGER_FEATURES], development.target)
    baseline_probability = baseline_model.predict_proba(
        validation[BASE_FEATURES])[:, 1]
    challenger_probability = challenger_model.predict_proba(
        validation[CHALLENGER_FEATURES])[:, 1]
    validation["baseline_probability"] = baseline_probability
    validation["challenger_probability"] = challenger_probability
    baseline_metrics = _metrics(validation.target, baseline_probability)
    challenger_metrics = _metrics(validation.target, challenger_probability)
    cluster_differences = _cluster_metric_differences(
        validation, baseline_probability, challenger_probability,
        bootstrap_draws, 18101)
    baseline_top = _daily_top(validation, "baseline_probability")
    challenger_top = _daily_top(validation, "challenger_probability")
    daily_comparison = _paired_daily_difference(
        baseline_top, challenger_top, bootstrap_draws, 18102)
    uplift_supported = bool(
        cluster_differences["brier_gain"]["ci95"][0] is not None
        and cluster_differences["brier_gain"]["ci95"][0] > 0
        and cluster_differences["log_loss_gain"]["ci95"][0] > 0
        and cluster_differences["auc_gain"]["ci95"][0] > 0
        and daily_comparison["bootstrap95"][0] > 0)

    unique = merged.sole_known_top_proxy.eq(1)
    report = {
        "version": VERSION,
        "status": "DIAGNOSTIC_ONLY",
        "target_contract": {
            "description": (
                "next-session touched limit AND not one-price AND B+2 net close mark > 0"),
            "no_touch_rows_retained_as_negative": True,
            "next_session_one_price_rows_retained_as_negative": True,
            "touchable_rows_with_unknown_b2_excluded": int(unknown.sum()),
            "actual_fills": 0,
            "complete_trades": 0,
        },
        "height_anchor_scope": {
            "rows": int(len(merged)), "sessions": int(merged.signal_date.nunique()),
            "unique_anchor_rows": int(unique.sum()),
            "co_high_anchor_rows": int((~unique).sum()),
            "all_anchor_proxy_success_rate": float(merged.target.mean()),
            "unique_anchor_proxy_success_rate": float(merged.loc[unique, "target"].mean()),
            "co_high_anchor_proxy_success_rate": float(merged.loc[~unique, "target"].mean()),
        },
        "split": {
            "method": "chronological_70_30_by_signal_date",
            "split_date": split_date,
            "development_rows": int(len(development)),
            "development_sessions": int(development.signal_date.nunique()),
            "validation_rows": int(len(validation)),
            "validation_sessions": int(validation.signal_date.nunique()),
            "validation_previously_inspected": True,
        },
        "models": {
            "boards_only": {"features": BASE_FEATURES, "metrics": baseline_metrics},
            "identity_emotion_price_volume": {
                "features": CHALLENGER_FEATURES, "metrics": challenger_metrics},
        },
        "date_clustered_metric_gains": cluster_differences,
        "daily_top1_within_height_anchors": daily_comparison,
        "height_anchor_buyability_uplift_supported": uplift_supported,
        "governance": {
            "height_anchor_can_auto_fill_relay_primary": False,
            "height_anchor_can_auto_fill_current_buy": False,
            "candidate_probability_claim_allowed": False,
            "direct_stock_reranking_allowed": False,
            "entry_authorized": False,
        },
        "interpretation": (
            "Highest-board identity is an emotion role, not proof of a buyable B+2 edge. "
            "Only if the corrected challenger beats boards-only metrics and daily selection "
            "under date-clustered intervals could a ranking claim be reconsidered."),
        "limitations": [
            "daily proxy cannot prove auction queue, fill, T+1 sellability, or realized exit",
            "historical candidate universe is not independently full-market complete",
            "features are retrospective reconstruction rather than frozen forward archives",
            "specific-theme, catalyst, competition identity, and abnormal-rule paths are absent",
            "validation dates were inspected by prior project research and are not sealed blind",
        ],
    }
    output = validation[[
        "signal_date", "code", "boards", "sole_known_top_proxy", "target",
        "baseline_probability", "challenger_probability",
    ]].copy()
    return output, report


def run(features_path: str, outcomes_path: str, output_dir: str,
        cutoff_exclusive: str = "2025-12-26") -> dict:
    feature_source, outcome_source, target = (
        Path(features_path), Path(outcomes_path), Path(output_dir))
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable V181 output")
    predictions, report = build(
        pd.read_csv(feature_source, dtype={"code": str}),
        pd.read_csv(outcome_source, dtype={"code": str}), cutoff_exclusive)
    report["source_hashes"] = {
        str(feature_source.resolve()): _sha256(feature_source),
        str(outcome_source.resolve()): _sha256(outcome_source),
    }
    target.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(target / "validation_predictions.csv", index=False,
                       encoding="utf-8-sig")
    report["predictions_sha256"] = _sha256(target / "validation_predictions.csv")
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--outcomes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cutoff-exclusive", default="2025-12-26")
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.outcomes, args.output,
                         args.cutoff_exclusive), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
