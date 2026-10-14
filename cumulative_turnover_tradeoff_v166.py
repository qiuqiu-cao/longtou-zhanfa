"""Explore whether relay-episode cumulative turnover adds out-of-sample value.

This is deliberately an exploratory audit.  It uses the corrected target table
that retains next-session no-touch rows as negatives, applies a trading-session
embargo, and fits preprocessing on each training fold only.  The validation
period has been inspected by earlier project work, so this module must never
label its result a fresh blind test or directly authorize production ranking.
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
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_FEATURES = [
    "boards", "turnover", "amount_to_float_cap", "one_price_day_proxy",
    "open_gap_pct", "amplitude_pct", "market_emotion_score",
    "market_failed_rate", "market_promotion_mid", "market_yesterday_premium",
]
TURNOVER_FEATURES = [
    "episode_turnover", "episode_average_turnover", "turnover_acceleration_log",
    "episode_turnover_log", "episode_turnover_log_sq", "episode_turnover_cv",
]
FOLDS = [
    ("2025Q1", "2025-01-01", "2025-04-01"),
    ("2025Q2", "2025-04-01", "2025-07-01"),
    ("2025Q3", "2025-07-01", "2025-10-01"),
    ("2025Q4", "2025-10-01", "2026-01-01"),
]


def _code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_cumulative_features(candidates: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    required_candidates = {"code", "signal_date", "boards"}
    required_daily = {"code", "date", "turnover"}
    if not required_candidates.issubset(candidates):
        raise ValueError(f"candidate fields missing: {sorted(required_candidates-set(candidates))}")
    if not required_daily.issubset(daily):
        raise ValueError(f"daily fields missing: {sorted(required_daily-set(daily))}")

    out = candidates.copy()
    out["code"] = _code(out["code"])
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["boards"] = pd.to_numeric(out["boards"], errors="coerce")
    bars = daily[["code", "date", "turnover"]].copy()
    bars["code"] = _code(bars["code"])
    bars["date"] = pd.to_datetime(bars["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    bars["turnover"] = pd.to_numeric(bars["turnover"], errors="coerce")
    bars = bars.dropna(subset=["date"]).drop_duplicates(["code", "date"], keep="last")

    calendar = sorted(bars["date"].unique())
    date_to_idx = {date: idx for idx, date in enumerate(calendar)}
    bars["session_idx"] = bars["date"].map(date_to_idx)
    out["session_idx"] = out["signal_date"].map(date_to_idx)

    lookup = bars[["code", "session_idx", "turnover"]]
    for lag in range(5):
        key = f"_lag_idx_{lag}"
        out[key] = out["session_idx"] - lag
        right = lookup.rename(columns={"session_idx": key, "turnover": f"turnover_lag{lag}"})
        out = out.merge(right, on=["code", key], how="left", validate="many_to_one")
        out = out.drop(columns=[key])

    episode_values: list[float] = []
    episode_cv: list[float] = []
    for row in out.itertuples(index=False):
        board = int(row.boards) if pd.notna(row.boards) else 0
        if board < 2 or board > 5:
            episode_values.append(np.nan)
            episode_cv.append(np.nan)
            continue
        values = np.array([getattr(row, f"turnover_lag{lag}") for lag in range(board)], dtype=float)
        if not np.isfinite(values).all():
            episode_values.append(np.nan)
            episode_cv.append(np.nan)
            continue
        total = float(values.sum())
        mean = float(values.mean())
        episode_values.append(total)
        episode_cv.append(float(values.std(ddof=0) / mean) if mean > 0 else np.nan)

    out["episode_turnover"] = episode_values
    out["episode_average_turnover"] = out["episode_turnover"] / out["boards"]
    ratio = out["turnover_lag0"] / out["turnover_lag1"].replace(0, np.nan)
    out["turnover_acceleration_log"] = np.log(ratio.where(ratio > 0))
    out["episode_turnover_log"] = np.log1p(out["episode_turnover"])
    out["episode_turnover_log_sq"] = out["episode_turnover_log"] ** 2
    out["episode_turnover_cv"] = episode_cv
    out["cumulative_turnover_status"] = np.where(
        out["episode_turnover"].notna(), "available", "unavailable_not_filled_zero"
    )
    return out.drop(columns=["session_idx"])


def learn_tier_thresholds(train: pd.DataFrame) -> dict[str, dict[str, float]]:
    thresholds: dict[str, dict[str, float]] = {}
    clean = train.dropna(subset=["episode_turnover", "boards"])
    for board, group in clean.groupby(clean["boards"].astype(int)):
        if len(group) < 30:
            continue
        q = group["episode_turnover"].quantile([1 / 3, 2 / 3])
        thresholds[str(board)] = {"low": float(q.iloc[0]), "high": float(q.iloc[1])}
    if len(clean) >= 30:
        q = clean["episode_turnover"].quantile([1 / 3, 2 / 3])
        thresholds["fallback"] = {"low": float(q.iloc[0]), "high": float(q.iloc[1])}
    return thresholds


def assign_tiers(frame: pd.DataFrame, thresholds: dict[str, dict[str, float]]) -> pd.Series:
    result = []
    for row in frame.itertuples(index=False):
        if pd.isna(row.episode_turnover) or pd.isna(row.boards):
            result.append("UNKNOWN")
            continue
        rule = thresholds.get(str(int(row.boards)), thresholds.get("fallback"))
        if not rule:
            result.append("UNKNOWN")
        elif row.episode_turnover <= rule["low"]:
            result.append("LOW")
        elif row.episode_turnover <= rule["high"]:
            result.append("MID")
        else:
            result.append("HIGH")
    return pd.Series(result, index=frame.index, dtype="object")


def _model(features: list[str]) -> Pipeline:
    prep = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), features)
    ], remainder="drop")
    return Pipeline([
        ("prep", prep),
        ("model", LogisticRegression(C=0.25, max_iter=2000, class_weight="balanced")),
    ])


def _metrics(y: pd.Series, probability: np.ndarray) -> dict:
    return {
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, probability)) if y.nunique() == 2 else None,
        "brier": float(brier_score_loss(y, probability)),
    }


def clustered_brier_delta_ci(predictions: pd.DataFrame, seed: int = 20260917,
                               repetitions: int = 2000) -> dict:
    if predictions.empty:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "sessions": 0}
    work = predictions.copy()
    work["delta"] = ((work["enhanced_probability"] - work["target"]) ** 2
                     - (work["baseline_probability"] - work["target"]) ** 2)
    by_date = work.groupby("signal_date", observed=True)["delta"].mean()
    rng = np.random.default_rng(seed)
    values = by_date.to_numpy(dtype=float)
    samples = np.empty(repetitions)
    for idx in range(repetitions):
        samples[idx] = rng.choice(values, size=len(values), replace=True).mean()
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "sessions": int(len(values)),
        "interpretation": "negative favors cumulative-turnover model",
    }


def _walk_forward(frame: pd.DataFrame, target_column: str, trade_only: bool,
                  embargo_sessions: int = 20) -> tuple[pd.DataFrame, list[dict]]:
    dates = sorted(frame["signal_date"].dropna().unique())
    date_position = {date: idx for idx, date in enumerate(dates)}
    all_predictions: list[pd.DataFrame] = []
    fold_reports: list[dict] = []
    for name, start, end in FOLDS:
        eligible_train_dates = [d for d in dates if d < start]
        if len(eligible_train_dates) <= embargo_sessions:
            continue
        train_last = eligible_train_dates[-embargo_sessions - 1]
        train = frame[frame["signal_date"] <= train_last].copy()
        test = frame[(frame["signal_date"] >= start) & (frame["signal_date"] < end)].copy()
        if trade_only:
            condition = (
                frame["next_session_touched_limit_LABEL_ONLY"].eq(True)
                & frame["next_session_one_price_limit_LABEL_ONLY"].eq(False)
                & frame["b2_net_close_mark_LABEL_ONLY"].notna()
            )
            train = frame.loc[train.index.intersection(frame.index[condition])].query(
                "signal_date <= @train_last").copy()
            test = frame.loc[test.index.intersection(frame.index[condition])].copy()
        train = train.dropna(subset=[target_column, "episode_turnover"])
        test = test.dropna(subset=[target_column, "episode_turnover"])
        if len(train) < 300 or len(test) < 50 or train[target_column].nunique() < 2 or test[target_column].nunique() < 2:
            fold_reports.append({"fold": name, "status": "INSUFFICIENT", "train_n": len(train), "test_n": len(test)})
            continue

        baseline = _model(BASE_FEATURES)
        enhanced = _model(BASE_FEATURES + TURNOVER_FEATURES)
        baseline.fit(train, train[target_column].astype(int))
        enhanced.fit(train, train[target_column].astype(int))
        p0 = baseline.predict_proba(test)[:, 1]
        p1 = enhanced.predict_proba(test)[:, 1]
        thresholds = learn_tier_thresholds(train)
        prediction = test[["code", "signal_date", "boards", "episode_turnover"]].copy()
        prediction["target"] = test[target_column].astype(int).to_numpy()
        prediction["baseline_probability"] = p0
        prediction["enhanced_probability"] = p1
        prediction["turnover_tier"] = assign_tiers(test, thresholds).to_numpy()
        prediction["fold"] = name
        if trade_only:
            prediction["b2_net_return"] = test["b2_net_close_mark_LABEL_ONLY"].to_numpy()
            prediction["b2_big_loss"] = test["b2_big_loss_mark_LABEL_ONLY"].to_numpy()
        all_predictions.append(prediction)
        base_metrics = _metrics(prediction["target"], p0)
        enhanced_metrics = _metrics(prediction["target"], p1)
        fold_reports.append({
            "fold": name, "status": "COMPLETE", "train_end": train_last,
            "embargo_sessions": embargo_sessions, "train_n": int(len(train)),
            "test_n": int(len(test)), "baseline": base_metrics,
            "enhanced": enhanced_metrics,
            "brier_delta_enhanced_minus_baseline": enhanced_metrics["brier"] - base_metrics["brier"],
            "auc_delta_enhanced_minus_baseline": enhanced_metrics["auc"] - base_metrics["auc"],
            "tier_thresholds_fit_on_train_only": thresholds,
        })
    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    return predictions, fold_reports


def _overall(predictions: pd.DataFrame) -> dict:
    if predictions.empty:
        return {"n": 0}
    return {
        "n": int(len(predictions)),
        "sessions": int(predictions["signal_date"].nunique()),
        "baseline": _metrics(predictions["target"], predictions["baseline_probability"].to_numpy()),
        "enhanced": _metrics(predictions["target"], predictions["enhanced_probability"].to_numpy()),
        "clustered_brier_delta": clustered_brier_delta_ci(predictions),
        "fold_brier_wins": int(sum(
            row["brier_delta_enhanced_minus_baseline"] < 0
            for row in []  # populated by caller
        )),
    }


def _tier_summary(identity: pd.DataFrame, trade: pd.DataFrame) -> list[dict]:
    rows = []
    for tier in ["LOW", "MID", "HIGH"]:
        a = identity[identity["turnover_tier"] == tier]
        b = trade[trade["turnover_tier"] == tier]
        rows.append({
            "tier": tier,
            "identity_n": int(len(a)),
            "promotion_rate": float(a["target"].mean()) if len(a) else None,
            "trade_n": int(len(b)),
            "b2_positive_rate": float(b["target"].mean()) if len(b) else None,
            "b2_mean_net_return": float(b["b2_net_return"].mean()) if len(b) else None,
            "b2_big_loss_rate": float(b["b2_big_loss"].mean()) if len(b) else None,
        })
    return rows


def run(features_path: Path, outcomes_path: Path, daily_path: Path,
        outcome_manifest_path: Path, target_contract_path: Path, output: Path) -> dict:
    manifest = json.loads(outcome_manifest_path.read_text(encoding="utf-8"))
    target_contract = json.loads(target_contract_path.read_text(encoding="utf-8"))
    no_touch = bool(
        target_contract.get("target_contract", {}).get("no_touch_rows_retained_as_negative")
        or target_contract.get("target_coverage", {}).get("no_touch_rows_retained_as_negative")
        or target_contract.get("no_touch_rows_retained_as_negative")
    )
    if not no_touch:
        raise ValueError("corrected no-touch-negative outcome manifest required")

    features = pd.read_csv(features_path, dtype={"code": str}, low_memory=False)
    outcomes = pd.read_csv(outcomes_path, dtype={"code": str}, low_memory=False)
    daily = pd.read_csv(daily_path, usecols=["date", "code", "turnover"], dtype={"code": str}, low_memory=False)
    frame = features.merge(outcomes, on=["code", "signal_date"], how="inner", validate="one_to_one")
    frame = frame[frame["boards"].between(2, 5, inclusive="both")].copy()
    frame = build_cumulative_features(frame, daily)
    frame["promotion_target"] = pd.to_numeric(frame["promotion_LABEL_ONLY"], errors="coerce")
    b2 = pd.to_numeric(frame["b2_net_close_mark_LABEL_ONLY"], errors="coerce")
    frame["b2_positive_target"] = np.where(b2.notna(), (b2 > 0).astype(int), np.nan)

    identity_pred, identity_folds = _walk_forward(frame, "promotion_target", trade_only=False)
    trade_pred, trade_folds = _walk_forward(frame, "b2_positive_target", trade_only=True)
    identity_overall = _overall(identity_pred)
    trade_overall = _overall(trade_pred)
    identity_complete = [f for f in identity_folds if f.get("status") == "COMPLETE"]
    trade_complete = [f for f in trade_folds if f.get("status") == "COMPLETE"]
    identity_overall["fold_brier_wins"] = int(sum(f["brier_delta_enhanced_minus_baseline"] < 0 for f in identity_complete))
    trade_overall["fold_brier_wins"] = int(sum(f["brier_delta_enhanced_minus_baseline"] < 0 for f in trade_complete))
    tiers = _tier_summary(identity_pred, trade_pred)
    by_tier = {row["tier"]: row for row in tiers}
    inverted_u = bool(
        all(by_tier[t]["promotion_rate"] is not None and by_tier[t]["b2_positive_rate"] is not None
            for t in ["LOW", "MID", "HIGH"])
        and by_tier["MID"]["promotion_rate"] > max(by_tier["LOW"]["promotion_rate"], by_tier["HIGH"]["promotion_rate"])
        and by_tier["MID"]["b2_positive_rate"] > max(by_tier["LOW"]["b2_positive_rate"], by_tier["HIGH"]["b2_positive_rate"])
    )
    identity_ci = identity_overall.get("clustered_brier_delta", {})
    trade_ci = trade_overall.get("clustered_brier_delta", {})
    evidence_gate = bool(
        identity_overall.get("n", 0) >= 1000
        and trade_overall.get("n", 0) >= 500
        and identity_ci.get("ci95_high") is not None and identity_ci["ci95_high"] < 0
        and trade_ci.get("ci95_high") is not None and trade_ci["ci95_high"] < 0
        and identity_overall["fold_brier_wins"] >= 3
        and trade_overall["fold_brier_wins"] >= 3
        and inverted_u
    )

    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "relay_cumulative_turnover_features.csv", index=False, encoding="utf-8-sig")
    identity_pred.to_csv(output / "identity_oos_predictions.csv", index=False, encoding="utf-8-sig")
    trade_pred.to_csv(output / "triggered_trade_oos_predictions.csv", index=False, encoding="utf-8-sig")
    report = {
        "version": "v166-cumulative-turnover-tradeoff-1",
        "status": "EXPLORATORY_ONLY",
        "hypothesis_id": "H020",
        "candidate_rows": int(len(frame)),
        "candidate_sessions": int(frame["signal_date"].nunique()),
        "feature_available_rows": int(frame["episode_turnover"].notna().sum()),
        "feature_missing_rows": int(frame["episode_turnover"].isna().sum()),
        "missing_policy": "unavailable; never filled with zero",
        "identity_target": "next-session close-limit promotion; no-touch rows retained as negatives",
        "trade_target": "B+2 net close mark positive, conditional on next-session touch and non-one-price proxy",
        "identity_walk_forward": {"overall": identity_overall, "folds": identity_folds},
        "triggered_trade_walk_forward": {"overall": trade_overall, "folds": trade_folds},
        "point_in_time_turnover_tiers": tiers,
        "inverted_u_observed_in_both_targets": inverted_u,
        "evidence_gate_passed": evidence_gate,
        "direct_model_use_allowed": False,
        "reason_direct_use_forbidden": (
            "The 2025 validation period has been repeatedly inspected and is not a fresh blind set; "
            "daily-bar entry and exit remain opportunity proxies rather than verified fills."
        ),
        "input_hashes": {
            "features": _sha256(features_path), "outcomes": _sha256(outcomes_path),
            "daily": _sha256(daily_path), "outcome_manifest": _sha256(outcome_manifest_path),
            "target_contract": _sha256(target_contract_path),
        },
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--outcomes", required=True)
    parser.add_argument("--daily", required=True)
    parser.add_argument("--outcome-manifest", required=True)
    parser.add_argument("--target-contract", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(Path(args.features), Path(args.outcomes), Path(args.daily),
                 Path(args.outcome_manifest), Path(args.target_contract), Path(args.output))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
