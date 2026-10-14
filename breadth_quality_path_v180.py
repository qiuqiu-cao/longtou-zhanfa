"""Evaluate B+1/B+2/B+3 market paths after breadth expansion.

This study extends V178 from a one-session diagnostic to aggregate market
paths.  It does not measure candidate returns: B+1/B+2/B+3 here refer only to
future market-emotion sessions.  Signal features are known at the close and
three future verified sessions are required without gaps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v180-breadth-quality-path-1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _numeric(series):
    return pd.to_numeric(series, errors="coerce")


def _mean(series):
    values = _numeric(series).dropna()
    return float(values.mean()) if len(values) else None


def _summary(frame: pd.DataFrame) -> dict:
    return {
        "sessions": int(len(frame)),
        "adverse_b1_rate": _mean(frame.adverse_b1),
        "adverse_any_b1_b2_rate": _mean(frame.adverse_any_b1_b2),
        "adverse_any_b1_b3_rate": _mean(frame.adverse_any_b1_b3),
        "no_adverse_b1_b3_rate": _mean(frame.no_adverse_b1_b3),
        "mean_median_premium_b1_b3": _mean(frame.mean_median_premium_b1_b3),
        "mean_middle_promotion_b1_b3": _mean(frame.mean_middle_promotion_b1_b3),
        "max_down_limits_b1_b3_mean": _mean(frame.max_down_limits_b1_b3),
    }


def _difference(frame: pd.DataFrame, column: str, seed: int,
                draws: int = 3000) -> dict:
    fragile = _numeric(frame.loc[frame.breadth_quality.eq(
        "FRAGILE_EXPANSION"), column]).dropna()
    clean = _numeric(frame.loc[frame.breadth_quality.eq(
        "CLEAN_EXPANSION"), column]).dropna()
    if fragile.empty or clean.empty:
        return {"metric": column, "estimate_fragile_minus_clean": None,
                "ci95": [None, None], "fragile_sessions": int(len(fragile)),
                "clean_sessions": int(len(clean))}
    estimate = float(fragile.mean() - clean.mean())
    rng = np.random.default_rng(seed)
    simulations = []
    for _ in range(draws):
        simulations.append(
            rng.choice(fragile.to_numpy(), len(fragile), replace=True).mean()
            - rng.choice(clean.to_numpy(), len(clean), replace=True).mean())
    return {"metric": column, "estimate_fragile_minus_clean": estimate,
            "ci95": [float(np.quantile(simulations, .025)),
                     float(np.quantile(simulations, .975))],
            "fragile_sessions": int(len(fragile)),
            "clean_sessions": int(len(clean))}


def build(market: pd.DataFrame,
          cutoff_exclusive: str = "2025-12-26") -> tuple[pd.DataFrame, dict]:
    required = {
        "date", "data_status", "limit_count", "failed_rate",
        "state_divergence", "state_retreat", "promotion_mid",
        "previous_limit_median_return", "down_limit_count",
    }
    if required - set(market):
        raise ValueError("market fields missing")
    frame = market.copy()
    frame["session"] = pd.to_datetime(frame.date, errors="coerce")
    if frame.session.isna().any() or frame.session.duplicated().any():
        raise ValueError("market dates must be valid and unique")
    frame = frame.sort_values("session").reset_index(drop=True)
    for column in required - {"date", "data_status"}:
        frame[column] = _numeric(frame[column])

    frame["previous_status"] = frame.data_status.shift(1)
    for horizon in (1, 2, 3):
        frame[f"status_b{horizon}"] = frame.data_status.shift(-horizon)
        frame[f"session_b{horizon}"] = frame.session.shift(-horizon).dt.strftime(
            "%Y-%m-%d")
    frame["signal_limit_change"] = frame.limit_count.diff()
    frame["signal_failed_rate_change"] = frame.failed_rate.diff()
    expansion = frame.signal_limit_change.gt(0)
    frame["breadth_quality"] = np.select(
        [expansion & frame.signal_failed_rate_change.gt(0),
         expansion & frame.signal_failed_rate_change.le(0)],
        ["FRAGILE_EXPANSION", "CLEAN_EXPANSION"], default="NO_EXPANSION")

    adverse = frame.state_divergence.eq(1) | frame.state_retreat.eq(1)
    for horizon in (1, 2, 3):
        frame[f"adverse_b{horizon}"] = adverse.shift(-horizon).astype(float)
    frame["adverse_any_b1_b2"] = (
        frame.adverse_b1.eq(1) | frame.adverse_b2.eq(1)).astype(float)
    frame["adverse_any_b1_b3"] = (
        frame.adverse_b1.eq(1) | frame.adverse_b2.eq(1)
        | frame.adverse_b3.eq(1)).astype(float)
    frame["no_adverse_b1_b3"] = 1 - frame.adverse_any_b1_b3
    premium_horizons = pd.concat([
        frame.previous_limit_median_return.shift(-horizon)
        for horizon in (1, 2, 3)], axis=1)
    promotion_horizons = pd.concat([
        frame.promotion_mid.shift(-horizon) for horizon in (1, 2, 3)], axis=1)
    down_limit_horizons = pd.concat([
        frame.down_limit_count.shift(-horizon) for horizon in (1, 2, 3)], axis=1)
    frame["mean_median_premium_b1_b3"] = premium_horizons.mean(axis=1)
    frame["mean_middle_promotion_b1_b3"] = promotion_horizons.mean(axis=1)
    frame["max_down_limits_b1_b3"] = down_limit_horizons.max(axis=1)

    eligible = (
        frame.session.lt(pd.Timestamp(cutoff_exclusive))
        & frame.data_status.eq("verified")
        & frame.previous_status.eq("verified")
        & frame.status_b1.eq("verified")
        & frame.status_b2.eq("verified")
        & frame.status_b3.eq("verified")
        & frame.breadth_quality.isin(["FRAGILE_EXPANSION", "CLEAN_EXPANSION"])
    )
    rows = frame.loc[eligible, [
        "date", "session_b1", "session_b2", "session_b3", "breadth_quality",
        "signal_limit_change", "signal_failed_rate_change", "adverse_b1",
        "adverse_b2", "adverse_b3", "adverse_any_b1_b2",
        "adverse_any_b1_b3", "no_adverse_b1_b3",
        "mean_median_premium_b1_b3", "mean_middle_promotion_b1_b3",
        "max_down_limits_b1_b3",
    ]].rename(columns={"date": "signal_date"}).reset_index(drop=True)
    if len(rows) < 150:
        raise ValueError("insufficient verified three-session expansion paths")

    dates = rows.signal_date.drop_duplicates().sort_values().tolist()
    split_index = max(1, min(len(dates) - 1, int(len(dates) * .70)))
    split_date = dates[split_index]
    discovery = rows[rows.signal_date.lt(split_date)]
    validation = rows[rows.signal_date.ge(split_date)]
    metrics = [
        "adverse_b1", "adverse_any_b1_b2", "adverse_any_b1_b3",
        "no_adverse_b1_b3", "mean_median_premium_b1_b3",
        "mean_middle_promotion_b1_b3", "max_down_limits_b1_b3",
    ]
    discovery_comparisons = {
        metric: _difference(discovery, metric, 18000 + index)
        for index, metric in enumerate(metrics)
    }
    validation_comparisons = {
        metric: _difference(validation, metric, 18020 + index)
        for index, metric in enumerate(metrics)
    }
    stable_horizons = {}
    for metric in ("adverse_b1", "adverse_any_b1_b2", "adverse_any_b1_b3"):
        left = discovery_comparisons[metric]["ci95"]
        right = validation_comparisons[metric]["ci95"]
        stable_horizons[metric] = bool(
            left[0] is not None and left[0] > 0
            and right[0] is not None and right[0] > 0)
    report = {
        "version": VERSION,
        "status": "DIAGNOSTIC_ONLY",
        "cutoff_exclusive": pd.Timestamp(cutoff_exclusive).strftime("%Y-%m-%d"),
        "session_rows": int(len(rows)),
        "split_date": split_date,
        "path_contract": {
            "b1_b2_b3_are_market_sessions_not_candidate_trade_returns": True,
            "three_consecutive_verified_future_sessions_required": True,
            "signal_known_at_close": True,
        },
        "state_summaries": {
            state: _summary(rows[rows.breadth_quality.eq(state)])
            for state in ("FRAGILE_EXPANSION", "CLEAN_EXPANSION")
        },
        "chronological_split": {
            "discovery_first_70pct_dates": {
                state: _summary(discovery[discovery.breadth_quality.eq(state)])
                for state in ("FRAGILE_EXPANSION", "CLEAN_EXPANSION")
            },
            "validation_last_30pct_dates": {
                state: _summary(validation[validation.breadth_quality.eq(state)])
                for state in ("FRAGILE_EXPANSION", "CLEAN_EXPANSION")
            },
        },
        "discovery_comparisons": discovery_comparisons,
        "validation_comparisons": validation_comparisons,
        "stable_adverse_path_horizons": stable_horizons,
        "any_two_or_three_session_path_stable": bool(
            stable_horizons["adverse_any_b1_b2"]
            or stable_horizons["adverse_any_b1_b3"]),
        "governance": {
            "two_to_three_session_probability_claim_allowed": False,
            "historical_scenario_path_use_allowed": True,
            "candidate_holding_period_inference_allowed": False,
            "direct_stock_reranking_allowed": False,
            "entry_authorized": False,
        },
        "interpretation": (
            "Breadth quality may describe a multi-session market risk path, but aggregate "
            "B+1/B+2/B+3 emotion outcomes cannot be converted into a candidate holding rule."),
        "limitations": [
            "aggregate provider pool completeness is not independently proven",
            "dates have been inspected by earlier project research and are not a sealed blind set",
            "overlapping three-session paths require future date-clustered confirmation",
            "no candidate-level theme, fill, cost, or executable exit path",
        ],
    }
    return rows, report


def run(input_path: str, output_dir: str,
        cutoff_exclusive: str = "2025-12-26") -> dict:
    source, target = Path(input_path), Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable V180 output")
    rows, report = build(pd.read_csv(source), cutoff_exclusive)
    report["source_hashes"] = {str(source.resolve()): _sha256(source)}
    target.mkdir(parents=True, exist_ok=False)
    rows.to_csv(target / "breadth_quality_path_rows.csv", index=False,
                encoding="utf-8-sig")
    report["rows_sha256"] = _sha256(target / "breadth_quality_path_rows.csv")
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/v3/market_emotion.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cutoff-exclusive", default="2025-12-26")
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output, args.cutoff_exclusive),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
