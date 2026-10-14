"""Test whether positive average premium hides negative median breadth.

All signal features are observed at the signal-day close.  Targets are shifted
to the next verified trading session.  The study is descriptive and cannot
authorize stock ranking or entry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v177-profit-concentration-emotion-1"


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
        "signal_mean_premium": _mean(frame.signal_mean_premium),
        "signal_median_premium": _mean(frame.signal_median_premium),
        "signal_mean_minus_median_gap": _mean(frame.profit_concentration_gap),
        "next_adverse_state_rate": _mean(frame.next_adverse_state),
        "next_failed_rate_mean": _mean(frame.next_failed_rate),
        "next_promotion_mid_mean": _mean(frame.next_promotion_mid),
        "next_previous_limit_median_return_mean": _mean(
            frame.next_previous_limit_median_return),
        "next_down_limit_count_mean": _mean(frame.next_down_limit_count),
    }


def _difference(frame: pd.DataFrame, column: str, left: str, right: str,
                seed: int, draws: int = 3000) -> dict:
    a = _numeric(frame.loc[frame.profit_state.eq(left), column]).dropna()
    b = _numeric(frame.loc[frame.profit_state.eq(right), column]).dropna()
    if a.empty or b.empty:
        return {"left": left, "right": right, "metric": column,
                "estimate": None, "ci95": [None, None],
                "sessions_left": int(len(a)), "sessions_right": int(len(b))}
    estimate = float(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    simulations = []
    for _ in range(draws):
        left_value = rng.choice(a.to_numpy(), len(a), replace=True).mean()
        right_value = rng.choice(b.to_numpy(), len(b), replace=True).mean()
        simulations.append(left_value - right_value)
    return {"left": left, "right": right, "metric": column,
            "estimate": estimate,
            "ci95": [float(np.quantile(simulations, .025)),
                     float(np.quantile(simulations, .975))],
            "sessions_left": int(len(a)), "sessions_right": int(len(b))}


def build(market: pd.DataFrame,
          cutoff_exclusive: str = "2025-12-26") -> tuple[pd.DataFrame, dict]:
    required = {
        "date", "data_status", "previous_limit_mean_return",
        "previous_limit_median_return", "state_divergence", "state_retreat",
        "failed_rate", "promotion_mid", "down_limit_count",
    }
    if required - set(market):
        raise ValueError("market fields missing")
    frame = market.copy()
    frame["signal_date"] = pd.to_datetime(
        frame.date, errors="coerce").dt.strftime("%Y-%m-%d")
    cutoff = pd.Timestamp(cutoff_exclusive).strftime("%Y-%m-%d")
    frame = frame[
        frame.signal_date.lt(cutoff) & frame.data_status.eq("verified")
    ].sort_values("signal_date").copy()
    if frame.signal_date.isna().any() or frame.signal_date.duplicated().any():
        raise ValueError("market dates must be valid and unique")
    frame["signal_mean_premium"] = _numeric(frame.previous_limit_mean_return)
    frame["signal_median_premium"] = _numeric(frame.previous_limit_median_return)
    frame["profit_concentration_gap"] = (
        frame.signal_mean_premium - frame.signal_median_premium)
    mean = frame.signal_mean_premium
    median = frame.signal_median_premium
    frame["profit_state"] = np.select(
        [(mean > 0) & (median > 0), (mean > 0) & (median <= 0),
         (mean <= 0) & (median <= 0), (mean <= 0) & (median > 0)],
        ["BROAD_POSITIVE", "NARROW_POSITIVE", "BROAD_NEGATIVE",
         "MEDIAN_POSITIVE_MEAN_NONPOSITIVE"], default="UNKNOWN")
    frame["next_session"] = frame.signal_date.shift(-1)
    frame["next_adverse_state"] = (
        _numeric(frame.state_divergence).shift(-1).eq(1)
        | _numeric(frame.state_retreat).shift(-1).eq(1)
    ).astype(float)
    # The last row has no next session and must remain unknown, not zero.
    frame.loc[frame.next_session.isna(), "next_adverse_state"] = np.nan
    frame["next_failed_rate"] = _numeric(frame.failed_rate).shift(-1)
    frame["next_promotion_mid"] = _numeric(frame.promotion_mid).shift(-1)
    frame["next_previous_limit_median_return"] = _numeric(
        frame.previous_limit_median_return).shift(-1)
    frame["next_down_limit_count"] = _numeric(frame.down_limit_count).shift(-1)
    rows = frame[[
        "signal_date", "next_session", "signal_mean_premium",
        "signal_median_premium", "profit_concentration_gap", "profit_state",
        "next_adverse_state", "next_failed_rate", "next_promotion_mid",
        "next_previous_limit_median_return", "next_down_limit_count",
    ]].copy()
    rows = rows[
        rows.profit_state.ne("UNKNOWN") & rows.next_session.notna()
    ].reset_index(drop=True)
    if rows.empty:
        raise ValueError("no complete signal/next-session rows")

    summaries = {
        name: _summary(group) for name, group in rows.groupby("profit_state")
    }
    comparisons = {
        "next_adverse_narrow_minus_broad_positive": _difference(
            rows, "next_adverse_state", "NARROW_POSITIVE", "BROAD_POSITIVE", 1771),
        "next_failed_rate_narrow_minus_broad_positive": _difference(
            rows, "next_failed_rate", "NARROW_POSITIVE", "BROAD_POSITIVE", 1772),
        "next_promotion_mid_narrow_minus_broad_positive": _difference(
            rows, "next_promotion_mid", "NARROW_POSITIVE", "BROAD_POSITIVE", 1773),
        "next_median_premium_narrow_minus_broad_positive": _difference(
            rows, "next_previous_limit_median_return",
            "NARROW_POSITIVE", "BROAD_POSITIVE", 1774),
        "next_down_limits_narrow_minus_broad_positive": _difference(
            rows, "next_down_limit_count", "NARROW_POSITIVE", "BROAD_POSITIVE", 1775),
    }
    dates = sorted(rows.signal_date.unique())
    split_date = dates[int(len(dates) * .7)]
    chronological = {}
    for label, part in {
        "discovery_first_70pct_dates": rows[rows.signal_date.lt(split_date)],
        "validation_last_30pct_dates": rows[rows.signal_date.ge(split_date)],
    }.items():
        chronological[label] = {
            name: _summary(group) for name, group in part.groupby("profit_state")
        }
    adverse = comparisons["next_adverse_narrow_minus_broad_positive"]
    validation = chronological["validation_last_30pct_dates"]
    validation_direction = None
    if "NARROW_POSITIVE" in validation and "BROAD_POSITIVE" in validation:
        left = validation["NARROW_POSITIVE"]["next_adverse_state_rate"]
        right = validation["BROAD_POSITIVE"]["next_adverse_state_rate"]
        if left is not None and right is not None:
            validation_direction = left - right
    association_stable = bool(
        adverse["estimate"] is not None and adverse["ci95"][0] > 0
        and validation_direction is not None and validation_direction > 0)
    report = {
        "version": VERSION, "status": "DIAGNOSTIC_ONLY",
        "cutoff_exclusive": cutoff, "session_rows": int(len(rows)),
        "signal_dates": int(rows.signal_date.nunique()), "split_date": split_date,
        "state_summaries": summaries,
        "bootstrap_comparisons": comparisons,
        "chronological_split": chronological,
        "narrow_profit_next_adverse_association_stable": association_stable,
        "validation_next_adverse_difference_narrow_minus_broad": validation_direction,
        "governance": {
            "mean_premium_alone_sufficient_for_supportive_emotion": False,
            "median_and_distribution_required": True,
            "profit_concentration_direct_stock_reranking_allowed": False,
            "profit_concentration_entry_authorized": False,
            "historical_emotion_scenario_use_allowed": True,
        },
        "interpretation": (
            "A positive mean with a non-positive median is a narrow-profit state: a few strong "
            "names may mask weak typical feedback. It must be evaluated separately from broad "
            "positive premium. The result is an environment diagnostic, not a stock score."
        ),
        "limitations": [
            "provider previous-limit pool completeness is not independently proven",
            "mean and median are close-to-close feedback, not executable entry returns",
            "single chronological split is not a sealed blind test",
            "no theme attribution, candidate execution, fees or T+1 exit path",
        ],
    }
    return rows, report


def run(market_path: Path, output_dir: Path,
        cutoff_exclusive: str = "2025-12-26") -> dict:
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite immutable V177 output")
    rows, report = build(pd.read_csv(market_path), cutoff_exclusive)
    output_dir.mkdir(parents=True)
    rows_path = output_dir / "profit_concentration_rows.csv"
    rows.to_csv(rows_path, index=False, encoding="utf-8-sig")
    report["source_hashes"] = {str(market_path.resolve()): _sha256(market_path)}
    report["rows_sha256"] = _sha256(rows_path)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", type=Path, required=True)
    parser.add_argument("--cutoff-exclusive", default="2025-12-26")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.market, args.output, args.cutoff_exclusive),
                     ensure_ascii=False, indent=2))
