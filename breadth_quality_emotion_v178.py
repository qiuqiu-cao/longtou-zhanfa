"""Audit whether expanding limit-up breadth is clean or fragile.

The signal is observed at the signal-day close.  A breadth expansion is
classified as fragile when the closing limit-up count rises while the failed-
board rate also rises from the immediately preceding verified session.  All
outcomes are shifted to the immediately following verified session.  This is
an aggregate emotion diagnostic and never a stock-ranking or entry model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v178-breadth-quality-emotion-1"


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
        "signal_limit_count_change_mean": _mean(frame.signal_limit_change),
        "signal_failed_rate_change_mean": _mean(frame.signal_failed_rate_change),
        "signal_relay_count_change_mean": _mean(frame.signal_relay_change),
        "signal_height_change_mean": _mean(frame.signal_height_change),
        "signal_first_board_share_mean": _mean(frame.signal_first_board_share),
        "next_adverse_state_rate": _mean(frame.next_adverse_state),
        "next_failed_rate_mean": _mean(frame.next_failed_rate),
        "next_promotion_mid_mean": _mean(frame.next_promotion_mid),
        "next_previous_limit_median_return_mean": _mean(
            frame.next_previous_limit_median_return),
        "next_down_limit_count_mean": _mean(frame.next_down_limit_count),
    }


def _difference(frame: pd.DataFrame, column: str, left: str, right: str,
                seed: int, draws: int = 3000) -> dict:
    a = _numeric(frame.loc[frame.breadth_quality.eq(left), column]).dropna()
    b = _numeric(frame.loc[frame.breadth_quality.eq(right), column]).dropna()
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
        "date", "data_status", "limit_count", "consecutive_count", "max_board",
        "failed_rate", "promotion_mid", "previous_limit_median_return",
        "down_limit_count", "state_divergence", "state_retreat",
    }
    if required - set(market):
        raise ValueError("market fields missing")
    frame = market.copy()
    frame["session"] = pd.to_datetime(frame.date, errors="coerce")
    if frame.session.isna().any() or frame.session.duplicated().any():
        raise ValueError("market dates must be valid and unique")
    frame = frame.sort_values("session").reset_index(drop=True)
    cutoff = pd.Timestamp(cutoff_exclusive)

    for column in (
        "limit_count", "consecutive_count", "max_board", "failed_rate",
        "promotion_mid", "previous_limit_median_return", "down_limit_count",
        "state_divergence", "state_retreat",
    ):
        frame[column] = _numeric(frame[column])

    frame["previous_status"] = frame.data_status.shift(1)
    frame["next_status"] = frame.data_status.shift(-1)
    frame["signal_limit_change"] = frame.limit_count.diff()
    frame["signal_failed_rate_change"] = frame.failed_rate.diff()
    frame["signal_relay_change"] = frame.consecutive_count.diff()
    frame["signal_height_change"] = frame.max_board.diff()
    frame["signal_first_board_count"] = frame.limit_count - frame.consecutive_count
    frame["signal_first_board_share"] = np.where(
        frame.limit_count.gt(0), frame.signal_first_board_count / frame.limit_count,
        np.nan)

    is_expansion = frame.signal_limit_change.gt(0)
    frame["breadth_quality"] = np.select(
        [is_expansion & frame.signal_failed_rate_change.gt(0),
         is_expansion & frame.signal_failed_rate_change.le(0)],
        ["FRAGILE_EXPANSION", "CLEAN_EXPANSION"], default="NO_EXPANSION")
    frame["low_only_warning"] = (
        is_expansion & frame.signal_relay_change.le(0)
        & frame.signal_height_change.le(0)
    )

    frame["next_session"] = frame.session.shift(-1).dt.strftime("%Y-%m-%d")
    frame["next_adverse_state"] = (
        frame.state_divergence.shift(-1).eq(1)
        | frame.state_retreat.shift(-1).eq(1)
    ).astype(float)
    frame["next_failed_rate"] = frame.failed_rate.shift(-1)
    frame["next_promotion_mid"] = frame.promotion_mid.shift(-1)
    frame["next_previous_limit_median_return"] = (
        frame.previous_limit_median_return.shift(-1))
    frame["next_down_limit_count"] = frame.down_limit_count.shift(-1)

    eligible = (
        frame.session.lt(cutoff)
        & frame.data_status.eq("verified")
        & frame.previous_status.eq("verified")
        & frame.next_status.eq("verified")
        & frame.breadth_quality.isin(["FRAGILE_EXPANSION", "CLEAN_EXPANSION"])
    )
    columns = [
        "date", "next_session", "breadth_quality", "low_only_warning",
        "signal_limit_change", "signal_failed_rate_change", "signal_relay_change",
        "signal_height_change", "signal_first_board_count",
        "signal_first_board_share", "next_adverse_state", "next_failed_rate",
        "next_promotion_mid", "next_previous_limit_median_return",
        "next_down_limit_count",
    ]
    rows = frame.loc[eligible, columns].rename(columns={"date": "signal_date"})
    rows = rows.reset_index(drop=True)
    if rows.empty:
        raise ValueError("no verified expansion sessions before cutoff")

    dates = rows.signal_date.drop_duplicates().sort_values().tolist()
    split_index = (0 if len(dates) == 1 else
                   max(1, min(len(dates) - 1, int(len(dates) * .70))))
    split_date = dates[split_index]
    discovery = rows[rows.signal_date.lt(split_date)]
    validation = rows[rows.signal_date.ge(split_date)]
    states = ["FRAGILE_EXPANSION", "CLEAN_EXPANSION"]
    comparisons = {
        "next_adverse_fragile_minus_clean": _difference(
            rows, "next_adverse_state", states[0], states[1], 17801),
        "next_failed_rate_fragile_minus_clean": _difference(
            rows, "next_failed_rate", states[0], states[1], 17802),
        "next_promotion_mid_fragile_minus_clean": _difference(
            rows, "next_promotion_mid", states[0], states[1], 17803),
        "next_median_premium_fragile_minus_clean": _difference(
            rows, "next_previous_limit_median_return", states[0], states[1], 17804),
        "next_down_limits_fragile_minus_clean": _difference(
            rows, "next_down_limit_count", states[0], states[1], 17805),
    }
    split_comparisons = {
        "discovery_next_adverse_fragile_minus_clean": _difference(
            discovery, "next_adverse_state", states[0], states[1], 17811),
        "validation_next_adverse_fragile_minus_clean": _difference(
            validation, "next_adverse_state", states[0], states[1], 17812),
        "discovery_next_failed_rate_fragile_minus_clean": _difference(
            discovery, "next_failed_rate", states[0], states[1], 17813),
        "validation_next_failed_rate_fragile_minus_clean": _difference(
            validation, "next_failed_rate", states[0], states[1], 17814),
    }
    discovery_adverse = split_comparisons[
        "discovery_next_adverse_fragile_minus_clean"]
    validation_adverse = split_comparisons[
        "validation_next_adverse_fragile_minus_clean"]
    discovery_failed = split_comparisons[
        "discovery_next_failed_rate_fragile_minus_clean"]
    validation_failed = split_comparisons[
        "validation_next_failed_rate_fragile_minus_clean"]
    validation_summaries = {
        state: _summary(validation[validation.breadth_quality.eq(state)])
        for state in states
    }
    validation_adverse_difference = None
    if all(validation_summaries[state]["next_adverse_state_rate"] is not None
           for state in states):
        validation_adverse_difference = (
            validation_summaries[states[0]]["next_adverse_state_rate"]
            - validation_summaries[states[1]]["next_adverse_state_rate"])
    report = {
        "version": VERSION,
        "status": "DIAGNOSTIC_ONLY",
        "cutoff_exclusive": pd.Timestamp(cutoff).strftime("%Y-%m-%d"),
        "session_rows": int(len(rows)),
        "split_date": split_date,
        "definition": {
            "breadth_expansion": "signal limit-up count change > 0",
            "fragile_expansion": (
                "breadth expansion and failed-board rate change > 0 versus the "
                "immediately preceding verified session"),
            "clean_expansion": (
                "breadth expansion and failed-board rate change <= 0 versus the "
                "immediately preceding verified session"),
            "low_only_warning": (
                "breadth expands while relay count and maximum board do not rise; "
                "descriptive only and not part of the primary split"),
        },
        "state_summaries": {
            state: _summary(rows[rows.breadth_quality.eq(state)]) for state in states
        },
        "bootstrap_comparisons": comparisons,
        "chronological_split_bootstrap_comparisons": split_comparisons,
        "chronological_split": {
            "discovery_first_70pct_dates": {
                state: _summary(discovery[discovery.breadth_quality.eq(state)])
                for state in states
            },
            "validation_last_30pct_dates": validation_summaries,
        },
        "fragile_expansion_adverse_association_stable": bool(
            discovery_adverse["ci95"][0] is not None
            and discovery_adverse["ci95"][0] > 0
            and validation_adverse["ci95"][0] is not None
            and validation_adverse["ci95"][0] > 0),
        "fragile_expansion_failed_rate_association_stable": bool(
            discovery_failed["ci95"][0] is not None
            and discovery_failed["ci95"][0] > 0
            and validation_failed["ci95"][0] is not None
            and validation_failed["ci95"][0] > 0),
        "validation_next_adverse_difference_fragile_minus_clean": (
            validation_adverse_difference),
        "governance": {
            "limit_count_growth_alone_sufficient_for_repair": False,
            "failed_board_rate_direction_required": True,
            "historical_emotion_scenario_use_allowed": True,
            "direct_stock_reranking_allowed": False,
            "entry_authorized": False,
        },
        "interpretation": (
            "More closing limit-ups do not by themselves prove a healthy repair. "
            "When breadth expands while the failed-board rate also rises, the new "
            "breadth coexists with worsening rejection pressure and must be separated "
            "from a clean expansion."),
        "limitations": [
            "aggregate provider pool completeness is not independently proven",
            "failed-rate changes are close-of-session diagnostics, not intraday entry signals",
            "single chronological split is not a sealed blind test",
            "no candidate-level theme attribution, fill, cost, or B+1/B+2/B+3 trade path",
        ],
    }
    return rows, report


def run(input_path: str, output_dir: str,
        cutoff_exclusive: str = "2025-12-26") -> dict:
    source = Path(input_path)
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable V178 output")
    rows, report = build(pd.read_csv(source), cutoff_exclusive)
    report["source_hashes"] = {str(source.resolve()): _sha256(source)}
    target.mkdir(parents=True, exist_ok=False)
    rows.to_csv(target / "breadth_quality_rows.csv", index=False, encoding="utf-8-sig")
    report["rows_sha256"] = _sha256(target / "breadth_quality_rows.csv")
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
