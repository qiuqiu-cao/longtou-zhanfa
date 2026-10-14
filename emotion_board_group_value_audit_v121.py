"""Time-separated audit of emotion-risk value for first boards versus relays.

The outcome is a conservative daily-bar *opportunity proxy*: a candidate whose
next-session high touched the inferred limit and was not a one-price limit-up is
marked from that hypothetical limit price to B+1/B+2/B+3 closes.  It is not an
actual fill or completed trade.
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


GROUPS = ["FIRST_BOARD", "CONSECUTIVE_BOARD"]
HORIZONS = [1, 2, 3]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _draw_blocks(values: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, len(values), size=int(np.ceil(len(values) / block)))
    indices = np.concatenate([
        (start + np.arange(block)) % len(values) for start in starts
    ])[:len(values)]
    return values[indices]


def _difference(high: np.ndarray, low: np.ndarray, repetitions=2000, block=5, seed=0) -> dict:
    high, low = high[np.isfinite(high)], low[np.isfinite(low)]
    if len(high) < 10 or len(low) < 10:
        return {"high_minus_low": None, "ci95": [None, None], "status": "INSUFFICIENT"}
    rng = np.random.default_rng(seed)
    draws = np.array([
        _draw_blocks(high, min(block, len(high)), rng).mean()
        - _draw_blocks(low, min(block, len(low)), rng).mean()
        for _ in range(repetitions)
    ])
    return {
        "high_minus_low": float(high.mean() - low.mean()),
        "ci95": [float(x) for x in np.quantile(draws, [.025, .975])],
        "status": "SESSION_LEVEL_BLOCK_BOOTSTRAP_DESCRIPTIVE",
    }


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes"})


def _metric(daily: pd.DataFrame, column: str, expected: str, seed: int) -> dict:
    high = daily.loc[daily.risk_group.eq("HIGH"), column].dropna().to_numpy(float)
    low = daily.loc[daily.risk_group.eq("LOW"), column].dropna().to_numpy(float)
    comparison = _difference(high, low, seed=seed)
    observed, ci = comparison["high_minus_low"], comparison["ci95"]
    direction = bool(observed is not None and (observed < 0 if expected == "LOWER" else observed > 0))
    strong = bool(direction and ci[0] is not None and (ci[1] < 0 if expected == "LOWER" else ci[0] > 0))
    return {
        "expected_high_risk_direction": expected,
        "high_risk_sessions": int(len(high)), "low_risk_sessions": int(len(low)),
        "high_risk_mean": float(high.mean()) if len(high) else None,
        "low_risk_mean": float(low.mean()) if len(low) else None,
        **comparison,
        "direction_pass": direction,
        "ci_excludes_zero_in_expected_direction": strong,
    }


def audit(features: pd.DataFrame, outcomes: pd.DataFrame, market: pd.DataFrame,
          split_fraction: float = .70) -> tuple[dict, pd.DataFrame]:
    feature_required = {"code", "signal_date", "board_group", "feature_status"}
    outcome_required = {"code", "signal_date", "entry_one_price_LABEL_ONLY", "price_path_state"}
    outcome_required |= {f"b{h}_net_close_mark_LABEL_ONLY" for h in HORIZONS}
    market_required = {"date", "emotion_adverse_next_probability", "emotion_forecast_status"}
    for label, frame, required in (
        ("features", features, feature_required), ("outcomes", outcomes, outcome_required),
        ("market", market, market_required),
    ):
        if required - set(frame):
            raise ValueError(f"{label} fields missing: {sorted(required - set(frame))}")
    features, outcomes, market = features.copy(), outcomes.copy(), market.copy()
    for frame in (features, outcomes):
        frame["code"] = frame.code.astype(str).str.zfill(6)
        if frame.duplicated(["code", "signal_date"]).any():
            raise ValueError("duplicate candidate/date key")
    market["signal_date"] = pd.to_datetime(market.date, errors="coerce").dt.strftime("%Y-%m-%d")
    if market.signal_date.duplicated().any():
        raise ValueError("duplicate market date")
    market["risk_probability"] = pd.to_numeric(
        market.emotion_adverse_next_probability, errors="coerce",
    )
    rows = features.merge(outcomes, on=["code", "signal_date"], validate="one_to_one")
    rows = rows.merge(
        market[["signal_date", "risk_probability", "emotion_forecast_status"]],
        on="signal_date", how="left", validate="many_to_one",
    )
    rows = rows[
        rows.board_group.isin(GROUPS)
        & rows.feature_status.eq("historical_reconstruction_not_forward_archive")
        & rows.price_path_state.eq("PRICE_OPPORTUNITY_PROXY_ONLY")
        & ~_truthy(rows.entry_one_price_LABEL_ONLY)
        & rows.risk_probability.notna()
    ].copy()
    dates = sorted(rows.signal_date.unique())
    if len(dates) < 250:
        raise ValueError("insufficient sessions for time-separated emotion audit")
    split = int(len(dates) * split_fraction)
    development_dates, evaluation_dates = dates[:split], dates[split:]
    daily_probability = rows.groupby("signal_date").risk_probability.first().sort_index()
    evaluation = rows[rows.signal_date.isin(evaluation_dates)].copy()
    rolling_groups, rolling_cuts = {}, {}
    for date in evaluation_dates:
        history = daily_probability.loc[daily_probability.index < date].tail(120)
        if len(history) < 60:
            raise ValueError("insufficient trailing probability history")
        low_cut, high_cut = float(history.quantile(.20)), float(history.quantile(.80))
        probability = float(daily_probability.loc[date])
        rolling_groups[date] = (
            "HIGH" if probability >= high_cut else "LOW" if probability <= low_cut else "MIDDLE"
        )
        rolling_cuts[date] = {"low": low_cut, "high": high_cut, "probability": probability}
    evaluation["risk_group"] = evaluation.signal_date.map(rolling_groups)
    for horizon in HORIZONS:
        mark = pd.to_numeric(evaluation[f"b{horizon}_net_close_mark_LABEL_ONLY"], errors="coerce")
        evaluation[f"b{horizon}_mark"] = mark
        evaluation[f"b{horizon}_positive"] = np.where(mark.notna(), mark.gt(0).astype(float), np.nan)
        evaluation[f"b{horizon}_large_loss"] = np.where(mark.notna(), mark.le(-.07).astype(float), np.nan)

    reports = {}
    for group_index, group in enumerate(GROUPS):
        candidates = evaluation[evaluation.board_group.eq(group)].copy()
        daily_columns = []
        named = {}
        for horizon in HORIZONS:
            for suffix in ("mark", "positive", "large_loss"):
                column = f"b{horizon}_{suffix}"
                daily_columns.append(column)
                named[column] = "LOWER" if suffix in {"mark"} else "LOWER" if suffix == "positive" else "HIGHER"
        daily = candidates.groupby(["signal_date", "risk_group"], as_index=False)[daily_columns].mean()
        metrics = {
            column: _metric(daily, column, expected, seed=100 * group_index + index)
            for index, (column, expected) in enumerate(named.items())
        }
        direction_passes = sum(item["direction_pass"] for item in metrics.values())
        strong_passes = sum(item["ci_excludes_zero_in_expected_direction"] for item in metrics.values())
        # A risk overlay must improve all three types (return, win frequency,
        # tail loss) across most horizons; a single favorable statistic is not enough.
        gate = bool(direction_passes >= 7 and strong_passes >= 3)
        reports[group] = {
            "candidate_rows": int(len(candidates)),
            "evaluation_sessions": int(candidates.signal_date.nunique()),
            "high_risk_candidate_rows": int(candidates.risk_group.eq("HIGH").sum()),
            "low_risk_candidate_rows": int(candidates.risk_group.eq("LOW").sum()),
            "metrics": metrics,
            "direction_pass_count": int(direction_passes),
            "strong_ci_pass_count": int(strong_passes),
            "risk_overlay_proxy_gate_passed": gate,
        }
    report = {
        "version": "v121-emotion-board-group-economic-audit-1",
        "status": "DIAGNOSTIC_ONLY",
        "scope": "mainboard historical reconstructed first/consecutive closing-limit candidates",
        "outcome": "hypothetical next-session non-one-price limit-touch fill to B+1/B+2/B+3 close marks",
        "split": {
            "method": "last_30pct_sessions_evaluated_against_trailing_120_session_probability_quintiles",
            "development_sessions": len(development_dates), "evaluation_sessions": len(evaluation_dates),
            "development_end": development_dates[-1], "evaluation_start": evaluation_dates[0],
            "risk_cutoff_lookback_sessions": 120,
            "first_evaluation_cutoffs": rolling_cuts[evaluation_dates[0]],
            "last_evaluation_cutoffs": rolling_cuts[evaluation_dates[-1]],
        },
        "board_groups": reports,
        "gate_rule": "at least 7/9 expected directions and 3/9 session-block-bootstrap CIs exclude zero",
        "positive_buy_factor_authorized": False,
        "deployment_authorized": False,
        "complete_trade_count": 0,
        "limitations": [
            "daily high touching limit does not establish an actual fill or queue priority",
            "features are historical reconstructions rather than original forward archives",
            "risk groups use trailing distributional quintiles and are not deployment thresholds",
            "session-level aggregation avoids overweighting broad limit-up days but loses candidate detail",
            "complete-trade profitability remains untested",
        ],
    }
    return report, evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--outcomes", required=True)
    parser.add_argument("--market", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {name: Path(getattr(args, name)) for name in ("features", "outcomes", "market")}
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable audit directory")
    report, evaluation = audit(
        pd.read_csv(paths["features"], dtype={"code": str}),
        pd.read_csv(paths["outcomes"], dtype={"code": str}),
        pd.read_csv(paths["market"]),
    )
    output.mkdir(parents=True)
    rows_path = output / "evaluation_rows_PROXY_ONLY.csv"
    evaluation.to_csv(rows_path, index=False, encoding="utf-8-sig")
    report["generated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    report["source_hashes"] = {str(path.resolve()): sha256(path) for path in paths.values()}
    report["evaluation_rows_sha256"] = sha256(rows_path)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
