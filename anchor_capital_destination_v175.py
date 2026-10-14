"""Audit where observable promotion migrates when the height anchor fails.

This is a session-level, descriptive study.  It uses the next-session outcomes
of the full closing-limit inventory to distinguish lower-relay promotion,
first-board promotion, both, and no observable internal destination.  Those
outcomes are labels, never signal-date features or stock-entry authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v175-anchor-capital-destination-1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _numeric(series):
    return pd.to_numeric(series, errors="coerce")


def _rate(series):
    values = _numeric(series).dropna()
    return float(values.mean()) if len(values) else None


def _summary(frame: pd.DataFrame) -> dict:
    return {
        "sessions": int(len(frame)),
        "anchor_advance_rate": _rate(frame.anchor_advanced),
        "lower_relay_promotion_rate_per_session": _rate(
            frame.any_lower_relay_promoted),
        "first_board_promotion_rate_per_session": _rate(
            frame.any_first_board_promoted),
        "next_adverse_state_rate": _rate(frame.next_adverse_state),
        "next_failed_rate_mean": float(_numeric(frame.next_failed_rate).mean())
        if _numeric(frame.next_failed_rate).notna().any() else None,
        "next_previous_limit_median_return_mean": float(
            _numeric(frame.next_previous_limit_median_return).mean())
        if _numeric(frame.next_previous_limit_median_return).notna().any() else None,
    }


def _difference(frame: pd.DataFrame, column: str, group_a: str, group_b: str,
                seed: int, draws: int = 3000) -> dict:
    a = _numeric(frame.loc[frame.destination_group.eq(group_a), column]).dropna()
    b = _numeric(frame.loc[frame.destination_group.eq(group_b), column]).dropna()
    if a.empty or b.empty:
        return {"group_a": group_a, "group_b": group_b, "metric": column,
                "estimate": None, "ci95": [None, None],
                "sessions_a": int(len(a)), "sessions_b": int(len(b))}
    estimate = float(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    sims = []
    for _ in range(draws):
        av = rng.choice(a.to_numpy(), size=len(a), replace=True).mean()
        bv = rng.choice(b.to_numpy(), size=len(b), replace=True).mean()
        sims.append(av - bv)
    return {"group_a": group_a, "group_b": group_b, "metric": column,
            "estimate": estimate,
            "ci95": [float(np.quantile(sims, .025)), float(np.quantile(sims, .975))],
            "sessions_a": int(len(a)), "sessions_b": int(len(b))}


def build(features: pd.DataFrame, outcomes: pd.DataFrame, market: pd.DataFrame,
          cutoff_exclusive: str = "2025-12-26") -> tuple[pd.DataFrame, dict]:
    feature_required = {
        "code", "signal_date", "boards", "start_known", "feature_status",
    }
    outcome_required = {"code", "signal_date", "promotion_LABEL_ONLY"}
    market_required = {
        "date", "data_status", "state_divergence", "state_retreat",
        "failed_rate", "previous_limit_median_return",
    }
    if feature_required - set(features):
        raise ValueError("feature fields missing")
    if outcome_required - set(outcomes):
        raise ValueError("outcome fields missing")
    if market_required - set(market):
        raise ValueError("market fields missing")

    feat, out, env = features.copy(), outcomes.copy(), market.copy()
    for frame in (feat, out):
        frame["code"] = frame.code.astype(str).str.zfill(6)
        frame["signal_date"] = pd.to_datetime(
            frame.signal_date, errors="coerce").dt.strftime("%Y-%m-%d")
    env["signal_date"] = pd.to_datetime(
        env.date, errors="coerce").dt.strftime("%Y-%m-%d")
    cutoff = pd.Timestamp(cutoff_exclusive).strftime("%Y-%m-%d")
    feat = feat[feat.signal_date.lt(cutoff)].copy()
    out = out[out.signal_date.lt(cutoff)].copy()
    env = env[env.signal_date.lt(cutoff) & env.data_status.eq("verified")].copy()
    if feat.duplicated(["signal_date", "code"]).any() or out.duplicated(
            ["signal_date", "code"]).any() or env.signal_date.duplicated().any():
        raise ValueError("inputs must be unique by point-in-time key")
    feat["boards"] = _numeric(feat.boards)
    feat["start_known"] = feat.start_known.map(
        lambda value: value is True or str(value).strip().lower() in {"1", "true"})
    feat = feat[feat.start_known & feat.boards.ge(1)].copy()
    joined = feat[["signal_date", "code", "boards", "feature_status"]].merge(
        out[["signal_date", "code", "promotion_LABEL_ONLY"]],
        on=["signal_date", "code"], how="inner", validate="one_to_one")
    joined["promotion_LABEL_ONLY"] = _numeric(joined.promotion_LABEL_ONLY)

    rows = []
    for signal_date, group in joined.groupby("signal_date", sort=True):
        if group.promotion_LABEL_ONLY.isna().any():
            continue
        maximum = int(group.boards.max())
        if maximum < 2:
            continue
        anchors = group[group.boards.eq(maximum)]
        first = group[group.boards.eq(1)]
        lower = group[group.boards.ge(2) & group.boards.lt(maximum)]
        anchor_advanced = bool(anchors.promotion_LABEL_ONLY.eq(1).any())
        lower_promoted = bool(lower.promotion_LABEL_ONLY.eq(1).any())
        first_promoted = bool(first.promotion_LABEL_ONLY.eq(1).any())
        first_rate = (float(first.promotion_LABEL_ONLY.mean()) if len(first) else None)
        lower_rate = (float(lower.promotion_LABEL_ONLY.mean()) if len(lower) else None)
        gap = (lower_rate - first_rate
               if lower_rate is not None and first_rate is not None else None)
        if anchor_advanced:
            destination = "ANCHOR_ADVANCED"
        elif not lower_promoted and not first_promoted:
            destination = "ANCHOR_FAILED_NO_INTERNAL_PROMOTION"
        elif lower_rate is None:
            destination = "ANCHOR_FAILED_FIRST_BOARD_ONLY_NO_LOWER_INVENTORY"
        elif gap >= .10:
            destination = "ANCHOR_FAILED_LOWER_RELAY_RATE_DOMINANT"
        elif gap <= -.10:
            destination = "ANCHOR_FAILED_FIRST_BOARD_RATE_DOMINANT"
        else:
            destination = "ANCHOR_FAILED_BALANCED_PROMOTION_RATES"
        rows.append({
            "signal_date": signal_date, "maximum_board": maximum,
            "anchor_count": int(len(anchors)), "anchor_advanced": anchor_advanced,
            "first_board_inventory": int(len(first)),
            "lower_relay_inventory": int(len(lower)),
            "first_board_promoted_count": int(first.promotion_LABEL_ONLY.sum()),
            "lower_relay_promoted_count": int(lower.promotion_LABEL_ONLY.sum()),
            "first_board_promotion_rate": first_rate,
            "lower_relay_promotion_rate": lower_rate,
            "lower_minus_first_promotion_rate": gap,
            "any_first_board_promoted": first_promoted,
            "any_lower_relay_promoted": lower_promoted,
            "capital_destination_label_only": destination,
        })
    sessions = pd.DataFrame(rows)
    if sessions.empty:
        raise ValueError("no complete height-anchor sessions")

    dates = sorted(env.signal_date.unique())
    next_date = {dates[index]: dates[index + 1] for index in range(len(dates) - 1)}
    next_env = env[["signal_date", "state_divergence", "state_retreat", "failed_rate",
                    "previous_limit_median_return"]].copy()
    next_env = next_env.rename(columns={
        "signal_date": "next_session",
        "failed_rate": "next_failed_rate",
        "previous_limit_median_return": "next_previous_limit_median_return",
    })
    next_env["next_adverse_state"] = (
        _numeric(next_env.state_divergence).eq(1)
        | _numeric(next_env.state_retreat).eq(1)
    ).astype(int)
    next_env = next_env.drop(columns=["state_divergence", "state_retreat"])
    sessions["next_session"] = sessions.signal_date.map(next_date)
    sessions = sessions.merge(next_env, on="next_session", how="left", validate="many_to_one")
    sessions["destination_group"] = sessions.capital_destination_label_only.str.replace(
        "ANCHOR_FAILED_", "", regex=False)

    failed = sessions[~sessions.anchor_advanced].copy()
    distributions = {
        key: int(value) for key, value in
        failed.capital_destination_label_only.value_counts().sort_index().items()
    }
    destination_summaries = {
        name: _summary(group) for name, group in
        failed.groupby("destination_group", sort=True)
    }
    comparisons = {
        "next_adverse_lower_minus_first_dominant": _difference(
            failed, "next_adverse_state", "LOWER_RELAY_RATE_DOMINANT",
            "FIRST_BOARD_RATE_DOMINANT", 1751),
        "next_failed_rate_lower_minus_first_dominant": _difference(
            failed, "next_failed_rate", "LOWER_RELAY_RATE_DOMINANT",
            "FIRST_BOARD_RATE_DOMINANT", 1752),
        "next_previous_limit_median_lower_minus_first_dominant": _difference(
            failed, "next_previous_limit_median_return", "LOWER_RELAY_RATE_DOMINANT",
            "FIRST_BOARD_RATE_DOMINANT", 1753),
    }
    unique_dates = sorted(sessions.signal_date.unique())
    split_date = unique_dates[int(len(unique_dates) * .7)]
    chronological = {}
    for label, frame in {
        "discovery_first_70pct_dates": sessions[sessions.signal_date.lt(split_date)],
        "validation_last_30pct_dates": sessions[sessions.signal_date.ge(split_date)],
    }.items():
        failure = frame[~frame.anchor_advanced]
        chronological[label] = {
            "all_sessions": _summary(frame),
            "anchor_failure_sessions": int(len(failure)),
            "failure_destination_counts": {
                key: int(value) for key, value in
                failure.capital_destination_label_only.value_counts().sort_index().items()
            },
        }
    adverse_cmp = comparisons["next_adverse_lower_minus_first_dominant"]
    median_cmp = comparisons[
        "next_previous_limit_median_lower_minus_first_dominant"]
    dominance_stable = bool(
        adverse_cmp["estimate"] is not None and adverse_cmp["ci95"][1] < 0
        and median_cmp["estimate"] is not None and median_cmp["ci95"][0] > 0)
    report = {
        "version": VERSION, "status": "DIAGNOSTIC_ONLY",
        "cutoff_exclusive": cutoff, "session_rows": int(len(sessions)),
        "signal_dates": int(sessions.signal_date.nunique()), "split_date": split_date,
        "anchor_failure_sessions": int(len(failed)),
        "anchor_advance_sessions": int(sessions.anchor_advanced.sum()),
        "failure_destination_counts": distributions,
        "destination_summaries": destination_summaries,
        "bootstrap_comparisons": comparisons,
        "chronological_split": chronological,
        "dominant_lane_margin_pp": 10.0,
        "lane_dominance_and_better_environment_association_stable": dominance_stable,
        "governance": {
            "anchor_failure_alone_implies_retreat": False,
            "same_session_destination_as_signal_date_feature_allowed": False,
            "automatic_high_to_low_entry_allowed": False,
            "stock_reranking_allowed": False,
            "historical_scenario_base_rates_allowed": True,
        },
        "interpretation": (
            "Failure of the height anchor is not itself the destination of capital. Because at "
            "least one first-board promotion is nearly automatic in a wide inventory, the audit "
            "compares lane promotion rates with a fixed ten-percentage-point dominance margin "
            "instead of using an any-stock flag. These are "
            "realized labels, so they explain the path after it occurs and cannot be leaked into "
            "a prior-close score or used to auto-buy a lower board."
        ),
        "limitations": [
            "promotion is a closing-board outcome, not intraday capital-flow or amount-share evidence",
            "same-theme versus cross-theme destination is unavailable without complete point-in-time themes",
            "current-universe and candidate-inventory completeness remain unproved",
            "historical security eligibility is incomplete",
            "no queue, verified fill, actual exit, catalyst timing or causal flow sequence",
        ],
    }
    return sessions, report


def run(features_path, outcomes_path, market_path, output_dir,
        cutoff_exclusive="2025-12-26") -> dict:
    paths = list(map(Path, (features_path, outcomes_path, market_path)))
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable capital-destination audit")
    rows, report = build(
        pd.read_csv(paths[0], dtype={"code": str}, low_memory=False),
        pd.read_csv(paths[1], dtype={"code": str}, low_memory=False),
        pd.read_csv(paths[2], low_memory=False), cutoff_exclusive,
    )
    target.mkdir(parents=True)
    rows_path = target / "session_destination_rows.csv"
    rows.to_csv(rows_path, index=False, encoding="utf-8-sig")
    report["source_hashes"] = {str(path.resolve()): _sha256(path) for path in paths}
    report["rows_sha256"] = _sha256(rows_path)
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--outcomes", required=True)
    parser.add_argument("--market", required=True)
    parser.add_argument("--cutoff-exclusive", default="2025-12-26")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.outcomes, args.market, args.output,
                         args.cutoff_exclusive), ensure_ascii=False, indent=2))
