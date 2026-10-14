"""Test whether ladder gaps mean scarcity or missing middle-board support.

This is a descriptive, date-clustered study.  Promotion/touch labels include
all height-anchor observations.  B+2 marks are conditional daily proxies and
are never described as executable trade returns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v172-ladder-gap-tradeoff-1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rate(series: pd.Series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def _summary(frame: pd.DataFrame) -> dict:
    b2 = pd.to_numeric(frame.b2_net_close_mark_LABEL_ONLY, errors="coerce")
    promotion = pd.to_numeric(frame.promotion_LABEL_ONLY, errors="coerce")
    touch = pd.to_numeric(frame.next_session_touched_limit_LABEL_ONLY, errors="coerce")
    big = pd.to_numeric(frame.b2_big_loss_mark_LABEL_ONLY, errors="coerce")
    return {
        "observations": int(len(frame)),
        "signal_dates": int(frame.signal_date.nunique()),
        "promotion_known": int(promotion.notna().sum()),
        "promotion_rate": _rate(promotion),
        "next_touch_known": int(touch.notna().sum()),
        "next_touch_rate": _rate(touch),
        "b2_proxy_known_conditional_on_touch": int(b2.notna().sum()),
        "b2_proxy_mean": float(b2.mean()) if b2.notna().any() else None,
        "b2_proxy_positive_rate": float(b2.gt(0).loc[b2.notna()].mean())
        if b2.notna().any() else None,
        "b2_proxy_big_loss_known": int(big.notna().sum()),
        "b2_proxy_big_loss_rate": _rate(big),
    }


def _date_metric(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    work = frame.assign(_value=values).dropna(subset=["_value"])
    return work.groupby("signal_date")._value.mean()


def _cluster_difference(frame: pd.DataFrame, column: str, group_a: str, group_b: str,
                        seed: int = 172, draws: int = 2000) -> dict:
    a = _date_metric(frame[frame.analysis_group.eq(group_a)], column)
    b = _date_metric(frame[frame.analysis_group.eq(group_b)], column)
    if a.empty or b.empty:
        return {"group_a": group_a, "group_b": group_b, "estimate": None,
                "ci95": [None, None], "cluster_dates_a": int(len(a)),
                "cluster_dates_b": int(len(b))}
    estimate = float(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    sims = []
    for _ in range(draws):
        av = rng.choice(a.to_numpy(), size=len(a), replace=True).mean()
        bv = rng.choice(b.to_numpy(), size=len(b), replace=True).mean()
        sims.append(av - bv)
    return {
        "group_a": group_a, "group_b": group_b, "estimate": estimate,
        "ci95": [float(np.quantile(sims, .025)), float(np.quantile(sims, .975))],
        "cluster_dates_a": int(len(a)), "cluster_dates_b": int(len(b)),
    }


def build(candidates: pd.DataFrame, outcomes: pd.DataFrame,
          market: pd.DataFrame, cutoff_exclusive: str = "2025-12-26") -> dict:
    candidate_required = {"code", "signal_date", "boards"}
    outcome_required = {
        "code", "signal_date", "promotion_LABEL_ONLY",
        "next_session_touched_limit_LABEL_ONLY", "b2_net_close_mark_LABEL_ONLY",
        "b2_big_loss_mark_LABEL_ONLY",
    }
    market_required = {
        "date", "data_status", "max_board", "ladder_gap_count", "promotion_mid",
        "previous_limit_tail_loss",
    }
    if candidate_required - set(candidates):
        raise ValueError("candidate fields missing")
    if outcome_required - set(outcomes):
        raise ValueError("outcome fields missing")
    if market_required - set(market):
        raise ValueError("market fields missing")

    cand = candidates.copy()
    out = outcomes.copy()
    env = market.copy()
    for frame in (cand, out):
        frame["code"] = frame.code.astype(str).str.zfill(6)
        frame["signal_date"] = pd.to_datetime(
            frame.signal_date, errors="coerce").dt.strftime("%Y-%m-%d")
    env["signal_date"] = pd.to_datetime(env.date, errors="coerce").dt.strftime("%Y-%m-%d")
    cutoff = pd.Timestamp(cutoff_exclusive).strftime("%Y-%m-%d")
    cand = cand[cand.signal_date.lt(cutoff)].copy()
    out = out[out.signal_date.lt(cutoff)].copy()
    env = env[env.signal_date.lt(cutoff) & env.data_status.eq("verified")].copy()
    if cand.duplicated(["signal_date", "code"]).any() or out.duplicated(
            ["signal_date", "code"]).any() or env.signal_date.duplicated().any():
        raise ValueError("inputs must be unique by point-in-time key")
    cand["boards"] = pd.to_numeric(cand.boards, errors="coerce")
    env["max_board"] = pd.to_numeric(env.max_board, errors="coerce")
    merged = cand.merge(
        env[["signal_date", "max_board", "ladder_gap_count", "promotion_mid",
             "previous_limit_tail_loss"]], on="signal_date", how="inner", validate="many_to_one")
    merged = merged[
        merged.boards.eq(merged.max_board) & merged.boards.ge(2)
    ].copy()
    merged = merged.merge(out, on=["signal_date", "code"], how="left",
                          validate="one_to_one", suffixes=("", "_outcome"))
    for column in ("ladder_gap_count", "promotion_mid", "previous_limit_tail_loss"):
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged["middle_damage"] = (
        merged.promotion_mid.lt(.25) | merged.previous_limit_tail_loss.le(-.07)
    )
    merged["gap_bucket"] = np.select(
        [merged.ladder_gap_count.ge(2), merged.ladder_gap_count.eq(1),
         merged.ladder_gap_count.eq(0)],
        ["GAP_2PLUS", "GAP_1", "NO_GAP"], default="UNKNOWN")
    merged["analysis_group"] = np.where(
        merged.gap_bucket.eq("GAP_2PLUS"),
        np.where(merged.middle_damage, "GAP_2PLUS_DAMAGED", "GAP_2PLUS_HEALTHY"),
        merged.gap_bucket,
    )
    merged = merged[merged.analysis_group.ne("UNKNOWN")].copy()
    if merged.empty:
        raise ValueError("no height-anchor observations after joins")

    groups = {
        name: _summary(group) for name, group in merged.groupby("analysis_group")
    }
    comparisons = {
        "promotion_damaged_vs_healthy_gap": _cluster_difference(
            merged, "promotion_LABEL_ONLY", "GAP_2PLUS_DAMAGED", "GAP_2PLUS_HEALTHY"),
        "touch_damaged_vs_healthy_gap": _cluster_difference(
            merged, "next_session_touched_limit_LABEL_ONLY",
            "GAP_2PLUS_DAMAGED", "GAP_2PLUS_HEALTHY", seed=173),
        "b2_proxy_damaged_vs_healthy_gap": _cluster_difference(
            merged, "b2_net_close_mark_LABEL_ONLY",
            "GAP_2PLUS_DAMAGED", "GAP_2PLUS_HEALTHY", seed=174),
        "promotion_all_gap2_vs_no_gap": _cluster_difference(
            merged.assign(analysis_group=np.where(
                merged.gap_bucket.eq("GAP_2PLUS"), "ALL_GAP_2PLUS", merged.analysis_group)),
            "promotion_LABEL_ONLY", "ALL_GAP_2PLUS", "NO_GAP", seed=175),
    }
    dates = sorted(merged.signal_date.unique())
    split_date = dates[int(len(dates) * .7)] if len(dates) > 1 else dates[0]
    chronological = {}
    for period, frame in {
        "discovery_first_70pct_dates": merged[merged.signal_date.lt(split_date)],
        "validation_last_30pct_dates": merged[merged.signal_date.ge(split_date)],
    }.items():
        chronological[period] = {
            name: _summary(group) for name, group in frame.groupby("analysis_group")
        }
    promotion_cmp = comparisons["promotion_damaged_vs_healthy_gap"]
    b2_cmp = comparisons["b2_proxy_damaged_vs_healthy_gap"]
    promotion_stable = (
        promotion_cmp["estimate"] is not None
        and promotion_cmp["ci95"][1] < 0
    )
    b2_stable = (
        b2_cmp["estimate"] is not None and b2_cmp["ci95"][1] < 0
    )
    factor_allowed = bool(promotion_stable and b2_stable)
    return {
        "version": VERSION, "status": "DIAGNOSTIC_ONLY",
        "cutoff_exclusive": cutoff, "height_anchor_observations": int(len(merged)),
        "signal_dates": int(merged.signal_date.nunique()), "split_date": split_date,
        "group_summaries": groups, "date_clustered_comparisons": comparisons,
        "chronological_split": chronological,
        "promotion_difference_stable": promotion_stable,
        "conditional_b2_difference_stable": b2_stable,
        "ladder_gap_positive_ranking_factor_allowed": factor_allowed,
        "interpretation": (
            "A ladder gap is not a monotonic positive factor. It may indicate scarcity only "
            "when middle-board promotion and tail-loss evidence remain healthy. Conditional B+2 "
            "marks are opportunity proxies after next-session touch, not executable returns."
        ),
        "limitations": [
            "current-universe survivorship bias remains",
            "historical security status is not independently complete",
            "candidate inventory valid_identity is unavailable and cannot establish historical eligibility",
            "B+2 proxy is conditional on a next-session price opportunity",
            "no queue, verified fill, actual exit, theme identity or catalyst timing",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--outcomes", required=True)
    parser.add_argument("--market", required=True)
    parser.add_argument("--cutoff-exclusive", default="2025-12-26")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = [Path(args.candidates), Path(args.outcomes), Path(args.market)]
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable ladder-gap report")
    report = build(
        pd.read_csv(paths[0], dtype={"code": str}),
        pd.read_csv(paths[1], dtype={"code": str}),
        pd.read_csv(paths[2]), args.cutoff_exclusive,
    )
    report["source_hashes"] = {str(path.resolve()): _sha256(path) for path in paths}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                      encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
