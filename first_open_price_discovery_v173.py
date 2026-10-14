"""Audit the first turnover board after one-price limit-up streaks.

The path is identified only from information available by the signal close:
the previous trading session was a one-price closing limit-up and the current
session is the next consecutive, non-one-price closing limit-up.  Outcomes are
kept separate: next-session promotion/touch include all rows, while B+2 marks
are explicitly conditional daily proxies and never executable trade returns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v173-first-open-price-discovery-1"
GROUPS = (
    "ORDINARY_TURNOVER_RELAY",
    "FIRST_OPEN_AFTER_1_ONE_PRICE",
    "FIRST_OPEN_AFTER_2PLUS_ONE_PRICE",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _summary(frame: pd.DataFrame) -> dict:
    promotion = _numeric(frame.promotion_LABEL_ONLY)
    touch = _numeric(frame.next_session_touched_limit_LABEL_ONLY)
    b2 = _numeric(frame.b2_net_close_mark_LABEL_ONLY)
    loss = _numeric(frame.b2_big_loss_mark_LABEL_ONLY)
    turnover = _numeric(frame.turnover)
    return {
        "observations": int(len(frame)), "signal_dates": int(frame.signal_date.nunique()),
        "promotion_known": int(promotion.notna().sum()),
        "promotion_rate": float(promotion.mean()) if promotion.notna().any() else None,
        "next_touch_known": int(touch.notna().sum()),
        "next_touch_rate": float(touch.mean()) if touch.notna().any() else None,
        "turnover_known": int(turnover.notna().sum()),
        "mean_turnover": float(turnover.mean()) if turnover.notna().any() else None,
        "b2_proxy_known_conditional_on_touch": int(b2.notna().sum()),
        "b2_proxy_mean": float(b2.mean()) if b2.notna().any() else None,
        "b2_proxy_positive_rate": float(b2.gt(0).loc[b2.notna()].mean())
        if b2.notna().any() else None,
        "b2_proxy_big_loss_known": int(loss.notna().sum()),
        "b2_proxy_big_loss_rate": float(loss.mean()) if loss.notna().any() else None,
    }


def _date_means(frame: pd.DataFrame, column: str) -> pd.Series:
    values = _numeric(frame[column])
    work = frame.assign(_metric=values).dropna(subset=["_metric"])
    return work.groupby("signal_date")._metric.mean()


def _cluster_comparison(frame: pd.DataFrame, column: str, left: str, right: str,
                        seed: int, repetitions: int = 2000) -> dict:
    a = _date_means(frame[frame.path_group.eq(left)], column)
    b = _date_means(frame[frame.path_group.eq(right)], column)
    if a.empty or b.empty:
        return {"left": left, "right": right, "estimate": None, "ci95": [None, None],
                "left_dates": int(len(a)), "right_dates": int(len(b))}
    estimate = float(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions)
    for idx in range(repetitions):
        draws[idx] = (
            rng.choice(a.to_numpy(), size=len(a), replace=True).mean()
            - rng.choice(b.to_numpy(), size=len(b), replace=True).mean()
        )
    return {"left": left, "right": right, "estimate": estimate,
            "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
            "left_dates": int(len(a)), "right_dates": int(len(b))}


def _previous_session_map(calendar: pd.DataFrame, cutoff: str) -> dict[str, str]:
    if not {"date", "data_status"} <= set(calendar):
        raise ValueError("calendar fields missing")
    valid = calendar[calendar.data_status.eq("verified")].copy()
    dates = sorted(pd.to_datetime(valid.date, errors="coerce").dropna().dt.strftime("%Y-%m-%d").unique())
    dates = [date for date in dates if date < cutoff]
    if len(dates) < 2:
        raise ValueError("verified trading calendar is insufficient")
    return {dates[index]: dates[index - 1] for index in range(1, len(dates))}


def identify(features: pd.DataFrame, calendar: pd.DataFrame, cutoff_exclusive: str) -> pd.DataFrame:
    required = {
        "date", "code", "boards", "one_word_proxy", "one_word_streak", "turnover",
        "bomb_count", "first_seal_minute", "last_seal_minute",
    }
    if required - set(features):
        raise ValueError("first-open feature fields missing")
    cutoff = pd.Timestamp(cutoff_exclusive).strftime("%Y-%m-%d")
    rows = features[list(required)].copy()
    rows["signal_date"] = pd.to_datetime(rows.date, errors="coerce").dt.strftime("%Y-%m-%d")
    rows["code"] = rows.code.astype(str).str.zfill(6)
    if rows.duplicated(["signal_date", "code"]).any():
        raise ValueError("feature keys must be unique")
    rows = rows[rows.signal_date.lt(cutoff)].copy()
    for field in ("boards", "one_word_proxy", "one_word_streak", "turnover",
                  "bomb_count", "first_seal_minute", "last_seal_minute"):
        rows[field] = _numeric(rows[field])
    previous = _previous_session_map(calendar, cutoff)
    rows["previous_session"] = rows.signal_date.map(previous)
    lag = rows[["signal_date", "code", "boards", "one_word_proxy", "one_word_streak"]].rename(
        columns={"signal_date": "previous_session", "boards": "previous_boards",
                 "one_word_proxy": "previous_one_word_proxy",
                 "one_word_streak": "previous_one_word_streak"})
    joined = rows.merge(lag, on=["previous_session", "code"], how="left",
                        validate="many_to_one")
    joined = joined[joined.boards.between(2, 5, inclusive="both")
                    & joined.one_word_proxy.eq(0)].copy()
    first_open = (joined.previous_one_word_proxy.eq(1)
                  & joined.boards.eq(joined.previous_boards + 1))
    joined["path_group"] = np.select(
        [first_open & joined.previous_one_word_streak.ge(2), first_open],
        ["FIRST_OPEN_AFTER_2PLUS_ONE_PRICE", "FIRST_OPEN_AFTER_1_ONE_PRICE"],
        default="ORDINARY_TURNOVER_RELAY",
    )
    joined["first_open_after_one_price"] = first_open
    joined["seal_span_minutes"] = joined.last_seal_minute - joined.first_seal_minute
    return joined.drop(columns=["date"])


def build(features: pd.DataFrame, outcomes: pd.DataFrame, calendar: pd.DataFrame,
          cutoff_exclusive: str = "2025-12-26") -> tuple[pd.DataFrame, dict]:
    outcome_fields = {
        "signal_date", "code", "promotion_LABEL_ONLY",
        "next_session_touched_limit_LABEL_ONLY", "b2_net_close_mark_LABEL_ONLY",
        "b2_big_loss_mark_LABEL_ONLY",
    }
    if outcome_fields - set(outcomes):
        raise ValueError("first-open outcome fields missing")
    paths = identify(features, calendar, cutoff_exclusive)
    labels = outcomes[list(outcome_fields)].copy()
    labels["signal_date"] = pd.to_datetime(
        labels.signal_date, errors="coerce").dt.strftime("%Y-%m-%d")
    labels["code"] = labels.code.astype(str).str.zfill(6)
    if labels.duplicated(["signal_date", "code"]).any():
        raise ValueError("outcome keys must be unique")
    joined = paths.merge(labels, on=["signal_date", "code"], how="inner",
                         validate="one_to_one")
    if joined.empty:
        raise ValueError("no first-open evaluation rows")
    summaries = {group: _summary(joined[joined.path_group.eq(group)]) for group in GROUPS}
    comparisons = {
        "promotion_long_vs_one_day": _cluster_comparison(
            joined, "promotion_LABEL_ONLY", GROUPS[2], GROUPS[1], 1731),
        "touch_long_vs_one_day": _cluster_comparison(
            joined, "next_session_touched_limit_LABEL_ONLY", GROUPS[2], GROUPS[1], 1732),
        "b2_long_vs_one_day": _cluster_comparison(
            joined, "b2_net_close_mark_LABEL_ONLY", GROUPS[2], GROUPS[1], 1733),
        "b2_loss_long_vs_one_day": _cluster_comparison(
            joined, "b2_big_loss_mark_LABEL_ONLY", GROUPS[2], GROUPS[1], 1734),
    }
    joined["first_open_combined"] = np.where(
        joined.first_open_after_one_price, "FIRST_OPEN_COMBINED", joined.path_group)
    combined = joined.assign(path_group=joined.first_open_combined)
    comparisons.update({
        "promotion_first_open_vs_ordinary": _cluster_comparison(
            combined, "promotion_LABEL_ONLY", "FIRST_OPEN_COMBINED", GROUPS[0], 1735),
        "b2_first_open_vs_ordinary": _cluster_comparison(
            combined, "b2_net_close_mark_LABEL_ONLY", "FIRST_OPEN_COMBINED", GROUPS[0], 1736),
    })

    dates = sorted(joined.signal_date.unique())
    split_date = dates[int(len(dates) * .7)]
    discovery = joined[joined.signal_date.lt(split_date)].copy()
    validation = joined[joined.signal_date.ge(split_date)].copy()
    first_discovery = discovery[discovery.first_open_after_one_price & discovery.turnover.notna()]
    thresholds = {}
    for board, part in first_discovery.groupby(first_discovery.boards.astype(int)):
        if len(part) >= 30:
            q = part.turnover.quantile([1 / 3, 2 / 3])
            thresholds[str(board)] = {"low": float(q.iloc[0]), "high": float(q.iloc[1])}
    if len(first_discovery) >= 30:
        q = first_discovery.turnover.quantile([1 / 3, 2 / 3])
        thresholds["fallback"] = {"low": float(q.iloc[0]), "high": float(q.iloc[1])}

    tiers = []
    for row in validation.itertuples(index=False):
        if not row.first_open_after_one_price or pd.isna(row.turnover):
            tiers.append("NOT_FIRST_OPEN_OR_UNKNOWN")
            continue
        rule = thresholds.get(str(int(row.boards)), thresholds.get("fallback"))
        if not rule:
            tiers.append("UNKNOWN")
        elif row.turnover <= rule["low"]:
            tiers.append("LOW")
        elif row.turnover <= rule["high"]:
            tiers.append("MID")
        else:
            tiers.append("HIGH")
    validation["turnover_tier"] = tiers
    validation_tiers = {
        tier: _summary(validation[validation.turnover_tier.eq(tier)])
        for tier in ("LOW", "MID", "HIGH")
    }
    long_promotion = comparisons["promotion_long_vs_one_day"]
    long_b2 = comparisons["b2_long_vs_one_day"]
    delayed_price_discovery_factor_allowed = bool(
        summaries[GROUPS[2]]["observations"] >= 100
        and long_promotion["ci95"][1] < 0 and long_b2["ci95"][1] < 0
    )
    report = {
        "version": VERSION, "status": "DIAGNOSTIC_ONLY",
        "hypothesis_id": "H049", "cutoff_exclusive": cutoff_exclusive,
        "evaluation_rows": int(len(joined)), "evaluation_sessions": int(joined.signal_date.nunique()),
        "split_date": split_date,
        "definitions": {
            GROUPS[0]: "current 2-5 board is non-one-price without an immediately preceding one-price board",
            GROUPS[1]: "previous session was a one-price board with streak 1; current consecutive board is non-one-price",
            GROUPS[2]: "previous session ended a one-price streak of at least 2; current consecutive board is non-one-price",
        },
        "group_summaries": summaries,
        "date_clustered_comparisons": comparisons,
        "turnover_thresholds_fit_on_discovery_only": thresholds,
        "validation_first_open_turnover_tiers": validation_tiers,
        "governance": {
            "delayed_price_discovery_factor_allowed": delayed_price_discovery_factor_allowed,
            "first_open_is_automatic_healthy_divergence": False,
            "turnover_is_monotonic_positive_factor": False,
            "stock_reranking_allowed": False, "entry_authorized": False,
        },
        "interpretation": (
            "First-open relays can retain identity premium, but a longer prior one-price streak "
            "does not prove a safer trade. Turnover on the first open is price discovery and "
            "potential delayed supply; it must be combined with intraday absorption, theme feedback "
            "and executable B+1/B+2/B+3 evidence."
        ),
        "limitations": [
            "one-price status is inferred from daily OHLC, not order queue data",
            "historical security eligibility and theme identity are incomplete",
            "seal-time and reopen-path fields are frequently unavailable",
            "B+2 marks are conditional daily opportunity proxies, not realized trades",
            "the validation period is time-split but not a pristine never-inspected blind test",
        ],
    }
    return joined, report


def run(features_path, outcomes_path, calendar_path, output_dir,
        cutoff_exclusive="2025-12-26") -> dict:
    paths = list(map(Path, (features_path, outcomes_path, calendar_path)))
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable first-open audit")
    rows, report = build(
        pd.read_csv(paths[0], dtype={"code": str}, low_memory=False),
        pd.read_csv(paths[1], dtype={"code": str}, low_memory=False),
        pd.read_csv(paths[2]), cutoff_exclusive,
    )
    target.mkdir(parents=True)
    rows_path = target / "first_open_evaluation_rows.csv"
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
    parser.add_argument("--calendar", required=True)
    parser.add_argument("--cutoff-exclusive", default="2025-12-26")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.outcomes, args.calendar, args.output,
                         args.cutoff_exclusive), ensure_ascii=False, indent=2))
