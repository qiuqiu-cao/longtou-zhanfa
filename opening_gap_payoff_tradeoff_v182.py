"""Audit next-session opening-gap strength versus buyable continuation payoff.

The opening gap is known only after the official open and therefore belongs to
the opening-confirmation layer, never the prior-close model.  Outcomes remain
daily proxies: no-touch and one-price-limit rows are explicit negatives; a
touchable row with unknown B+2 mark is excluded.  No actual fill or complete
trade is claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v182-opening-gap-payoff-tradeoff-1"
BANDS = ["LE_0", "GT_0_LE_5", "GT_5_LE_7", "GT_7"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")
    values = series.astype(str).str.strip().str.lower()
    return values.map({
        "true": True, "1": True, "1.0": True,
        "false": False, "0": False, "0.0": False,
        "nan": pd.NA, "none": pd.NA, "": pd.NA,
    }).astype("boolean")


def _band(values: pd.Series) -> pd.Series:
    return pd.cut(values, [-np.inf, 0, 5, 7, np.inf], labels=BANDS,
                  right=True).astype(str)


def _summary(frame: pd.DataFrame) -> dict:
    result = {}
    for band in BANDS:
        rows = frame[frame.opening_gap_band.eq(band)]
        touchable = rows[rows.touchable]
        big_known = touchable[touchable.b1_big_loss.notna()]
        result[band] = {
            "rows": int(len(rows)),
            "sessions": int(rows.signal_date.nunique()),
            "combined_buyable_b2_positive_rate": (
                None if rows.empty else float(rows.target.mean())),
            "touchable_rate": None if rows.empty else float(rows.touchable.mean()),
            "one_price_unbuyable_rate": (
                None if rows.empty else float(rows.one_price.mean())),
            "conditional_b2_positive_rate": (
                None if touchable.empty else float(touchable.b2_positive.mean())),
            "conditional_b1_big_loss_rate": (
                None if big_known.empty else float(big_known.b1_big_loss.mean())),
            "conditional_mean_b2_net_close_mark": (
                None if touchable.empty else float(touchable.b2_mark.mean())),
        }
    return result


def _daily_aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["big_known"] = work.touchable & work.b1_big_loss.notna()
    work["big_loss_sum"] = np.where(
        work.big_known, work.b1_big_loss.fillna(False).astype(int), 0)
    work["b2_touch_sum"] = np.where(work.touchable, work.b2_mark, 0.0)
    work["b2_touch_count"] = work.touchable.astype(int)
    return work.groupby(["signal_date", "opening_gap_band"], as_index=False).agg(
        rows=("target", "size"), target_sum=("target", "sum"),
        one_price_sum=("one_price", "sum"), touch_sum=("touchable", "sum"),
        b2_positive_sum=("b2_positive", "sum"), big_known_sum=("big_known", "sum"),
        big_loss_sum=("big_loss_sum", "sum"), b2_touch_sum=("b2_touch_sum", "sum"),
        b2_touch_count=("b2_touch_count", "sum"),
    )


def _rates(counts: pd.Series) -> dict:
    def ratio(numerator: str, denominator: str):
        den = float(counts.get(denominator, 0))
        return None if den <= 0 else float(counts.get(numerator, 0) / den)
    return {
        "combined": ratio("target_sum", "rows"),
        "one_price": ratio("one_price_sum", "rows"),
        "conditional_b2_positive": ratio("b2_positive_sum", "touch_sum"),
        "conditional_b1_big_loss": ratio("big_loss_sum", "big_known_sum"),
        "conditional_mean_b2": ratio("b2_touch_sum", "b2_touch_count"),
    }


def _difference(a: dict, b: dict) -> dict:
    return {
        key: None if a[key] is None or b[key] is None else float(a[key] - b[key])
        for key in a
    }


def _clustered_difference(frame: pd.DataFrame, band_a: str, band_b: str,
                          draws: int, seed: int) -> dict:
    daily = _daily_aggregate(frame)
    dates = sorted(frame.signal_date.unique())
    columns = [
        "rows", "target_sum", "one_price_sum", "touch_sum", "b2_positive_sum",
        "big_known_sum", "big_loss_sum", "b2_touch_sum", "b2_touch_count",
    ]
    matrices = {}
    for band in (band_a, band_b):
        indexed = daily[daily.opening_gap_band.eq(band)].set_index("signal_date")
        matrices[band] = indexed.reindex(dates)[columns].fillna(0).to_numpy(float)
    direct = {}
    for band in (band_a, band_b):
        direct[band] = _rates(pd.Series(matrices[band].sum(axis=0), index=columns))
    estimate = _difference(direct[band_a], direct[band_b])
    rng = np.random.default_rng(seed)
    simulated = {key: [] for key in estimate}
    for _ in range(draws):
        indices = rng.integers(0, len(dates), len(dates))
        rates_a = _rates(pd.Series(matrices[band_a][indices].sum(axis=0), index=columns))
        rates_b = _rates(pd.Series(matrices[band_b][indices].sum(axis=0), index=columns))
        diff = _difference(rates_a, rates_b)
        for key, value in diff.items():
            if value is not None:
                simulated[key].append(value)
    return {
        "comparison": f"{band_a}_MINUS_{band_b}",
        "date_clustered": True,
        "metrics": {
            key: {
                "estimate": value,
                "ci95": [None, None] if not simulated[key] else [
                    float(np.quantile(simulated[key], .025)),
                    float(np.quantile(simulated[key], .975)),
                ],
            }
            for key, value in estimate.items()
        },
    }


def build(features: pd.DataFrame, outcomes: pd.DataFrame, entry_bars: pd.DataFrame,
          cutoff_exclusive: str = "2026-01-01", split_fraction: float = .70,
          bootstrap_draws: int = 2000) -> tuple[pd.DataFrame, dict]:
    feature_required = {"code", "signal_date", "boards", "board_group"}
    outcome_required = {
        "code", "signal_date", "entry_date_LABEL_ONLY",
        "next_session_touched_limit_LABEL_ONLY",
        "next_session_one_price_limit_LABEL_ONLY", "b1_big_loss_mark_LABEL_ONLY",
        "b2_net_close_mark_LABEL_ONLY", "actual_entry_verified",
        "complete_trade_profit_LABEL_ONLY",
    }
    bar_required = {"code", "date", "open", "pre_close"}
    if feature_required - set(features):
        raise ValueError("feature fields missing")
    if outcome_required - set(outcomes):
        raise ValueError("outcome fields missing")
    if bar_required - set(entry_bars):
        raise ValueError("entry-bar fields missing")
    left, right, bars = features.copy(), outcomes.copy(), entry_bars.copy()
    for frame, date_column in ((left, "signal_date"), (right, "signal_date"),
                               (bars, "date")):
        frame["code"] = frame.code.astype(str).str.zfill(6)
        frame[date_column] = pd.to_datetime(
            frame[date_column], errors="coerce").dt.strftime("%Y-%m-%d")
        if frame[["code", date_column]].isna().any().any():
            raise ValueError("invalid candidate or bar key")
        if frame.duplicated(["code", date_column]).any():
            raise ValueError("candidate or bar keys must be unique")
    merged = left.merge(right, on=["code", "signal_date"], how="inner",
                        validate="one_to_one")
    merged["entry_date_LABEL_ONLY"] = pd.to_datetime(
        merged.entry_date_LABEL_ONLY, errors="coerce").dt.strftime("%Y-%m-%d")
    merged = merged.merge(
        bars, left_on=["code", "entry_date_LABEL_ONLY"], right_on=["code", "date"],
        how="left", validate="one_to_one")
    merged["boards"] = pd.to_numeric(merged.boards, errors="coerce")
    merged["open"] = pd.to_numeric(merged.open, errors="coerce")
    merged["pre_close"] = pd.to_numeric(merged.pre_close, errors="coerce")
    merged["entry_open_gap_pct"] = (merged.open / merged.pre_close - 1) * 100
    eligible_pre_open = (
        merged.boards.between(1, 5)
        & merged.signal_date.lt(pd.Timestamp(cutoff_exclusive).strftime("%Y-%m-%d"))
    )
    opening_gap_missing = int((
        eligible_pre_open
        & (merged.entry_open_gap_pct.isna() | ~merged.pre_close.gt(0))
    ).sum())
    merged = merged[
        eligible_pre_open
        & merged.entry_open_gap_pct.notna()
        & merged.pre_close.gt(0)
    ].copy()
    if merged.empty:
        raise ValueError("no opening-gap observations")
    if _boolean(merged.actual_entry_verified).fillna(False).any():
        raise ValueError("opening-gap proxy unexpectedly claims verified entries")
    if pd.to_numeric(merged.complete_trade_profit_LABEL_ONLY,
                     errors="coerce").notna().any():
        raise ValueError("opening-gap proxy unexpectedly contains complete trades")
    touched = _boolean(merged.next_session_touched_limit_LABEL_ONLY)
    one_price = _boolean(merged.next_session_one_price_limit_LABEL_ONLY)
    b2 = pd.to_numeric(merged.b2_net_close_mark_LABEL_ONLY, errors="coerce")
    touchable = touched.eq(True) & one_price.eq(False)
    unknown_b2 = touchable & b2.isna()
    excluded_unknown_b2 = int(unknown_b2.sum())
    merged = merged[~unknown_b2].copy()
    touched = touched.loc[merged.index]
    one_price = one_price.loc[merged.index]
    b2 = b2.loc[merged.index]
    merged["touchable"] = touchable.loc[merged.index].astype(bool)
    merged["one_price"] = one_price.eq(True).astype(bool)
    merged["b2_mark"] = b2
    merged["b2_positive"] = (merged.touchable & b2.gt(0)).astype(bool)
    merged["target"] = merged.b2_positive
    merged["b1_big_loss"] = _boolean(
        merged.b1_big_loss_mark_LABEL_ONLY).astype("boolean")
    merged["opening_gap_band"] = _band(merged.entry_open_gap_pct)
    dates = sorted(merged.signal_date.unique())
    if len(dates) < 200:
        raise ValueError("insufficient opening-gap sessions")
    split_index = max(1, min(len(dates) - 1, int(len(dates) * split_fraction)))
    split_date = dates[split_index]
    development = merged[merged.signal_date.lt(split_date)].copy()
    validation = merged[merged.signal_date.ge(split_date)].copy()
    groups = {
        "ALL": pd.Series(True, index=merged.index),
        "FIRST_BOARD": merged.board_group.eq("FIRST_BOARD"),
        "CONSECUTIVE_BOARD": merged.board_group.eq("CONSECUTIVE_BOARD"),
    }
    summaries, comparisons = {}, {}
    for group, mask in groups.items():
        summaries[group] = {
            "development": _summary(development[mask.loc[development.index]]),
            "validation": _summary(validation[mask.loc[validation.index]]),
        }
        comparisons[group] = {
            "development_5_7_vs_gt_7": _clustered_difference(
                development[mask.loc[development.index]], "GT_5_LE_7", "GT_7",
                bootstrap_draws, 18201),
            "validation_5_7_vs_gt_7": _clustered_difference(
                validation[mask.loc[validation.index]], "GT_5_LE_7", "GT_7",
                bootstrap_draws, 18202),
            "validation_0_5_vs_5_7": _clustered_difference(
                validation[mask.loc[validation.index]], "GT_0_LE_5", "GT_5_LE_7",
                bootstrap_draws, 18203),
        }
    all_dev = comparisons["ALL"]["development_5_7_vs_gt_7"]["metrics"]
    all_val = comparisons["ALL"]["validation_5_7_vs_gt_7"]["metrics"]
    report = {
        "version": VERSION,
        "status": "DIAGNOSTIC_ONLY",
        "feature_contract": {
            "entry_open_gap_available_not_before": (
                "after_opening_call_auction_match_approximately_09:25"),
            "continuous_auction_begins_at": "09:30",
            "prior_close_model_use_allowed": False,
            "auction_09_25_equivalence_claim_allowed": False,
            "data_status": "INFERRED_FROM_THIRD_PARTY_DAILY_OHLC",
        },
        "target_contract": {
            "description": (
                "next-session touched limit AND not one-price AND B+2 net close mark > 0"),
            "no_touch_rows_retained_as_negative": True,
            "next_session_one_price_rows_retained_as_negative": True,
            "touchable_rows_with_unknown_b2_excluded": excluded_unknown_b2,
            "actual_fills": 0,
            "complete_trades": 0,
        },
        "scope": {
            "rows": int(len(merged)), "sessions": int(merged.signal_date.nunique()),
            "first_board_rows": int(merged.board_group.eq("FIRST_BOARD").sum()),
            "relay_rows": int(merged.board_group.eq("CONSECUTIVE_BOARD").sum()),
            "opening_gap_missing_rows_excluded": opening_gap_missing,
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
        "band_definitions": {
            "LE_0": "opening gap <= 0%",
            "GT_0_LE_5": "0% < opening gap <= 5%",
            "GT_5_LE_7": "5% < opening gap <= 7%",
            "GT_7": "opening gap > 7%",
        },
        "summaries": summaries,
        "date_clustered_comparisons": comparisons,
        "directional_replication": {
            "five_to_seven_combined_better_than_above_seven_in_both_periods": bool(
                all_dev["combined"]["estimate"] > 0
                and all_val["combined"]["estimate"] > 0),
            "five_to_seven_big_loss_lower_than_above_seven_in_both_periods": bool(
                all_dev["conditional_b1_big_loss"]["estimate"] < 0
                and all_val["conditional_b1_big_loss"]["estimate"] < 0),
            "five_to_seven_vs_above_seven_both_ci_confirmed": bool(
                all_dev["combined"]["ci95"][0] is not None
                and all_dev["combined"]["ci95"][0] > 0
                and all_val["combined"]["ci95"][0] > 0
                and all_dev["conditional_b1_big_loss"]["ci95"][1] < 0
                and all_val["conditional_b1_big_loss"]["ci95"][1] < 0),
        },
        "governance": {
            "opening_gap_direct_entry_allowed": False,
            "opening_gap_direct_stock_reranking_allowed": False,
            "above_seven_default_risk_flag_allowed": True,
            "zero_to_five_automatic_buy_allowed": False,
            "five_to_seven_automatic_buy_allowed": False,
            "requires_intraday_absorption_reseal_and_theme_confirmation": True,
        },
        "interpretation": (
            "Opening strength is a trade-off, not a monotonic buy factor. The 5-7% band "
            "has higher touch identity, while >7% contains more one-price/unbuyable rows and "
            "worse conditional loss. Neither 0-5% nor 5-7% authorizes entry without actual "
            "absorption, reseal, theme support and a valid price ceiling."),
        "limitations": [
            "official open is known only after the opening call-auction match and cannot be used in the prior-close model",
            "daily OHLC cannot prove auction match amount, queue, fill, VWAP path, or reseal",
            "historical candidate universe is not independently full-market complete",
            "daily limit-touch entry and close marks are proxies rather than complete trades",
            "validation dates were inspected and are not sealed blind",
        ],
    }
    output_columns = [
        "code", "signal_date", "entry_date_LABEL_ONLY", "boards", "board_group",
        "entry_open_gap_pct", "opening_gap_band", "touchable", "one_price",
        "target", "b1_big_loss", "b2_mark",
    ]
    return merged[output_columns].sort_values(["signal_date", "code"]), report


def load_entry_bars(paths: list[Path], keys: set[tuple[str, str]]) -> pd.DataFrame:
    collected, seen = [], set()
    for path in paths:
        for chunk in pd.read_csv(
                path, usecols=["date", "code", "open", "pre_close"],
                dtype={"code": str}, chunksize=250_000):
            chunk["code"] = chunk.code.astype(str).str.zfill(6)
            keep = [
                (code, str(date)) in keys and (code, str(date)) not in seen
                for code, date in zip(chunk.code, chunk.date)
            ]
            selected = chunk.loc[keep].copy()
            seen.update(zip(selected.code, selected.date.astype(str)))
            if not selected.empty:
                collected.append(selected)
    if not collected:
        return pd.DataFrame(columns=["date", "code", "open", "pre_close"])
    return pd.concat(collected, ignore_index=True)


def run(features_path: str | Path, outcomes_path: str | Path,
        bars_paths: list[str | Path], output_dir: str | Path,
        cutoff_exclusive: str = "2026-01-01", bootstrap_draws: int = 2000) -> dict:
    features_path, outcomes_path = Path(features_path), Path(outcomes_path)
    bar_paths = [Path(path) for path in bars_paths]
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable V182 output")
    features = pd.read_csv(features_path, dtype={"code": str})
    outcomes = pd.read_csv(outcomes_path, dtype={"code": str})
    keys = set(zip(outcomes.code.astype(str).str.zfill(6),
                   outcomes.entry_date_LABEL_ONLY.astype(str)))
    bars = load_entry_bars(bar_paths, keys)
    rows, report = build(features, outcomes, bars, cutoff_exclusive,
                         bootstrap_draws=bootstrap_draws)
    target.mkdir(parents=True, exist_ok=False)
    predictions = target / "opening_gap_observations.csv"
    rows.to_csv(predictions, index=False, encoding="utf-8-sig")
    report["source_hashes"] = {
        str(path.resolve()): _sha256(path)
        for path in [features_path, outcomes_path, *bar_paths]
    }
    report["observations_sha256"] = _sha256(predictions)
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--outcomes", required=True)
    parser.add_argument("--bars", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cutoff-exclusive", default="2026-01-01")
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.outcomes, args.bars, args.output,
                         args.cutoff_exclusive, args.bootstrap_draws),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
