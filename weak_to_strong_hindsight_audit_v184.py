"""Audit the hindsight trap in low-open 'weak-to-strong' narratives.

Conditioning a low-open candidate on the fact that it later touched limit uses
future information unavailable at 09:30.  This audit keeps the full low-open
denominator beside the later-touch conditional denominator and forbids the
conditional result from becoming an opening-time buy rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from opening_gap_payoff_tradeoff_v182 import _clustered_difference, _summary


VERSION = "v184-weak-to-strong-hindsight-audit-1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.astype(str).str.strip().str.lower().map({
        "true": True, "1": True, "1.0": True,
        "false": False, "0": False, "0.0": False,
    })


def _inflation(summary: dict) -> dict:
    combined = summary.get("combined_buyable_b2_positive_rate")
    conditional = summary.get("conditional_b2_positive_rate")
    touchable = summary.get("touchable_rate")
    return {
        "full_opening_denominator_success_rate": combined,
        "later_touch_conditioned_b2_positive_rate": conditional,
        "touchable_rate": touchable,
        "conditioning_inflation_multiple": (
            None if combined in (None, 0) or conditional is None
            else float(conditional / combined)),
        "conditioning_inflation_percentage_points": (
            None if combined is None or conditional is None
            else float(conditional - combined)),
        "later_touch_is_future_information_at_09_30": True,
    }


def build(observations: pd.DataFrame, v182_report: dict,
          split_fraction: float = .70, bootstrap_draws: int = 2000
          ) -> tuple[pd.DataFrame, dict]:
    required = {
        "code", "signal_date", "boards", "board_group", "opening_gap_band",
        "touchable", "one_price", "target", "b1_big_loss", "b2_mark",
    }
    if required - set(observations):
        raise ValueError("V182 observation fields missing")
    if (v182_report.get("version") != "v182-opening-gap-payoff-tradeoff-1"
            or v182_report.get("status") != "DIAGNOSTIC_ONLY"
            or v182_report.get("target_contract", {}).get(
                "no_touch_rows_retained_as_negative") is not True
            or v182_report.get("target_contract", {}).get(
                "next_session_one_price_rows_retained_as_negative") is not True
            or v182_report.get("target_contract", {}).get("actual_fills") != 0
            or v182_report.get("target_contract", {}).get("complete_trades") != 0):
        raise ValueError("V182 source contract is invalid")
    frame = observations.copy()
    frame["code"] = frame.code.astype(str).str.zfill(6)
    frame["signal_date"] = pd.to_datetime(
        frame.signal_date, errors="coerce").dt.strftime("%Y-%m-%d")
    if frame[["code", "signal_date"]].isna().any().any():
        raise ValueError("invalid candidate key")
    if frame.duplicated(["code", "signal_date"]).any():
        raise ValueError("candidate keys must be unique")
    for column in ("touchable", "one_price", "target"):
        frame[column] = _bool(frame[column])
        if frame[column].isna().any():
            raise ValueError("invalid V182 boolean observation")
        frame[column] = frame[column].astype(bool)
    frame["b1_big_loss"] = _bool(frame.b1_big_loss).astype("boolean")
    frame["b2_mark"] = pd.to_numeric(frame.b2_mark, errors="coerce")
    frame["b2_positive"] = frame.target
    if not set(frame.opening_gap_band.unique()).issubset(
            {"LE_0", "GT_0_LE_5", "GT_5_LE_7", "GT_7"}):
        raise ValueError("unexpected opening-gap band")
    dates = sorted(frame.signal_date.unique())
    if len(dates) < 200:
        raise ValueError("insufficient weak-to-strong sessions")
    split_index = max(1, min(len(dates) - 1, int(len(dates) * split_fraction)))
    split_date = dates[split_index]
    development = frame[frame.signal_date.lt(split_date)].copy()
    validation = frame[frame.signal_date.ge(split_date)].copy()
    groups = {
        "ALL": pd.Series(True, index=frame.index),
        "FIRST_BOARD": frame.board_group.eq("FIRST_BOARD"),
        "CONSECUTIVE_BOARD": frame.board_group.eq("CONSECUTIVE_BOARD"),
    }
    summaries, comparisons, inflation = {}, {}, {}
    for group, mask in groups.items():
        dev = development[mask.loc[development.index]]
        val = validation[mask.loc[validation.index]]
        summaries[group] = {
            "development": _summary(dev), "validation": _summary(val),
        }
        comparisons[group] = {
            "development_low_vs_0_5": _clustered_difference(
                dev, "LE_0", "GT_0_LE_5", bootstrap_draws, 18401),
            "validation_low_vs_0_5": _clustered_difference(
                val, "LE_0", "GT_0_LE_5", bootstrap_draws, 18402),
        }
        inflation[group] = {
            "development_low_open": _inflation(summaries[group]["development"]["LE_0"]),
            "validation_low_open": _inflation(summaries[group]["validation"]["LE_0"]),
            "development_0_5_open": _inflation(
                summaries[group]["development"]["GT_0_LE_5"]),
            "validation_0_5_open": _inflation(
                summaries[group]["validation"]["GT_0_LE_5"]),
        }
    all_dev = comparisons["ALL"]["development_low_vs_0_5"]["metrics"]
    all_val = comparisons["ALL"]["validation_low_vs_0_5"]["metrics"]
    conditional_payoff_edge_supported = bool(
        all_dev["conditional_b2_positive"]["estimate"] > 0
        and all_val["conditional_b2_positive"]["estimate"] > 0
        and all_dev["conditional_b2_positive"]["ci95"][0] > 0
        and all_val["conditional_b2_positive"]["ci95"][0] > 0
        and all_dev["conditional_b1_big_loss"]["estimate"] < 0
        and all_val["conditional_b1_big_loss"]["estimate"] < 0
        and all_dev["conditional_b1_big_loss"]["ci95"][1] < 0
        and all_val["conditional_b1_big_loss"]["ci95"][1] < 0)
    report = {
        "version": VERSION,
        "status": "DIAGNOSTIC_ONLY",
        "source_contract": {
            "v182_version": v182_report["version"],
            "no_touch_rows_retained_as_negative": True,
            "next_session_one_price_rows_retained_as_negative": True,
            "actual_fills": 0, "complete_trades": 0,
        },
        "scope": {
            "rows": int(len(frame)), "sessions": int(frame.signal_date.nunique()),
            "low_open_rows": int(frame.opening_gap_band.eq("LE_0").sum()),
            "zero_to_five_rows": int(frame.opening_gap_band.eq("GT_0_LE_5").sum()),
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
        "denominator_contract": {
            "opening_time_denominator": "all candidates in the opening-gap band",
            "later_touch_conditional_denominator": (
                "future subset that later touched limit and was not one-price"),
            "later_touch_available_at_09_30": False,
            "conditional_result_may_not_be_presented_as_opening_win_rate": True,
        },
        "summaries": summaries,
        "date_clustered_low_vs_zero_to_five": comparisons,
        "hindsight_inflation": inflation,
        "low_open_conditional_payoff_edge_supported": conditional_payoff_edge_supported,
        "governance": {
            "later_touch_may_be_used_as_09_30_feature": False,
            "low_open_automatic_weak_to_strong_buy_allowed": False,
            "low_open_automatic_reject_allowed": False,
            "requires_live_reclaim_vwap_volume_and_theme_confirmation": True,
            "direct_stock_reranking_allowed": False,
            "entry_authorized": False,
        },
        "interpretation": (
            "Low/flat opens have a much smaller full-denominator chance of reaching a buyable "
            "B+2-positive path. The apparently attractive payoff among names that later touch "
            "limit is conditioned on future information and cannot be known at 09:30. A low open "
            "is neither an automatic buy nor an automatic rejection; it remains watch-only until "
            "a real reclaim, volume, VWAP and theme confirmation occurs."),
        "limitations": [
            "daily OHLC cannot identify reclaim time, VWAP, pullback volume, or actual fill",
            "candidate universe is not independently full-market complete",
            "validation dates were inspected and are not sealed blind",
        ],
    }
    columns = [
        "code", "signal_date", "boards", "board_group", "opening_gap_band",
        "touchable", "one_price", "target", "b1_big_loss", "b2_mark",
    ]
    return frame[columns].sort_values(["signal_date", "code"]), report


def run(observations_path: str | Path, v182_report_path: str | Path,
        output_dir: str | Path, bootstrap_draws: int = 2000) -> dict:
    observations_path, v182_report_path = Path(observations_path), Path(v182_report_path)
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable V184 output")
    rows, report = build(
        pd.read_csv(observations_path, dtype={"code": str}),
        json.loads(v182_report_path.read_text(encoding="utf-8")),
        bootstrap_draws=bootstrap_draws)
    target.mkdir(parents=True, exist_ok=False)
    output = target / "weak_to_strong_observations.csv"
    rows.to_csv(output, index=False, encoding="utf-8-sig")
    report["source_hashes"] = {
        str(path.resolve()): _sha256(path)
        for path in (observations_path, v182_report_path)
    }
    report["observations_sha256"] = _sha256(output)
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--v182-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    print(json.dumps(run(args.observations, args.v182_report, args.output,
                         args.bootstrap_draws), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
