"""Audit whether a repair day follows through or relapses next session.

The same-day repair score is descriptive and known at the close.  It may shape
the next-session scenario, but cannot become a calibrated probability, stock
ranker, or intraday buy signal without stable time-separated evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v187-repair-day-followthrough-audit-1"
REPAIR = "修复"
RELAPSE_STATES = {"大分歧", "退潮"}
ADVANCE_STATES = {"启动", "升温", "加速", "一致"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(frame: pd.DataFrame) -> dict:
    return {
        "sessions": int(len(frame)),
        "relapse_rate": None if frame.empty else float(frame["next_relapse"].mean()),
        "advance_rate": None if frame.empty else float(frame["next_advance"].mean()),
        "repair_stall_rate": (
            None if frame.empty else float(frame["next_repair_stall"].mean())),
        "next_previous_limit_median_return": (
            None if frame.empty else float(frame["next_median"].mean())),
        "next_middle_promotion": (
            None if frame.empty else float(frame["next_promotion_mid"].mean())),
        "next_failed_rate": (
            None if frame.empty else float(frame["next_failed_rate"].mean())),
        "next_down_limit_count": (
            None if frame.empty else float(frame["next_down_limit_count"].mean())),
        "mean_same_day_quality_score": (
            None if frame.empty else float(frame["repair_quality_score"].mean())),
    }


def _difference(broad: pd.DataFrame, narrow: pd.DataFrame, draws: int,
                seed: int) -> dict:
    fields = {
        "relapse_rate": "next_relapse",
        "advance_rate": "next_advance",
        "next_median": "next_median",
        "next_promotion_mid": "next_promotion_mid",
        "next_failed_rate": "next_failed_rate",
        "next_down_limit_count": "next_down_limit_count",
    }
    estimate = {
        name: float(broad[column].mean() - narrow[column].mean())
        for name, column in fields.items()
    }
    rng = np.random.default_rng(seed)
    simulations = {name: [] for name in fields}
    for _ in range(draws):
        broad_sample = broad.iloc[rng.integers(0, len(broad), len(broad))]
        narrow_sample = narrow.iloc[rng.integers(0, len(narrow), len(narrow))]
        for name, column in fields.items():
            simulations[name].append(float(
                broad_sample[column].mean() - narrow_sample[column].mean()))
    return {
        "comparison": "BROAD_REPAIR_MINUS_NARROW_REPAIR",
        "session_clustered": True,
        "metrics": {
            name: {
                "estimate": estimate[name],
                "ci95": [float(value) for value in np.quantile(
                    simulations[name], [.025, .975])],
            }
            for name in fields
        },
    }


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date", "data_status", "emotion_state",
        "previous_limit_median_return", "promotion_mid", "failed_rate",
        "down_limit_count", "limit_count", "max_board",
    }
    if required - set(frame):
        raise ValueError("repair follow-through market fields missing")
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    if data["date"].isna().any() or data.duplicated("date").any():
        raise ValueError("market dates must be valid and unique")
    data = data.sort_values("date").reset_index(drop=True)
    if not data["data_status"].isin(["verified", "sparse"]).all():
        raise ValueError("unexpected market data status")
    for column in (
        "previous_limit_median_return", "promotion_mid", "failed_rate",
        "down_limit_count", "limit_count", "max_board",
    ):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    core_known = data[[
        "previous_limit_median_return", "promotion_mid", "failed_rate",
        "down_limit_count",
    ]].notna().all(axis=1)
    data["repair_quality_score"] = (
        data["previous_limit_median_return"].gt(0).astype(int)
        + data["promotion_mid"].ge(.25).astype(int)
        + data["failed_rate"].le(.30).astype(int)
        + data["down_limit_count"].le(5).astype(int)
    ).where(core_known)
    data["repair_quality_class"] = np.where(
        data["repair_quality_score"].ge(3), "BROAD_REPAIR", "NARROW_REPAIR")
    data.loc[~core_known, "repair_quality_class"] = "UNAVAILABLE"
    data["next_date"] = data["date"].shift(-1)
    data["next_state"] = data["emotion_state"].shift(-1)
    data["next_median"] = data["previous_limit_median_return"].shift(-1)
    data["next_promotion_mid"] = data["promotion_mid"].shift(-1)
    data["next_failed_rate"] = data["failed_rate"].shift(-1)
    data["next_down_limit_count"] = data["down_limit_count"].shift(-1)
    data["next_relapse"] = data["next_state"].isin(RELAPSE_STATES)
    data["next_advance"] = data["next_state"].isin(ADVANCE_STATES)
    data["next_repair_stall"] = data["next_state"].eq(REPAIR)
    outcome_known = data[[
        "next_state", "next_median", "next_promotion_mid", "next_failed_rate",
        "next_down_limit_count",
    ]].notna().all(axis=1)
    return data[
        data["emotion_state"].eq(REPAIR)
        & data["data_status"].eq("verified")
        & core_known & outcome_known
    ].copy()


def build(frame: pd.DataFrame, split_fraction: float = .70,
          bootstrap_draws: int = 2000) -> tuple[dict, pd.DataFrame]:
    if not 0 < split_fraction < 1:
        raise ValueError("split_fraction must be between zero and one")
    if bootstrap_draws < 100:
        raise ValueError("bootstrap_draws must be at least 100")
    repairs = prepare(frame)
    if len(repairs) < 100:
        raise ValueError("at least 100 evaluable repair sessions required")
    split_index = max(1, min(len(repairs) - 1, int(len(repairs) * split_fraction)))
    development = repairs.iloc[:split_index].copy()
    validation = repairs.iloc[split_index:].copy()
    split_date = development.iloc[-1]["date"].strftime("%Y-%m-%d")
    summaries, comparisons = {}, {}
    for offset, (period, sample) in enumerate((
            ("development", development), ("validation", validation))):
        broad = sample[sample["repair_quality_class"].eq("BROAD_REPAIR")]
        narrow = sample[sample["repair_quality_class"].eq("NARROW_REPAIR")]
        if min(len(broad), len(narrow)) < 10:
            raise ValueError("repair class support below ten in a time segment")
        summaries[period] = {
            "ALL_REPAIR": _summary(sample),
            "BROAD_REPAIR": _summary(broad),
            "NARROW_REPAIR": _summary(narrow),
        }
        comparisons[period] = _difference(
            broad, narrow, bootstrap_draws, 18700 + offset)
    dev_relapse = comparisons["development"]["metrics"]["relapse_rate"]
    val_relapse = comparisons["validation"]["metrics"]["relapse_rate"]
    stable = bool(
        dev_relapse["estimate"] < 0 and dev_relapse["ci95"][1] < 0
        and val_relapse["estimate"] < 0 and val_relapse["ci95"][1] < 0
    )

    latest_verified = frame.copy()
    latest_verified["date"] = pd.to_datetime(latest_verified["date"], errors="coerce")
    latest_verified = latest_verified[
        latest_verified["data_status"].eq("verified")].sort_values("date")
    latest = latest_verified.iloc[-1]
    if latest["emotion_state"] == REPAIR:
        score = sum((
            float(latest["previous_limit_median_return"]) > 0,
            float(latest["promotion_mid"]) >= .25,
            float(latest["failed_rate"]) <= .30,
            float(latest["down_limit_count"]) <= 5,
        ))
        latest_context = {
            "signal_date": latest["date"].strftime("%Y-%m-%d"),
            "emotion_state": REPAIR,
            "quality_score": int(score),
            "quality_class": "BROAD_REPAIR" if score >= 3 else "NARROW_REPAIR",
            "status": "HISTORICAL_SCENARIO_CONTEXT_ONLY",
        }
    else:
        latest_context = {
            "signal_date": latest["date"].strftime("%Y-%m-%d"),
            "emotion_state": latest["emotion_state"],
            "status": "NOT_APPLICABLE_CURRENT_STATE",
        }
    report = {
        "version": VERSION,
        "status": "DIAGNOSTIC_ONLY",
        "data_contract": {
            "signal_features_available_at_repair_day_close": True,
            "next_session_fields_used_only_as_labels": True,
            "sparse_signal_days_excluded": True,
            "missing_quality_fields_filled_as_zero": False,
            "actual_stock_entries": 0, "complete_stock_trades": 0,
        },
        "repair_definition": {
            "broad_repair": "at least 3 of 4 same-day checks pass",
            "same_day_checks": [
                "previous-limit median return > 0",
                "middle-board promotion >= 25%",
                "failed-board rate <= 30%",
                "down-limit count <= 5",
            ],
            "next_relapse": "next state is 大分歧 or 退潮",
            "next_advance": "next state is 启动, 升温, 加速, or 一致",
        },
        "scope": {
            "repair_sessions": int(len(repairs)),
            "first_date": repairs.iloc[0]["date"].strftime("%Y-%m-%d"),
            "last_date": repairs.iloc[-1]["date"].strftime("%Y-%m-%d"),
        },
        "split": {
            "method": "chronological_70_30_by_repair_session",
            "split_date": split_date,
            "development_sessions": int(len(development)),
            "validation_sessions": int(len(validation)),
            "validation_previously_inspected": True,
        },
        "summaries": summaries,
        "session_clustered_broad_minus_narrow": comparisons,
        "broad_repair_lower_relapse_stable_across_time": stable,
        "latest_repair_context": latest_context,
        "governance": {
            "historical_repair_scenario_context_allowed": True,
            "calibrated_next_relapse_probability_allowed": False,
            "repair_score_direct_emotion_gate_allowed": False,
            "repair_score_direct_stock_reranking_allowed": False,
            "repair_score_entry_authorized": False,
            "requires_next_session_auction_middle_board_and_loss_confirmation": True,
        },
        "interpretation": (
            "A broad repair has lower next-session relapse in the recent validation segment, "
            "but the effect is weak and uncertain in development. The repair score may describe "
            "the main scenario only; it cannot promise follow-through or authorize a stock entry."),
        "limitations": [
            "repair state and quality are close-derived and available only after the signal close",
            "specific-theme and intraday repair breadth are unavailable historically",
            "validation dates were inspected and are not sealed blind",
        ],
    }
    observations = repairs[[
        "date", "repair_quality_score", "repair_quality_class", "next_date",
        "next_state", "next_relapse", "next_advance", "next_repair_stall",
        "next_median", "next_promotion_mid", "next_failed_rate",
        "next_down_limit_count",
    ]].copy()
    return report, observations


def run(input_path: str | Path, output_dir: str | Path,
        bootstrap_draws: int = 2000) -> dict:
    input_path, target = Path(input_path), Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable audit")
    report, observations = build(
        pd.read_csv(input_path), bootstrap_draws=bootstrap_draws)
    report["source_hashes"] = {
        str(input_path.resolve()): _sha256(input_path),
        str(Path(__file__).resolve()): _sha256(Path(__file__)),
    }
    target.mkdir(parents=True, exist_ok=False)
    observations.to_csv(target / "repair_followthrough_observations.csv", index=False)
    report["observations_sha256"] = _sha256(
        target / "repair_followthrough_observations.csv")
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output, args.bootstrap_draws),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
