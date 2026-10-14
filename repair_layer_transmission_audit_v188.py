"""Audit whether repair strength is transmitted across low, middle and height layers.

The layer score is known only after the signal close.  Thresholds are frozen
from the chronological development segment.  Next-session fields are labels,
never signal features or entry authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v188-repair-layer-transmission-audit-1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(frame: pd.DataFrame) -> dict:
    return {
        "sessions": int(len(frame)),
        "next_relapse_rate": None if frame.empty else float(frame.next_relapse.mean()),
        "next_advance_rate": None if frame.empty else float(frame.next_advance.mean()),
        "next_previous_limit_median_return": (
            None if frame.empty else float(frame.next_median.mean())),
        "next_failed_rate": (
            None if frame.empty else float(frame.next_failed_rate.mean())),
        "next_down_limit_count": (
            None if frame.empty else float(frame.next_down_limit_count.mean())),
        "mean_layer_score": (
            None if frame.empty else float(frame.repair_layer_score.mean())),
    }


def _difference(left: pd.DataFrame, right: pd.DataFrame, draws: int,
                seed: int, label: str) -> dict:
    fields = {
        "next_relapse_rate": "next_relapse",
        "next_advance_rate": "next_advance",
        "next_median": "next_median",
        "next_failed_rate": "next_failed_rate",
        "next_down_limit_count": "next_down_limit_count",
    }
    if min(len(left), len(right)) < 4:
        return {
            "comparison": label,
            "session_clustered": True,
            "adequate_for_interval": False,
            "minimum_group_sessions": int(min(len(left), len(right))),
            "metrics": {},
        }
    rng = np.random.default_rng(seed)
    simulations = {name: [] for name in fields}
    estimates = {
        name: float(left[column].mean() - right[column].mean())
        for name, column in fields.items()
    }
    for _ in range(draws):
        left_sample = left.iloc[rng.integers(0, len(left), len(left))]
        right_sample = right.iloc[rng.integers(0, len(right), len(right))]
        for name, column in fields.items():
            simulations[name].append(float(
                left_sample[column].mean() - right_sample[column].mean()))
    return {
        "comparison": label,
        "session_clustered": True,
        "adequate_for_interval": min(len(left), len(right)) >= 10,
        "minimum_group_sessions": int(min(len(left), len(right))),
        "metrics": {
            name: {
                "estimate": estimates[name],
                "ci95": [float(value) for value in np.quantile(
                    simulations[name], [.025, .975])],
            }
            for name in fields
        },
    }


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date", "data_status", "state_repair", "state_divergence",
        "state_retreat", "state_start", "state_warming",
        "state_acceleration", "state_consensus", "promotion_1to2",
        "promotion_mid", "space_feedback", "previous_limit_median_return",
        "failed_rate", "down_limit_count",
    }
    if required - set(frame):
        raise ValueError("repair layer market fields missing")
    data = frame.copy()
    data["date"] = pd.to_datetime(data.date, errors="coerce")
    if data.date.isna().any() or data.duplicated("date").any():
        raise ValueError("market dates must be valid and unique")
    data = data.sort_values("date").reset_index(drop=True)
    if not data.data_status.isin(["verified", "sparse"]).all():
        raise ValueError("unexpected market data status")
    numeric = [
        "state_repair", "state_divergence", "state_retreat", "state_start",
        "state_warming", "state_acceleration", "state_consensus",
        "promotion_1to2", "promotion_mid", "space_feedback",
        "previous_limit_median_return", "failed_rate", "down_limit_count",
    ]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    same_day_known = data[[
        "promotion_1to2", "promotion_mid", "space_feedback",
    ]].notna().all(axis=1)
    next_state_known = data[[
        "state_divergence", "state_retreat", "state_start", "state_warming",
        "state_acceleration", "state_consensus",
    ]].shift(-1).notna().all(axis=1)
    data["next_date"] = data.date.shift(-1)
    data["next_relapse"] = (
        data.state_divergence.shift(-1).eq(1)
        | data.state_retreat.shift(-1).eq(1))
    data["next_advance"] = (
        data.state_start.shift(-1).eq(1)
        | data.state_warming.shift(-1).eq(1)
        | data.state_acceleration.shift(-1).eq(1)
        | data.state_consensus.shift(-1).eq(1))
    data["next_median"] = data.previous_limit_median_return.shift(-1)
    data["next_failed_rate"] = data.failed_rate.shift(-1)
    data["next_down_limit_count"] = data.down_limit_count.shift(-1)
    outcome_known = data[[
        "next_median", "next_failed_rate", "next_down_limit_count",
    ]].notna().all(axis=1) & next_state_known
    return data[
        data.state_repair.eq(1)
        & data.data_status.eq("verified")
        & same_day_known & outcome_known
    ].copy()


def _apply_layers(frame: pd.DataFrame, first_threshold: float,
                  middle_threshold: float) -> pd.DataFrame:
    result = frame.copy()
    result["first_board_layer_confirmed"] = result.promotion_1to2.ge(
        first_threshold)
    result["middle_board_layer_confirmed"] = result.promotion_mid.ge(
        middle_threshold)
    result["height_layer_confirmed"] = result.space_feedback.gt(0)
    result["repair_layer_score"] = (
        result.first_board_layer_confirmed.astype(int)
        + result.middle_board_layer_confirmed.astype(int)
        + result.height_layer_confirmed.astype(int))
    result["repair_layer_class"] = np.where(
        result.repair_layer_score.eq(3), "SYNCHRONIZED_3",
        "PARTIAL_0_TO_2")
    return result


def build(frame: pd.DataFrame, split_fraction: float = .70,
          bootstrap_draws: int = 2000) -> tuple[dict, pd.DataFrame]:
    if not 0 < split_fraction < 1:
        raise ValueError("split_fraction must be between zero and one")
    if bootstrap_draws < 100:
        raise ValueError("bootstrap_draws must be at least 100")
    repairs = prepare(frame).reset_index(drop=True)
    if len(repairs) < 100:
        raise ValueError("at least 100 evaluable repair sessions required")
    split_index = max(1, min(len(repairs) - 1, int(len(repairs) * split_fraction)))
    raw_development = repairs.iloc[:split_index].copy()
    first_threshold = float(raw_development.promotion_1to2.median())
    middle_threshold = float(raw_development.promotion_mid.median())
    repairs = _apply_layers(repairs, first_threshold, middle_threshold)
    development = repairs.iloc[:split_index].copy()
    validation = repairs.iloc[split_index:].copy()

    summaries, comparisons = {}, {}
    for offset, (period, sample) in enumerate((
            ("development", development), ("validation", validation))):
        synchronized = sample[sample.repair_layer_class.eq("SYNCHRONIZED_3")]
        partial = sample[sample.repair_layer_class.eq("PARTIAL_0_TO_2")]
        middle = sample[sample.middle_board_layer_confirmed]
        no_middle = sample[~sample.middle_board_layer_confirmed]
        summaries[period] = {
            "ALL_REPAIR": _summary(sample),
            "SYNCHRONIZED_3": _summary(synchronized),
            "PARTIAL_0_TO_2": _summary(partial),
            "MIDDLE_CONFIRMED": _summary(middle),
            "MIDDLE_UNCONFIRMED": _summary(no_middle),
        }
        comparisons[period] = {
            "synchronized_minus_partial": _difference(
                synchronized, partial, bootstrap_draws, 18800 + offset,
                "SYNCHRONIZED_3_MINUS_PARTIAL_0_TO_2"),
            "middle_confirmed_minus_unconfirmed": _difference(
                middle, no_middle, bootstrap_draws, 18810 + offset,
                "MIDDLE_CONFIRMED_MINUS_UNCONFIRMED"),
        }

    def stable(comparison_name: str) -> bool:
        items = [comparisons[period][comparison_name]
                 for period in ("development", "validation")]
        if not all(item["adequate_for_interval"] for item in items):
            return False
        relapse = [item["metrics"].get("next_relapse_rate", {})
                   for item in items]
        return bool(all(metric and metric["estimate"] < 0
                        and metric["ci95"][1] < 0 for metric in relapse))

    latest_verified = frame[frame.data_status.eq("verified")].copy()
    latest_verified["date"] = pd.to_datetime(latest_verified.date, errors="coerce")
    latest = latest_verified.sort_values("date").iloc[-1]
    if float(latest.state_repair) == 1:
        first_confirmed = float(latest.promotion_1to2) >= first_threshold
        middle_confirmed = float(latest.promotion_mid) >= middle_threshold
        height_confirmed = float(latest.space_feedback) > 0
        latest_context = {
            "signal_date": latest.date.strftime("%Y-%m-%d"),
            "first_board_layer_confirmed": bool(first_confirmed),
            "middle_board_layer_confirmed": bool(middle_confirmed),
            "height_layer_confirmed": bool(height_confirmed),
            "repair_layer_score": int(sum((
                first_confirmed, middle_confirmed, height_confirmed))),
            "repair_layer_class": (
                "SYNCHRONIZED_3" if all((first_confirmed, middle_confirmed,
                                         height_confirmed))
                else "PARTIAL_0_TO_2"),
            "status": "HISTORICAL_SCENARIO_CONTEXT_ONLY",
        }
    else:
        latest_context = {
            "signal_date": latest.date.strftime("%Y-%m-%d"),
            "status": "NOT_APPLICABLE_CURRENT_STATE",
        }

    synchronized_stable = stable("synchronized_minus_partial")
    middle_stable = stable("middle_confirmed_minus_unconfirmed")
    report = {
        "version": VERSION,
        "status": "DIAGNOSTIC_ONLY",
        "data_contract": {
            "same_day_layers_available_at_repair_close": True,
            "thresholds_frozen_from_development_only": True,
            "next_session_fields_used_only_as_labels": True,
            "missing_layer_fields_filled_as_zero": False,
            "actual_stock_entries": 0,
            "complete_stock_trades": 0,
        },
        "layer_definition": {
            "first_board_layer": "promotion_1to2 >= development median",
            "middle_board_layer": "promotion_mid >= development median",
            "height_layer": "space_feedback > 0",
            "synchronized": "all three layers confirmed",
        },
        "frozen_development_thresholds": {
            "promotion_1to2_median": first_threshold,
            "promotion_mid_median": middle_threshold,
            "space_feedback_strictly_above": 0.0,
        },
        "scope": {
            "repair_sessions": int(len(repairs)),
            "first_date": repairs.iloc[0].date.strftime("%Y-%m-%d"),
            "last_date": repairs.iloc[-1].date.strftime("%Y-%m-%d"),
        },
        "split": {
            "method": "chronological_70_30_by_repair_session",
            "split_date": development.iloc[-1].date.strftime("%Y-%m-%d"),
            "development_sessions": int(len(development)),
            "validation_sessions": int(len(validation)),
            "validation_previously_inspected": True,
        },
        "summaries": summaries,
        "session_clustered_comparisons": comparisons,
        "synchronized_repair_lower_relapse_stable_across_time": synchronized_stable,
        "middle_confirmation_lower_relapse_stable_across_time": middle_stable,
        "latest_repair_layer_context": latest_context,
        "governance": {
            "historical_repair_layer_scenario_context_allowed": True,
            "calibrated_layer_followthrough_probability_allowed": False,
            "repair_layer_score_direct_emotion_gate_allowed": False,
            "middle_board_confirmation_positive_ranking_factor_allowed": False,
            "repair_layer_direct_stock_reranking_allowed": False,
            "repair_layer_position_size_increase_allowed": False,
            "repair_layer_entry_authorized": False,
            "requires_next_session_auction_middle_board_loss_and_theme_confirmation": True,
        },
        "interpretation": (
            "Synchronized low, middle and height repair is directionally associated with less "
            "next-session relapse, but the recent synchronized validation group is too small. "
            "Middle-board confirmation also fails stable time-separated separation. Layer "
            "structure may describe a scenario only and cannot rank stocks or authorize entry."),
        "limitations": [
            "repair states and layer fields are close-derived",
            "development medians are descriptive thresholds, not optimized trading cutoffs",
            "the recent synchronized validation group is below ten sessions",
            "specific-theme propagation and intraday order are unavailable historically",
            "validation dates were inspected and are not sealed blind",
        ],
    }
    observations = repairs[[
        "date", "promotion_1to2", "promotion_mid", "space_feedback",
        "first_board_layer_confirmed", "middle_board_layer_confirmed",
        "height_layer_confirmed", "repair_layer_score", "repair_layer_class",
        "next_date", "next_relapse", "next_advance", "next_median",
        "next_failed_rate", "next_down_limit_count",
    ]].copy()
    return report, observations


def run(input_path: str | Path, output_dir: str | Path,
        bootstrap_draws: int = 2000) -> dict:
    source, target = Path(input_path), Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable audit")
    report, observations = build(
        pd.read_csv(source), bootstrap_draws=bootstrap_draws)
    report["source_hashes"] = {
        str(source.resolve()): _sha256(source),
        str(Path(__file__).resolve()): _sha256(Path(__file__)),
    }
    target.mkdir(parents=True, exist_ok=False)
    observation_path = target / "repair_layer_observations.csv"
    observations.to_csv(observation_path, index=False)
    report["observations_sha256"] = _sha256(observation_path)
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
