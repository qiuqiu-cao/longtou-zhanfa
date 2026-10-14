"""Audit when aggregate high-to-low rotation is healthy versus collapse-masked.

This is a market-window audit, not a stock selector.  Same-day close features
define the window; all next-session fields remain outcome labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v189-high-to-low-ecology-audit-1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(frame: pd.DataFrame) -> dict:
    return {
        "sessions": int(len(frame)),
        "next_adverse_state_rate": (
            None if frame.empty else float(frame.next_adverse.mean())),
        "next_first_board_promotion": (
            None if frame.empty else float(frame.next_promotion_1to2.mean())),
        "next_middle_promotion": (
            None if frame.empty else float(frame.next_promotion_mid.mean())),
        "next_previous_limit_median_return": (
            None if frame.empty else float(frame.next_median.mean())),
        "next_failed_rate": (
            None if frame.empty else float(frame.next_failed_rate.mean())),
        "next_height_feedback": (
            None if frame.empty else float(frame.next_space_feedback.mean())),
        "mean_ecology_score": (
            None if frame.empty else float(frame.ecology_score.mean())),
    }


def _difference(healthy: pd.DataFrame, fragile: pd.DataFrame,
                draws: int, seed: int) -> dict:
    fields = {
        "next_adverse_state_rate": "next_adverse",
        "next_first_board_promotion": "next_promotion_1to2",
        "next_middle_promotion": "next_promotion_mid",
        "next_median": "next_median",
        "next_failed_rate": "next_failed_rate",
        "next_height_feedback": "next_space_feedback",
    }
    rng = np.random.default_rng(seed)
    simulations = {name: [] for name in fields}
    estimates = {
        name: float(healthy[column].mean() - fragile[column].mean())
        for name, column in fields.items()
    }
    for _ in range(draws):
        left = healthy.iloc[rng.integers(0, len(healthy), len(healthy))]
        right = fragile.iloc[rng.integers(0, len(fragile), len(fragile))]
        for name, column in fields.items():
            simulations[name].append(float(
                left[column].mean() - right[column].mean()))
    return {
        "comparison": "HEALTHY_ROTATION_WINDOW_MINUS_COLLAPSE_MASKED",
        "session_clustered": True,
        "minimum_group_sessions": int(min(len(healthy), len(fragile))),
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
        "date", "data_status", "state_divergence", "state_retreat",
        "max_board", "space_feedback", "promotion_1to2", "promotion_mid",
        "previous_limit_median_return", "failed_rate", "down_limit_count",
    }
    if required - set(frame):
        raise ValueError("high-to-low ecology market fields missing")
    data = frame.copy()
    data["date"] = pd.to_datetime(data.date, errors="coerce")
    if data.date.isna().any() or data.duplicated("date").any():
        raise ValueError("market dates must be valid and unique")
    data = data.sort_values("date").reset_index(drop=True)
    if not data.data_status.isin(["verified", "sparse"]).all():
        raise ValueError("unexpected market data status")
    numeric = [
        "state_divergence", "state_retreat", "max_board", "space_feedback",
        "promotion_1to2", "promotion_mid", "previous_limit_median_return",
        "failed_rate", "down_limit_count",
    ]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    same_day_known = data[[
        "max_board", "space_feedback", "promotion_1to2", "promotion_mid",
        "previous_limit_median_return", "failed_rate", "down_limit_count",
    ]].notna().all(axis=1)
    data["next_date"] = data.date.shift(-1)
    data["next_adverse"] = (
        data.state_divergence.shift(-1).eq(1)
        | data.state_retreat.shift(-1).eq(1))
    data["next_promotion_1to2"] = data.promotion_1to2.shift(-1)
    data["next_promotion_mid"] = data.promotion_mid.shift(-1)
    data["next_median"] = data.previous_limit_median_return.shift(-1)
    data["next_failed_rate"] = data.failed_rate.shift(-1)
    data["next_space_feedback"] = data.space_feedback.shift(-1)
    outcome_known = data[[
        "next_promotion_1to2", "next_promotion_mid", "next_median",
        "next_failed_rate", "next_space_feedback",
    ]].notna().all(axis=1)
    return data[
        data.data_status.eq("verified") & same_day_known & outcome_known
    ].copy()


def _apply_window(frame: pd.DataFrame, low_strength_threshold: float) -> pd.DataFrame:
    result = frame.copy()
    result["height_risk_increased"] = (
        result.max_board.ge(3) & result.space_feedback.le(0))
    result["low_layer_strong"] = result.promotion_1to2.ge(
        low_strength_threshold)
    result["ecology_score"] = (
        result.previous_limit_median_return.gt(0).astype(int)
        + result.failed_rate.le(.30).astype(int)
        + result.down_limit_count.le(5).astype(int)
        + result.promotion_mid.ge(.25).astype(int))
    result["high_to_low_window"] = (
        result.height_risk_increased & result.low_layer_strong)
    result["high_to_low_ecology_class"] = np.where(
        result.ecology_score.ge(3), "HEALTHY_ROTATION_WINDOW",
        "COLLAPSE_MASKED_LOW_STRENGTH")
    return result


def build(frame: pd.DataFrame, split_fraction: float = .70,
          bootstrap_draws: int = 2000) -> tuple[dict, pd.DataFrame]:
    if not 0 < split_fraction < 1:
        raise ValueError("split_fraction must be between zero and one")
    if bootstrap_draws < 100:
        raise ValueError("bootstrap_draws must be at least 100")
    all_sessions = prepare(frame).reset_index(drop=True)
    if len(all_sessions) < 400:
        raise ValueError("at least 400 evaluable market sessions required")
    split_index = max(1, min(len(all_sessions) - 1,
                             int(len(all_sessions) * split_fraction)))
    development_all = all_sessions.iloc[:split_index]
    low_threshold = float(development_all.promotion_1to2.median())
    all_sessions = _apply_window(all_sessions, low_threshold)
    development = all_sessions.iloc[:split_index]
    validation = all_sessions.iloc[split_index:]

    summaries, comparisons = {}, {}
    for offset, (period, sample) in enumerate((
            ("development", development), ("validation", validation))):
        cohort = sample[sample.high_to_low_window].copy()
        healthy = cohort[
            cohort.high_to_low_ecology_class.eq("HEALTHY_ROTATION_WINDOW")]
        fragile = cohort[
            cohort.high_to_low_ecology_class.eq(
                "COLLAPSE_MASKED_LOW_STRENGTH")]
        if min(len(healthy), len(fragile)) < 10:
            raise ValueError("high-to-low ecology support below ten in a time segment")
        summaries[period] = {
            "ALL_HIGH_TO_LOW_WINDOWS": _summary(cohort),
            "HEALTHY_ROTATION_WINDOW": _summary(healthy),
            "COLLAPSE_MASKED_LOW_STRENGTH": _summary(fragile),
        }
        comparisons[period] = _difference(
            healthy, fragile, bootstrap_draws, 18900 + offset)

    dev_adverse = comparisons["development"]["metrics"][
        "next_adverse_state_rate"]
    val_adverse = comparisons["validation"]["metrics"][
        "next_adverse_state_rate"]
    stable = bool(
        dev_adverse["estimate"] < 0 and dev_adverse["ci95"][1] < 0
        and val_adverse["estimate"] < 0 and val_adverse["ci95"][1] < 0)

    latest_source = frame[frame.data_status.eq("verified")].copy()
    latest_source["date"] = pd.to_datetime(latest_source.date, errors="coerce")
    for column in (
            "max_board", "space_feedback", "promotion_1to2", "promotion_mid",
            "previous_limit_median_return", "failed_rate", "down_limit_count"):
        latest_source[column] = pd.to_numeric(
            latest_source[column], errors="coerce")
    latest = _apply_window(
        latest_source.dropna(subset=[
            "date", "max_board", "space_feedback", "promotion_1to2",
            "promotion_mid", "previous_limit_median_return", "failed_rate",
            "down_limit_count",
        ]).sort_values("date"), low_threshold).iloc[-1]
    latest_context = {
        "signal_date": latest.date.strftime("%Y-%m-%d"),
        "height_risk_increased": bool(latest.height_risk_increased),
        "low_layer_strong": bool(latest.low_layer_strong),
        "ecology_score": int(latest.ecology_score),
        "high_to_low_window_active": bool(latest.high_to_low_window),
        "ecology_class": (
            latest.high_to_low_ecology_class
            if latest.high_to_low_window else "NOT_ACTIVE"),
        "status": "HISTORICAL_MARKET_WINDOW_CONTEXT_ONLY",
    }
    report = {
        "version": VERSION,
        "status": "DIAGNOSTIC_ONLY",
        "data_contract": {
            "same_day_window_features_available_at_close": True,
            "low_strength_threshold_frozen_from_development_only": True,
            "next_session_fields_used_only_as_labels": True,
            "specific_theme_alignment_available": False,
            "stock_level_supplement_identity_available": False,
            "actual_stock_entries": 0,
            "complete_stock_trades": 0,
        },
        "window_definition": {
            "height_risk_increased": "max_board >= 3 and space_feedback <= 0",
            "low_layer_strong": "promotion_1to2 >= frozen development median",
            "ecology_checks": [
                "previous-limit median return > 0",
                "failed-board rate <= 30%",
                "down-limit count <= 5",
                "middle-board promotion >= 25%",
            ],
            "healthy_rotation_window": "at least 3 of 4 ecology checks pass",
            "collapse_masked": "fewer than 3 ecology checks pass",
        },
        "frozen_development_thresholds": {
            "promotion_1to2_median": low_threshold,
        },
        "scope": {
            "evaluable_market_sessions": int(len(all_sessions)),
            "high_to_low_windows": int(all_sessions.high_to_low_window.sum()),
        },
        "split": {
            "method": "chronological_70_30_by_all_evaluable_sessions",
            "split_date": development.iloc[-1].date.strftime("%Y-%m-%d"),
            "development_sessions": int(len(development)),
            "validation_sessions": int(len(validation)),
            "validation_previously_inspected": True,
        },
        "summaries": summaries,
        "session_clustered_healthy_minus_collapse_masked": comparisons,
        "healthy_high_to_low_lower_adverse_stable_across_time": stable,
        "latest_market_context": latest_context,
        "governance": {
            "historical_high_to_low_market_scenario_allowed": True,
            "calibrated_high_to_low_window_probability_allowed": False,
            "high_to_low_window_direct_emotion_gate_allowed": False,
            "high_to_low_window_direct_stock_reranking_allowed": False,
            "high_to_low_stock_selection_allowed": False,
            "same_theme_supplement_mapping_allowed": False,
            "position_size_increase_allowed": False,
            "entry_authorized": False,
            "requires_specific_theme_candidate_auction_and_open_confirmation": True,
        },
        "interpretation": (
            "Healthy ecology around height weakness is materially better in the recent segment, "
            "but development shows no adverse-state separation. The aggregate window may frame "
            "a high-to-low scenario only; without specific-theme identity and live confirmation "
            "it cannot select a supplement stock or authorize entry."),
        "limitations": [
            "the audit is aggregate market-level and cannot identify a same-theme supplement",
            "specific-theme membership, catalyst persistence and propagation order are absent",
            "validation dates were inspected and are not sealed blind",
            "next-session lane metrics are labels and cannot be used at the signal close",
        ],
    }
    observations = all_sessions[all_sessions.high_to_low_window][[
        "date", "max_board", "space_feedback", "promotion_1to2",
        "promotion_mid", "previous_limit_median_return", "failed_rate",
        "down_limit_count", "ecology_score", "high_to_low_ecology_class",
        "next_date", "next_adverse", "next_promotion_1to2",
        "next_promotion_mid", "next_median", "next_failed_rate",
        "next_space_feedback",
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
    path = target / "high_to_low_ecology_observations.csv"
    observations.to_csv(path, index=False)
    report["observations_sha256"] = _sha256(path)
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
