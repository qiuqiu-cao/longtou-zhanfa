"""Test whether opening-gap meaning changes stably across emotion regimes.

This is a diagnostic interaction audit built on V182 corrected daily proxies.
The official open is available only after the opening call-auction match;
continuous auction starts at 09:30. Regime interaction
cannot substitute for auction flow, intraday absorption, theme confirmation,
an actual fill, or a complete trade.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from opening_gap_payoff_tradeoff_v182 import (
    _clustered_difference, _daily_aggregate, _difference, _rates, _summary,
)


VERSION = "v183-opening-gap-emotion-interaction-1"
STATE_FLAGS = {
    "START": "market_state_start",
    "WARMING": "market_state_warming",
    "ACCELERATION": "market_state_acceleration",
    "CONSENSUS": "market_state_consensus",
    "DIVERGENCE": "market_state_divergence",
    "RETREAT": "market_state_retreat",
    "REPAIR": "market_state_repair",
}
REGIME_GROUPS = {
    "SUPPORTIVE": {"START", "WARMING", "REPAIR"},
    "HOT": {"ACCELERATION", "CONSENSUS"},
    "ADVERSE": {"DIVERGENCE", "RETREAT"},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.astype(str).str.strip().str.lower().map({
        "true": True, "1": True, "1.0": True,
        "false": False, "0": False, "0.0": False,
    })


def _same_direction_and_nonzero_ci(development: dict, validation: dict,
                                    metric: str) -> bool:
    dev = development["metrics"][metric]
    val = validation["metrics"][metric]
    if dev["estimate"] is None or val["estimate"] is None:
        return False
    same_positive = dev["estimate"] > 0 and val["estimate"] > 0
    same_negative = dev["estimate"] < 0 and val["estimate"] < 0
    dev_excludes = dev["ci95"][0] > 0 or dev["ci95"][1] < 0
    val_excludes = val["ci95"][0] > 0 or val["ci95"][1] < 0
    return bool((same_positive or same_negative) and dev_excludes and val_excludes)


def _difference_in_differences(frame: pd.DataFrame, regime_a: str, regime_b: str,
                               metric: str, draws: int, seed: int) -> dict:
    columns = [
        "rows", "target_sum", "one_price_sum", "touch_sum", "b2_positive_sum",
        "big_known_sum", "big_loss_sum", "b2_touch_sum", "b2_touch_count",
    ]
    packed = {}
    for regime in (regime_a, regime_b):
        subset = frame[frame.regime_group.eq(regime)]
        dates = sorted(subset.signal_date.unique())
        daily = _daily_aggregate(subset)
        matrices = {}
        for band in ("GT_5_LE_7", "GT_7"):
            indexed = daily[daily.opening_gap_band.eq(band)].set_index("signal_date")
            matrices[band] = indexed.reindex(dates)[columns].fillna(0).to_numpy(float)
        rate_a = _rates(pd.Series(matrices["GT_5_LE_7"].sum(axis=0), index=columns))
        rate_b = _rates(pd.Series(matrices["GT_7"].sum(axis=0), index=columns))
        packed[regime] = {
            "dates": dates, "matrices": matrices,
            "estimate": _difference(rate_a, rate_b)[metric],
        }
    estimate_a = packed[regime_a]["estimate"]
    estimate_b = packed[regime_b]["estimate"]
    if estimate_a is None or estimate_b is None:
        return {"estimate": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    simulations = []
    for _ in range(draws):
        parts = []
        for regime in (regime_a, regime_b):
            item = packed[regime]
            indices = rng.integers(0, len(item["dates"]), len(item["dates"]))
            band_rates = {}
            for band in ("GT_5_LE_7", "GT_7"):
                counts = pd.Series(
                    item["matrices"][band][indices].sum(axis=0), index=columns)
                band_rates[band] = _rates(counts)
            parts.append(_difference(
                band_rates["GT_5_LE_7"], band_rates["GT_7"])[metric])
        if parts[0] is not None and parts[1] is not None:
            simulations.append(parts[0] - parts[1])
    return {
        "comparison": f"{regime_a}_EFFECT_MINUS_{regime_b}_EFFECT",
        "metric": metric,
        "estimate": float(estimate_a - estimate_b),
        "ci95": [float(np.quantile(simulations, .025)),
                 float(np.quantile(simulations, .975))],
    }


def build(observations: pd.DataFrame, features: pd.DataFrame,
          split_fraction: float = .70, bootstrap_draws: int = 1000) -> tuple[pd.DataFrame, dict]:
    observation_required = {
        "code", "signal_date", "opening_gap_band", "touchable", "one_price",
        "target", "b1_big_loss", "b2_mark",
    }
    feature_required = {"code", "signal_date", *STATE_FLAGS.values()}
    if observation_required - set(observations):
        raise ValueError("V182 observation fields missing")
    if feature_required - set(features):
        raise ValueError("emotion-state feature fields missing")
    left, right = observations.copy(), features.copy()
    for frame in (left, right):
        frame["code"] = frame.code.astype(str).str.zfill(6)
        frame["signal_date"] = pd.to_datetime(
            frame.signal_date, errors="coerce").dt.strftime("%Y-%m-%d")
        if frame[["code", "signal_date"]].isna().any().any():
            raise ValueError("invalid candidate key")
        if frame.duplicated(["code", "signal_date"]).any():
            raise ValueError("candidate keys must be unique")
    merged = left.merge(
        right[["code", "signal_date", *STATE_FLAGS.values()]],
        on=["code", "signal_date"], how="left", validate="one_to_one")
    for column in STATE_FLAGS.values():
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    active_count = merged[list(STATE_FLAGS.values())].eq(1).sum(axis=1)
    invalid_state_rows = int(active_count.ne(1).sum())
    merged = merged[active_count.eq(1)].copy()
    inverse = {column: state for state, column in STATE_FLAGS.items()}
    merged["emotion_state"] = merged[list(STATE_FLAGS.values())].idxmax(axis=1).map(inverse)
    state_to_group = {
        state: group for group, states in REGIME_GROUPS.items() for state in states
    }
    merged["regime_group"] = merged.emotion_state.map(state_to_group)
    if merged.regime_group.isna().any():
        raise ValueError("unmapped emotion state")
    for column in ("touchable", "one_price", "target"):
        merged[column] = _bool(merged[column])
        if merged[column].isna().any():
            raise ValueError("invalid V182 boolean observation")
        merged[column] = merged[column].astype(bool)
    merged["b1_big_loss"] = _bool(merged.b1_big_loss).astype("boolean")
    merged["b2_mark"] = pd.to_numeric(merged.b2_mark, errors="coerce")
    merged["b2_positive"] = merged.target
    dates = sorted(merged.signal_date.unique())
    if len(dates) < 200:
        raise ValueError("insufficient emotion-interaction sessions")
    split_index = max(1, min(len(dates) - 1, int(len(dates) * split_fraction)))
    split_date = dates[split_index]
    development = merged[merged.signal_date.lt(split_date)].copy()
    validation = merged[merged.signal_date.ge(split_date)].copy()
    summaries, comparisons, consistency = {}, {}, {}
    for regime in REGIME_GROUPS:
        dev = development[development.regime_group.eq(regime)]
        val = validation[validation.regime_group.eq(regime)]
        summaries[regime] = {
            "development": _summary(dev), "validation": _summary(val),
        }
        dev_comparison = _clustered_difference(
            dev, "GT_5_LE_7", "GT_7", bootstrap_draws, 18301)
        val_comparison = _clustered_difference(
            val, "GT_5_LE_7", "GT_7", bootstrap_draws, 18302)
        comparisons[regime] = {
            "development_5_7_vs_gt_7": dev_comparison,
            "validation_5_7_vs_gt_7": val_comparison,
        }
        consistency[regime] = {
            metric: _same_direction_and_nonzero_ci(
                dev_comparison, val_comparison, metric)
            for metric in (
                "combined", "one_price", "conditional_b2_positive",
                "conditional_b1_big_loss", "conditional_mean_b2")
        }
    heterogeneity = {
        "development_supportive_vs_adverse_combined": _difference_in_differences(
            development, "SUPPORTIVE", "ADVERSE", "combined",
            bootstrap_draws, 18311),
        "validation_supportive_vs_adverse_combined": _difference_in_differences(
            validation, "SUPPORTIVE", "ADVERSE", "combined",
            bootstrap_draws, 18312),
        "development_supportive_vs_adverse_big_loss": _difference_in_differences(
            development, "SUPPORTIVE", "ADVERSE", "conditional_b1_big_loss",
            bootstrap_draws, 18313),
        "validation_supportive_vs_adverse_big_loss": _difference_in_differences(
            validation, "SUPPORTIVE", "ADVERSE", "conditional_b1_big_loss",
            bootstrap_draws, 18314),
    }
    regime_interaction_supported = bool(
        any(consistency[regime]["combined"] for regime in consistency)
        and any(consistency[regime]["conditional_b1_big_loss"] for regime in consistency)
        and heterogeneity["development_supportive_vs_adverse_combined"]["ci95"][0]
        * heterogeneity["development_supportive_vs_adverse_combined"]["ci95"][1] > 0
        and heterogeneity["validation_supportive_vs_adverse_combined"]["ci95"][0]
        * heterogeneity["validation_supportive_vs_adverse_combined"]["ci95"][1] > 0)
    report = {
        "version": VERSION,
        "status": "DIAGNOSTIC_ONLY",
        "input_contract": {
            "v182_corrected_observations_required": True,
            "official_open_available_only_after_opening_call_match": True,
            "continuous_auction_begins_at_09_30": True,
            "emotion_state_known_by_signal_close": True,
            "actual_fills": 0, "complete_trades": 0,
        },
        "scope": {
            "rows": int(len(merged)), "sessions": int(merged.signal_date.nunique()),
            "invalid_or_nonunique_state_rows_excluded": invalid_state_rows,
            "regime_rows": {
                regime: int(merged.regime_group.eq(regime).sum())
                for regime in REGIME_GROUPS
            },
            "regime_sessions": {
                regime: int(merged.loc[
                    merged.regime_group.eq(regime), "signal_date"].nunique())
                for regime in REGIME_GROUPS
            },
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
        "regime_definition": {
            group: sorted(states) for group, states in REGIME_GROUPS.items()
        },
        "summaries": summaries,
        "date_clustered_comparisons": comparisons,
        "temporal_consistency": consistency,
        "difference_in_differences": heterogeneity,
        "opening_gap_emotion_interaction_supported": regime_interaction_supported,
        "governance": {
            "emotion_can_override_above_seven_risk_flag": False,
            "opening_gap_emotion_interaction_factor_allowed": False,
            "supportive_regime_automatic_high_open_buy_allowed": False,
            "adverse_regime_automatic_high_open_cancel_allowed": False,
            "direct_stock_reranking_allowed": False,
            "entry_authorized": False,
        },
        "interpretation": (
            "The 5-7% versus above-7% payoff difference changes across regimes and time "
            "segments; supportive, hot and adverse labels do not provide a stable waiver or "
            "automatic veto. Above-7% remains a buyability risk flag in every regime, while "
            "actual absorption, reseal, theme synchronization and price ceiling remain required."),
        "limitations": [
            "emotion states are retrospective historical reconstructions, not frozen forward labels",
            "daily OHLC cannot observe auction queues, VWAP path, intraday absorption, or fills",
            "candidate universe is not independently full-market complete",
            "validation dates were inspected and are not sealed blind",
        ],
    }
    columns = [
        "code", "signal_date", "opening_gap_band", "emotion_state", "regime_group",
        "touchable", "one_price", "target", "b1_big_loss", "b2_mark",
    ]
    return merged[columns].sort_values(["signal_date", "code"]), report


def run(observations_path: str | Path, features_path: str | Path,
        output_dir: str | Path, bootstrap_draws: int = 1000) -> dict:
    observations_path, features_path = Path(observations_path), Path(features_path)
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable V183 output")
    rows, report = build(
        pd.read_csv(observations_path, dtype={"code": str}),
        pd.read_csv(features_path, dtype={"code": str}),
        bootstrap_draws=bootstrap_draws)
    target.mkdir(parents=True, exist_ok=False)
    output = target / "opening_gap_emotion_observations.csv"
    rows.to_csv(output, index=False, encoding="utf-8-sig")
    report["source_hashes"] = {
        str(path.resolve()): _sha256(path)
        for path in (observations_path, features_path)
    }
    report["observations_sha256"] = _sha256(output)
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(run(args.observations, args.features, args.output,
                         args.bootstrap_draws), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
