"""Audit the base-rate trap in days with many first-board limit-ups.

The probability that *at least one* first board promotes rises mechanically with
the number of first boards.  That event-level statement is not the same as the
promotion probability of an individual candidate and cannot rank a stock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v185-first-board-supply-base-rate-audit-1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "sessions": 0, "mean_first_board_supply": None,
            "known_candidates": 0, "promotions": 0,
            "candidate_promotion_rate": None,
            "daily_any_promotion_rate": None,
            "mean_daily_promotion_rate": None,
        }
    known = int(frame["known"].sum())
    promotions = int(frame["promotions"].sum())
    return {
        "sessions": int(len(frame)),
        "mean_first_board_supply": float(frame["supply"].mean()),
        "known_candidates": known,
        "promotions": promotions,
        "candidate_promotion_rate": promotions / known if known else None,
        "daily_any_promotion_rate": float(frame["any_promotion"].mean()),
        "mean_daily_promotion_rate": float(frame["daily_promotion_rate"].mean()),
    }


def _bootstrap_difference(high: pd.DataFrame, low: pd.DataFrame, draws: int,
                          seed: int) -> dict:
    def candidate_rate(frame: pd.DataFrame) -> float:
        return float(frame["promotions"].sum() / frame["known"].sum())

    estimate = candidate_rate(high) - candidate_rate(low)
    rng = np.random.default_rng(seed)
    simulations = []
    any_simulations = []
    for _ in range(draws):
        high_sample = high.iloc[rng.integers(0, len(high), len(high))]
        low_sample = low.iloc[rng.integers(0, len(low), len(low))]
        simulations.append(candidate_rate(high_sample) - candidate_rate(low_sample))
        any_simulations.append(
            float(high_sample["any_promotion"].mean()
                  - low_sample["any_promotion"].mean()))
    return {
        "comparison": "HIGH_SUPPLY_MINUS_LOW_SUPPLY",
        "candidate_promotion_rate_difference": estimate,
        "candidate_promotion_rate_ci95": [
            float(value) for value in np.quantile(simulations, [.025, .975])],
        "daily_any_promotion_rate_difference": float(
            high["any_promotion"].mean() - low["any_promotion"].mean()),
        "daily_any_promotion_rate_ci95": [
            float(value) for value in np.quantile(any_simulations, [.025, .975])],
        "date_clustered": True,
    }


def build(candidates: pd.DataFrame, outcomes: pd.DataFrame, outcome_manifest: dict,
          split_fraction: float = .70, bootstrap_draws: int = 2000) -> tuple[dict, pd.DataFrame]:
    if not 0 < split_fraction < 1:
        raise ValueError("split_fraction must be between zero and one")
    if bootstrap_draws < 100:
        raise ValueError("bootstrap_draws must be at least 100")
    contract = outcome_manifest.get("target_contract", {})
    target_coverage = outcome_manifest.get("target_coverage", {})
    if not (contract.get("no_touch_rows_retained_as_negative") is True
            or target_coverage.get("no_touch_rows_retained_as_negative") is True
            or outcome_manifest.get("no_touch_rows_retained_as_negative") is True):
        raise ValueError("corrected no-touch-negative outcome contract required")
    required_candidates = {"code", "signal_date", "board_group"}
    required_outcomes = {"code", "signal_date", "promotion_LABEL_ONLY"}
    if required_candidates - set(candidates) or required_outcomes - set(outcomes):
        raise ValueError("candidate or outcome fields missing")

    first = candidates.loc[
        candidates["board_group"].eq("FIRST_BOARD"),
        ["code", "signal_date"],
    ].copy()
    first["code"] = first["code"].astype(str).str.zfill(6)
    labels = outcomes[["code", "signal_date", "promotion_LABEL_ONLY"]].copy()
    labels["code"] = labels["code"].astype(str).str.zfill(6)
    if first.duplicated(["code", "signal_date"]).any():
        raise ValueError("duplicate first-board candidate")
    if labels.duplicated(["code", "signal_date"]).any():
        raise ValueError("duplicate outcome label")
    merged = first.merge(labels, on=["code", "signal_date"], how="left",
                         validate="one_to_one")
    merged["promotion_LABEL_ONLY"] = pd.to_numeric(
        merged["promotion_LABEL_ONLY"], errors="coerce")
    daily = merged.groupby("signal_date", as_index=False).agg(
        supply=("code", "size"),
        known=("promotion_LABEL_ONLY", "count"),
        promotions=("promotion_LABEL_ONLY", "sum"),
    )
    # A zero-known terminal date has no next-session label and cannot be evaluated.
    daily = daily[daily["known"].gt(0)].copy().sort_values("signal_date")
    if len(daily) < 100:
        raise ValueError("at least 100 evaluable sessions required")
    daily["promotions"] = daily["promotions"].astype(int)
    daily["known_coverage"] = daily["known"] / daily["supply"]
    daily["daily_promotion_rate"] = daily["promotions"] / daily["known"]
    daily["any_promotion"] = daily["promotions"].gt(0)

    split_index = max(1, min(len(daily) - 1, int(len(daily) * split_fraction)))
    split_date = str(daily.iloc[split_index - 1]["signal_date"])
    development = daily.iloc[:split_index].copy()
    validation = daily.iloc[split_index:].copy()
    q25, q75 = [float(value) for value in development["supply"].quantile([.25, .75])]

    summaries = {}
    comparisons = {}
    for offset, (period, frame) in enumerate((
            ("development", development), ("validation", validation))):
        low = frame[frame["supply"].le(q25)]
        high = frame[frame["supply"].gt(q75)]
        if low.empty or high.empty:
            raise ValueError("fixed supply buckets empty in a time segment")
        summaries[period] = {
            "ALL": _summary(frame), "LOW_SUPPLY": _summary(low),
            "HIGH_SUPPLY": _summary(high),
        }
        comparisons[period] = _bootstrap_difference(
            high, low, bootstrap_draws, 18500 + offset)

    any_success_is_discriminative = bool(
        summaries["development"]["ALL"]["daily_any_promotion_rate"] < .90
        or summaries["validation"]["ALL"]["daily_any_promotion_rate"] < .90)
    positive_edge_stable = bool(all(
        comparisons[period]["candidate_promotion_rate_difference"] > 0
        and comparisons[period]["candidate_promotion_rate_ci95"][0] > 0
        for period in ("development", "validation")
    ))
    report = {
        "version": VERSION,
        "status": "DIAGNOSTIC_ONLY",
        "target_contract": {
            "source_no_touch_rows_retained_as_negative": True,
            "missing_promotion_labels_filled_as_zero": False,
            "actual_fills": 0, "complete_trades": 0,
        },
        "scope": {
            "first_board_rows": int(len(merged)),
            "evaluable_sessions": int(len(daily)),
            "known_promotion_labels": int(merged["promotion_LABEL_ONLY"].notna().sum()),
            "mean_daily_known_coverage": float(daily["known_coverage"].mean()),
        },
        "split": {
            "method": "chronological_70_30_by_signal_date",
            "split_date": split_date,
            "development_sessions": int(len(development)),
            "validation_sessions": int(len(validation)),
            "validation_previously_inspected": True,
        },
        "frozen_supply_thresholds_from_development": {
            "low_supply_max": q25, "high_supply_strictly_above": q75,
        },
        "summaries": summaries,
        "date_clustered_high_minus_low": comparisons,
        "daily_any_promotion_is_discriminative": any_success_is_discriminative,
        "high_first_board_supply_positive_edge_supported": positive_edge_stable,
        "governance": {
            "any_first_board_promoted_may_rank_a_candidate": False,
            "first_board_supply_positive_ranking_factor_allowed": False,
            "high_supply_automatic_emotion_upgrade_allowed": False,
            "high_supply_automatic_reject_allowed": False,
            "requires_candidate_theme_identity_and_execution_confirmation": True,
            "direct_stock_reranking_allowed": False,
            "entry_authorized": False,
        },
        "interpretation": (
            "With dozens of first boards, at least one promotion is almost guaranteed and is "
            "therefore a weak discriminator. High supply does not prove better per-candidate "
            "odds; its candidate promotion rate is directionally lower in both time segments, "
            "but uncertainty remains. Breadth must be separated from candidate quality and "
            "from verified theme structure."),
        "limitations": [
            "current-universe survivorship bias",
            "historical security eligibility is not independently reverified",
            "promotion is a close-to-close identity label, not a complete executable trade",
            "validation dates were inspected and are not sealed blind",
        ],
    }
    return report, daily


def run(candidates_path: str | Path, outcomes_path: str | Path,
        manifest_path: str | Path, output_dir: str | Path,
        bootstrap_draws: int = 2000) -> dict:
    candidates_path, outcomes_path, manifest_path = map(
        Path, (candidates_path, outcomes_path, manifest_path))
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable audit")
    report, daily = build(
        pd.read_csv(candidates_path, dtype={"code": str}),
        pd.read_csv(outcomes_path, dtype={"code": str}),
        json.loads(manifest_path.read_text(encoding="utf-8")),
        bootstrap_draws=bootstrap_draws,
    )
    report["source_hashes"] = {
        str(path.resolve()): _sha256(path)
        for path in (candidates_path, outcomes_path, manifest_path, Path(__file__))
    }
    target.mkdir(parents=True, exist_ok=False)
    daily.to_csv(target / "daily_first_board_supply.csv", index=False)
    report["daily_observations_sha256"] = _sha256(
        target / "daily_first_board_supply.csv")
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--outcomes", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    print(json.dumps(run(args.candidates, args.outcomes, args.manifest,
                         args.output, args.bootstrap_draws),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
