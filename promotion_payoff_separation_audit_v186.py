"""Separate next-session promotion identity from post-entry B+2/B+3 payoff.

Promotion is useful for describing a stock's continuing identity.  It is not an
executable-trade target: a promoted name may be one-price and unavailable, or
may fail to preserve a positive mark after a hypothetical limit-price fill.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v186-promotion-payoff-separation-audit-1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")
    return series.astype(str).str.strip().str.lower().map({
        "true": True, "1": True, "1.0": True,
        "false": False, "0": False, "0.0": False,
        "nan": pd.NA, "none": pd.NA, "": pd.NA,
    }).astype("boolean")


def _summary(frame: pd.DataFrame) -> dict:
    promotion_known = frame[frame["promotion"].notna()]
    paired_b2 = frame[frame["promotion"].notna() & frame["b2_target_known"]]
    paired_both = frame[frame["promotion"].notna() & frame["both_target_known"]]
    promoted_b2 = paired_b2[paired_b2["promotion"]]
    promoted_both = paired_both[paired_both["promotion"]]
    promoted_buyable = frame[
        frame["promotion"].eq(True) & frame["touchable"]
        & frame["b1_big_loss"].notna()
    ]
    return {
        "rows": int(len(frame)),
        "sessions": int(frame["signal_date"].nunique()),
        "promotion_known_rows": int(len(promotion_known)),
        "promotion_rate": (
            None if promotion_known.empty
            else float(promotion_known["promotion"].mean())),
        "paired_b2_rows": int(len(paired_b2)),
        "full_denominator_buyable_b2_positive_rate": (
            None if paired_b2.empty else float(paired_b2["b2_positive"].mean())),
        "promoted_rows_in_paired_b2": int(len(promoted_b2)),
        "promoted_and_buyable_b2_positive_rate": (
            None if promoted_b2.empty else float(promoted_b2["b2_positive"].mean())),
        "promoted_without_buyable_b2_positive_rate": (
            None if promoted_b2.empty else float((~promoted_b2["b2_positive"]).mean())),
        "promoted_one_price_unbuyable_rate": (
            None if promoted_b2.empty else float(promoted_b2["one_price"].mean())),
        "paired_b2_b3_rows": int(len(paired_both)),
        "full_denominator_buyable_b2_b3_both_positive_rate": (
            None if paired_both.empty else float(paired_both["both_positive"].mean())),
        "promoted_and_buyable_b2_b3_both_positive_rate": (
            None if promoted_both.empty
            else float(promoted_both["both_positive"].mean())),
        "conditional_b1_big_loss_rate_among_promoted_buyable": (
            None if promoted_buyable.empty
            else float(promoted_buyable["b1_big_loss"].mean())),
        "touchable_rate": None if frame.empty else float(frame["touchable"].mean()),
    }


def _clustered_identity_minus_payoff(frame: pd.DataFrame, draws: int,
                                      seed: int) -> dict:
    paired = frame[frame["promotion"].notna() & frame["b2_target_known"]].copy()
    if paired.empty:
        raise ValueError("no paired promotion and B+2 targets")
    paired["promotion_int"] = paired["promotion"].astype(int)
    paired["b2_positive_int"] = paired["b2_positive"].astype(int)
    paired["promoted_no_b2"] = (
        paired["promotion"] & ~paired["b2_positive"]).astype(int)
    daily = paired.groupby("signal_date", as_index=False).agg(
        rows=("code", "size"), promotion_sum=("promotion_int", "sum"),
        b2_positive_sum=("b2_positive_int", "sum"),
        promoted_no_b2_sum=("promoted_no_b2", "sum"),
    )

    def rates(sample: pd.DataFrame) -> tuple[float, float]:
        rows = float(sample["rows"].sum())
        promoted = float(sample["promotion_sum"].sum())
        return (
            float((sample["promotion_sum"].sum()
                   - sample["b2_positive_sum"].sum()) / rows),
            float(sample["promoted_no_b2_sum"].sum() / promoted)
            if promoted else np.nan,
        )

    difference, promoted_failure = rates(daily)
    rng = np.random.default_rng(seed)
    diffs, failures = [], []
    for _ in range(draws):
        sample = daily.iloc[rng.integers(0, len(daily), len(daily))]
        diff, failure = rates(sample)
        diffs.append(diff)
        if not np.isnan(failure):
            failures.append(failure)
    return {
        "date_clustered": True,
        "promotion_rate_minus_full_b2_positive_rate": difference,
        "promotion_minus_b2_ci95": [
            float(value) for value in np.quantile(diffs, [.025, .975])],
        "promoted_without_buyable_b2_positive_rate": promoted_failure,
        "promoted_without_b2_ci95": [
            float(value) for value in np.quantile(failures, [.025, .975])],
    }


def build(candidates: pd.DataFrame, outcomes: pd.DataFrame, corrected_contract: dict,
          split_fraction: float = .70, bootstrap_draws: int = 2000
          ) -> tuple[dict, pd.DataFrame]:
    if not 0 < split_fraction < 1:
        raise ValueError("split_fraction must be between zero and one")
    if bootstrap_draws < 100:
        raise ValueError("bootstrap_draws must be at least 100")
    coverage = corrected_contract.get("target_coverage", {})
    target = corrected_contract.get("target_contract", {})
    if not (coverage.get("no_touch_rows_retained_as_negative") is True
            or target.get("no_touch_rows_retained_as_negative") is True
            or corrected_contract.get("no_touch_rows_retained_as_negative") is True):
        raise ValueError("corrected no-touch-negative contract required")
    required_candidates = {"code", "signal_date", "boards", "board_group"}
    required_outcomes = {
        "code", "signal_date", "promotion_LABEL_ONLY",
        "next_session_touched_limit_LABEL_ONLY",
        "next_session_one_price_limit_LABEL_ONLY",
        "b1_big_loss_mark_LABEL_ONLY", "b2_net_close_mark_LABEL_ONLY",
        "b3_net_close_mark_LABEL_ONLY", "actual_entry_verified",
        "complete_trade_profit_LABEL_ONLY",
    }
    if required_candidates - set(candidates) or required_outcomes - set(outcomes):
        raise ValueError("candidate or outcome fields missing")
    left = candidates[list(required_candidates)].copy()
    right = outcomes[list(required_outcomes)].copy()
    for frame in (left, right):
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        if frame.duplicated(["code", "signal_date"]).any():
            raise ValueError("duplicate candidate-outcome key")
    merged = left.merge(right, on=["code", "signal_date"], how="inner",
                        validate="one_to_one")
    merged["boards"] = pd.to_numeric(merged["boards"], errors="coerce")
    merged = merged[merged["boards"].between(1, 5)].copy()
    if _boolean(merged["actual_entry_verified"]).fillna(False).any():
        raise ValueError("proxy unexpectedly claims actual fills")
    if pd.to_numeric(merged["complete_trade_profit_LABEL_ONLY"],
                     errors="coerce").notna().any():
        raise ValueError("proxy unexpectedly contains complete trades")

    merged["promotion"] = _boolean(merged["promotion_LABEL_ONLY"])
    touched = _boolean(merged["next_session_touched_limit_LABEL_ONLY"])
    one_price = _boolean(merged["next_session_one_price_limit_LABEL_ONLY"])
    payoff_observable = touched.notna() & one_price.notna()
    unknown_touch_rows_excluded = int((~payoff_observable).sum())
    merged = merged[payoff_observable].copy()
    touched = touched.loc[merged.index]
    one_price = one_price.loc[merged.index]
    merged["touchable"] = (touched.eq(True) & one_price.eq(False)).astype(bool)
    merged["one_price"] = one_price.eq(True).astype(bool)
    merged["b1_big_loss"] = _boolean(merged["b1_big_loss_mark_LABEL_ONLY"])
    b2 = pd.to_numeric(merged["b2_net_close_mark_LABEL_ONLY"], errors="coerce")
    b3 = pd.to_numeric(merged["b3_net_close_mark_LABEL_ONLY"], errors="coerce")
    merged["b2_target_known"] = (~merged["touchable"] | b2.notna()).astype(bool)
    merged["both_target_known"] = (
        ~merged["touchable"] | (b2.notna() & b3.notna())).astype(bool)
    merged["b2_positive"] = (merged["touchable"] & b2.gt(0)).astype(bool)
    merged["b3_positive"] = (merged["touchable"] & b3.gt(0)).astype(bool)
    merged["both_positive"] = (
        merged["b2_positive"] & merged["b3_positive"]).astype(bool)

    evaluable_dates = sorted(merged.loc[
        merged["promotion"].notna(), "signal_date"].unique())
    if len(evaluable_dates) < 200:
        raise ValueError("insufficient evaluable sessions")
    split_index = max(1, min(len(evaluable_dates) - 1,
                             int(len(evaluable_dates) * split_fraction)))
    split_date = str(evaluable_dates[split_index - 1])
    development = merged[merged["signal_date"].le(split_date)].copy()
    validation = merged[merged["signal_date"].gt(split_date)].copy()
    groups = {
        "ALL": lambda x: x,
        "FIRST_BOARD": lambda x: x[x["board_group"].eq("FIRST_BOARD")],
        "CONSECUTIVE_BOARD": lambda x: x[x["board_group"].eq("CONSECUTIVE_BOARD")],
    }
    summaries, comparisons = {}, {}
    for offset, (group, selector) in enumerate(groups.items()):
        summaries[group] = {
            "development": _summary(selector(development)),
            "validation": _summary(selector(validation)),
        }
        comparisons[group] = {
            "development": _clustered_identity_minus_payoff(
                selector(development), bootstrap_draws, 18600 + offset * 2),
            "validation": _clustered_identity_minus_payoff(
                selector(validation), bootstrap_draws, 18601 + offset * 2),
        }
    board_level = {}
    for boards in range(1, 6):
        board_level[str(boards)] = {
            "development": _summary(development[development["boards"].eq(boards)]),
            "validation": _summary(validation[validation["boards"].eq(boards)]),
        }
    report = {
        "version": VERSION,
        "status": "DIAGNOSTIC_ONLY",
        "target_contract": {
            "no_touch_rows_retained_as_negative": True,
            "one_price_rows_retained_as_negative": True,
            "touchable_unknown_b2_excluded_from_b2_denominator": True,
            "unknown_touch_rows_excluded_not_filled": True,
            "missing_promotion_labels_filled_as_zero": False,
            "actual_fills": 0, "complete_trades": 0,
        },
        "scope": {
            "rows": int(len(merged)),
            "sessions": int(len(evaluable_dates)),
            "promotion_known_rows": int(merged["promotion"].notna().sum()),
            "b2_target_known_rows": int(merged["b2_target_known"].sum()),
            "b2_b3_target_known_rows": int(merged["both_target_known"].sum()),
            "unknown_touch_rows_excluded": unknown_touch_rows_excluded,
        },
        "split": {
            "method": "chronological_70_30_by_signal_date",
            "split_date": split_date,
            "development_sessions": int(development["signal_date"].nunique()),
            "validation_sessions": int(validation["signal_date"].nunique()),
            "validation_previously_inspected": True,
        },
        "summaries": summaries,
        "board_level_summaries": board_level,
        "date_clustered_identity_minus_payoff": comparisons,
        "promotion_is_buyable_b2_payoff_equivalent": False,
        "promotion_is_buyable_b2_b3_sustained_equivalent": False,
        "governance": {
            "promotion_probability_may_be_presented_as_trade_win_probability": False,
            "promotion_probability_may_replace_b2_b3_payoff_target": False,
            "board_level_promotion_rate_may_directly_rank_entries": False,
            "requires_separate_buyability_payoff_and_big_loss_models": True,
            "direct_stock_reranking_allowed": False,
            "entry_authorized": False,
        },
        "interpretation": (
            "Promotion describes continuing identity, not an executable post-fill payoff. "
            "A majority of promoted names fail the full-denominator buyable B+2-positive path, "
            "and fewer preserve positive marks through both B+2 and B+3. Promotion, buyability, "
            "multi-session payoff and big-loss risk must remain separate targets."),
        "limitations": [
            "daily proxy without verified auction queue or actual fill",
            "current-universe survivorship bias and incomplete historical eligibility",
            "validation dates were inspected and are not sealed blind",
        ],
    }
    observations = merged[[
        "code", "signal_date", "boards", "board_group", "promotion", "touchable",
        "one_price", "b1_big_loss", "b2_target_known", "both_target_known",
        "b2_positive", "b3_positive", "both_positive",
    ]].copy()
    return report, observations


def run(candidates_path: str | Path, outcomes_path: str | Path,
        contract_path: str | Path, output_dir: str | Path,
        bootstrap_draws: int = 2000) -> dict:
    candidates_path, outcomes_path, contract_path = map(
        Path, (candidates_path, outcomes_path, contract_path))
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable audit")
    report, observations = build(
        pd.read_csv(candidates_path, dtype={"code": str}),
        pd.read_csv(outcomes_path, dtype={"code": str}),
        json.loads(contract_path.read_text(encoding="utf-8")),
        bootstrap_draws=bootstrap_draws,
    )
    report["source_hashes"] = {
        str(path.resolve()): _sha256(path)
        for path in (candidates_path, outcomes_path, contract_path, Path(__file__))
    }
    target.mkdir(parents=True, exist_ok=False)
    observations.to_csv(target / "promotion_payoff_observations.csv", index=False)
    report["observations_sha256"] = _sha256(
        target / "promotion_payoff_observations.csv")
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--outcomes", required=True)
    parser.add_argument("--corrected-contract", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    print(json.dumps(run(args.candidates, args.outcomes, args.corrected_contract,
                         args.output, args.bootstrap_draws),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
