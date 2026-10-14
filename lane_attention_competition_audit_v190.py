"""Audit attention competition between first-board and relay lanes.

The two lanes compete for scarce short-term capital, but they do not share the
same identity target.  This audit uses only same-day closing-pool composition
as the exposure and keeps next-session promotion and B+2/B+3 outcomes strictly
on the label side.  It is a market-level diagnostic: without point-in-time
specific-theme history it cannot name a cross-theme winner or authorize entry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v190-lane-attention-competition-audit-1"


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


def _prepare(candidates: pd.DataFrame, outcomes: pd.DataFrame,
             contract: dict) -> pd.DataFrame:
    coverage = contract.get("target_coverage", {})
    target = contract.get("target_contract", {})
    if not (coverage.get("no_touch_rows_retained_as_negative") is True
            or target.get("no_touch_rows_retained_as_negative") is True
            or contract.get("no_touch_rows_retained_as_negative") is True):
        raise ValueError("corrected no-touch-negative contract required")
    candidate_fields = {"code", "signal_date", "boards", "board_group"}
    outcome_fields = {
        "code", "signal_date", "promotion_LABEL_ONLY",
        "next_session_touched_limit_LABEL_ONLY",
        "next_session_one_price_limit_LABEL_ONLY",
        "b1_big_loss_mark_LABEL_ONLY", "b2_net_close_mark_LABEL_ONLY",
        "b3_net_close_mark_LABEL_ONLY", "actual_entry_verified",
        "complete_trade_profit_LABEL_ONLY",
    }
    missing = ((candidate_fields - set(candidates))
               | (outcome_fields - set(outcomes)))
    if missing:
        raise ValueError(f"required fields missing: {sorted(missing)}")
    left = candidates[list(candidate_fields)].copy()
    right = outcomes[list(outcome_fields)].copy()
    for frame in (left, right):
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        frame["signal_date"] = pd.to_datetime(
            frame["signal_date"], errors="raise").dt.strftime("%Y-%m-%d")
        if frame.duplicated(["code", "signal_date"]).any():
            raise ValueError("duplicate candidate key")
    left["boards"] = pd.to_numeric(left["boards"], errors="coerce")
    left = left[left["boards"].between(1, 5)].copy()
    merged = left.merge(right, on=["code", "signal_date"], how="inner",
                        validate="one_to_one")
    if _boolean(merged["actual_entry_verified"]).fillna(False).any():
        raise ValueError("diagnostic source unexpectedly claims actual fills")
    if pd.to_numeric(merged["complete_trade_profit_LABEL_ONLY"],
                     errors="coerce").notna().any():
        raise ValueError("diagnostic source unexpectedly contains complete trades")

    daily = merged.groupby("signal_date", as_index=False).agg(
        first_supply=("board_group", lambda x: int((x == "FIRST_BOARD").sum())),
        relay_supply=("board_group", lambda x: int((x == "CONSECUTIVE_BOARD").sum())),
        max_board=("boards", "max"),
    )
    daily["total_supply"] = daily["first_supply"] + daily["relay_supply"]
    daily["relay_share"] = daily["relay_supply"] / daily["total_supply"]
    merged = merged.merge(daily, on="signal_date", how="left", validate="many_to_one")
    merged = merged[(merged["first_supply"] > 0) & (merged["relay_supply"] > 0)].copy()

    merged["promotion"] = _boolean(merged["promotion_LABEL_ONLY"])
    touched = _boolean(merged["next_session_touched_limit_LABEL_ONLY"])
    one_price = _boolean(merged["next_session_one_price_limit_LABEL_ONLY"])
    merged["path_known"] = touched.notna() & one_price.notna()
    merged["touchable"] = (
        touched.eq(True) & one_price.eq(False)).fillna(False).astype(bool)
    merged["one_price"] = one_price.eq(True).fillna(False).astype(bool)
    b1_loss = _boolean(merged["b1_big_loss_mark_LABEL_ONLY"])
    b2 = pd.to_numeric(merged["b2_net_close_mark_LABEL_ONLY"], errors="coerce")
    b3 = pd.to_numeric(merged["b3_net_close_mark_LABEL_ONLY"], errors="coerce")
    merged["b1_big_loss"] = b1_loss
    merged["b2_target_known"] = merged["path_known"] & (~merged["touchable"] | b2.notna())
    merged["b3_target_known"] = merged["path_known"] & (~merged["touchable"] | b3.notna())
    merged["b2_positive"] = (merged["touchable"] & b2.gt(0)).fillna(False).astype(bool)
    merged["b3_positive"] = (merged["touchable"] & b3.gt(0)).fillna(False).astype(bool)
    return merged.sort_values(["signal_date", "code"]).reset_index(drop=True)


def _summary(frame: pd.DataFrame) -> dict:
    promotion = frame[frame["promotion"].notna()]
    b2 = frame[frame["b2_target_known"]]
    b3 = frame[frame["b3_target_known"]]
    touchable = frame[frame["touchable"] & frame["b1_big_loss"].notna()]
    return {
        "rows": int(len(frame)),
        "sessions": int(frame["signal_date"].nunique()),
        "promotion_known_rows": int(len(promotion)),
        "promotion_rate": None if promotion.empty else float(promotion["promotion"].mean()),
        "b2_target_known_rows": int(len(b2)),
        "full_denominator_buyable_b2_positive_rate": (
            None if b2.empty else float(b2["b2_positive"].mean())),
        "b3_target_known_rows": int(len(b3)),
        "full_denominator_buyable_b3_positive_rate": (
            None if b3.empty else float(b3["b3_positive"].mean())),
        "touchable_rate": None if frame.empty else float(frame["touchable"].mean()),
        "touchable_b1_big_loss_known_rows": int(len(touchable)),
        "conditional_b1_big_loss_rate": (
            None if touchable.empty else float(touchable["b1_big_loss"].mean())),
    }


def _daily_rates(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["promotion_known"] = work["promotion"].notna().astype(int)
    work["promotion_value"] = work["promotion"].fillna(False).astype(int)
    work["b2_known"] = work["b2_target_known"].astype(int)
    work["b2_value"] = work["b2_positive"].astype(int)
    work["b3_known"] = work["b3_target_known"].astype(int)
    work["b3_value"] = work["b3_positive"].astype(int)
    work["touchable_value"] = work["touchable"].astype(int)
    work["loss_known"] = (work["touchable"] & work["b1_big_loss"].notna()).astype(int)
    work["loss_value"] = (
        work["touchable"] & work["b1_big_loss"].eq(True)
    ).fillna(False).astype(int)
    return work.groupby("signal_date", as_index=False).agg(
        rows=("code", "size"),
        promotion_known=("promotion_known", "sum"),
        promotion_value=("promotion_value", "sum"),
        b2_known=("b2_known", "sum"), b2_value=("b2_value", "sum"),
        b3_known=("b3_known", "sum"), b3_value=("b3_value", "sum"),
        touchable_value=("touchable_value", "sum"),
        loss_known=("loss_known", "sum"), loss_value=("loss_value", "sum"),
    )


def _rates(daily: pd.DataFrame) -> dict[str, float]:
    def ratio(num: str, den: str) -> float:
        denominator = float(daily[den].sum())
        return float(daily[num].sum() / denominator) if denominator else np.nan
    return {
        "promotion_rate": ratio("promotion_value", "promotion_known"),
        "buyable_b2_positive_rate": ratio("b2_value", "b2_known"),
        "buyable_b3_positive_rate": ratio("b3_value", "b3_known"),
        "touchable_rate": ratio("touchable_value", "rows"),
        "conditional_b1_big_loss_rate": ratio("loss_value", "loss_known"),
    }


def _clustered_difference(high: pd.DataFrame, low: pd.DataFrame, draws: int,
                          seed: int) -> dict:
    high_daily, low_daily = _daily_rates(high), _daily_rates(low)
    if high_daily.empty or low_daily.empty:
        raise ValueError("empty exposure bucket")
    high_rates, low_rates = _rates(high_daily), _rates(low_daily)
    rng = np.random.default_rng(seed)
    simulations = {key: [] for key in high_rates}
    for _ in range(draws):
        hs = high_daily.iloc[rng.integers(0, len(high_daily), len(high_daily))]
        ls = low_daily.iloc[rng.integers(0, len(low_daily), len(low_daily))]
        hr, lr = _rates(hs), _rates(ls)
        for key in simulations:
            difference = hr[key] - lr[key]
            if np.isfinite(difference):
                simulations[key].append(difference)
    output = {"comparison": "HIGH_PRESSURE_MINUS_LOW_PRESSURE", "date_clustered": True}
    for key in simulations:
        estimate = high_rates[key] - low_rates[key]
        values = simulations[key]
        output[key + "_difference"] = None if not np.isfinite(estimate) else float(estimate)
        output[key + "_ci95"] = (None if not values else
            [float(x) for x in np.quantile(values, [.025, .975])])
    return output


def build(candidates: pd.DataFrame, outcomes: pd.DataFrame, contract: dict,
          split_fraction: float = .70, bootstrap_draws: int = 2000
          ) -> tuple[dict, pd.DataFrame]:
    if not 0 < split_fraction < 1:
        raise ValueError("split_fraction must be between zero and one")
    if bootstrap_draws < 100:
        raise ValueError("bootstrap_draws must be at least 100")
    frame = _prepare(candidates, outcomes, contract)
    dates = sorted(frame["signal_date"].unique())
    if len(dates) < 200:
        raise ValueError("at least 200 both-lane sessions required")
    split_index = max(1, min(len(dates) - 1, int(len(dates) * split_fraction)))
    split_date = str(dates[split_index - 1])
    development = frame[frame["signal_date"] <= split_date].copy()
    validation = frame[frame["signal_date"] > split_date].copy()
    daily_development = development[[
        "signal_date", "first_supply", "relay_supply", "relay_share"
    ]].drop_duplicates("signal_date")
    relay_share_low, relay_share_high = [float(x) for x in
        daily_development["relay_share"].quantile([.25, .75])]
    first_supply_low, first_supply_high = [float(x) for x in
        daily_development["first_supply"].quantile([.25, .75])]

    summaries, comparisons = {}, {}
    for offset, (period, segment) in enumerate((
            ("development", development), ("validation", validation))):
        first = segment[segment["board_group"].eq("FIRST_BOARD")]
        relay = segment[segment["board_group"].eq("CONSECUTIVE_BOARD")]
        first_low = first[first["relay_share"] <= relay_share_low]
        first_high = first[first["relay_share"] >= relay_share_high]
        relay_low = relay[relay["first_supply"] <= first_supply_low]
        relay_high = relay[relay["first_supply"] >= first_supply_high]
        summaries[period] = {
            "FIRST_BOARD_UNDER_RELAY_PRESSURE": {
                "LOW": _summary(first_low), "HIGH": _summary(first_high)},
            "RELAY_UNDER_FIRST_BOARD_SUPPLY": {
                "LOW": _summary(relay_low), "HIGH": _summary(relay_high)},
        }
        comparisons[period] = {
            "FIRST_BOARD_UNDER_RELAY_PRESSURE": _clustered_difference(
                first_high, first_low, bootstrap_draws, 19000 + offset * 2),
            "RELAY_UNDER_FIRST_BOARD_SUPPLY": _clustered_difference(
                relay_high, relay_low, bootstrap_draws, 19001 + offset * 2),
        }

    def stable_negative(path: str, metric: str) -> bool:
        for period in ("development", "validation"):
            item = comparisons[period][path]
            estimate, interval = item[metric + "_difference"], item[metric + "_ci95"]
            if estimate is None or interval is None or estimate >= 0 or interval[1] >= 0:
                return False
        return True

    first_suppression = stable_negative(
        "FIRST_BOARD_UNDER_RELAY_PRESSURE", "buyable_b2_positive_rate")
    relay_suppression = stable_negative(
        "RELAY_UNDER_FIRST_BOARD_SUPPLY", "buyable_b2_positive_rate")
    report = {
        "version": VERSION, "status": "DIAGNOSTIC_ONLY",
        "target_contract": {
            "exposure_uses_same_day_pool_only": True,
            "next_session_fields_are_labels_only": True,
            "no_touch_rows_retained_as_negative": True,
            "one_price_rows_retained_as_negative": True,
            "actual_fills": 0, "complete_trades": 0,
        },
        "scope": {
            "rows": int(len(frame)), "both_lane_sessions": int(len(dates)),
            "first_board_rows": int(frame["board_group"].eq("FIRST_BOARD").sum()),
            "relay_rows": int(frame["board_group"].eq("CONSECUTIVE_BOARD").sum()),
        },
        "split": {
            "method": "chronological_70_30_by_signal_date",
            "split_date": split_date,
            "development_sessions": int(development["signal_date"].nunique()),
            "validation_sessions": int(validation["signal_date"].nunique()),
            "validation_previously_inspected": True,
        },
        "frozen_thresholds_from_development": {
            "low_relay_share_max": relay_share_low,
            "high_relay_share_at_least": relay_share_high,
            "low_first_board_supply_max": first_supply_low,
            "high_first_board_supply_at_least": first_supply_high,
        },
        "summaries": summaries,
        "date_clustered_high_minus_low": comparisons,
        "stable_first_board_b2_suppression_by_relay_pressure": first_suppression,
        "stable_relay_b2_suppression_by_first_board_supply": relay_suppression,
        "governance": {
            "common_first_relay_winner_score_allowed": False,
            "market_lane_pressure_may_name_a_stock_winner": False,
            "market_lane_pressure_may_replace_specific_theme_evidence": False,
            "direct_stock_reranking_allowed": False,
            "entry_authorized": False,
        },
        "interpretation": (
            "First boards and relays occupy different attention lanes. Same-day lane supply "
            "can describe crowding pressure, but only a stable development-and-validation "
            "effect could justify retaining it as a market context factor. It cannot identify "
            "which specific theme or stock won the race."),
        "limitations": [
            "no historical point-in-time specific-theme identities",
            "daily proxy without auction queue or verified fills",
            "current-universe survivorship bias and incomplete historical eligibility",
            "validation dates were inspected and are not sealed blind",
        ],
    }
    observations = frame[[
        "code", "signal_date", "boards", "board_group", "first_supply",
        "relay_supply", "relay_share", "max_board", "promotion", "touchable",
        "one_price", "b1_big_loss", "b2_target_known", "b2_positive",
        "b3_target_known", "b3_positive",
    ]].copy()
    return report, observations


def run(candidates_path: str | Path, outcomes_path: str | Path,
        contract_path: str | Path, output_dir: str | Path,
        bootstrap_draws: int = 2000) -> dict:
    paths = [Path(candidates_path), Path(outcomes_path), Path(contract_path)]
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable V190 audit")
    report, observations = build(
        pd.read_csv(paths[0], dtype={"code": str}),
        pd.read_csv(paths[1], dtype={"code": str}),
        json.loads(paths[2].read_text(encoding="utf-8")),
        bootstrap_draws=bootstrap_draws,
    )
    report["source_hashes"] = {
        str(path.resolve()): _sha256(path) for path in [*paths, Path(__file__)]}
    target.mkdir(parents=True, exist_ok=False)
    observations.to_csv(target / "lane_attention_observations.csv", index=False)
    report["observations_sha256"] = _sha256(
        target / "lane_attention_observations.csv")
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
    print(json.dumps(run(args.candidates, args.outcomes,
                         args.corrected_contract, args.output,
                         args.bootstrap_draws), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
