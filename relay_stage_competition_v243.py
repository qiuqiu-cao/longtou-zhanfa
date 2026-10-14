"""Evaluate V242 expectations through auction, open, attack and seal stages.

The winner is determined by the highest resulting board that actually confirms
by 10:00, not by raw auction gap or first-touch time alone.  A 3-board name can
take over only when every 4-board candidate fails confirmation.  Market-wide
high/middle negative feedback converts any survivor into a retreat survivor,
not a proven core switch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


VERSION = "v243-relay-stage-competition-1"
TZ = ZoneInfo("Asia/Shanghai")
VERIFIED = {"VERIFIED", "PRIMARY_VERIFIED", "OFFICIAL_VERIFIED"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _truthy(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() in {
        "1", "true", "yes", "y"
    }


def _number(value: Any, field: str) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        raise ValueError(f"missing numeric field: {field}")
    return float(parsed)


def _dt(value: str) -> datetime:
    stamp = datetime.fromisoformat(str(value))
    if stamp.tzinfo is None:
        raise ValueError("timestamp must include timezone")
    return stamp.astimezone(TZ)


def _time_leq(value: Any, boundary: str) -> bool:
    text = str(value or "").strip()
    return bool(text) and text <= boundary


def _candidate(row: pd.Series, decision_date: str) -> dict:
    if str(row.trade_date) != decision_date:
        raise ValueError("candidate trade_date mismatch")
    if _dt(row.available_at) > _dt(f"{decision_date}T10:00:59+08:00"):
        raise ValueError("candidate evidence arrived after 10:00 decision cutoff")
    source_verified = str(row.source_status).strip().upper() in VERIFIED
    gap = _number(row.auction_gap_pct, "auction_gap_pct")
    amount_ratio = _number(row.auction_amount_ratio_pct, "auction_amount_ratio_pct")
    decay = _number(row.match_amount_decay_proxy, "match_amount_decay_proxy")
    failed = int(_number(row.failed_seal_episodes_before_1000,
                         "failed_seal_episodes_before_1000"))
    sync = _number(row.theme_sync_ratio_1000, "theme_sync_ratio_1000")
    coverage = _number(row.theme_coverage_ratio_1000, "theme_coverage_ratio_1000")
    peers = int(_number(row.theme_peer_count_1000, "theme_peer_count_1000"))
    if not 0 <= decay <= 1 or not 0 <= sync <= 1 or not 0 <= coverage <= 1:
        raise ValueError("decay, sync and coverage must be within [0,1]")

    low = None if pd.isna(row.normal_gap_low) else float(row.normal_gap_low)
    high = None if pd.isna(row.normal_gap_high) else float(row.normal_gap_high)
    expectation_type = str(row.expectation_type)
    reclaimed = _truthy(row.reclaimed_previous_close_0935)
    above_vwap = _truthy(row.above_vwap_0935)
    vwap_slope = _truthy(row.vwap_slope_positive_0935)
    sealed_935 = _truthy(row.sealed_at_0935)
    sealed_1000 = _truthy(row.sealed_at_1000)
    if low is None or high is None:
        auction_class = "PRICE_DISCOVERY_NO_NUMERIC_EXPECTATION"
        auction_acceptable = False
        recovered_from_under = False
        overheat_absorbed = False
    elif gap < low:
        auction_class = "UNDER_EXPECTATION"
        recovered_from_under = reclaimed and above_vwap and vwap_slope and sealed_1000
        auction_acceptable = recovered_from_under
        overheat_absorbed = False
    elif gap > high:
        auction_class = "ABOVE_EXPECTATION_POSSIBLE_OVERDRAFT"
        overheat_absorbed = amount_ratio >= 3 and decay <= .30 and sealed_935
        auction_acceptable = overheat_absorbed
        recovered_from_under = False
    else:
        auction_class = "WITHIN_OWN_EXPECTATION"
        auction_acceptable = True
        recovered_from_under = False
        overheat_absorbed = False

    price_discovery = expectation_type in {
        "FIRST_OPEN_PRICE_DISCOVERY", "SUPPLY_RELEASE_WATCH_ONLY"}
    open_confirmed = (
        above_vwap and vwap_slope and failed < 2
        and (auction_acceptable or price_discovery)
    )
    attack_confirmed = (
        _time_leq(row.first_limit_touch_time, "10:00")
        and _time_leq(row.first_seal_time, "10:00") and sealed_1000
    )
    ladder_confirmed = (
        coverage >= .7 and peers >= 2 and sync >= .6
        and (_truthy(row.candidate_led_theme_by_time_order)
             or _truthy(row.theme_strengthened_after_candidate))
    )
    identity_confirmed = source_verified and open_confirmed and attack_confirmed and ladder_confirmed
    tradable_confirmed = identity_confirmed and _truthy(row.trade_expectation_allowed)
    failure = []
    if not source_verified: failure.append("SOURCE_NOT_VERIFIED")
    if not auction_acceptable and not price_discovery: failure.append("AUCTION_EXPECTATION_NOT_RECOVERED")
    if not above_vwap: failure.append("BELOW_VWAP")
    if not vwap_slope: failure.append("VWAP_SLOPE_NOT_POSITIVE")
    if failed >= 2: failure.append("TWO_OR_MORE_FAILED_SEALS")
    if not attack_confirmed: failure.append("NO_CONFIRMED_ATTACK_AND_SEAL_BY_1000")
    if not ladder_confirmed: failure.append("THEME_LEAD_OR_LADDER_NOT_CONFIRMED")
    if identity_confirmed:
        stage = "1000_SEAL_AND_THEME_LEAD_CONFIRMED"
    elif open_confirmed and attack_confirmed:
        stage = "1000_INDIVIDUAL_SEAL_WITHOUT_THEME_LEAD"
    elif open_confirmed:
        stage = "0935_OPEN_CONFIRMED_ONLY"
    elif auction_acceptable:
        stage = "0925_AUCTION_ONLY"
    else:
        stage = "EXPECTATION_FAILED"
    return {
        "source_verified": source_verified, "auction_gap_pct": gap,
        "auction_amount_ratio_pct": amount_ratio, "match_amount_decay_proxy": decay,
        "auction_class": auction_class, "recovered_from_under_expectation": recovered_from_under,
        "overheat_absorbed": overheat_absorbed, "open_confirmed_0935": open_confirmed,
        "attack_confirmed_1000": attack_confirmed, "ladder_confirmed_1000": ladder_confirmed,
        "identity_confirmed_1000": identity_confirmed,
        "tradable_confirmed_1000": tradable_confirmed,
        "stage_at_1000": stage, "failure_reasons": ";".join(failure),
        "time_order_is_causation": False,
    }


def evaluate(matrix: pd.DataFrame, v242: dict, observations: pd.DataFrame | None,
             market: dict | None, decision_date: str) -> tuple[pd.DataFrame, dict]:
    matrix = matrix.copy(); matrix["code"] = matrix.code.astype(str).str.zfill(6)
    codes = set(matrix.code)
    result = {
        "version": VERSION, "signal_date": v242.get("signal_date"),
        "decision_date": decision_date, "full_denominator": len(matrix),
        "decision": "DATA_UNAVAILABLE", "primary_expectation_status": "UNAVAILABLE",
        "highest_identity_winners": [], "next_winner_after_primary_failure": None,
        "entry_authorized": False, "replacement_trade_allowed": False,
        "level2_used": False,
    }
    if observations is None or observations.empty or market is None:
        result["missing_codes"] = sorted(codes)
        result["missing_market_context"] = market is None
        return pd.DataFrame(), result
    observations = observations.copy(); observations["code"] = observations.code.astype(str).str.zfill(6)
    if observations.code.duplicated().any():
        raise ValueError("duplicate candidate observations")
    observed = set(observations.code)
    if observed != codes:
        result["decision"] = "DATA_UNAVAILABLE_INCOMPLETE_3_4_BOARD_DENOMINATOR"
        result["missing_codes"] = sorted(codes - observed)
        result["extra_codes"] = sorted(observed - codes)
        return pd.DataFrame(), result
    if set(observations.signal_date.astype(str)) != {str(v242.get("signal_date"))}:
        raise ValueError("observation signal_date mismatch")
    if str(market.get("trade_date")) != decision_date:
        raise ValueError("market context trade_date mismatch")
    if _dt(market["available_at"]) > _dt(f"{decision_date}T10:00:59+08:00"):
        raise ValueError("market context arrived after decision cutoff")
    market_verified = str(market.get("source_status", "")).strip().upper() in VERIFIED
    systemic_negative = market_verified and (
        _truthy(market.get("height_anchor_severe_negative"))
        and (_truthy(market.get("middle_board_batch_negative"))
             or _truthy(market.get("previous_core_batch_negative")))
    )
    evaluations = [{"code": row.code, **_candidate(row, decision_date)}
                   for _, row in observations.merge(
                       matrix[["code", "normal_gap_low", "normal_gap_high",
                               "expectation_type", "trade_expectation_allowed"]],
                       on="code", validate="one_to_one").iterrows()]
    evaluated = matrix.merge(pd.DataFrame(evaluations), on="code", validate="one_to_one")
    confirmed = evaluated[evaluated.identity_confirmed_1000].copy()
    primary_obj = v242.get("primary_close_expectation") or {}
    primary_code = str(primary_obj.get("code", "")).zfill(6)
    primary_row = evaluated[evaluated.code.eq(primary_code)]
    primary_confirmed = bool(len(primary_row) == 1 and primary_row.iloc[0].identity_confirmed_1000)
    result["primary_expectation_status"] = (
        "CONFIRMED" if primary_confirmed else "FAILED_OR_UNCONFIRMED")
    if not primary_row.empty:
        result["primary_failure_reasons"] = str(primary_row.iloc[0].failure_reasons)
    result["market_gate"] = {
        "source_verified": market_verified, "systemic_negative": systemic_negative,
        "height_anchor_severe_negative": _truthy(market.get("height_anchor_severe_negative")),
        "middle_board_batch_negative": _truthy(market.get("middle_board_batch_negative")),
        "previous_core_batch_negative": _truthy(market.get("previous_core_batch_negative")),
    }
    if confirmed.empty:
        result["decision"] = "NO_WINNER_RETREAT_CASH"
        return evaluated, result
    highest_start = int(confirmed.boards.max())
    highest = confirmed[confirmed.boards.eq(highest_start)].sort_values(
        ["winner_hierarchy_rank", "code"], na_position="last")
    result["highest_identity_winners"] = [
        {"code": row.code, "name": row["name"], "starting_boards": int(row.boards),
         "resulting_boards": int(row.boards) + 1,
         "trade_expectation_allowed": _truthy(row.trade_expectation_allowed)}
        for _, row in highest.iterrows()
    ]
    if not primary_confirmed:
        first = highest.iloc[0]
        result["next_winner_after_primary_failure"] = {
            "code": first.code, "name": first["name"],
            "unique": len(highest) == 1,
            "observation_only_not_replacement_trade": True,
        }
    if not market_verified:
        result["decision"] = "MARKET_CONTEXT_UNVERIFIED_IDENTITY_ONLY"
        result["tradable_winner"] = None
    elif systemic_negative:
        result["decision"] = "RETREAT_SURVIVOR_NOT_PROVEN_CORE_SWITCH"
        result["tradable_winner"] = None
    elif len(highest) > 1:
        result["decision"] = "SAME_HEIGHT_CO_WINNERS_NO_UNIQUE_LEADER"
        result["tradable_winner"] = None
    else:
        winner = highest.iloc[0]
        if int(winner.boards) == 3 and not bool(
                evaluated[evaluated.boards.eq(4)].identity_confirmed_1000.any()):
            decision = "UNIQUE_3_TO_4_CROSS_HEIGHT_TAKEOVER"
        elif winner.code == primary_code:
            decision = "PRIMARY_PATH_EXPECTATION_CONFIRMED"
        else:
            decision = "UNIQUE_SAME_HEIGHT_RIVAL_WINNER"
        result["decision"] = decision
        result["tradable_winner"] = (
            {"code": winner.code, "name": winner["name"],
             "paper_confirmation_only": True}
            if _truthy(winner.tradable_confirmed_1000) else None)
    return evaluated, result


def run(matrix_path: str, report_path: str, decision_date: str, output_dir: str,
        observations_path: str | None = None, market_path: str | None = None) -> dict:
    matrix_file, report_file = Path(matrix_path), Path(report_path)
    v242 = json.loads(report_file.read_text(encoding="utf-8-sig"))
    if v242.get("version") != "v242-relay-path-expectation-1":
        raise ValueError("V242 report version mismatch")
    if v242.get("artifact_hashes", {}).get(matrix_file.name) != _sha(matrix_file):
        raise ValueError("V242 matrix hash mismatch")
    matrix = pd.read_csv(matrix_file, dtype={"code": str})
    observations = pd.read_csv(observations_path, dtype={"code": str}) if observations_path else None
    market = json.loads(Path(market_path).read_text(encoding="utf-8-sig")) if market_path else None
    evaluated, result = evaluate(matrix, v242, observations, market, decision_date)
    sources = [matrix_file, report_file]
    if observations_path: sources.append(Path(observations_path))
    if market_path: sources.append(Path(market_path))
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable V243 output")
    target.mkdir(parents=True)
    detail = target / "stage_competition.csv"
    evaluated.to_csv(detail, index=False, encoding="utf-8-sig")
    result["source_hashes"] = {str(path.resolve()): _sha(path) for path in sources}
    result["artifact_hashes"] = {detail.name: _sha(detail)}
    (target / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", required=True); ap.add_argument("--report", required=True)
    ap.add_argument("--decision-date", required=True); ap.add_argument("--observations")
    ap.add_argument("--market"); ap.add_argument("--output", required=True)
    ns = ap.parse_args()
    print(json.dumps(run(ns.matrix, ns.report, ns.decision_date, ns.output,
                         ns.observations, ns.market), ensure_ascii=False, indent=2))
