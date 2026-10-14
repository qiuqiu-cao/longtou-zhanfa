#!/usr/bin/env python3
"""Freeze one machine-readable next-session shadow plan without authorizing an order."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo


SHANGHAI = ZoneInfo("Asia/Shanghai")
VERSION = "v18-shadow-conditional-plan-1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def next_limit(previous_close: float) -> float:
    return float((Decimal(str(previous_close)) * Decimal("1.10")).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP))


def freeze(review: dict, code: str, previous_close: float, trade_date: str,
           paper_shares: int = 100) -> dict:
    code = str(code).zfill(6)
    if paper_shares <= 0 or paper_shares % 100:
        raise ValueError("paper shares must be a positive 100-share lot")
    candidates = list(review.get("first_board_candidates", [])) + list(review.get("relay_candidates", []))
    matches = [row for row in candidates if str(row.get("code")).zfill(6) == code]
    if len(matches) != 1:
        raise ValueError("shadow primary must exist exactly once in the full candidate pool")
    selected = matches[0]
    if selected.get("board_group") != "CONSECUTIVE_BOARD" or not 2 <= int(selected["boards"]) <= 5:
        raise ValueError("V18 shadow relay must be a 2-5 board candidate")
    if selected.get("active_hard_risks"):
        raise ValueError("active hard-risk candidate cannot be frozen as shadow primary")
    if selected.get("theme_status") != "verified_same_day_group":
        raise ValueError("shadow primary requires source-bound same-day theme evidence")
    result = deepcopy(review)
    for row in list(result.get("first_board_candidates", [])) + list(result.get("relay_candidates", [])):
        if str(row.get("code")).zfill(6) != code:
            continue
        row["auction_conditions"] = [
            {"field": "quote_data_status", "op": "eq", "value": "VERIFIED_PUBLIC_QUOTE_FRESH"},
            {"field": "auction_gap_proxy", "op": "between", "lower": 0.00, "upper": 0.07},
            {"field": "public_level1_buyability", "op": "not_in",
             "values": ["UNBUYABLE_ONE_WORD_PROXY", "UNAVAILABLE"]},
            {"field": "agriculture_positive_peer_count", "op": "gte", "value": 2,
             "peer_codes": ["600108", "601952", "002470", "002312", "603077"]},
        ]
        row["opening_conditions"] = [
            {"field": "minute_data_status", "op": "eq", "value": "VERIFIED_PUBLIC_MINUTE_BAR_FRESH"},
            {"field": "close_to_session_vwap_proxy", "op": "gte", "value": 0.0},
            {"field": "vwap_slope_5m_proxy", "op": "gt", "value": 0.0},
            {"field": "failed_seal_twice_proxy", "op": "eq", "value": False},
            {"field": "reseal_minute_proxy", "op": "between_time", "lower": "09:30", "upper": "10:00"},
        ]
        row["cancel_conditions"] = [
            {"field": "auction_gap_proxy", "op": "outside", "lower": 0.00, "upper": 0.07},
            {"field": "agriculture_positive_peer_count", "op": "lt", "value": 2},
            {"field": "below_vwap_expanding_proxy", "op": "eq", "value": True},
            {"field": "failed_seal_twice_proxy", "op": "eq", "value": True},
            {"field": "reseal_minute_proxy", "op": "after_or_missing", "value": "10:00"},
        ]
        row["entry_permission"] = "SHADOW_WATCH_ONLY"
        row["execution_state"] = "FROZEN_CONDITIONAL_SHADOW"
    price = next_limit(previous_close)
    result["status"] = "WATCH_SHADOW_ONLY"
    result["trade_date"] = trade_date
    result["selection"] = {
        "validated_threshold": None,
        "conditional_primary": {
            "code": code, "name": selected.get("name"), "boards": int(selected["boards"]),
            "plan_role": "RESEARCH_SHADOW_PRIMARY_NOT_DIRECT_RECOMMENDATION",
        },
        "reference_close": float(previous_close),
        "maximum_acceptable_price": price,
        "entry_style": "FIRST_OR_SECOND_RESEAL_AFTER_09_30_NO_LATER_THAN_10_00",
        "position_shares": paper_shares,
        "position_scope": "PAPER_OBSERVATION_UNIT_NOT_USER_ACCOUNT_POSITION",
        "holding_plan": {
            "B": "T+1 locked; no fictitious same-day stop",
            "B+1": "exit at earliest executable price if weak-open/VWAP/sector failure; otherwise reassess at close",
            "B+2": "hold only if close remains above entry after costs and theme/core identity remains intact",
            "B+3": "mandatory time exit by 14:50 unless a separately frozen extension rule exists",
            "sell_blocked": "mark to close and carry until first executable session; do not hide the loss",
        },
    }
    result["shadow_plan"] = {
        "version": VERSION,
        "frozen_at": datetime.now(SHANGHAI).isoformat(),
        "signal_date": review.get("signal_date"), "trade_date": trade_date,
        "candidate_code": code, "paper_only": True,
        "direct_entry_authorized": False, "actual_position_shares": 0,
        "no_level2_fill_assumption": "at-limit observation is not a fill without queue/broker evidence",
        "rules_frozen_before_trade_date": True,
    }
    result["direct_recommendation"] = False
    result["order_submitted"] = False
    result["position_created"] = False
    result["guarantee_of_premium"] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", required=True)
    parser.add_argument("--code", required=True)
    parser.add_argument("--previous-close", type=float, required=True)
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--paper-shares", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, target = Path(args.review), Path(args.output)
    if target.exists():
        raise FileExistsError("refusing to overwrite frozen shadow plan")
    payload = freeze(json.loads(source.read_text(encoding="utf-8")), args.code,
                     args.previous_close, args.trade_date, args.paper_shares)
    payload["shadow_plan"]["source_review"] = str(source.resolve())
    payload["shadow_plan"]["source_review_sha256"] = sha(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload_hash = sha(target)
    manifest = {"version": VERSION, "plan": str(target.resolve()), "plan_sha256": payload_hash,
                "paper_only": True, "direct_entry_authorized": False}
    target.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
