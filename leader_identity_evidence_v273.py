"""Deterministic leader-identity transitions from timestamped evidence.

Identity does not manufacture a next-board probability or authorize a purchase.
Missing evidence stays unknown. Highest board is only a space/emotion anchor.
"""
from __future__ import annotations

from next_session_expectation_contract_v271 import stamp

VERSION = "v273-evidence-leader-identity-1"
FACTS = {"theme_verified", "sector_support", "first_board_turnover_verified",
         "same_height_active_lead", "genuine_divergence", "absorption_verified",
         "reseal_verified", "competition_denominator_complete", "all_rivals_failed",
         "old_cycle_dependency", "old_leader_stable", "independent_catalyst",
         "old_cycle_loss_cleared", "new_ladder_persistent", "crossed_old_height",
         "fundamental_hard_risk", "severe_market_contagion"}


def judge(boards, is_highest, evidence, asof):
    if not isinstance(boards, int) or isinstance(boards, bool) or boards < 1:
        raise ValueError("positive integer board count required")
    if not isinstance(is_highest, bool):
        raise ValueError("highest-board flag must be boolean")
    cutoff = stamp(asof)
    facts = {}
    for item in evidence:
        key = item["fact"]
        if key not in FACTS or key in facts:
            raise ValueError("unknown or duplicate identity evidence")
        observed, available = stamp(item["observed_at"]), stamp(item["available_at"])
        if observed > available or available > cutoff:
            raise ValueError("identity evidence not available at decision time")
        if not item.get("source") or item.get("status") not in {"verified", "inferred", "unavailable"}:
            raise ValueError("evidence source and status required")
        if item.get("value") is not None and not isinstance(item["value"], bool):
            raise ValueError("identity fact must be true/false/unknown")
        facts[key] = item["value"] if item["status"] == "verified" else None

    def passed(keys):
        return all(facts.get(k) is True for k in keys)

    required_seed = ["theme_verified", "sector_support", "first_board_turnover_verified", "same_height_active_lead"]
    required_core = required_seed + ["genuine_divergence", "absorption_verified", "reseal_verified"]
    missing = [k for k in (required_seed if boards == 2 else required_core) if facts.get(k) is None]
    hard_risk = facts.get("fundamental_hard_risk") is True or facts.get("severe_market_contagion") is True
    role = "SPACE_ANCHOR" if is_highest else "CHALLENGER"
    if boards == 1:
        role = "START_CANDIDATE"
    elif boards == 2 and passed(required_seed):
        role = "LEADER_SEED"
    elif boards >= 3 and passed(required_core):
        role = "TURNOVER_CORE"
        if passed(["competition_denominator_complete", "all_rivals_failed"]):
            role = "POSITION_WINNER"
        if passed(["old_cycle_dependency", "old_leader_stable"]):
            role = "SUPPLEMENT"
        elif passed(["independent_catalyst", "old_cycle_loss_cleared", "new_ladder_persistent", "crossed_old_height"]):
            role = "NEW_CYCLE_LEADER_CONFIRMED"
    return {"version": VERSION, "identity": role, "missing_facts": missing,
            "hard_risk": hard_risk, "cycle_status": "NEW_CYCLE_CONFIRMED" if role == "NEW_CYCLE_LEADER_CONFIRMED" else "UNCONFIRMED",
            "competition_status": "POSITION_CONFIRMED" if passed(["competition_denominator_complete", "all_rivals_failed"]) else "COMPETING_OR_UNKNOWN",
            "evidence_status": "VERIFIED_REQUIRED_FACTS" if not missing else "INCOMPLETE",
            "decision": "CANCELLED" if hard_risk else "WATCH",
            "next_board_probability": None, "trade_authorized": False,
            "identity_invalidation": ["theme/core support fails", "rival remains stronger", "absorbed divergence not verified", "hard risk or market contagion"]}
