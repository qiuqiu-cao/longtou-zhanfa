"""Map distinct leader roles without turning research ratings into buy signals.

The mapper is intentionally deterministic.  It consumes a verified complete
closing limit-up pool and separates:

* an emotion anchor (observed height/market influence),
* a 100%-severe-abnormal *path* research candidate, and
* a same-specific-theme high-to-low supplement candidate.

No output is a calibrated probability or an execution instruction.  Missing
theme, official-rule, scenario, or full-pool evidence fails closed.
"""
from __future__ import annotations

import math

from leader_target_screen_v17 import timestamp


IDENTITY_PRIORITY = {
    "height_anchor": 0,
    "unique_survivor": 1,
    "pioneer": 2,
    "capacity_core": 3,
    "supplement": 4,
}
HIGH_CORE_IDENTITIES = {"height_anchor", "unique_survivor", "pioneer", "capacity_core"}
THEME_STATES = {"EXPANDING", "STABLE", "DIVERGING", "COLLAPSED"}
RISK_BANDS = {"LOW", "MIXED", "HIGH"}


def _finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _valid_candidate(row):
    boards = _finite(row.get("boards"))
    return (
        bool(row.get("candidate_id"))
        and str(row.get("code", "")).zfill(6).isdigit()
        and boards is not None and boards >= 1 and boards.is_integer()
        and row.get("pool_assignment") == "CLOSING_LIMIT_UP"
        and row.get("is_mainboard") is True
        and row.get("is_st") is False
        and row.get("is_delisting_period") is False
        and row.get("security_status") == "verified"
        and row.get("limit_up_status") == "verified"
        and row.get("identity_status") == "verified"
    )


def _theme_valid(row):
    return (
        row.get("theme_status") in {
            "verified_same_day_group", "verified_explicit_source_roster"
        }
        and row.get("theme_granularity") == "specific"
        and bool(row.get("theme_group_id"))
        and bool(row.get("theme"))
        and row.get("theme_state") in THEME_STATES
    )


def _role_view(row, role, rationale):
    return {
        "role": role,
        "candidate_id": row["candidate_id"],
        "code": str(row["code"]).zfill(6),
        "name": row.get("name"),
        "boards": int(float(row["boards"])),
        "theme": row.get("theme"),
        "theme_group_id": row.get("theme_group_id"),
        "competition_identity": row.get("competition_identity"),
        "tradability_observation": row.get("tradability"),
        "rationale": rationale,
        "calibrated_probability": None,
        "buy_signal": False,
    }


def _rank_low_supplements(normalized, high):
    """Return eligible same-specific-theme low candidates for one high core."""
    lows = []
    for row in normalized:
        support = _finite(row.get("theme_support_ex_self"))
        rivals = _finite(row.get("same_height_peer_count"))
        if (
            row["candidate_id"] != high["candidate_id"]
            and row["theme_group_id"] == high["theme_group_id"]
            and _theme_valid(row)
            and row["theme_state"] != "COLLAPSED"
            and row["boards"] < high["boards"] and row["boards"] <= 2
            and row.get("tradability") == "buyable"
            and row.get("one_price_board") is False
            and support is not None and support >= 1
            and rivals is not None and rivals >= 0
            and row.get("competition_identity") in {
                "unique_survivor", "pioneer", "capacity_core"
            }
        ):
            lows.append((
                IDENTITY_PRIORITY[row["competition_identity"]],
                -support,
                rivals,
                -row["boards"],
                row["candidate_id"],
                row,
            ))
    lows.sort(key=lambda item: item[:-1])
    return [item[-1] for item in lows]


def _high_low_mappings(normalized):
    """Create an explicit mapping record for every identified high-position core."""
    highs = [
        row for row in normalized
        if row["boards"] >= 3 and (
            row.get("is_emotion_anchor") is True
            or row.get("competition_identity") in HIGH_CORE_IDENTITIES
        )
    ]
    highs.sort(key=lambda row: (
        -row["boards"], IDENTITY_PRIORITY.get(row.get("competition_identity"), 99),
        row["candidate_id"],
    ))
    mappings = []
    for high in highs:
        high_view = _role_view(
            high, "HIGH_POSITION_CORE",
            ["identified high-position closing-limit core requiring an explicit low mapping"],
        )
        blockers = []
        if not _theme_valid(high):
            blockers.append("HIGH_CORE_SPECIFIC_THEME_NOT_VERIFIED")
        elif high["theme_state"] == "COLLAPSED":
            blockers.append("HIGH_CORE_THEME_COLLAPSED")
        lows = [] if blockers else _rank_low_supplements(normalized, high)
        if not lows and not blockers:
            blockers.append("NO_ELIGIBLE_SAME_THEME_LOW_CORE")
        low_view = None
        if lows:
            low = lows[0]
            low_view = _role_view(
                low,
                "HIGH_TO_LOW_SAME_THEME_SUPPLEMENT",
                [
                    f"mapped from high-position core {high['candidate_id']}",
                    "same verified specific theme rather than broad-concept association",
                    "low-position identity is pioneer/core, buyable at observation, and not one-price",
                    "mapping is invalid if the high core or specific theme collapses next session",
                ],
            )
            low_view["mapped_anchor_id"] = high["candidate_id"]
        mappings.append({
            "high_core": high_view,
            "mapping_status": "MAPPED_RESEARCH_ONLY" if low_view else "NO_ELIGIBLE_MAPPING",
            "supplement_candidate": low_view,
            "blocking_reasons": blockers,
            "capital_migration_hypothesis": (
                "same-specific-theme capital may rotate from the high core only if theme and "
                "profit ecology remain alive; this is not inferred from low board count alone"
            ),
            "next_session_confirmation_required": [
                "high core risk rises without theme collapse",
                "mapped low outperforms same-theme low rivals with real turnover",
                "theme middle/capacity feedback does not deteriorate",
            ],
            "invalidation_conditions": [
                "high core and theme ladder collapse together",
                "mapped low loses same-theme competition",
                "one-price lock or no verifiable fill path",
            ],
            "buy_signal": False,
        })
    return mappings


def map_roles(payload, asof):
    cutoff = timestamp(asof)
    universe = payload.get("universe", {})
    emotion = payload.get("next_session_emotion_risk", {})
    rows = universe.get("candidates", [])
    issues = []

    signal_date = universe.get("signal_date")
    ids = [row.get("candidate_id") for row in rows]
    if (
        universe.get("status") != "VERIFIED_FULL_CLOSING_LIMIT_POOL"
        or universe.get("universe_complete") is not True
        or universe.get("closing_limit_count") != len(rows)
        or not rows or len(ids) != len(set(ids))
        or any(not _valid_candidate(row) for row in rows)
    ):
        issues.append("FULL_POOL_NOT_VERIFIED")
    try:
        observed = timestamp(emotion["observed_at"])
        available = timestamp(emotion["available_at"])
        emotion_valid = (
            emotion.get("model_status") == "DIAGNOSTIC_ONLY"
            and emotion.get("risk_band") in RISK_BANDS
            and emotion.get("policy_effect") == "RISK_ONLY_NO_POSITIVE_RANK_BONUS"
            and emotion.get("signal_date") == signal_date
            and observed <= available <= cutoff
            and emotion.get("data_status") in {"verified", "inferred"}
            and bool(emotion.get("source"))
        )
    except (KeyError, TypeError, ValueError):
        emotion_valid = False
    if not emotion_valid:
        issues.append("NEXT_SESSION_EMOTION_RISK_UNAVAILABLE")

    if issues:
        return {
            "status": "DATA_LIMITED",
            "model_status": "DIAGNOSTIC_ONLY",
            "blocking_reasons": sorted(set(issues)),
            "emotion_anchor": None,
            "hundred_percent_path_candidate": None,
            "high_low_supplement_candidate": None,
            "high_low_supplement_mappings": [],
            "buy_signal": False,
        }

    normalized = [dict(row, code=str(row["code"]).zfill(6), boards=int(float(row["boards"])))
                  for row in rows]
    anchors = [row for row in normalized if row.get("is_emotion_anchor") is True]
    anchors.sort(key=lambda row: (
        -row["boards"],
        IDENTITY_PRIORITY.get(row.get("competition_identity"), 99),
        row["candidate_id"],
    ))
    anchor = anchors[0] if anchors else None
    anchor_view = None if anchor is None else _role_view(
        anchor,
        "EMOTION_ANCHOR",
        ["verified closing height/influence identity", "not automatically a buy candidate"],
    )

    path_candidates = []
    for row in normalized:
        steps = _finite(row.get("scenario_limit_boards_to_100"))
        support = _finite(row.get("theme_support_ex_self"))
        rivals = _finite(row.get("same_height_peer_count"))
        distance = _finite(row.get("distance_100_pp"))
        if (
            row["boards"] >= 2
            and _theme_valid(row)
            and row["theme_state"] != "COLLAPSED"
            and row.get("abnormal_rule_status") == "verified"
            and row.get("indicator_status") == "verified"
            and row.get("scenario_status") == "verified_inputs_counterfactual"
            and row.get("target_status") == "TARGET_RESEARCH_CANDIDATE"
            and steps is not None and steps.is_integer() and 1 <= steps <= 3
            and distance is not None and distance > 0
            and support is not None and support >= 1
            and rivals is not None and rivals >= 0
            and row.get("competition_identity") in IDENTITY_PRIORITY
        ):
            path_candidates.append((
                IDENTITY_PRIORITY[row["competition_identity"]],
                -support,
                int(steps),
                rivals,
                -row["boards"],
                row["candidate_id"],
                row,
            ))
    path_candidates.sort(key=lambda item: item[:-1])
    path = path_candidates[0][-1] if path_candidates else None
    path_view = None
    if path is not None:
        path_view = _role_view(
            path,
            "HUNDRED_PERCENT_ABNORMAL_PATH_POTENTIAL",
            [
                "official rule and current distance are verified",
                "specific theme has at least one excluding-self support stock",
                "counterfactual limit-up path reaches threshold within three sessions",
                "competition identity and theme support rank before distance",
            ],
        )
        path_view.update({
            "distance_100_pp": float(path["distance_100_pp"]),
            "scenario_limit_boards_to_100": int(float(path["scenario_limit_boards_to_100"])),
            "interpretation": "scenario_path_not_hit_probability_and_not_buy_reason",
        })

    mappings = _high_low_mappings(normalized)
    anchor_mapping = next((
        item for item in mappings
        if anchor is not None
        and item["high_core"]["candidate_id"] == anchor["candidate_id"]
    ), None)
    low_view = anchor_mapping["supplement_candidate"] if anchor_mapping else None

    risk_band = emotion["risk_band"]
    return {
        "status": "ROLE_RESEARCH_READY",
        "model_status": "DIAGNOSTIC_ONLY",
        "asof": cutoff.isoformat(),
        "signal_date": signal_date,
        "next_session_risk_band": risk_band,
        "selection_effect": (
            "HARDEN_CONFIRMATION_OR_CASH" if risk_band == "HIGH"
            else "NO_POSITIVE_BONUS" if risk_band == "LOW"
            else "REQUIRE_ROTATION_AND_THEME_CONFIRMATION"
        ),
        "emotion_anchor": anchor_view,
        "hundred_percent_path_candidate": path_view,
        "high_low_supplement_candidate": low_view,
        "high_low_supplement_mappings": mappings,
        "high_low_mapping_coverage": {
            "identified_high_cores": len(mappings),
            "mapped_high_cores": sum(item["supplement_candidate"] is not None for item in mappings),
            "unmapped_high_cores": sum(item["supplement_candidate"] is None for item in mappings),
        },
        "role_distinction": {
            "emotion_anchor_is_buy": False,
            "hundred_percent_path_is_buy": False,
            "high_low_mapping_is_buy": False,
        },
        "ranking_rule": (
            "verified competition identity, theme support, scenario reachability, rivalry, height; "
            "never distance alone"
        ),
        "calibrated_probability": None,
        "buy_signal": False,
        "blocking_reasons": [],
    }
