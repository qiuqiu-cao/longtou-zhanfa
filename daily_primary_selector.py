"""Select at most one conditional daily primary from a locked full pool.

This module never submits an order.  It fails closed when pool coverage, model
stack, emotion, opening confirmation, risk evidence, or sizing inputs are not
complete.  Failure of the predeclared rank-one candidate resolves to cash; it
does not promote an intraday backup.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from leader_target_screen_v17 import finite, timestamp


ACTIVE_REGIMES = {"启动", "升温", "加速", "修复"}
GROUPS = {"FIRST_BOARD", "CONSECUTIVE_BOARD"}
FALSE_STRENGTH_FLAGS = {
    "auction_match_decay", "high_open_without_amount", "theme_desync",
    "continuous_one_price_first_open", "explosive_volume_stall",
    "repeated_failed_seal", "late_weak_seal", "below_vwap_expanding",
}
EMOTION_DIMENSIONS = {"money_effect", "broken_loss_effect", "height_feedback",
                      "theme_divergence", "repair_quality"}
SAME_HEIGHT_FACTORS = {
    "turnover_absorption_quality", "theme_purity", "sector_leadership",
    "competitor_elimination", "announcement_soft_risk",
}
ANNOUNCEMENT_CLEAR_ASSURANCE = {
    "OFFICIAL_FULL_TEXT_VERIFIED",
    "OFFICIAL_INDEX_VERIFIED_NO_ROWS",
}


def _fresh(block, cutoff, trade_date):
    try:
        observed, available = timestamp(block["observed_at"]), timestamp(block["available_at"])
        max_age = finite(block["max_age_seconds"])
        return (max_age is not None and max_age >= 0 and observed <= available <= cutoff
                and observed.strftime("%Y-%m-%d") == trade_date
                and (cutoff - observed).total_seconds() <= max_age
                and block.get("data_status") == "verified" and bool(block.get("source")))
    except (KeyError, TypeError, ValueError):
        return False


def _risk_policy(policy, cutoff):
    try:
        frozen, available = timestamp(policy["frozen_at"]), timestamp(policy["available_at"])
        risk, gross, theme, stress = [finite(policy[key]) for key in
                                      ("single_trade_risk_fraction", "gross_exposure_fraction",
                                       "theme_exposure_fraction", "stress_loss_fraction")]
        equity, current_gross, current_theme = [finite(policy[key]) for key in
                                                ("account_equity", "existing_gross_exposure_cash",
                                                 "existing_theme_exposure_cash")]
        caps = policy["regime_position_caps"]
        max_adverse = policy["maximum_adverse_dimensions"]
        valid = (bool(policy.get("policy_id")) and policy.get("data_status") == "verified"
                 and bool(policy.get("source")) and frozen <= available <= cutoff
                 and all(value is not None for value in (risk, gross, theme, stress, equity, current_gross, current_theme))
                 and 0 < risk <= .02 and 0 < gross <= 1 and 0 < theme <= 1 and 0 < stress <= 1
                 and equity > 0 and 0 <= current_gross < equity and 0 <= current_theme < equity
                 and isinstance(caps, dict) and set(caps) == ACTIVE_REGIMES
                 and all(finite(value) is not None and 0 <= finite(value) <= 1 for value in caps.values())
                 and isinstance(max_adverse, int) and not isinstance(max_adverse, bool) and 0 <= max_adverse <= 2)
        if not valid:
            raise ValueError
        return {"risk": risk, "gross": gross, "theme": theme, "stress": stress,
                "equity": equity, "current_gross": current_gross, "current_theme": current_theme,
                "caps": {key: finite(value) for key, value in caps.items()}, "max_adverse": max_adverse}
    except (KeyError, TypeError, ValueError):
        return None


def _research_choice(rows, scores, group=None):
    """Expose a transparent rank-one research object, never a buy signal."""
    eligible = [row for row in rows if group is None or row["board_group"] == group]
    if not eligible:
        return None
    row = eligible[0]
    return {
        "candidate_id": row["candidate_id"], "code": row["code"],
        "name": row.get("name"), "boards": row["boards"],
        "board_group": row["board_group"],
        "rank_score": scores[row["candidate_id"]],
        "rank_score_meaning": "opportunity_proxy_score_not_complete_trade_probability",
        "status": "RESEARCH_RANK_ONE_REQUIRES_NEXT_SESSION_CONFIRMATION",
        "buy_signal": False,
    }


def _same_height_contract(block, candidate_map, signal_date, cutoff):
    """Validate exact group scope without allowing a heuristic champion."""
    issues, row_map = [], {}
    rows = block.get("rows", []) if isinstance(block, dict) else []
    if (block.get("status") != "FROZEN_POINT_IN_TIME_COMPLETE"
            or block.get("signal_date") != signal_date
            or block.get("seal_speed_or_zero_failed_seals_can_select_champion") is not False
            or block.get("factor_use_status") != "EVIDENCE_GATE_ONLY_NOT_POSITIVE_WEIGHT"):
        issues.append("SAME_HEIGHT_COMPARISON_CONTRACT_INVALID")
    ids = [row.get("candidate_id") for row in rows]
    if set(ids) != set(candidate_map) or len(ids) != len(set(ids)):
        issues.append("SAME_HEIGHT_COMPARISON_NOT_EXACT_FULL_POOL")
        return issues, row_map
    expected_groups = {}
    for candidate_id, candidate in candidate_map.items():
        group_id = f"{signal_date}:H{candidate['boards']}"
        expected_groups.setdefault(group_id, set()).add(candidate_id)
    for row in rows:
        candidate_id = row.get("candidate_id")
        candidate = candidate_map[candidate_id]
        group_id = f"{signal_date}:H{candidate['boards']}"
        members = row.get("group_member_ids")
        factor_status = row.get("factor_evidence_status", {})
        try:
            evidence_time_valid = timestamp(row["evidence_available_at"]) <= cutoff
        except (KeyError, TypeError, ValueError):
            evidence_time_valid = False
        valid = (
            row.get("height_group_id") == group_id
            and row.get("same_height_group_size") == len(expected_groups[group_id])
            and isinstance(members, list) and set(members) == expected_groups[group_id]
            and len(members) == len(set(members))
            and row.get("group_scope_complete") is True
            and row.get("five_factor_evidence_complete") is True
            and set(factor_status) == SAME_HEIGHT_FACTORS
            and all(value in {"VERIFIED", "POINT_IN_TIME_VERIFIED"}
                    for value in factor_status.values())
            and evidence_time_valid
            and row.get("official_champion") is False
        )
        if not valid:
            issues.append("SAME_HEIGHT_GROUP_OR_FIVE_FACTOR_EVIDENCE_INCOMPLETE")
        row_map[candidate_id] = row
    return issues, row_map


def _first_board_dual_target_contract(block, candidate_map, signal_date, cutoff):
    """Bind first boards to two independently validated target probabilities."""
    issues, row_map = [], {}
    rows = block.get("rows", []) if isinstance(block, dict) else []
    first_ids = {key for key, row in candidate_map.items()
                 if row["board_group"] == "FIRST_BOARD"}
    if (block.get("status") != "FORWARD_VALIDATED_DUAL_TARGET"
            or block.get("signal_date") != signal_date
            or block.get("targets_trained_separately") is not True
            or block.get("b2_target") != "EXECUTABLE_NET_RETURN_POSITIVE"
            or block.get("b1_risk_target") != "NET_LOSS_AT_LEAST_7_PERCENT"
            or block.get("probabilities_calibrated_for_deployment") is not True):
        issues.append("FIRST_BOARD_DUAL_TARGET_NOT_FORWARD_VALIDATED")
    ids = [row.get("candidate_id") for row in rows]
    if set(ids) != first_ids or len(ids) != len(set(ids)):
        issues.append("FIRST_BOARD_DUAL_TARGET_NOT_EXACT_POOL")
        return issues, row_map
    for row in rows:
        profit = finite(row.get("b2_positive_probability"))
        loss = finite(row.get("b1_big_loss_probability"))
        try:
            available = timestamp(row["available_at"])
            time_valid = available <= cutoff
        except (KeyError, TypeError, ValueError):
            time_valid = False
        if (profit is None or loss is None or not 0 <= profit <= 1 or not 0 <= loss <= 1
                or row.get("inside_frozen_feature_support") is not True or not time_valid):
            issues.append("FIRST_BOARD_DUAL_TARGET_ROW_INVALID")
        row_map[row["candidate_id"]] = row
    return issues, row_map


def _preopen_announcement_contract(block, top, signal_date, trade_date, cutoff):
    """Require the frozen top-1 to survive a point-in-time pre-open notice scan."""
    if not isinstance(block, dict) or not top:
        return ["PREOPEN_ANNOUNCEMENT_DATA_UNAVAILABLE"]
    if block.get("decision") == "HARD_RISK_REJECT":
        return ["PREOPEN_ANNOUNCEMENT_HARD_RISK"]
    issues = []
    try:
        bound_at = timestamp(block["bound_at"])
        signal_open = timestamp(f"{signal_date}T15:05:00+08:00")
        deadline = timestamp(f"{trade_date}T09:15:00+08:00")
        time_valid = signal_open <= bound_at < deadline and bound_at <= cutoff
    except (KeyError, TypeError, ValueError):
        time_valid = False
    if (block.get("version") != "v14-announcement-forward-1"
            or block.get("signal_date") != signal_date
            or block.get("trade_date") != trade_date
            or str(block.get("candidate_code", "")).zfill(6) != top["code"]
            or not block.get("binding_hash") or not time_valid):
        issues.append("PREOPEN_ANNOUNCEMENT_BINDING_INVALID")
    if block.get("coverage_complete") is not True:
        issues.append("PREOPEN_ANNOUNCEMENT_DATA_UNAVAILABLE")
    if block.get("decision") != "NO_HARD_RISK_FOUND":
        issues.append("PREOPEN_ANNOUNCEMENT_NOT_CLEARED")
    if block.get("evidence_assurance") not in ANNOUNCEMENT_CLEAR_ASSURANCE:
        issues.append("PREOPEN_ANNOUNCEMENT_FULLTEXT_NOT_VERIFIED")
    return issues


def select(payload, asof):
    cutoff = timestamp(asof)
    issues = []
    universe = payload.get("universe", {})
    ranking = payload.get("ranking", {})
    confirmation = payload.get("top1_opening_confirmation", {})
    emotion = payload.get("emotion", {})
    stack = payload.get("model_stack_lock", {})
    acceptance = payload.get("system_acceptance", {})
    policy = payload.get("risk_policy", {})
    signal_date, trade_date = universe.get("signal_date"), payload.get("trade_date")
    if not signal_date or not trade_date or cutoff.strftime("%Y-%m-%d") != trade_date:
        issues.append("INVALID_SESSION_BINDING")

    candidates = universe.get("candidates", [])
    ids, candidate_map = [], {}
    for row in candidates:
        candidate_id = row.get("candidate_id")
        code = str(row.get("code", "")).zfill(6)
        boards = finite(row.get("boards"))
        expected = "FIRST_BOARD" if boards == 1 else "CONSECUTIVE_BOARD" if boards is not None and boards >= 2 and boards.is_integer() else None
        if (not candidate_id or candidate_id in candidate_map or not code.isdigit() or len(code) != 6
                or expected is None or row.get("board_group") != expected
                or row.get("pool_assignment") != "CLOSING_LIMIT_UP"
                or row.get("is_mainboard") is not True or row.get("is_st") is not False
                or row.get("is_delisting_period") is not False or row.get("no_price_limit") is not False
                or row.get("security_status") != "verified" or row.get("limit_up_status") != "verified"
                or row.get("identity_status") != "verified"):
            issues.append("INVALID_OR_DUPLICATE_POOL_CANDIDATE")
            continue
        normalized = dict(row, code=code, boards=int(boards))
        candidate_map[candidate_id] = normalized
        ids.append(candidate_id)
    declared_counts = universe.get("board_level_counts", {})
    actual_counts = {}
    for row in candidate_map.values():
        actual_counts[str(row["boards"])] = actual_counts.get(str(row["boards"]), 0) + 1
    failed_ids = universe.get("failed_board_candidate_ids")
    failed_separate = (isinstance(failed_ids, list) and len(failed_ids) == len(set(failed_ids))
                       and universe.get("failed_board_count") == len(failed_ids)
                       and not (set(failed_ids) & set(ids)))
    anchor_ids = universe.get("emotion_anchor_ids")
    anchors_valid = (isinstance(anchor_ids, list) and len(anchor_ids) == len(set(anchor_ids))
                     and set(anchor_ids) <= set(ids))
    if (universe.get("status") != "VERIFIED_FULL_CLOSING_LIMIT_POOL"
            or universe.get("universe_complete") is not True
            or universe.get("point_in_time_status_complete") is not True
            or universe.get("all_observed_board_levels_reported") is not True
            or universe.get("failed_board_pool_mixed_with_main") is not False
            or universe.get("closing_limit_count") != len(candidates)
            or declared_counts != actual_counts or not candidates or not failed_separate or not anchors_valid):
        issues.append("FULL_POOL_COVERAGE_NOT_VERIFIED")

    same_height_issues, same_height_map = _same_height_contract(
        payload.get("same_height_comparison", {}), candidate_map, signal_date, cutoff)
    issues.extend(same_height_issues)
    dual_target_issues, dual_target_map = _first_board_dual_target_contract(
        payload.get("first_board_dual_target", {}), candidate_map, signal_date, cutoff)
    issues.extend(dual_target_issues)

    rank_rows = ranking.get("rows", [])
    rank_ids = [row.get("candidate_id") for row in rank_rows]
    valid_scores = {}
    for row in rank_rows:
        score = finite(row.get("opportunity_proxy_score"))
        if score is None or not 0 <= score <= 1 or not isinstance(row.get("is_daily_top1"), bool):
            issues.append("INVALID_RANK_SCORE")
        else:
            valid_scores[row.get("candidate_id")] = score
    if (ranking.get("status") != "CURRENT_FULL_POOL_RANK_SCORES_FROZEN"
            or ranking.get("signal_date") != signal_date
            or ranking.get("score_is_complete_trade_probability") is not False
            or set(rank_ids) != set(ids) or len(rank_ids) != len(set(rank_ids))):
        issues.append("RANKING_DOES_NOT_COVER_EXACT_POOL")
    marked = [row.get("candidate_id") for row in rank_rows if row.get("is_daily_top1") is True]
    expected_top = min(valid_scores, key=lambda key: (-valid_scores[key], key)) if valid_scores else None
    if len(marked) != 1 or marked[0] != expected_top:
        issues.append("INVALID_FROZEN_DAILY_TOP1")
    top_id = expected_top
    top = candidate_map.get(top_id)
    issues.extend(_preopen_announcement_contract(
        payload.get("preopen_announcement_binding"), top,
        signal_date, trade_date, cutoff,
    ))

    if (stack.get("status") != "MODEL_STACK_READY_FOR_LOCKED_BLIND_PAPER"
            or stack.get("stack_audit", {}).get("can_feed_daily_decision_contract") is not True
            or stack.get("deployment_allowed") is not False):
        issues.append("MODEL_STACK_NOT_READY")
    if acceptance.get("status") not in {"PAPER_TRADE_READY", "FORWARD_VALIDATED"}:
        issues.append("SYSTEM_ACCEPTANCE_NOT_READY")
    try:
        if timestamp(stack["frozen_at"]) > timestamp(ranking["scored_at"]) or timestamp(ranking["scored_at"]) > cutoff:
            issues.append("MODEL_STACK_OR_RANK_TIME_INVALID")
    except (KeyError, TypeError, ValueError):
        issues.append("MODEL_STACK_OR_RANK_TIME_INVALID")

    states = emotion.get("dimension_states", {})
    adverse = sum(state == "ADVERSE" for state in states.values()) if isinstance(states, dict) else 99
    regime = emotion.get("market_regime")
    if (emotion.get("entry_support_eligible") is not True or set(states) != EMOTION_DIMENSIONS
            or any(state not in {"SUPPORTIVE", "MIXED", "ADVERSE"} for state in states.values())
            or emotion.get("asof") != cutoff.isoformat()):
        issues.append("EMOTION_EVIDENCE_NOT_ACTIONABLE")
    if (regime not in ACTIVE_REGIMES or emotion.get("pattern") == "BROAD_ADVERSE_HYPOTHESIS"
            or states.get("broken_loss_effect") == "ADVERSE"):
        issues.append("EMOTION_HARD_BLOCK")

    risk = _risk_policy(policy, cutoff)
    if risk is None:
        issues.append("RISK_POLICY_INVALID")
    elif adverse > risk["max_adverse"]:
        issues.append("TOO_MANY_ADVERSE_EMOTION_DIMENSIONS")

    if confirmation.get("candidate_id") != top_id or (top and str(confirmation.get("code", "")).zfill(6) != top["code"]):
        issues.append("OPENING_CONFIRMATION_NOT_FOR_FROZEN_TOP1")
    if not _fresh(confirmation, cutoff, trade_date):
        issues.append("OPENING_CONFIRMATION_NOT_FRESH_VERIFIED")
    profit = finite(confirmation.get("complete_trade_profit_probability"))
    large_loss = finite(confirmation.get("large_loss_probability"))
    if top and top["board_group"] == "FIRST_BOARD":
        dual = dual_target_map.get(top_id, {})
        b2_probability = finite(confirmation.get("b2_positive_probability"))
        b1_loss_probability = finite(confirmation.get("b1_big_loss_probability"))
        bound_b2, bound_b1 = finite(dual.get("b2_positive_probability")), finite(
            dual.get("b1_big_loss_probability"))
        if (confirmation.get("probability_target_scope")
                != "FIRST_BOARD_B2_POSITIVE_AND_B1_BIG_LOSS"
                or b2_probability is None or b1_loss_probability is None
                or bound_b2 is None or bound_b1 is None
                or not math.isclose(b2_probability, bound_b2, abs_tol=1e-12)
                or not math.isclose(b1_loss_probability, bound_b1, abs_tol=1e-12)):
            issues.append("FIRST_BOARD_CONFIRMATION_NOT_BOUND_TO_DUAL_TARGET")
        profit, large_loss = b2_probability, b1_loss_probability
    chosen = stack.get("evidence", {}).get("frozen_threshold", {}).get("chosen_thresholds", {})
    profit_cut, loss_cut = finite(chosen.get("profit_threshold")), finite(chosen.get("large_loss_threshold"))
    if (profit is None or large_loss is None or profit_cut is None or loss_cut is None
            or not 0 <= profit <= 1 or not 0 <= large_loss <= 1
            or profit < profit_cut or large_loss > loss_cut
            or confirmation.get("inside_frozen_feature_support") is not True):
        issues.append("PROFIT_OR_LARGE_LOSS_GATE_FAILED")
    for field in ("auction_conditions_passed", "opening_conditions_passed", "theme_sync_passed",
                  "competition_confirmation_passed"):
        if confirmation.get(field) is not True:
            issues.append(field.upper() + "_FAILED")
    flags = confirmation.get("false_strength_flags", {})
    if set(flags) != FALSE_STRENGTH_FLAGS or any(not isinstance(value, bool) for value in flags.values()):
        issues.append("FALSE_STRENGTH_EVIDENCE_INCOMPLETE")
    elif any(flags.values()):
        issues.append("OBSERVABLE_FALSE_STRENGTH_RISK")
    theme = confirmation.get("theme", {})
    try:
        theme_valid = (theme.get("claim_status") in {"verified_business", "market_narrative"}
                       and theme.get("granularity") == "specific" and bool(theme.get("name"))
                       and bool(theme.get("source")) and timestamp(theme["available_at"]) <= cutoff)
    except (KeyError, TypeError, ValueError):
        theme_valid = False
    if not theme_valid:
        issues.append("THEME_EVIDENCE_NOT_ACTIONABLE")
    if confirmation.get("hard_risk_block") is not False or confirmation.get("execution_state") != "BUYABLE":
        issues.append("HARD_RISK_OR_UNBUYABLE")

    entry, max_price, invalidation = [finite(confirmation.get(key)) for key in
                                      ("planned_entry_price", "maximum_acceptable_price", "invalidation_price")]
    quantity = notional = risk_cash = None
    if (any(value is None for value in (entry, max_price, invalidation)) or entry <= 0
            or invalidation <= 0 or invalidation >= entry or entry > max_price):
        issues.append("INVALID_ENTRY_PRICE_BOUNDS")
    elif risk is not None and regime in ACTIVE_REGIMES:
        loss_fraction = max((entry - invalidation) / entry, risk["stress"])
        caps = [risk["equity"] * risk["caps"][regime],
                risk["equity"] * risk["risk"] / loss_fraction,
                risk["equity"] * risk["gross"] - risk["current_gross"],
                risk["equity"] * risk["theme"] - risk["current_theme"]]
        allowed_cash = max(0.0, min(caps))
        quantity = math.floor(allowed_cash / entry / 100) * 100
        notional = quantity * entry
        risk_cash = notional * loss_fraction
        if quantity < 100:
            issues.append("RISK_BUDGET_BELOW_ONE_LOT")

    ranking_integrity_failures = {
        "INVALID_OR_DUPLICATE_POOL_CANDIDATE", "FULL_POOL_COVERAGE_NOT_VERIFIED",
        "INVALID_RANK_SCORE", "RANKING_DOES_NOT_COVER_EXACT_POOL",
        "INVALID_FROZEN_DAILY_TOP1", "SAME_HEIGHT_COMPARISON_NOT_EXACT_FULL_POOL",
        "SAME_HEIGHT_COMPARISON_CONTRACT_INVALID", "FIRST_BOARD_DUAL_TARGET_NOT_EXACT_POOL",
    }
    blocking = sorted(set(issues))
    eligible = not blocking
    mode = ("CONDITIONAL_USER_DECISION" if acceptance.get("status") == "FORWARD_VALIDATED"
            else "LOCKED_PAPER_PLAN" if acceptance.get("status") == "PAPER_TRADE_READY" else "NO_PLAN")
    primary = ({"candidate_id": top_id, "code": top["code"], "name": top.get("name"),
                "boards": top["boards"], "board_group": top["board_group"],
                "theme": theme.get("name"), "rank_score": valid_scores[top_id],
                "rank_score_meaning": "opportunity_proxy_score_not_complete_trade_probability",
                "complete_trade_profit_probability": profit, "large_loss_probability": large_loss,
                "probability_target_scope": confirmation.get("probability_target_scope"),
                "planned_entry_price": entry, "maximum_acceptable_price": max_price,
                "invalidation_price": invalidation, "max_shares": quantity,
                "planned_notional": notional, "planned_stress_loss_cash": risk_cash,
                "conditional": True} if eligible else None)
    ordered = sorted(candidate_map.values(), key=lambda row: (-valid_scores.get(row["candidate_id"], -1), row["candidate_id"]))
    ranking_valid = not (set(blocking) & ranking_integrity_failures)
    research_shortlist = {
        "status": ("VALID_FULL_POOL_RESEARCH_RANKING_NOT_BUY_SIGNAL" if ranking_valid
                   else "UNAVAILABLE_RANKING_INTEGRITY_FAILED"),
        "first_board_choice": (_research_choice(ordered, valid_scores, "FIRST_BOARD")
                               if ranking_valid else None),
        "relay_choice": (_research_choice(ordered, valid_scores, "CONSECUTIVE_BOARD")
                         if ranking_valid else None),
        "all_limit_up_choice": (_research_choice(ordered, valid_scores)
                                if ranking_valid else None),
        "selection_rule": "highest frozen opportunity proxy score; tie breaks by candidate_id",
        "same_height_comparison_rule": (
            "all candidates at the same height are one exact group; seal speed and zero failed seals cannot name a champion"),
        "buy_signal": False,
        "may_upgrade_only_if_same_all_limit_up_choice_passes_every_entry_gate": True,
        "failed_top1_means_cash_not_intraday_backup": True,
    }
    return {"status": "CONDITIONAL_PRIMARY_READY" if eligible else "CASH",
            "mode": mode if eligible else "CASH", "asof": cutoff.isoformat(),
            "signal_date": signal_date, "trade_date": trade_date,
            "primary_choice": primary, "blocking_reasons": blocking,
            "research_shortlist": research_shortlist,
            "roles": {"emotion_anchor": anchor_ids if anchors_valid else [],
                      "leader_potential": [top_id] if top_id else [],
                      "current_buy_opportunity": [top_id] if eligible else []},
            "candidate_groups": {group: [row["candidate_id"] for row in ordered if row["board_group"] == group]
                                 for group in GROUPS},
            "same_height_groups": {
                group_id: sorted({candidate_id for candidate_id, row in same_height_map.items()
                                  if row.get("height_group_id") == group_id})
                for group_id in sorted({row.get("height_group_id") for row in same_height_map.values()
                                        if row.get("height_group_id")})},
            "frozen_top1_failed_means_cash": True, "automatic_backup_allowed": False,
            "direct_recommendation": False, "order_submitted": False, "position_created": False,
            "guarantee_of_premium": False,
            "probability_scope": "policy-conditional complete trades inside frozen support only"}


def run(input_path, output_path, asof):
    source, target = Path(input_path), Path(output_path)
    if target.exists():
        raise FileExistsError("choose a new immutable daily primary decision path")
    result = select(json.loads(source.read_text(encoding="utf-8")), asof)
    result["source_hashes"] = {str(source.resolve()): hashlib.sha256(source.read_bytes()).hexdigest(),
                               str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--asof", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output, args.asof), ensure_ascii=False, indent=2))
