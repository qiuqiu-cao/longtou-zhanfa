"""Audit supplied sell fills against exit rules frozen before entry.

This module checks declared evidence only.  It neither invents an exit fill nor
certifies that a supplied broker/paper record is authentic.
"""
from leader_target_screen_v17 import timestamp, finite
from entry_plan_audit import numeric_condition


REQUIRED_SCENARIOS = {
    "strong_continuation": "HOLD_TO_NEXT_REVIEW",
    "normal_divergence": "HOLD_IF_RULES_PASS",
    "trend_invalidation": "SELL_ALL",
}
BLOCKED_STATES = {"locked_limit_down", "suspended", "no_sell_liquidity"}


def _index(rows, key):
    result = {}
    for row in rows:
        value = row.get(key)
        if not value or value in result:
            raise ValueError("missing or duplicate evidence key")
        result[value] = row
    return result


def _closed_session_offsets(sessions, entry_day, cutoff):
    if sessions != sorted(set(sessions)) or entry_day not in sessions:
        raise ValueError("sorted unique sessions including entry required")
    origin = sessions.index(entry_day)
    return {
        day: index-origin for index, day in enumerate(sessions)
        if index >= origin and timestamp(day+"T15:00:00+08:00") <= cutoff
    }


def _blocked_evidence(confirmation, cutoff):
    if confirmation.get("sell_blocked") is not True:
        return False
    evidence = confirmation.get("sellability_evidence", {})
    try:
        observed = timestamp(evidence["observed_at"])
        available = timestamp(evidence["available_at"])
        return (observed <= available <= cutoff and evidence.get("state") in BLOCKED_STATES
                and evidence.get("data_status") == "verified" and bool(evidence.get("source"))
                and bool(evidence.get("evidence_id")))
    except (KeyError, TypeError, ValueError):
        return False


def _validate_policy(plan, first_buy_at):
    issues = []
    try:
        if not timestamp(plan["frozen_at"]) <= timestamp(plan["available_at"]) < first_buy_at:
            issues.append("EXIT_POLICY_NOT_FROZEN_BEFORE_ENTRY")
        maximum = finite(plan["maximum_holding_session_offset"])
        if maximum is None or not maximum.is_integer() or maximum < 1:
            raise ValueError("invalid holding offset")
        maximum = int(maximum)
        rules = plan["exit_rules"]
        if not isinstance(rules, list) or not rules:
            raise ValueError("exit rules required")
        rule_map = {}
        for rule in rules:
            rule_id = rule["rule_id"]
            if not rule_id or rule_id in rule_map:
                raise ValueError("duplicate exit rule")
            if rule.get("effect") not in {"HOLD", "SELL_ALL"}:
                raise ValueError("invalid exit rule effect")
            start, end = finite(rule["earliest_session_offset"]), finite(rule["latest_session_offset"])
            if (start is None or end is None or not start.is_integer() or not end.is_integer()
                    or start < 1 or end < start):
                raise ValueError("invalid exit rule window")
            if rule.get("trigger_mode") not in {"NUMERIC", "TIME_STOP"}:
                raise ValueError("invalid trigger mode")
            if rule["trigger_mode"] == "NUMERIC" and not isinstance(rule.get("condition_rule"), dict):
                raise ValueError("numeric condition required")
            rule_map[rule_id] = rule
        scenarios = plan["holding_scenarios"]
        for name, action in REQUIRED_SCENARIOS.items():
            scenario = scenarios[name]
            references = scenario["rule_ids"]
            expected_effect = "SELL_ALL" if action == "SELL_ALL" else "HOLD"
            if (scenario.get("action") != action or not references
                    or any(r not in rule_map or rule_map[r].get("effect") != expected_effect for r in references)):
                raise ValueError("invalid holding scenario")
        time_stops = [r for r in rules if r.get("trigger_mode") == "TIME_STOP" and r.get("effect") == "SELL_ALL"]
        if not any(int(finite(r["earliest_session_offset"])) == maximum for r in time_stops):
            issues.append("MISSING_FROZEN_TIME_STOP_AT_MAXIMUM_HOLD")
        beyond = plan.get("allow_beyond_b3") is True
        if maximum > 3 and not beyond:
            issues.append("HOLD_BEYOND_B3_NOT_PREAUTHORIZED")
        if beyond:
            references = plan.get("beyond_b3_rule_ids")
            if not references or any(r not in rule_map or rule_map[r].get("effect") != "HOLD" for r in references):
                issues.append("INVALID_BEYOND_B3_CONTINUATION_RULES")
        return maximum, rule_map, issues
    except (KeyError, TypeError, ValueError):
        return None, {}, issues+["INVALID_OR_MISSING_FROZEN_EXIT_POLICY"]


def audit(fills, plans, confirmations, sessions, asof):
    """Return plan-level declared exit-rule compliance without dropping failures."""
    cutoff = timestamp(asof)
    plan_map = _index(plans, "plan_id") if plans else {}
    confirmation_map = _index(confirmations, "exit_confirmation_id") if confirmations else {}
    grouped = {}
    for fill in sorted(fills, key=lambda row: timestamp(row["filled_at"])):
        grouped.setdefault(fill.get("plan_id"), []).append(fill)
    results = []
    for plan_id, plan_fills in grouped.items():
        buys = [row for row in plan_fills if row.get("side") == "BUY"]
        if not buys:
            continue
        issues = []
        plan = plan_map.get(plan_id)
        first_buy_at = min(timestamp(row["filled_at"]) for row in buys)
        entry_day = first_buy_at.strftime("%Y-%m-%d")
        offsets = _closed_session_offsets(sessions, entry_day, cutoff)
        latest_closed_offset = max(offsets.values(), default=0)
        bought = sum(float(row["shares"]) for row in buys)
        sells = [row for row in plan_fills if row.get("side") == "SELL"]
        sold = sum(float(row["shares"]) for row in sells)
        remaining = bought-sold
        if remaining < -1e-8:
            issues.append("EXIT_QUANTITY_EXCEEDS_ENTRY")
        if plan is None:
            maximum, rule_map = None, {}
            issues.append("MISSING_EXIT_PLAN")
        else:
            maximum, rule_map, policy_issues = _validate_policy(plan, first_buy_at)
            issues.extend(policy_issues)
            if any(plan.get("code") != row.get("code") or plan.get("kind") != row.get("kind") for row in plan_fills):
                issues.append("EXIT_PLAN_IDENTITY_OR_ACCOUNT_MISMATCH")
        confirmation_results = []
        blocked_active = False
        used = set()
        for sell in sells:
            cid = sell.get("exit_confirmation_id")
            confirmation = confirmation_map.get(cid)
            local = []
            rule_result = None
            if confirmation is None:
                local.append("MISSING_EXIT_CONFIRMATION")
            else:
                used.add(cid)
                rule = rule_map.get(confirmation.get("rule_id"))
                if rule is None or rule.get("effect") != "SELL_ALL":
                    local.append("UNKNOWN_OR_NONSELL_EXIT_RULE")
                try:
                    observed = timestamp(confirmation["observed_at"])
                    available = timestamp(confirmation["available_at"])
                    expires = timestamp(confirmation["expires_at"])
                    executed = timestamp(sell["filled_at"])
                    day = executed.strftime("%Y-%m-%d")
                    offset = offsets.get(day)
                    if not observed <= available < executed <= expires:
                        local.append("INVALID_CONFIRMATION_FILL_SEQUENCE")
                    if (confirmation.get("plan_id") != plan_id or confirmation.get("code") != sell.get("code")
                            or confirmation.get("kind") != sell.get("kind")):
                        local.append("EXIT_CONFIRMATION_IDENTITY_MISMATCH")
                    if (confirmation.get("data_status") != "verified" or not confirmation.get("source")
                            or not confirmation.get("evidence_id")):
                        local.append("UNVERIFIED_EXIT_CONFIRMATION")
                    if rule is not None:
                        start, end = int(finite(rule["earliest_session_offset"])), int(finite(rule["latest_session_offset"]))
                        if offset is None or not start <= offset <= end:
                            local.append("EXIT_OUTSIDE_FROZEN_RULE_WINDOW")
                        if rule.get("trigger_mode") == "TIME_STOP":
                            rule_result = dict(status="PASS" if offset is not None and offset >= start else "FAIL")
                        else:
                            rule_result = numeric_condition(rule["condition_rule"], confirmation.get("measurement", {}),
                                                            confirmation["available_at"])
                        if rule_result["status"] != "PASS":
                            local.append("EXIT_TRIGGER_EVIDENCE_NOT_PASSED")
                except (KeyError, TypeError, ValueError):
                    local.append("INVALID_OR_MISSING_EXIT_AUDIT_FIELDS")
            issues.extend(local)
            confirmation_results.append(dict(exit_confirmation_id=cid, fill_evidence_id=sell.get("evidence_id"),
                                             status="PASS" if not local else "ISSUES_FOUND", issues=local,
                                             trigger_audit=rule_result))
        for cid, confirmation in confirmation_map.items():
            if confirmation.get("plan_id") != plan_id or cid in used:
                continue
            if _blocked_evidence(confirmation, cutoff):
                local = []
                rule_result = None
                rule = rule_map.get(confirmation.get("rule_id"))
                try:
                    observed = timestamp(confirmation["observed_at"])
                    available = timestamp(confirmation["available_at"])
                    day = observed.strftime("%Y-%m-%d")
                    offset = offsets.get(day)
                    if not observed <= available <= cutoff:
                        local.append("INVALID_BLOCKED_CONFIRMATION_SEQUENCE")
                    if (confirmation.get("code") != buys[0].get("code")
                            or confirmation.get("kind") != buys[0].get("kind")
                            or confirmation.get("data_status") != "verified" or not confirmation.get("source")
                            or not confirmation.get("evidence_id")):
                        local.append("UNVERIFIED_OR_MISMATCHED_BLOCKED_CONFIRMATION")
                    if rule is None or rule.get("effect") != "SELL_ALL":
                        local.append("UNKNOWN_OR_NONSELL_BLOCKED_RULE")
                    else:
                        start, end = int(finite(rule["earliest_session_offset"])), int(finite(rule["latest_session_offset"]))
                        if offset is None or not start <= offset <= end:
                            local.append("BLOCKED_EXIT_OUTSIDE_FROZEN_RULE_WINDOW")
                        if rule.get("trigger_mode") == "TIME_STOP":
                            rule_result = dict(status="PASS" if offset is not None and offset >= start else "FAIL")
                        else:
                            rule_result = numeric_condition(rule["condition_rule"], confirmation.get("measurement", {}),
                                                            confirmation["available_at"])
                        if rule_result["status"] != "PASS":
                            local.append("BLOCKED_EXIT_TRIGGER_NOT_PASSED")
                except (KeyError, TypeError, ValueError):
                    local.append("INVALID_OR_MISSING_BLOCKED_EXIT_FIELDS")
                if not local:
                    blocked_active = True
                else:
                    issues.extend(local)
                confirmation_results.append(dict(exit_confirmation_id=cid, fill_evidence_id=None,
                                                 status="EXIT_BLOCKED_DECLARED" if not local else "ISSUES_FOUND",
                                                 issues=local, trigger_audit=rule_result))
        if maximum is not None and remaining > 1e-8 and latest_closed_offset >= maximum:
            if blocked_active:
                terminal_state = "EXIT_BLOCKED_UNRESOLVED"
            else:
                terminal_state = "MANDATORY_EXIT_NOT_COMPLETED"
                issues.append(terminal_state)
        elif remaining > 1e-8:
            terminal_state = "OPEN_WITHIN_FROZEN_WINDOW"
        else:
            terminal_state = "CLOSED"
        results.append(dict(plan_id=plan_id, status="DECLARED_EXIT_RULES_MATCH" if not issues else "ISSUES_FOUND",
                            issues=issues, entry_session=entry_day, latest_closed_session_offset=latest_closed_offset,
                            maximum_holding_session_offset=maximum, bought_shares=bought, sold_shares=sold,
                            remaining_shares=max(0, remaining), terminal_state=terminal_state,
                            exit_confirmations=confirmation_results, execution_independently_verified=False))
    return dict(status="NO_TRADES" if not results else "DECLARED_EXIT_RULES_MATCH" if all(not row["issues"] for row in results)
                else "ISSUES_FOUND", trades=results,
                declared_compliance_count=sum(not row["issues"] for row in results),
                independently_verified=False, strategy_win_rate=None,
                limitation="freeze times, trigger measurements, sellability and fills are supplied by caller")
