"""Validate an immutable daily leader report against the full decision contract.

Structural validity is separate from permission to enter.  A report can be
complete and still correctly resolve to cash when evidence or system gates are
insufficient.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from leader_target_screen_v17 import timestamp, finite


EMOTION_DIMENSIONS = {
    "money_effect": {"yesterday_limit_premium", "first_board_promotion", "buyable_profit_rate"},
    "broken_loss_effect": {"broken_tail_return", "broken_down_limit_rate", "contagion_scope"},
    "height_feedback": {"high_feedback", "middle_feedback", "low_feedback"},
    "theme_divergence": {"core_feedback", "follower_feedback", "withdrawal_scope"},
    "repair_quality": {"repair_close_retention", "repair_volume_support", "theme_sync"},
}
ROLE_NAMES = ("emotion_anchor", "leader_potential", "current_buy_opportunity")
SCENARIO_ACTIONS = {
    "strong_continuation": "HOLD_TO_NEXT_REVIEW",
    "normal_divergence": "HOLD_IF_RULES_PASS",
    "trend_invalidation": "SELL_ALL",
}
EVIDENCE_STATUSES = {"verified", "delayed", "inferred", "unavailable"}
GROUPS = {"FIRST_BOARD", "CONSECUTIVE_BOARD"}
COMPETITION_IDENTITIES = {"pioneer", "turnover_core", "height_anchor", "capacity_core", "supplement", "unique_survivor"}
HEIGHT_ROLES = {"CURRENT_EMOTION_ANCHOR", "POTENTIAL_NEXT_HIGHEST", "BOTH", "NONE"}
PREMIUM_SPACE_LEVELS = {"HIGH", "MEDIUM", "LOW", "UNAVAILABLE"}
BUYABILITY_LEVELS = {"CONDITIONAL", "ANCHOR_ONLY", "UNBUYABLE", "REJECT"}


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _condition_valid(rule):
    try:
        threshold = finite(rule["threshold"])
        window = finite(rule["window_seconds"])
        age = finite(rule["max_age_seconds"])
        return (_text(rule["condition_id"]) and _text(rule["metric"]) and _text(rule["unit"])
                and _text(rule["definition_id"]) and rule["operator"] in {">", ">=", "<", "<=", "=="}
                and threshold is not None and window is not None and age is not None and window >= 0 and age >= 0)
    except (KeyError, TypeError, ValueError):
        return False


def _evidence_valid(item, cutoff, report_date):
    try:
        observed, available = timestamp(item["observed_at"]), timestamp(item["available_at"])
        status = item["data_status"]
        value = item["value"]
        if isinstance(value, float) and not math.isfinite(value):
            return False, False
        age_limit = finite(item.get("max_age_seconds", 300))
        denominator, observed_count, missing_count = [item[key] for key in ("denominator", "observed_count", "missing_count")]
        counts_valid = (all(isinstance(v,int) and not isinstance(v,bool) and v >= 0
                            for v in (denominator,observed_count,missing_count))
                        and denominator >= 1 and observed_count >= 1
                        and observed_count+missing_count == denominator)
        valid = (age_limit is not None and age_limit >= 0
                 and observed <= available <= cutoff and observed.strftime("%Y-%m-%d") == report_date
                 and status in EVIDENCE_STATUSES and _text(item["source"])
                 and _text(item["definition_id"]) and value is not None and counts_valid)
        fresh_verified = (valid and status == "verified" and
                          (cutoff-observed).total_seconds() <= age_limit and observed_count/denominator >= .9)
        return valid, fresh_verified
    except (KeyError, TypeError, ValueError):
        return False, False


def _candidate_issues(candidate, cutoff, report_date):
    issues = []
    try:
        code = str(candidate["code"]).zfill(6)
        boards = finite(candidate["boards"])
        if (not code.isdigit() or len(code) != 6 or code == "000000" or not _text(candidate.get("name"))
                or boards is None or boards < 1 or not boards.is_integer()):
            issues.append("INVALID_CANDIDATE_ID_OR_BOARD")
        expected = "FIRST_BOARD" if boards == 1 else "CONSECUTIVE_BOARD"
        if candidate.get("board_group") != expected:
            issues.append("BOARD_GROUP_MISMATCH")
        if candidate.get("pool_assignment") != "CLOSING_LIMIT_UP":
            issues.append("FOCUS_CANDIDATE_NOT_IN_CLOSING_POOL")
        if candidate.get("competition_identity") not in COMPETITION_IDENTITIES:
            issues.append("MISSING_COMPETITION_IDENTITY")
        if candidate.get("height_role") not in HEIGHT_ROLES:
            issues.append("INVALID_HEIGHT_ROLE")
        theme = candidate["theme"]
        if (not _text(theme.get("name")) or theme.get("granularity") != "specific"
                or theme.get("claim_status") not in {"verified_business", "market_narrative", "rumor"}
                or not _text(theme.get("source"))):
            issues.append("INVALID_THEME_EVIDENCE")
        if timestamp(theme["available_at"]) > cutoff or timestamp(theme["published_at"]) > timestamp(theme["available_at"]):
            issues.append("FUTURE_THEME_EVIDENCE")
        for field in ("continuation_logic", "opposition_evidence", "competitor_invalidation", "risk_factors"):
            value = candidate.get(field)
            if not isinstance(value, list) or not value or any(not _text(item) for item in value):
                issues.append("MISSING_"+field.upper())
        for field in ("auction_conditions", "opening_conditions", "cancel_conditions"):
            rules = candidate.get(field)
            if not isinstance(rules, list) or not rules or not all(_condition_valid(rule) for rule in rules):
                issues.append("INVALID_"+field.upper())
        entry = candidate["entry_plan"]
        start, end = timestamp(entry["entry_start"]), timestamp(entry["entry_end"])
        lower, upper, slip, shares = [finite(entry[key]) for key in ("min_price", "max_price", "allowed_slippage", "max_shares")]
        if (start <= cutoff or start > end or start.strftime("%Y-%m-%d") != end.strftime("%Y-%m-%d")
                or any(v is None for v in (lower, upper, slip, shares))
                or lower <= 0 or upper < lower or slip < 0 or shares <= 0 or not shares.is_integer() or shares % 100
                or not _text(entry.get("risk_policy_id"))):
            issues.append("INVALID_ENTRY_PLAN")
        premium = candidate["premium_space"]
        if (premium.get("assessment") not in PREMIUM_SPACE_LEVELS
                or premium.get("evidence_status") not in EVIDENCE_STATUSES
                or not _text(premium.get("risk_reward_conclusion"))
                or not isinstance(premium.get("drivers"), list) or not premium.get("drivers")
                or any(not _text(item) for item in premium.get("drivers", []))
                or not isinstance(premium.get("exhaustion_signals"), list)
                or not premium.get("exhaustion_signals")
                or any(not _text(item) for item in premium.get("exhaustion_signals", []))):
            issues.append("INVALID_PREMIUM_SPACE_ASSESSMENT")
        anchor_plan = candidate["anchor_entry_plan"]
        condition_ids = {rule["condition_id"] for field in
                         ("auction_conditions", "opening_conditions", "cancel_conditions")
                         for rule in candidate.get(field, []) if isinstance(rule, dict) and "condition_id" in rule}
        if (anchor_plan.get("buyability") not in BUYABILITY_LEVELS
                or not all(_text(anchor_plan.get(field)) for field in
                           ("setup_type", "high_open_action", "flat_open_action",
                            "low_open_action", "sector_break_action"))
                or not isinstance(anchor_plan.get("entry_trigger_ids"), list)
                or not anchor_plan.get("entry_trigger_ids")
                or not set(anchor_plan.get("entry_trigger_ids", [])).issubset(condition_ids)
                or not isinstance(anchor_plan.get("invalid_trigger_ids"), list)
                or not anchor_plan.get("invalid_trigger_ids")
                or not set(anchor_plan.get("invalid_trigger_ids", [])).issubset(condition_ids)
                or finite(anchor_plan.get("max_price")) != upper
                or not isinstance(anchor_plan.get("expected_holding_sessions"), list)
                or len(anchor_plan.get("expected_holding_sessions", [])) != 2
                or any(finite(item) is None or finite(item) < 0
                       for item in anchor_plan.get("expected_holding_sessions", []))):
            issues.append("INVALID_ANCHOR_ENTRY_PLAN")
        scenarios = candidate["holding_scenarios"]
        for name, action in SCENARIO_ACTIONS.items():
            scenario = scenarios[name]
            if scenario.get("action") != action or not scenario.get("rule_ids"):
                issues.append("INVALID_HOLDING_SCENARIO_"+name.upper())
        if candidate.get("execution_state") not in {"CONDITIONAL", "UNBUYABLE", "DATA_LIMITED"}:
            issues.append("INVALID_EXECUTION_STATE")
        if candidate.get("model_probability") is not None and candidate.get("probability_status") != "calibrated_out_of_sample":
            issues.append("UNVALIDATED_PROBABILITY_PRESENTED")
    except (KeyError, TypeError, ValueError):
        issues.append("INVALID_OR_MISSING_CANDIDATE_FIELDS")
    return sorted(set(issues))


def audit(report, asof=None):
    issues = []
    try:
        cutoff = timestamp(asof or report["asof"])
        if asof is not None and cutoff != timestamp(report["asof"]):
            issues.append("ASOF_MISMATCH")
        report_date = report["report_date"]
        if cutoff.strftime("%Y-%m-%d") != report_date:
            issues.append("REPORT_DATE_MISMATCH")
        if report.get("stage") not in {"CLOSING_PLAN", "AUCTION_CONFIRMATION", "OPENING_CONFIRMATION"}:
            issues.append("INVALID_STAGE")
    except (KeyError, TypeError, ValueError):
        return dict(status="INVALID_REPORT", issues=["INVALID_REPORT_TIME"], entry_permission="NO_ENTER")
    quality = report.get("data_quality", {})
    if (not quality.get("sources") or not quality.get("verified_fields")
            or not isinstance(quality.get("unavailable_fields"), list)
            or not _text(quality.get("execution_limitation"))):
        issues.append("INCOMPLETE_DATA_QUALITY_BLOCK")
    universe = report.get("universe_summary", {})
    levels = universe.get("board_level_counts", {})
    try:
        closing = int(universe["closing_limit_count"])
        first = int(universe["first_board_count"])
        if closing < 0 or first < 0 or first != int(levels.get("1", -1)) or sum(int(v) for v in levels.values()) != closing:
            issues.append("INCONSISTENT_UNIVERSE_COUNTS")
        if universe.get("failed_board_pool_mixed_with_main") is not False:
            issues.append("FAILED_BOARD_POOL_NOT_SEPARATE")
        if universe.get("all_observed_board_levels_reported") is not True:
            issues.append("BOARD_LEVEL_COVERAGE_NOT_REPORTED")
    except (KeyError, TypeError, ValueError):
        issues.append("INVALID_UNIVERSE_SUMMARY")
    emotion = report.get("emotion_analysis", {})
    actionable_emotion = True
    for dimension, required_metrics in EMOTION_DIMENSIONS.items():
        block = emotion.get(dimension, {})
        if not all(_text(block.get(field)) for field in ("interpretation", "trade_effect", "invalidation")):
            issues.append("INCOMPLETE_EMOTION_"+dimension.upper())
            actionable_emotion = False
            continue
        evidence = block.get("evidence", [])
        metrics = {row.get("metric") for row in evidence if isinstance(row, dict)}
        if not required_metrics <= metrics:
            issues.append("MISSING_EMOTION_METRICS_"+dimension.upper())
            actionable_emotion = False
        for item in evidence:
            valid, fresh = _evidence_valid(item, cutoff, report_date)
            if not valid:
                issues.append("INVALID_EMOTION_EVIDENCE_"+dimension.upper())
            if not fresh:
                actionable_emotion = False
    focus = report.get("focus_candidates", [])
    candidate_codes = set()
    candidate_audits = []
    groups = {name: [] for name in GROUPS}
    for candidate in focus:
        code = str(candidate.get("code", "")).zfill(6)
        if code in candidate_codes:
            cissues = ["DUPLICATE_FOCUS_CANDIDATE"]
        else:
            candidate_codes.add(code)
            cissues = _candidate_issues(candidate, cutoff, report_date)
        if candidate.get("board_group") in groups:
            groups[candidate["board_group"]].append(code)
        candidate_audits.append(dict(code=code, issues=cissues, status="PASS" if not cissues else "ISSUES_FOUND"))
        issues.extend(f"CANDIDATE_{code}_{item}" for item in cissues)
    declared_groups = report.get("candidate_groups", {})
    for group in GROUPS:
        declared = declared_groups.get(group)
        if (not isinstance(declared, list)
                or sorted(str(code).zfill(6) for code in declared) != sorted(groups[group])):
            issues.append("CANDIDATE_GROUP_MISMATCH_"+group)
    roles = report.get("roles", {})
    for role in ROLE_NAMES:
        values = roles.get(role)
        normalized = [str(code).zfill(6) for code in values] if isinstance(values, list) else []
        if (not isinstance(values, list) or len(normalized) != len(set(normalized))
                or any(code not in candidate_codes for code in normalized)):
            issues.append("INVALID_ROLE_"+role.upper())
    outlook = report.get("height_outlook", {})
    try:
        current_height = finite(outlook["current_highest_board"])
        current_anchors = [str(code).zfill(6) for code in outlook["current_anchor_codes"]]
        potential = [str(code).zfill(6) for code in outlook["potential_next_highest_codes"]]
        if (current_height is None or current_height < 0 or not current_height.is_integer()
                or current_anchors != [str(code).zfill(6) for code in roles.get("emotion_anchor", [])]
                or potential != [str(code).zfill(6) for code in roles.get("leader_potential", [])]
                or outlook.get("role_overlap_allowed") is not True
                or not _text(outlook.get("anchor_premium_conclusion"))
                or not isinstance(outlook.get("rotation_confirmations"), list)
                or not outlook.get("rotation_confirmations")
                or any(not _text(item) for item in outlook.get("rotation_confirmations", []))
                or not isinstance(outlook.get("failure_conditions"), list)
                or not outlook.get("failure_conditions")
                or any(not _text(item) for item in outlook.get("failure_conditions", []))):
            issues.append("INVALID_HEIGHT_OUTLOOK")
    except (KeyError, TypeError, ValueError):
        issues.append("INVALID_HEIGHT_OUTLOOK")
    primary = report.get("primary_choice")
    system_status = report.get("system_status")
    permission = "NO_ENTER"
    if primary is None:
        if report.get("final_decision") != "CASH":
            issues.append("NO_PRIMARY_MUST_RESOLVE_TO_CASH")
    else:
        try:
            code = str(primary["code"]).zfill(6)
            if (code not in {str(c).zfill(6) for c in roles.get("current_buy_opportunity", [])}
                    or code not in candidate_codes or primary.get("conditional") is not True
                    or not _text(primary.get("relative_advantage")) or not _text(primary.get("trigger_summary"))
                    or primary.get("on_failure") not in {"CASH", "PREDECLARED_BACKUP"}
                    or report.get("final_decision") != "CONDITIONAL_PRIMARY"):
                issues.append("INVALID_PRIMARY_CHOICE")
            target = next((candidate for candidate in focus if str(candidate.get("code", "")).zfill(6) == code), {})
            primary_clean = not next((row["issues"] for row in candidate_audits if row["code"] == code), ["missing"])
            theme_ok = target.get("theme", {}).get("claim_status") != "rumor"
            premium_ok = target.get("premium_space", {}).get("assessment") in {"HIGH", "MEDIUM"}
            buyability_ok = target.get("anchor_entry_plan", {}).get("buyability") == "CONDITIONAL"
            if (not issues and actionable_emotion and primary_clean and theme_ok
                    and premium_ok and buyability_ok
                    and system_status in {"PAPER_TRADE_READY", "FORWARD_VALIDATED"}
                    and report.get("stage") in {"AUCTION_CONFIRMATION", "OPENING_CONFIRMATION"}
                    and quality.get("direct_recommendation_allowed") is False):
                permission = "CONDITIONAL_ENTRY_PLAN_ONLY"
        except (KeyError, TypeError, ValueError):
            issues.append("INVALID_PRIMARY_CHOICE")
    if quality.get("direct_recommendation_allowed") is not False:
        issues.append("DIRECT_RECOMMENDATION_MUST_REMAIN_FALSE")
    if system_status not in {"DIAGNOSTIC_ONLY", "PAPER_TRADE_READY", "FORWARD_VALIDATED"}:
        issues.append("INVALID_SYSTEM_STATUS")
    if issues:
        permission = "NO_ENTER"
    status = "INVALID_REPORT" if issues else "VALID_CASH_REPORT" if primary is None else "VALID_CONDITIONAL_REPORT"
    return dict(status=status, entry_permission=permission, issues=sorted(set(issues)),
                actionable_emotion_evidence=actionable_emotion, candidate_audits=candidate_audits,
                first_board_focus_count=len(groups["FIRST_BOARD"]), relay_focus_count=len(groups["CONSECUTIVE_BOARD"]),
                guarantee=False, order_submitted=False)


def run(input_path, output_path, asof=None):
    source, target = Path(input_path), Path(output_path)
    if target.exists():
        raise FileExistsError("choose a new immutable daily audit path")
    report = json.loads(source.read_text(encoding="utf-8"))
    result = audit(report, asof)
    result["source_hashes"] = {
        str(source.resolve()): hashlib.sha256(source.read_bytes()).hexdigest(),
        str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--asof")
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output, args.asof), ensure_ascii=False, indent=2))
