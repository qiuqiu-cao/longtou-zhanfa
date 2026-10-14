"""Versioned evidence links between next-session outlook and stock selection.

Validates research plans, not predictive accuracy or permission to trade.
"""
from daily_decision_contract import _condition_valid, _evidence_valid, timestamp


PATHS = {"CONSENSUS_CONTINUES", "CONSENSUS_TO_DIVERGENCE", "DIVERGENCE_CONTINUES",
         "DIVERGENCE_TO_REPAIR", "HIGH_LOW_ROTATION", "UNAVAILABLE"}


def audit_outlook(report):
    issues = []
    outlook = report.get("next_session_outlook", {})
    try:
        cutoff = timestamp(report["asof"])
        if timestamp(outlook["target_session"] + "T00:00:00+08:00").date() <= cutoff.date():
            issues.append("TARGET_NOT_FUTURE_SESSION")
        evidence = outlook.get("evidence", [])
        lookup = {}
        for item in evidence:
            key = item.get("evidence_id")
            valid, _ = _evidence_valid(item, cutoff, report["report_date"])
            if not key or key in lookup or not valid:
                issues.append("INVALID_OR_DUPLICATE_OUTLOOK_EVIDENCE")
            lookup[key] = item
        if outlook.get("main_path") not in PATHS:
            issues.append("INVALID_MAIN_PATH")
        if outlook.get("alternative_path") not in PATHS or outlook.get("alternative_path") == outlook.get("main_path"):
            issues.append("MISSING_DISTINCT_ALTERNATIVE")
        if outlook.get("probability") is not None:
            issues.append("UNCALIBRATED_OUTLOOK_PROBABILITY")
        for field in ("support_evidence_ids", "opposing_evidence_ids"):
            refs = outlook.get(field, [])
            if not refs or any(ref not in lookup for ref in refs):
                issues.append("INVALID_" + field.upper())
        for field in ("confirmation_conditions", "invalidation_conditions"):
            rules = outlook.get(field, [])
            if not rules or not all(_condition_valid(rule) for rule in rules):
                issues.append("INVALID_" + field.upper())
        focus = {str(c["code"]).zfill(6): c for c in report.get("focus_candidates", [])}
        links = outlook.get("selection_links", [])
        codes = [str(link.get("code", "")).zfill(6) for link in links]
        if len(codes) != len(set(codes)) or set(codes) != set(focus):
            issues.append("SELECTION_LINK_COVERAGE_MISMATCH")
        for link, code in zip(links, codes):
            if link.get("board_group") != focus.get(code, {}).get("board_group"):
                issues.append("SELECTION_GROUP_MISMATCH")
            for field in ("continuation_driver", "exhaustion_risk", "relative_advantage",
                          "main_path_action", "alternative_path_action"):
                if not isinstance(link.get(field), str) or not link[field].strip():
                    issues.append("MISSING_SELECTION_" + field.upper())
            for field in ("support_evidence_ids", "opposing_evidence_ids"):
                refs = link.get(field, [])
                if not refs or any(ref not in lookup for ref in refs):
                    issues.append("INVALID_SELECTION_" + field.upper())
            candidate = focus.get(code, {})
            for field, candidate_field in (("entry_condition_ids", "opening_conditions"),
                                           ("cancel_condition_ids", "cancel_conditions")):
                declared = {r.get("condition_id") for r in candidate.get(candidate_field, [])}
                refs = link.get(field, [])
                if not refs or not set(refs) <= declared:
                    issues.append("UNLINKED_" + field.upper())
    except (KeyError, TypeError, ValueError, AttributeError):
        issues.append("MALFORMED_OUTLOOK")
    return {"version": "emotion-selection-contract-1", "issues": sorted(set(issues)),
            "status": "INVALID" if issues else "STRUCTURALLY_VALID_RESEARCH_PLAN",
            "predictive_accuracy_verified": False, "entry_authorized": False,
            "calendar_session_verified": False}
