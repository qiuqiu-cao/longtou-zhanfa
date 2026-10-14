"""Ensure the narrative daily report and quantitative primary decision agree."""
import argparse
import hashlib
import json
from pathlib import Path

from daily_decision_contract import audit as audit_report
from leader_target_screen_v17 import finite, timestamp


def audit(report, selection, asof):
    cutoff = timestamp(asof)
    report_result = audit_report(report, asof)
    issues = []
    if selection.get("asof") != cutoff.isoformat() or report.get("asof") != cutoff.isoformat():
        issues.append("ASOF_NOT_IDENTICAL")
    if selection.get("trade_date") != report.get("report_date"):
        issues.append("TRADE_DATE_NOT_IDENTICAL")
    if selection.get("direct_recommendation") is not False or selection.get("order_submitted") is not False:
        issues.append("SELECTOR_MUST_NOT_RECOMMEND_OR_SUBMIT_ORDER")
    selected = selection.get("primary_choice")
    reported = report.get("primary_choice")
    if selection.get("status") == "CASH":
        if selected is not None or reported is not None or report.get("final_decision") != "CASH":
            issues.append("CASH_DECISIONS_DIVERGE")
        if report_result.get("status") != "VALID_CASH_REPORT":
            issues.append("CASH_REPORT_CONTRACT_FAILED")
    elif selection.get("status") == "CONDITIONAL_PRIMARY_READY":
        try:
            code = str(selected["code"]).zfill(6)
            if (str(reported["code"]).zfill(6) != code or reported.get("conditional") is not True
                    or report.get("final_decision") != "CONDITIONAL_PRIMARY"
                    or report_result.get("status") != "VALID_CONDITIONAL_REPORT"
                    or report_result.get("entry_permission") != "CONDITIONAL_ENTRY_PLAN_ONLY"):
                issues.append("CONDITIONAL_DECISIONS_DIVERGE")
            candidate = next(row for row in report.get("focus_candidates", []) if str(row.get("code", "")).zfill(6) == code)
            entry = candidate["entry_plan"]
            if (finite(entry.get("max_price")) != finite(selected.get("maximum_acceptable_price"))
                    or finite(entry.get("max_shares")) != finite(selected.get("max_shares"))
                    or code not in {str(value).zfill(6) for value in report.get("roles", {}).get("current_buy_opportunity", [])}):
                issues.append("ENTRY_OR_ROLE_FIELDS_DIVERGE")
            if selected.get("rank_score_meaning") != "opportunity_proxy_score_not_complete_trade_probability":
                issues.append("RANK_SCORE_MEANING_CHANGED")
        except (KeyError, StopIteration, TypeError, ValueError):
            issues.append("CONDITIONAL_PACKAGE_FIELDS_MISSING")
    else:
        issues.append("INVALID_SELECTOR_STATUS")
    if report.get("system_status") not in {"PAPER_TRADE_READY", "FORWARD_VALIDATED"}:
        # A diagnostic cash report remains valid, but never a ready decision package.
        issues.append("REPORT_SYSTEM_NOT_READY")
    ready = not issues
    mode = selection.get("mode")
    status = ("PACKAGE_READY_FOR_USER_REVIEW" if ready and mode == "CONDITIONAL_USER_DECISION"
              else "PACKAGE_READY_FOR_LOCKED_PAPER" if ready else "PACKAGE_NOT_ACTIONABLE")
    return {"status": status, "issues": sorted(set(issues)), "report_contract": report_result,
            "primary_code": str(selected["code"]).zfill(6) if ready and selected else None,
            "direct_recommendation": False, "order_submitted": False,
            "guarantee": False, "selector_and_narrative_match": ready}


def run(report_path, selection_path, output_path, asof):
    report_path, selection_path, target = map(Path, (report_path, selection_path, output_path))
    if target.exists():
        raise FileExistsError("choose a new immutable daily package audit")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    result = audit(report, selection, asof)
    result["source_hashes"] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (report_path, selection_path, Path(__file__))}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--asof", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.report, args.selection, args.output, args.asof), ensure_ascii=False, indent=2))
