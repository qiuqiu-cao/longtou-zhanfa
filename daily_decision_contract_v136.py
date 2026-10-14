"""Closing-plan audit combining the existing contract and emotion-selection links.

New version only: never rewrites historical audits or authorizes execution.
"""
import argparse
import hashlib
import json
from pathlib import Path

from daily_decision_contract import audit as audit_base
from emotion_selection_contract import audit_outlook


def audit(report):
    base = audit_base(report)
    outlook = audit_outlook(report)
    issues = list(base["issues"]) + ["OUTLOOK_" + x for x in outlook["issues"]]
    if report.get("stage") != "CLOSING_PLAN":
        issues.append("V136_REQUIRES_CLOSING_PLAN")
    return dict(version="daily-decision-contract-136", status="INVALID_REPORT" if issues else
                "VALID_CONDITIONAL_RESEARCH_PLAN", issues=sorted(set(issues)),
                base_audit=base, outlook_audit=outlook, entry_permission="NO_ENTER",
                predictive_accuracy_verified=False, order_submitted=False)


def run(source, target):
    source, target = Path(source), Path(target)
    result = audit(json.loads(source.read_text(encoding="utf-8")))
    dependencies = [source, Path(__file__), Path(__file__).with_name("daily_decision_contract.py"),
                    Path(__file__).with_name("emotion_selection_contract.py")]
    result["source_hashes"] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in dependencies}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.input, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(1 if result["issues"] else 0)
