"""Bind a frozen next-session expectation to its exact daily review.

The adapter records missing plans and rejects incomplete/cross-date bindings.
It neither fits a model nor changes trading permission. CLI timestamps cannot
be backdated; historical replay remains distinct from forward registration.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from next_session_expectation_contract_v271 import (
    digest, load_registered, now, stamp, validate_plan,
)

VERSION = "v277-review-forecast-binding-2"
KEY = "next_session_expectation"


def review_digest(review):
    core = {k: v for k, v in review.items() if k != KEY}
    return digest(core)


def audit(review, registered=None):
    result = {"version": VERSION, "review_hash": review_digest(review),
              "status": "NOT_REGISTERED", "issues": [],
              "forward_accuracy_eligible": False, "trade_authorized": False,
              "original_trade_status": review.get("status"),
              "automatic_replacement_allowed": False}
    if registered is None:
        result["issues"] = ["FROZEN_NEXT_SESSION_PLAN_NOT_PROVIDED"]
        return result
    content = {k: v for k, v in registered.items() if k != "sha256"}
    if digest(content) != registered.get("sha256"):
        raise ValueError("frozen plan hash mismatch")
    plan = registered["plan"]
    validation = validate_plan(plan, registered["recorded_at"])
    issues = []
    if validation["forecast_status"] != registered.get("forecast_status"):
        issues.append("REGISTRATION_STATUS_MISMATCH")
    if plan.get("review_hash") != result["review_hash"]:
        issues.append("EXACT_REVIEW_HASH_MISMATCH")
    if review.get("signal_date") != plan["signal_date"]:
        issues.append("SIGNAL_DATE_MISMATCH")
    if stamp(review["asof"]) > stamp(plan.get("knowledge_cutoff", plan["evidence_cutoff"])):
        issues.append("REVIEW_NOT_AVAILABLE_AT_EVIDENCE_CUTOFF")
    pool = review.get("relay_candidates", [])
    current = {r["code"]: r["boards"] for r in pool}
    expected = {r["code"]: r["boards"] for r in plan["candidates"]}
    if len(current) != len(pool):
        issues.append("DUPLICATE_REVIEW_CANDIDATE")
    if set(current) != set(expected):
        issues.append("RELAY_DENOMINATOR_MISMATCH")
    elif current != expected:
        issues.append("CANDIDATE_BOARD_MISMATCH")
    if review.get("pool_summary", {}).get("relay") != len(pool):
        issues.append("REVIEW_RELAY_COUNT_MISMATCH")
    primary = next((r for r in plan["candidates"]
                    if r["code"] == plan.get("primary_candidate")), None)
    if primary and primary["promotion_call"] != "PROMOTE":
        issues.append("PRIMARY_IS_NOT_A_POSITIVE_PROMOTION_CALL")
    # Linking does not prove exchange-wide coverage, point-in-time security
    # status or theme completeness. Preserve the report's original scope.
    result.update({"status": "INVALID_BINDING" if issues else "BOUND",
                   "issues": sorted(set(issues)),
                   "registration_hash": registered["sha256"],
                   "forecast_status": validation["forecast_status"],
                   "signal_date": plan["signal_date"], "target_date": plan["target_date"],
                   "market_primary": plan["market"]["primary"],
                   "market_direction": plan["market"].get("direction"),
                   "primary_candidate": plan.get("primary_candidate"),
                   "identity_primary_candidate": plan.get("identity_primary_candidate"),
                   "non_one_word_primary_candidate": plan.get("non_one_word_primary_candidate"),
                   "knowledge_cutoff": validation["knowledge_cutoff"],
                   "relay_denominator": len(pool),
                   "uncertain_calls": sum(r["promotion_call"] == "UNCERTAIN" for r in plan["candidates"]),
                   "scope": review.get("decision_scope", "UNSPECIFIED"),
                   "execution_state": "PENDING_OPEN_CONFIRMATION",
                   "forward_accuracy_eligible": not issues and validation["forecast_status"] == "FORWARD_REGISTERED",
                   "accuracy_claim_allowed": False,
                   "required_outcome": "EXACT_TARGET_SESSION_VERIFIED_COMPLETE_SCOPE",
                   "probability": plan["market"].get("probability") if not issues else None})
    return result


def attach(review, registered=None):
    review[KEY] = audit(review, registered)
    return review


def markdown(binding):
    return ("\n## 隔日预期留档核对\n\n"
            f"- 核对状态：`{binding['status']}`；事前登记可待结算：`{binding['forward_accuracy_eligible']}`。\n"
            f"- 缺口：`{', '.join(binding['issues']) or '无绑定错误'}`。\n"
            "- 事前登记仅表示预测已保存；准确率需下一交易日结算，交易仍须竞价和盘中确认。\n")


def run(review_path, output, registered_path=None):
    target = Path(output)
    if target.exists():
        raise FileExistsError("refusing to overwrite forecast binding")
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    registered = load_registered(registered_path) if registered_path else None
    bound = attach(deepcopy(review), registered)
    bound[KEY]["binding_recorded_at"] = now().isoformat()
    target.mkdir(parents=True, exist_ok=False)
    (target / "review.json").write_text(json.dumps(bound, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (target / "binding.md").write_text(markdown(bound[KEY]), encoding="utf-8")
    return bound[KEY]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", required=True)
    parser.add_argument("--registered")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.review, args.output, args.registered), ensure_ascii=True))
