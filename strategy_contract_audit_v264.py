"""Audit the V263 leader strategy against the user's research contract.

This is a fail-closed evidence audit. It does not promote diagnostic proxies
to live trading permission; unavailable theme, auction, intraday and fill
evidence are reported explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(report: dict) -> dict:
    stages = report.get("aggregate", {})
    required_stages = {"A_first_board", "B_second_board", "C_third_board"}
    stage_ok = required_stages.issubset(stages) and all(
        all(k in stages[name].get("models", {}).get("model", {})
            for k in ("top1", "top3", "top5"))
        for name in required_stages
    )
    model_stage = stages.get("B_second_board", {}).get("models", {}).get("model", {})
    forward = model_stage.get("top1_forward_bar_proxy", {})
    forward_ok = all(forward.get(f"{h}d", {}).get("known_rows", 0) > 0
                     for h in (1, 3, 5))
    env_ok = all(
        bool(stages.get(name, {}).get("models", {}).get("model", {}).get(
            "top1_by_environment")) for name in required_stages
    )
    failure_ok = all("failure_category_counts_top1" in stages.get(name, {})
                     for name in required_stages)
    interpretation = report.get("interpretation", {})
    audit_rows = [
        {"requirement": "A/B/C Top1/Top3/Top5 identity metrics", "status": "PASS" if stage_ok else "MISSING",
         "evidence": "aggregate.*.models.model.top1/top3/top5"},
        {"requirement": "future max board and future gain", "status": "PASS" if stage_ok else "MISSING",
         "evidence": "aggregate.*.models.model.top1_future_max_board_mean/top1_future_gain_mean"},
        {"requirement": "1/3/5 day return and risk proxy", "status": "PASS" if forward_ok else "MISSING",
         "evidence": "aggregate.B_second_board.models.model.top1_forward_bar_proxy"},
        {"requirement": "environment-stratified results", "status": "PASS" if env_ok else "MISSING",
         "evidence": "aggregate.*.models.model.top1_by_environment"},
        {"requirement": "failure taxonomy", "status": "PASS" if failure_ok else "MISSING",
         "evidence": "aggregate.*.failure_category_counts_top1"},
        {"requirement": "verified topic-leader label", "status": "UNAVAILABLE",
         "evidence": report.get("data_audit", {}).get("theme_status")},
        {"requirement": "auction/Level-2/intraday execution", "status": "UNAVAILABLE",
         "evidence": "No timestamped auction, queue, VWAP or tick fields in V263 features"},
        {"requirement": "actual fills and executable win rate", "status": "UNAVAILABLE",
         "evidence": f"actual_fill_count={interpretation.get('actual_fill_count')}"},
        {"requirement": "pristine blind test", "status": "NOT_MET",
         "evidence": f"is_pristine_blind_test={interpretation.get('is_pristine_blind_test')}"},
        {"requirement": "survivorship-free universe", "status": "NOT_MET",
         "evidence": f"survivorship_risk={interpretation.get('survivorship_risk')}"},
    ]
    live_ready = all(row["status"] == "PASS" for row in audit_rows)
    return {
        "version": "v264-strategy-contract-audit-1",
        "status": "DIAGNOSTIC_ONLY",
        "live_ready": live_ready,
        "decision": "CASH" if not live_ready else "CONDITIONAL_REVIEW",
        "requirements": audit_rows,
        "source_report_status": report.get("status"),
        "source_version": report.get("version"),
        "governance": {
            "strategy_upgrade_allowed": False,
            "strategy_upgrade_reason": "identity diagnostics improved coverage, but theme/execution/blind/universe gates are not passed",
            "topic_leader_unavailable_blocks_topic_claim": True,
            "execution_evidence_unavailable_blocks_entry_authorization": True,
            "diagnostic_identity_may_not_be_presented_as_trade_win_rate": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("immutable output already exists")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = audit(report)
    result["source_hash"] = {str(args.report.resolve()): _sha(args.report)}
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "live_ready": result["live_ready"],
                      "decision": result["decision"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
