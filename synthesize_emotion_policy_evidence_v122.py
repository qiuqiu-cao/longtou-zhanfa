"""Synthesize aggregate-market and board-group economic evidence for policy use."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthesize(aggregate: dict, board: dict) -> dict:
    if aggregate.get("status") != "DIAGNOSTIC_ONLY" or board.get("status") != "DIAGNOSTIC_ONLY":
        raise ValueError("emotion evidence must be diagnostic")
    groups = board.get("board_groups", {})
    if set(groups) != {"FIRST_BOARD", "CONSECUTIVE_BOARD"}:
        raise ValueError("first-board and consecutive-board evidence both required")
    aggregate_passed = bool(aggregate.get("economic_gate_passed"))
    first_passed = bool(groups["FIRST_BOARD"].get("risk_overlay_proxy_gate_passed"))
    relay_passed = bool(groups["CONSECUTIVE_BOARD"].get("risk_overlay_proxy_gate_passed"))
    complete_trades = int(board.get("complete_trade_count", 0))
    return {
        "version": "v122-emotion-policy-evidence-synthesis-1",
        "status": "DIAGNOSTIC_ONLY",
        "evidence_summary": {
            "aggregate_market_damage_gate_passed": aggregate_passed,
            "aggregate_direction_pass_count": aggregate.get("direction_pass_count"),
            "aggregate_strong_ci_pass_count": aggregate.get("strong_ci_pass_count"),
            "first_board_return_proxy_gate_passed": first_passed,
            "first_board_direction_pass_count": groups["FIRST_BOARD"].get("direction_pass_count"),
            "first_board_strong_ci_pass_count": groups["FIRST_BOARD"].get("strong_ci_pass_count"),
            "relay_return_proxy_gate_passed": relay_passed,
            "relay_direction_pass_count": groups["CONSECUTIVE_BOARD"].get("direction_pass_count"),
            "relay_strong_ci_pass_count": groups["CONSECUTIVE_BOARD"].get("strong_ci_pass_count"),
            "complete_trade_count": complete_trades,
        },
        "interpretation": {
            "market_level": (
                "emotion risk is supported as an aggregate damage warning"
                if aggregate_passed else "aggregate damage warning lacks its prespecified evidence gate"
            ),
            "candidate_level": (
                "risk probability has not shown stable economic separation after conditioning on a non-one-price limit touch"
            ),
            "selection_effect": (
                "surviving candidates in adverse markets may be unusually strong; this is an inference, not proof of tradability"
            ),
        },
        "policy_governance": {
            "aggregate_risk_warning_allowed": aggregate_passed,
            "emotion_positive_buy_bonus_allowed": False,
            "claim_emotion_veto_improves_first_board_returns": first_passed and complete_trades > 0,
            "claim_emotion_veto_improves_relay_returns": relay_passed and complete_trades > 0,
            "blanket_relay_veto_promoted_to_live": False,
            "existing_frozen_paper_veto": "KEEP_PAPER_ONLY_PENDING_COMPLETE_TRADE_FORWARD_EVIDENCE",
            "allowed_uses": [
                "raise confirmation strictness",
                "reduce research position cap",
                "require stronger theme and VWAP synchronization",
                "default to cash when opening evidence is missing",
            ],
            "forbidden_claims": [
                "high risk guarantees relay loss",
                "low risk guarantees candidate premium",
                "aggregate market damage proves candidate-level return improvement",
            ],
        },
        "next_validation_gate": {
            "minimum_completed_executable_trades_total": 100,
            "minimum_completed_trades_each_board_group": 30,
            "required_outcomes": [
                "complete-trade net win rate", "mean net return", "large-loss rate",
                "profit factor", "maximum drawdown",
            ],
            "time_ordered": True,
            "frozen_rule_required": True,
        },
        "deployment_authorized": False,
        "limitations": [
            "aggregate outcomes are daily market statistics",
            "candidate outcomes are daily-bar opportunity proxies without actual fills",
            "the two audits condition on different populations and are not contradictory causal estimates",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-audit", required=True)
    parser.add_argument("--board-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    aggregate_path, board_path, output = map(
        Path, (args.aggregate_audit, args.board_audit, args.output),
    )
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable synthesis")
    report = synthesize(
        json.loads(aggregate_path.read_text(encoding="utf-8")),
        json.loads(board_path.read_text(encoding="utf-8")),
    )
    report["source_hashes"] = {
        str(aggregate_path.resolve()): sha256(aggregate_path),
        str(board_path.resolve()): sha256(board_path),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
