"""V267 bridge: add cycle/fundamental cleanliness to the existing review state.

The V266 bridge remains untouched for reproducibility.  This wrapper adds the
V267 context report as an additional diagnostic gate and keeps trade
authorization permanently false.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from strategy_review_bridge_v266 import _load, build_bridge as build_v266


VERSION = "v267-strategy-review-bridge-1"


def build_bridge_v267(
    review: dict[str, Any],
    strategy_audit: dict[str, Any],
    input_contract: dict[str, Any],
    cycle_gate: dict[str, Any],
    *,
    review_path: str | Path | None = None,
    strategy_path: str | Path | None = None,
    contract_path: str | Path | None = None,
    cycle_gate_path: str | Path | None = None,
    reference_asof: str | None = None,
) -> dict[str, Any]:
    report = build_v266(
        review, strategy_audit, input_contract,
        review_path=review_path, strategy_path=strategy_path,
        contract_path=contract_path, reference_asof=reference_asof,
    )
    context_review = cycle_gate.get("cycle_fundamental_context_review") or {}
    context_allowed = context_review.get("allowed") is True
    blockers = list(report.get("blocking_reasons") or [])
    if not context_allowed:
        blockers.append("V267_CYCLE_FUNDAMENTAL_CONTEXT_UNAVAILABLE")
    report["version"] = VERSION
    report["blocking_reasons"] = sorted(set(blockers))
    report["cycle_fundamental"] = {
        "context_review_allowed": context_allowed,
        "cycle_review": cycle_gate.get("cycle_review"),
        "fundamental_cleanliness_review": cycle_gate.get("fundamental_cleanliness_review"),
        "row_state_counts": cycle_gate.get("row_state_counts"),
    }
    report.setdefault("gates", {})["cycle_fundamental_context_review_allowed"] = context_allowed
    report.setdefault("governance", {})["cycle_fundamental_is_diagnostic_only"] = True
    report.setdefault("governance", {})["trade_authorization_requires_live_verified_fill_and_exit"] = True
    report["source_hashes"].update({
        str(cycle_gate_path): cycle_gate.get("source_hash")
        for cycle_gate_path in [cycle_gate_path]
        if cycle_gate_path and cycle_gate.get("source_hash")
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-review", required=True)
    parser.add_argument("--strategy-audit", required=True)
    parser.add_argument("--input-contract", required=True)
    parser.add_argument("--cycle-fundamental-gate", required=True)
    parser.add_argument("--reference-asof")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable V267 bridge output")
    report = build_bridge_v267(
        _load(args.daily_review), _load(args.strategy_audit),
        _load(args.input_contract), _load(args.cycle_fundamental_gate),
        review_path=args.daily_review, strategy_path=args.strategy_audit,
        contract_path=args.input_contract, cycle_gate_path=args.cycle_fundamental_gate,
        reference_asof=args.reference_asof,
    )
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"version": VERSION, "decision": report["decision"], "context_review_allowed": report["cycle_fundamental"]["context_review_allowed"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
