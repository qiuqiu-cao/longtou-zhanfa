"""Frozen research-only 10:00 cumulative-VWAP shadow gate."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


VERSION = "v139-absorption-forward-shadow-1"


def classify(asof, minute_status, close_to_vwap):
    if asof != "10:00":
        return "NOT_DUE_BEFORE_1000"
    if minute_status != "VERIFIED_PUBLIC_MINUTE_BAR_FRESH" or close_to_vwap is None:
        return "DATA_LIMITED"
    try:
        value = float(close_to_vwap)
    except (TypeError, ValueError):
        return "DATA_LIMITED"
    return "SHADOW_PASS_ABOVE_OR_AT_VWAP" if value >= 0 else "SHADOW_VETO_BELOW_VWAP"


def protocol():
    return {
        "version": VERSION, "status": "FROZEN_RESEARCH_ONLY",
        "effective_for_new_records_after_freeze": True,
        "observation_time": "10:00 Asia/Shanghai",
        "metric": "last completed public one-minute close / cumulative session VWAP - 1",
        "pass_rule": ">= 0", "veto_rule": "< 0", "missing_rule": "DATA_LIMITED",
        "scope": "all frozen close-review focus candidates, first and consecutive boards reported separately",
        "live_decision_effect": "NONE_SHADOW_ONLY",
        "entry_authorized": False, "automatic_order": False,
        "retuning_before_gate": False,
        "minimum_forward_gate": {
            "settled_candidate_records": 100, "each_board_group": 30,
            "required_metrics": ["complete-trade net win rate", "mean net return", "large-loss rate",
                                 "profit factor", "account maximum drawdown"],
            "actual_or_conservative_fill_required": True,
        },
        "limitations": ["public one-minute OHLCV is not tick or Level-2 evidence",
                        "VWAP pass is not a buy signal", "historical association is not causal proof"],
    }


def freeze(output: Path):
    if output.exists():
        raise FileExistsError("refusing to overwrite frozen protocol")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = protocol()
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"path": str(output.resolve()), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.output), ensure_ascii=False, indent=2))
