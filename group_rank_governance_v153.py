"""Govern first-board and relay rank-uplift claims after the no-touch fix."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


VERSION = "v153-group-rank-governance-1"
CHALLENGER = "identity_emotion_price_volume"
BASELINE = "highest_board_baseline"
GROUPS = ("FIRST_BOARD", "CONSECUTIVE_BOARD")


def evaluate(report, minimum_sessions=100):
    coverage = report.get("target_coverage", {})
    if (
        report.get("status") != "EXPLORATORY_ONLY"
        or coverage.get("explicit_touch_field_used") is not True
        or coverage.get("no_touch_rows_retained_as_negative") is not True
    ):
        raise ValueError("walk-forward report must use explicit no-touch negatives")
    aggregate = report.get("aggregate_selected_by_board_group", {})
    if CHALLENGER not in aggregate or BASELINE not in aggregate:
        raise ValueError("required frozen challenger and baseline missing")
    groups = {}
    for group in GROUPS:
        candidate = aggregate[CHALLENGER].get(group)
        baseline = aggregate[BASELINE].get(group)
        if not candidate or not baseline:
            raise ValueError("board-group evidence missing")
        paired = candidate.get("paired_vs_group_baseline", {})
        ci = paired.get("bootstrap95", [None, None])
        n = int(paired.get("paired_sessions", 0))
        uplift = paired.get("difference")
        supported = bool(
            n >= minimum_sessions and uplift is not None and uplift > 0
            and ci[0] is not None and ci[0] > 0
        )
        b2 = candidate.get("path_proxy", {}).get("b2", {})
        groups[group] = {
            "paired_sessions": n,
            "challenger_proxy_success_rate": candidate.get("proxy_success_rate"),
            "baseline_proxy_success_rate": baseline.get("proxy_success_rate"),
            "paired_difference": uplift,
            "paired_bootstrap95": ci,
            "rank_uplift_supported": supported,
            "conditional_b2_known": b2.get("known"),
            "conditional_b2_positive_rate": b2.get("positive_rate"),
            "conditional_b2_large_loss_rate": b2.get("large_loss_rate"),
            "conditional_b2_mean_return": b2.get("mean_net_close_mark"),
            "conditional_path_warning": (
                "B+2 path statistics condition on a hypothetical touch/entry path and are not a full-pool trade rate"
            ),
        }
    all_supported = all(item["rank_uplift_supported"] for item in groups.values())
    return {
        "version": VERSION,
        "status": "GROUP_RANK_UPLIFT_SUPPORTED_RESEARCH_ONLY" if all_supported else "GROUP_RANK_UPLIFT_NOT_PROVEN",
        "challenger": CHALLENGER,
        "baseline": BASELINE,
        "target": "next-session non-one-price touch and positive B+2 close-mark proxy",
        "target_no_touch_negative_fixed": True,
        "groups": groups,
        "first_board_probability_improvement_claim_allowed": groups["FIRST_BOARD"]["rank_uplift_supported"],
        "relay_probability_improvement_claim_allowed": groups["CONSECUTIVE_BOARD"]["rank_uplift_supported"],
        "complete_trade_probability_claim_allowed": False,
        "deployment_allowed": False,
        "next_action": (
            "add V150 point-in-time specific-theme and competition factors, then rerun on a new isolated period"
            if not all_supported else
            "freeze challenger and collect a new untouched full-market period with actual execution evidence"
        ),
        "limitations": [
            "historical universe is not proven full-market complete",
            "features were reconstructed after the fact",
            "auction, queue, actual fills and frozen exits are absent",
            "the same exploratory windows have already been inspected",
        ],
    }


def run(input_path, output_path, minimum_sessions=100):
    source, target = Path(input_path), Path(output_path)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable governance report")
    result = evaluate(json.loads(source.read_text(encoding="utf-8")), minimum_sessions)
    result["source"] = {"path": str(source.resolve()),
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-sessions", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output, args.minimum_sessions),
                     ensure_ascii=False, indent=2))

