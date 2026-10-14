"""Audit whether reconstructed emotion features add stable ranking value."""
import argparse
import hashlib
import json
from pathlib import Path


def _number(value, label):
    if not isinstance(value, (int, float)):
        raise ValueError("missing numeric emotion audit field: " + label)
    return float(value)


def audit(rank_report, selective_report):
    if rank_report.get("target") != selective_report.get("target"):
        raise ValueError("rank and selective targets differ")
    if rank_report.get("status") != "EXPLORATORY_ONLY":
        raise ValueError("rank evidence is not exploratory proxy output")
    base = rank_report["groups"]["identity"]
    emotion = rank_report["groups"]["identity_plus_emotion"]
    all_delta = {
        "candidate_auc_delta": (_number(emotion["all_candidates"]["candidate_auc"], "emotion auc")
                                - _number(base["all_candidates"]["candidate_auc"], "identity auc")),
        "candidate_brier_improvement": (_number(base["all_candidates"]["candidate_brier"], "identity brier")
                                        - _number(emotion["all_candidates"]["candidate_brier"], "emotion brier")),
        "daily_top1_success_delta": (_number(emotion["daily_top1"]["proxy_success_rate"], "emotion top1")
                                     - _number(base["daily_top1"]["proxy_success_rate"], "identity top1")),
    }
    windows = {}
    stable_active_uplift = True
    for name, evidence in selective_report["windows"].items():
        all_rate = _number(evidence["unconditional_top1"]["proxy_precision"], name + " all")
        active_rate = _number(evidence["emotion_active_top1"]["proxy_precision"], name + " active")
        delta = active_rate - all_rate
        windows[name] = {
            "all_session_top1_proxy_rate": all_rate,
            "active_state_top1_proxy_rate": active_rate,
            "active_state_delta": delta,
            "active_state_selections": evidence["emotion_active_top1"]["selections"],
        }
        stable_active_uplift &= delta > 0
    additive_stable = (all_delta["candidate_auc_delta"] > 0
                       and all_delta["candidate_brier_improvement"] > 0
                       and all_delta["daily_top1_success_delta"] > 0)
    passed = additive_stable and stable_active_uplift
    return {
        "status": "EMOTION_POSITIVE_FACTOR_ACCEPTED" if passed else "EMOTION_POSITIVE_FACTOR_REJECTED",
        "target": rank_report["target"],
        "factor_scope": "reconstructed_same_day_emotion_for_full_pool_opportunity_ranking",
        "identity_to_emotion_deltas": all_delta,
        "active_state_filter_windows": windows,
        "additive_rank_uplift_stable": additive_stable,
        "active_state_filter_uplift_stable": stable_active_uplift,
        "positive_buy_score_authorized": passed,
        "risk_context_use_only": not passed,
        "deployment_authorized": False,
        "interpretation": (
            "Emotion may still control permission, sizing, and confirmation strictness, but this evidence "
            "does not authorize a positive stock-level buy-score bonus."
        ),
        "limitations": [
            "reconstructed historical features are not original forward archives",
            "the outcome is a price-opportunity proxy rather than a completed trade",
            "the later research window has already been inspected and is not blind",
            "risk-gating value needs a separately labeled complete-trade loss audit",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-report", type=Path, required=True)
    parser.add_argument("--selective-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite immutable emotion audit")
    result = audit(json.loads(args.rank_report.read_text(encoding="utf-8")),
                   json.loads(args.selective_report.read_text(encoding="utf-8")))
    result["source_hashes"] = {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (args.rank_report, args.selective_report, Path(__file__))
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
