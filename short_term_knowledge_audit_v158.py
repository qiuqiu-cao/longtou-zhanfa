"""Audit the short-term knowledge hypothesis library before model use."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED = {
    "hypothesis_id", "experience", "source_url", "observable_proxy",
    "expected_direction", "counterexample", "test_status", "notes",
}
PRODUCTION_READY_STATUS = "样本外稳定通过"


def audit(frame: pd.DataFrame) -> dict:
    missing = REQUIRED - set(frame)
    if missing:
        raise ValueError(f"knowledge fields missing: {sorted(missing)}")
    if frame.empty or frame.hypothesis_id.duplicated().any():
        raise ValueError("knowledge library must be non-empty with unique ids")
    empty = {}
    for column in REQUIRED:
        bad = frame[column].isna() | frame[column].astype(str).str.strip().eq("")
        if bad.any():
            empty[column] = frame.loc[bad, "hypothesis_id"].tolist()
    invalid_urls = frame.loc[
        ~frame.source_url.astype(str).str.startswith(("https://", "http://")),
        "hypothesis_id",
    ].tolist()
    no_counterexample = frame.loc[
        frame.counterexample.astype(str).str.len().lt(4), "hypothesis_id"
    ].tolist()
    production = frame.test_status.astype(str).eq(PRODUCTION_READY_STATUS)
    categories = {
        "emotion_and_loss_transmission": ["H001", "H004", "H006", "H009"],
        "identity_and_theme_feedback": ["H002", "H005", "H007"],
        "cross_theme_and_high_low_switch": ["H008", "H010", "H012"],
        "intraday_microstructure": ["H003", "H011", "H013"],
        "execution_and_cost": ["H014", "H020", "H022"],
        "regulation_and_observability": ["H015", "H016"],
        "target_and_validation": ["H017"],
        "expectation_and_auction": ["H018", "H024", "H028"],
        "competition_and_attention_transfer": ["H019", "H023"],
        "late_board_and_repair": ["H021", "H025"],
        "regime_size_and_exit_asymmetry": ["H026", "H027"],
        "post_close_disclosure_and_flow_proxies": ["H029", "H030"],
        "entry_payoff_and_failed_seal_path": ["H031", "H032"],
        "normalization_and_theme_deduplication": ["H033", "H034", "H035"],
        "event_identity_and_propagation_order": ["H036", "H037", "H038"],
        "capacity_and_t1_risk": ["H039", "H040"],
        "theme_membership_driver_and_expectation_separation": [
            "H041", "H042", "H043", "H044", "H045"],
        "confidence_aware_cross_theme_races": ["H046", "H047", "H048"],
        "delayed_price_discovery_and_ladder_gaps": ["H049", "H050"],
        "rotation_flow_and_intraday_causality": ["H051", "H052", "H053"],
        "profit_concentration_and_distribution": ["H054"],
        "breadth_expansion_quality": ["H055"],
        "height_anchor_identity_vs_buyability": ["H056"],
        "opening_gap_strength_payoff_tradeoff": ["H057"],
        "opening_gap_emotion_interaction": ["H058"],
        "weak_to_strong_hindsight_conditioning": ["H059"],
        "first_board_supply_base_rate": ["H060"],
        "repair_day_followthrough_quality": ["H061"],
        "repair_layer_transmission": ["H062"],
        "high_to_low_ecology_window": ["H063"],
        "first_relay_lane_attention_competition": ["H064"],
        "severe_abnormal_reset_evidence_hierarchy": ["H065"],
        "fast_board_structure_vs_control_intent": ["H066"],
        "trading_calendar_vs_observation_dates": ["H067"],
        "theme_evidence_attention_and_coverage_bias": ["H068"],
        "win_rate_expectancy_and_tail_risk": ["H069"],
        "selective_tail_risk_veto_and_coverage": ["H070"],
        "opening_information_and_execution_causality": ["H071"],
        "t1_unexitable_loss_and_down_limit_queue": ["H072"],
        "exit_information_and_timing_causality": ["H073"],
        "broad_index_vs_short_term_ecology": ["H074"],
    }
    present = set(frame.hypothesis_id.astype(str))
    missing_categories = {
        name: sorted(set(ids) - present) for name, ids in categories.items()
        if set(ids) - present
    }
    passed = not empty and not invalid_urls and not no_counterexample and not missing_categories
    return {
        "version": "v203-short-term-knowledge-audit-34",
        "status": "HYPOTHESIS_LIBRARY_READY_FOR_TESTING" if passed else "INVALID_LIBRARY",
        "hypothesis_count": int(len(frame)),
        "production_ready_count": int(production.sum()),
        "direct_model_use_allowed": bool(production.all() and len(frame) > 0),
        "empty_fields": empty,
        "invalid_source_url_ids": invalid_urls,
        "missing_counterexample_ids": no_counterexample,
        "missing_category_hypotheses": missing_categories,
        "categories": categories,
        "policy": (
            "Community and practitioner claims generate hypotheses only. Each must have a "
            "point-in-time proxy, counterexample, isolated out-of-sample test and execution-cost audit."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="forum_hypotheses.csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable knowledge audit")
    report = audit(pd.read_csv(args.input))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
