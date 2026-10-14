"""Build an evidence-first daily full-pool review; never submit an order."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

import symbolic_name_theme_v217 as symbolic_name_theme


STATE_FLAGS = {
    "启动": "state_start", "升温": "state_warming", "加速": "state_acceleration",
    "一致": "state_consensus", "大分歧": "state_divergence", "退潮": "state_retreat",
    "修复": "state_repair",
}
COMPETITION_IDENTITIES = {
    "pioneer", "turnover_core", "height_anchor", "capacity_core",
    "supplement", "unique_survivor",
}


def _float(row, key):
    value = pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else None


def _display_percent(value, digits=1):
    """Render a percentage without turning an unavailable value into zero."""
    if value is None:
        return "不可用"
    return f"{value:.{digits}%}"


def _display_number(value):
    return "不可用" if value is None else str(value)


def _integer(row, key):
    value = _float(row, key)
    return int(value) if value is not None and value.is_integer() else None


def _truthy(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "1.0", "true", "yes"}


def _local_naive_timestamp(value):
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return pd.NaT
    if pd.isna(stamp):
        return pd.NaT
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("Asia/Shanghai").tz_localize(None)
    return stamp


def _height_anchor_buyability_summary(evidence):
    """Bind research evidence without turning height identity into a trade signal."""
    if evidence is None:
        return {
            "status": "UNAVAILABLE",
            "source_version": None,
            "corrected_target_no_touch_negative": None,
            "validation_sessions": None,
            "boards_only_daily_top1_success_rate": None,
            "challenger_daily_top1_success_rate": None,
            "challenger_uplift_supported": False,
            "candidate_probability_claim_allowed": False,
            "entry_authorized": False,
            "interpretation": "最高板仅保留情绪身份；没有绑定纠正标签研究时，不得推导可买性。",
        }
    required = {
        "version", "status", "target_contract", "split", "models",
        "daily_top1_within_height_anchors", "height_anchor_buyability_uplift_supported",
        "governance", "interpretation",
    }
    if required - set(evidence):
        raise ValueError("height-anchor buyability evidence fields missing")
    target = evidence.get("target_contract", {})
    split = evidence.get("split", {})
    top1 = evidence.get("daily_top1_within_height_anchors", {})
    governance = evidence.get("governance", {})
    if (evidence.get("version") != "v181-height-anchor-buyability-1"
            or evidence.get("status") != "DIAGNOSTIC_ONLY"):
        raise ValueError("height-anchor buyability evidence version or status invalid")
    if (target.get("no_touch_rows_retained_as_negative") is not True
            or target.get("next_session_one_price_rows_retained_as_negative") is not True
            or target.get("actual_fills") != 0
            or target.get("complete_trades") != 0):
        raise ValueError("height-anchor target contract is not fail-closed")
    forbidden = (
        "height_anchor_can_auto_fill_relay_primary",
        "height_anchor_can_auto_fill_current_buy",
        "candidate_probability_claim_allowed",
        "direct_stock_reranking_allowed",
        "entry_authorized",
    )
    if any(governance.get(key) is not False for key in forbidden):
        raise ValueError("height-anchor evidence cannot authorize ranking or entry")
    if evidence.get("height_anchor_buyability_uplift_supported") is not False:
        raise ValueError("unsupported height-anchor uplift cannot be promoted")
    if split.get("validation_previously_inspected") is not True:
        raise ValueError("height-anchor validation inspection status must be explicit")
    return {
        "status": "BOUND_DIAGNOSTIC_ONLY",
        "source_version": evidence["version"],
        "corrected_target_no_touch_negative": True,
        "validation_sessions": split.get("validation_sessions"),
        "boards_only_daily_top1_success_rate": top1.get("baseline_success_rate"),
        "challenger_daily_top1_success_rate": top1.get("challenger_success_rate"),
        "challenger_uplift_supported": False,
        "candidate_probability_claim_allowed": False,
        "entry_authorized": False,
        "validation_previously_inspected": True,
        "interpretation": evidence.get("interpretation"),
    }


def _opening_gap_payoff_summary(evidence):
    """Bind opening-gap research as confirmation risk, never prior-close alpha."""
    if evidence is None:
        return {
            "status": "UNAVAILABLE", "source_version": None,
            "validation_sessions": None, "above_seven_one_price_rate": None,
            "five_to_seven_big_loss_rate": None, "above_seven_big_loss_rate": None,
            "prior_close_model_use_allowed": False,
            "opening_gap_direct_entry_allowed": False,
            "above_seven_default_risk_flag_allowed": False,
        }
    required = {
        "version", "status", "feature_contract", "target_contract", "split",
        "summaries", "directional_replication", "governance", "interpretation",
    }
    if required - set(evidence):
        raise ValueError("opening-gap payoff evidence fields missing")
    feature = evidence.get("feature_contract", {})
    target = evidence.get("target_contract", {})
    governance = evidence.get("governance", {})
    if (evidence.get("version") != "v182-opening-gap-payoff-tradeoff-1"
            or evidence.get("status") != "DIAGNOSTIC_ONLY"):
        raise ValueError("opening-gap payoff evidence version or status invalid")
    if (feature.get("prior_close_model_use_allowed") is not False
            or feature.get("auction_09_25_equivalence_claim_allowed") is not False
            or feature.get("entry_open_gap_available_not_before")
            != "after_opening_call_auction_match_approximately_09:25"
            or feature.get("continuous_auction_begins_at") != "09:30"):
        raise ValueError("opening-gap evidence leaks after-close information")
    if (target.get("no_touch_rows_retained_as_negative") is not True
            or target.get("next_session_one_price_rows_retained_as_negative") is not True
            or target.get("actual_fills") != 0
            or target.get("complete_trades") != 0):
        raise ValueError("opening-gap target contract is not fail-closed")
    if (governance.get("opening_gap_direct_entry_allowed") is not False
            or governance.get("opening_gap_direct_stock_reranking_allowed") is not False
            or governance.get("zero_to_five_automatic_buy_allowed") is not False
            or governance.get("five_to_seven_automatic_buy_allowed") is not False
            or governance.get(
                "requires_intraday_absorption_reseal_and_theme_confirmation") is not True):
        raise ValueError("opening-gap evidence cannot authorize ranking or entry")
    validation = evidence.get("summaries", {}).get("ALL", {}).get("validation", {})
    five_to_seven = validation.get("GT_5_LE_7", {})
    above_seven = validation.get("GT_7", {})
    if not five_to_seven or not above_seven:
        raise ValueError("opening-gap validation bands missing")
    return {
        "status": "BOUND_DIAGNOSTIC_ONLY",
        "source_version": evidence["version"],
        "validation_sessions": evidence.get("split", {}).get("validation_sessions"),
        "above_seven_one_price_rate": above_seven.get("one_price_unbuyable_rate"),
        "five_to_seven_big_loss_rate": five_to_seven.get(
            "conditional_b1_big_loss_rate"),
        "above_seven_big_loss_rate": above_seven.get(
            "conditional_b1_big_loss_rate"),
        "five_to_seven_combined_rate": five_to_seven.get(
            "combined_buyable_b2_positive_rate"),
        "above_seven_combined_rate": above_seven.get(
            "combined_buyable_b2_positive_rate"),
        "payoff_advantage_ci_confirmed": evidence.get(
            "directional_replication", {}).get(
                "five_to_seven_vs_above_seven_both_ci_confirmed"),
        "prior_close_model_use_allowed": False,
        "opening_gap_direct_entry_allowed": False,
        "above_seven_default_risk_flag_allowed": bool(
            governance.get("above_seven_default_risk_flag_allowed")),
        "interpretation": evidence.get("interpretation"),
    }


def _opening_gap_emotion_summary(evidence):
    """Bind emotion interaction evidence without allowing a regime override."""
    if evidence is None:
        return {
            "status": "UNAVAILABLE", "source_version": None,
            "interaction_supported": False,
            "emotion_can_override_above_seven_risk_flag": False,
            "entry_authorized": False,
        }
    required = {
        "version", "status", "input_contract", "scope", "split",
        "temporal_consistency", "difference_in_differences",
        "opening_gap_emotion_interaction_supported", "governance", "interpretation",
    }
    if required - set(evidence):
        raise ValueError("opening-gap emotion interaction fields missing")
    contract = evidence.get("input_contract", {})
    governance = evidence.get("governance", {})
    if (evidence.get("version") != "v183-opening-gap-emotion-interaction-1"
            or evidence.get("status") != "DIAGNOSTIC_ONLY"):
        raise ValueError("opening-gap emotion evidence version or status invalid")
    if (contract.get("v182_corrected_observations_required") is not True
            or contract.get(
                "official_open_available_only_after_opening_call_match") is not True
            or contract.get("continuous_auction_begins_at_09_30") is not True
            or contract.get("actual_fills") != 0
            or contract.get("complete_trades") != 0):
        raise ValueError("opening-gap emotion input contract is invalid")
    forbidden = (
        "emotion_can_override_above_seven_risk_flag",
        "opening_gap_emotion_interaction_factor_allowed",
        "supportive_regime_automatic_high_open_buy_allowed",
        "adverse_regime_automatic_high_open_cancel_allowed",
        "direct_stock_reranking_allowed", "entry_authorized",
    )
    if any(governance.get(key) is not False for key in forbidden):
        raise ValueError("opening-gap emotion evidence cannot authorize a regime override")
    if evidence.get("opening_gap_emotion_interaction_supported") is not False:
        raise ValueError("unsupported opening-gap emotion interaction cannot be promoted")
    return {
        "status": "BOUND_DIAGNOSTIC_ONLY",
        "source_version": evidence["version"],
        "validation_sessions": evidence.get("split", {}).get("validation_sessions"),
        "regime_sessions": evidence.get("scope", {}).get("regime_sessions", {}),
        "interaction_supported": False,
        "emotion_can_override_above_seven_risk_flag": False,
        "opening_gap_emotion_interaction_factor_allowed": False,
        "entry_authorized": False,
        "interpretation": evidence.get("interpretation"),
    }


def _weak_to_strong_hindsight_summary(evidence):
    """Bind low-open denominator evidence without leaking later-touch information."""
    if evidence is None:
        return {
            "status": "UNAVAILABLE", "source_version": None,
            "validation_sessions": None,
            "full_low_open_success_rate": None,
            "later_touch_conditioned_rate": None,
            "conditioning_inflation_multiple": None,
            "later_touch_may_be_used_as_09_30_feature": False,
            "low_open_automatic_weak_to_strong_buy_allowed": False,
            "low_open_automatic_reject_allowed": False,
            "entry_authorized": False,
        }
    required = {
        "version", "status", "source_contract", "split", "denominator_contract",
        "hindsight_inflation", "low_open_conditional_payoff_edge_supported",
        "governance", "interpretation",
    }
    if required - set(evidence):
        raise ValueError("weak-to-strong hindsight evidence fields missing")
    source = evidence.get("source_contract", {})
    denominator = evidence.get("denominator_contract", {})
    governance = evidence.get("governance", {})
    if (evidence.get("version") != "v184-weak-to-strong-hindsight-audit-1"
            or evidence.get("status") != "DIAGNOSTIC_ONLY"):
        raise ValueError("weak-to-strong hindsight evidence version or status invalid")
    if (source.get("v182_version") != "v182-opening-gap-payoff-tradeoff-1"
            or source.get("no_touch_rows_retained_as_negative") is not True
            or source.get("next_session_one_price_rows_retained_as_negative") is not True
            or source.get("actual_fills") != 0
            or source.get("complete_trades") != 0):
        raise ValueError("weak-to-strong source contract is not fail-closed")
    if (denominator.get("later_touch_available_at_09_30") is not False
            or denominator.get(
                "conditional_result_may_not_be_presented_as_opening_win_rate") is not True):
        raise ValueError("weak-to-strong denominator contract permits future leakage")
    forbidden = (
        "later_touch_may_be_used_as_09_30_feature",
        "low_open_automatic_weak_to_strong_buy_allowed",
        "low_open_automatic_reject_allowed",
        "direct_stock_reranking_allowed", "entry_authorized",
    )
    if any(governance.get(key) is not False for key in forbidden):
        raise ValueError("weak-to-strong evidence cannot authorize ranking or entry")
    if governance.get(
            "requires_live_reclaim_vwap_volume_and_theme_confirmation") is not True:
        raise ValueError("weak-to-strong evidence must require live confirmation")
    if evidence.get("low_open_conditional_payoff_edge_supported") is not False:
        raise ValueError("unsupported low-open payoff edge cannot be promoted")
    validation = evidence.get("hindsight_inflation", {}).get(
        "ALL", {}).get("validation_low_open", {})
    if not validation:
        raise ValueError("weak-to-strong validation denominator missing")
    return {
        "status": "BOUND_DIAGNOSTIC_ONLY",
        "source_version": evidence["version"],
        "validation_sessions": evidence.get("split", {}).get("validation_sessions"),
        "full_low_open_success_rate": validation.get(
            "full_opening_denominator_success_rate"),
        "later_touch_conditioned_rate": validation.get(
            "later_touch_conditioned_b2_positive_rate"),
        "conditioning_inflation_multiple": validation.get(
            "conditioning_inflation_multiple"),
        "later_touch_may_be_used_as_09_30_feature": False,
        "low_open_automatic_weak_to_strong_buy_allowed": False,
        "low_open_automatic_reject_allowed": False,
        "requires_live_reclaim_vwap_volume_and_theme_confirmation": True,
        "entry_authorized": False,
        "interpretation": evidence.get("interpretation"),
    }


def _first_board_supply_summary(evidence):
    """Bind first-board supply as a breadth base-rate warning, never stock alpha."""
    if evidence is None:
        return {
            "status": "UNAVAILABLE", "source_version": None,
            "validation_sessions": None, "validation_any_promotion_rate": None,
            "validation_low_supply_candidate_rate": None,
            "validation_high_supply_candidate_rate": None,
            "first_board_supply_positive_ranking_factor_allowed": False,
            "entry_authorized": False,
        }
    required = {
        "version", "status", "target_contract", "split",
        "frozen_supply_thresholds_from_development", "summaries",
        "date_clustered_high_minus_low", "daily_any_promotion_is_discriminative",
        "high_first_board_supply_positive_edge_supported", "governance",
        "interpretation",
    }
    if required - set(evidence):
        raise ValueError("first-board supply evidence fields missing")
    target = evidence.get("target_contract", {})
    governance = evidence.get("governance", {})
    if (evidence.get("version") != "v185-first-board-supply-base-rate-audit-1"
            or evidence.get("status") != "DIAGNOSTIC_ONLY"):
        raise ValueError("first-board supply evidence version or status invalid")
    if (target.get("source_no_touch_rows_retained_as_negative") is not True
            or target.get("missing_promotion_labels_filled_as_zero") is not False
            or target.get("actual_fills") != 0
            or target.get("complete_trades") != 0):
        raise ValueError("first-board supply target contract is not fail-closed")
    forbidden = (
        "any_first_board_promoted_may_rank_a_candidate",
        "first_board_supply_positive_ranking_factor_allowed",
        "high_supply_automatic_emotion_upgrade_allowed",
        "high_supply_automatic_reject_allowed",
        "direct_stock_reranking_allowed", "entry_authorized",
    )
    if any(governance.get(key) is not False for key in forbidden):
        raise ValueError("first-board supply evidence cannot authorize ranking or entry")
    if governance.get(
            "requires_candidate_theme_identity_and_execution_confirmation") is not True:
        raise ValueError("first-board supply evidence must require candidate confirmation")
    if evidence.get("daily_any_promotion_is_discriminative") is not False:
        raise ValueError("non-discriminative any-promotion flag cannot be promoted")
    if evidence.get("high_first_board_supply_positive_edge_supported") is not False:
        raise ValueError("unsupported first-board supply edge cannot be promoted")
    validation = evidence.get("summaries", {}).get("validation", {})
    if not all(key in validation for key in ("ALL", "LOW_SUPPLY", "HIGH_SUPPLY")):
        raise ValueError("first-board supply validation summaries missing")
    return {
        "status": "BOUND_DIAGNOSTIC_ONLY",
        "source_version": evidence["version"],
        "validation_sessions": evidence.get("split", {}).get("validation_sessions"),
        "low_supply_max": evidence.get(
            "frozen_supply_thresholds_from_development", {}).get("low_supply_max"),
        "high_supply_strictly_above": evidence.get(
            "frozen_supply_thresholds_from_development", {}).get(
                "high_supply_strictly_above"),
        "validation_any_promotion_rate": validation["ALL"].get(
            "daily_any_promotion_rate"),
        "validation_low_supply_candidate_rate": validation["LOW_SUPPLY"].get(
            "candidate_promotion_rate"),
        "validation_high_supply_candidate_rate": validation["HIGH_SUPPLY"].get(
            "candidate_promotion_rate"),
        "first_board_supply_positive_ranking_factor_allowed": False,
        "high_supply_automatic_emotion_upgrade_allowed": False,
        "entry_authorized": False,
        "interpretation": evidence.get("interpretation"),
    }


def _promotion_payoff_summary(evidence):
    """Bind promotion/payoff separation without turning identity into a buy win rate."""
    if evidence is None:
        return {
            "status": "UNAVAILABLE", "source_version": None,
            "validation_sessions": None, "promotion_rate": None,
            "full_buyable_b2_positive_rate": None,
            "promoted_without_buyable_b2_positive_rate": None,
            "promoted_b2_b3_both_positive_rate": None,
            "promotion_probability_may_be_presented_as_trade_win_probability": False,
            "entry_authorized": False,
        }
    required = {
        "version", "status", "target_contract", "split", "summaries",
        "date_clustered_identity_minus_payoff",
        "promotion_is_buyable_b2_payoff_equivalent",
        "promotion_is_buyable_b2_b3_sustained_equivalent",
        "governance", "interpretation",
    }
    if required - set(evidence):
        raise ValueError("promotion-payoff separation evidence fields missing")
    target = evidence.get("target_contract", {})
    governance = evidence.get("governance", {})
    if (evidence.get("version") != "v186-promotion-payoff-separation-audit-1"
            or evidence.get("status") != "DIAGNOSTIC_ONLY"):
        raise ValueError("promotion-payoff evidence version or status invalid")
    if (target.get("no_touch_rows_retained_as_negative") is not True
            or target.get("one_price_rows_retained_as_negative") is not True
            or target.get("touchable_unknown_b2_excluded_from_b2_denominator") is not True
            or target.get("unknown_touch_rows_excluded_not_filled") is not True
            or target.get("missing_promotion_labels_filled_as_zero") is not False
            or target.get("actual_fills") != 0
            or target.get("complete_trades") != 0):
        raise ValueError("promotion-payoff target contract is not fail-closed")
    forbidden = (
        "promotion_probability_may_be_presented_as_trade_win_probability",
        "promotion_probability_may_replace_b2_b3_payoff_target",
        "board_level_promotion_rate_may_directly_rank_entries",
        "direct_stock_reranking_allowed", "entry_authorized",
    )
    if any(governance.get(key) is not False for key in forbidden):
        raise ValueError("promotion-payoff evidence cannot authorize ranking or entry")
    if governance.get("requires_separate_buyability_payoff_and_big_loss_models") is not True:
        raise ValueError("promotion-payoff evidence must preserve separate targets")
    if (evidence.get("promotion_is_buyable_b2_payoff_equivalent") is not False
            or evidence.get(
                "promotion_is_buyable_b2_b3_sustained_equivalent") is not False):
        raise ValueError("promotion cannot be promoted as payoff-equivalent")
    validation = evidence.get("summaries", {}).get("ALL", {}).get("validation", {})
    relay = evidence.get("summaries", {}).get(
        "CONSECUTIVE_BOARD", {}).get("validation", {})
    if not validation or not relay:
        raise ValueError("promotion-payoff validation summaries missing")
    return {
        "status": "BOUND_DIAGNOSTIC_ONLY",
        "source_version": evidence["version"],
        "validation_sessions": evidence.get("split", {}).get("validation_sessions"),
        "promotion_rate": validation.get("promotion_rate"),
        "full_buyable_b2_positive_rate": validation.get(
            "full_denominator_buyable_b2_positive_rate"),
        "promoted_without_buyable_b2_positive_rate": validation.get(
            "promoted_without_buyable_b2_positive_rate"),
        "promoted_b2_b3_both_positive_rate": validation.get(
            "promoted_and_buyable_b2_b3_both_positive_rate"),
        "relay_promoted_without_buyable_b2_positive_rate": relay.get(
            "promoted_without_buyable_b2_positive_rate"),
        "promotion_probability_may_be_presented_as_trade_win_probability": False,
        "promotion_probability_may_replace_b2_b3_payoff_target": False,
        "requires_separate_buyability_payoff_and_big_loss_models": True,
        "entry_authorized": False,
        "interpretation": evidence.get("interpretation"),
    }


def _repair_followthrough_summary(evidence):
    """Bind repair-day follow-through as scenario context, never an entry gate."""
    if evidence is None:
        return {
            "status": "UNAVAILABLE", "source_version": None,
            "latest_signal_date": None, "latest_quality_score": None,
            "latest_quality_class": None, "development_relapse_gap": None,
            "validation_relapse_gap": None,
            "lower_relapse_stable_across_time": False,
            "historical_scenario_context_allowed": False,
            "calibrated_probability_allowed": False,
            "direct_emotion_gate_allowed": False,
            "direct_stock_reranking_allowed": False,
            "entry_authorized": False,
        }
    required = {
        "version", "status", "data_contract", "repair_definition", "scope",
        "split", "summaries", "session_clustered_broad_minus_narrow",
        "broad_repair_lower_relapse_stable_across_time",
        "latest_repair_context", "governance", "interpretation",
    }
    if required - set(evidence):
        raise ValueError("repair follow-through evidence fields missing")
    contract = evidence.get("data_contract", {})
    governance = evidence.get("governance", {})
    if (evidence.get("version") != "v187-repair-day-followthrough-audit-1"
            or evidence.get("status") != "DIAGNOSTIC_ONLY"):
        raise ValueError("repair follow-through evidence version or status invalid")
    if (contract.get("signal_features_available_at_repair_day_close") is not True
            or contract.get("next_session_fields_used_only_as_labels") is not True
            or contract.get("sparse_signal_days_excluded") is not True
            or contract.get("missing_quality_fields_filled_as_zero") is not False
            or contract.get("actual_stock_entries") != 0
            or contract.get("complete_stock_trades") != 0):
        raise ValueError("repair follow-through data contract is not fail-closed")
    if evidence.get("broad_repair_lower_relapse_stable_across_time") is not False:
        raise ValueError("unstable repair follow-through cannot be promoted")
    forbidden = (
        "calibrated_next_relapse_probability_allowed",
        "repair_score_direct_emotion_gate_allowed",
        "repair_score_direct_stock_reranking_allowed",
        "repair_score_entry_authorized",
    )
    if any(governance.get(key) is not False for key in forbidden):
        raise ValueError("repair follow-through evidence cannot authorize ranking or entry")
    if (governance.get("historical_repair_scenario_context_allowed") is not True
            or governance.get(
                "requires_next_session_auction_middle_board_and_loss_confirmation") is not True):
        raise ValueError("repair follow-through evidence must remain confirmation-dependent")
    comparisons = evidence.get("session_clustered_broad_minus_narrow", {})
    development = comparisons.get("development", {}).get("metrics", {}).get(
        "relapse_rate", {})
    validation = comparisons.get("validation", {}).get("metrics", {}).get(
        "relapse_rate", {})
    validation_summary = evidence.get("summaries", {}).get("validation", {})
    if not development or not validation or not all(
            key in validation_summary for key in ("BROAD_REPAIR", "NARROW_REPAIR")):
        raise ValueError("repair follow-through comparison summaries missing")
    latest = evidence.get("latest_repair_context", {})
    return {
        "status": "BOUND_DIAGNOSTIC_ONLY",
        "source_version": evidence["version"],
        "repair_sessions": evidence.get("scope", {}).get("repair_sessions"),
        "validation_sessions": evidence.get("split", {}).get("validation_sessions"),
        "latest_signal_date": latest.get("signal_date"),
        "latest_quality_score": latest.get("quality_score"),
        "latest_quality_class": latest.get("quality_class"),
        "development_broad_relapse_rate": evidence.get(
            "summaries", {}).get("development", {}).get(
                "BROAD_REPAIR", {}).get("relapse_rate"),
        "development_narrow_relapse_rate": evidence.get(
            "summaries", {}).get("development", {}).get(
                "NARROW_REPAIR", {}).get("relapse_rate"),
        "validation_broad_relapse_rate": validation_summary[
            "BROAD_REPAIR"].get("relapse_rate"),
        "validation_narrow_relapse_rate": validation_summary[
            "NARROW_REPAIR"].get("relapse_rate"),
        "development_relapse_gap": development.get("estimate"),
        "development_relapse_gap_ci95": development.get("ci95"),
        "validation_relapse_gap": validation.get("estimate"),
        "validation_relapse_gap_ci95": validation.get("ci95"),
        "lower_relapse_stable_across_time": False,
        "historical_scenario_context_allowed": True,
        "calibrated_probability_allowed": False,
        "direct_emotion_gate_allowed": False,
        "direct_stock_reranking_allowed": False,
        "requires_next_session_confirmation": True,
        "entry_authorized": False,
        "interpretation": evidence.get("interpretation"),
    }


def _repair_layer_summary(evidence):
    """Bind low/middle/height repair transmission as descriptive context only."""
    if evidence is None:
        return {
            "status": "UNAVAILABLE", "source_version": None,
            "latest_layer_score": None, "latest_layer_class": None,
            "synchronized_stable": False, "middle_confirmation_stable": False,
            "historical_scenario_context_allowed": False,
            "calibrated_probability_allowed": False,
            "direct_emotion_gate_allowed": False,
            "direct_stock_reranking_allowed": False,
            "position_size_increase_allowed": False,
            "entry_authorized": False,
        }
    required = {
        "version", "status", "data_contract", "layer_definition",
        "frozen_development_thresholds", "scope", "split", "summaries",
        "session_clustered_comparisons",
        "synchronized_repair_lower_relapse_stable_across_time",
        "middle_confirmation_lower_relapse_stable_across_time",
        "latest_repair_layer_context", "governance", "interpretation",
    }
    if required - set(evidence):
        raise ValueError("repair layer evidence fields missing")
    contract = evidence.get("data_contract", {})
    governance = evidence.get("governance", {})
    if (evidence.get("version") != "v188-repair-layer-transmission-audit-1"
            or evidence.get("status") != "DIAGNOSTIC_ONLY"):
        raise ValueError("repair layer evidence version or status invalid")
    if (contract.get("same_day_layers_available_at_repair_close") is not True
            or contract.get("thresholds_frozen_from_development_only") is not True
            or contract.get("next_session_fields_used_only_as_labels") is not True
            or contract.get("missing_layer_fields_filled_as_zero") is not False
            or contract.get("actual_stock_entries") != 0
            or contract.get("complete_stock_trades") != 0):
        raise ValueError("repair layer data contract is not fail-closed")
    if (evidence.get(
            "synchronized_repair_lower_relapse_stable_across_time") is not False
            or evidence.get(
                "middle_confirmation_lower_relapse_stable_across_time") is not False):
        raise ValueError("unstable repair layer evidence cannot be promoted")
    forbidden = (
        "calibrated_layer_followthrough_probability_allowed",
        "repair_layer_score_direct_emotion_gate_allowed",
        "middle_board_confirmation_positive_ranking_factor_allowed",
        "repair_layer_direct_stock_reranking_allowed",
        "repair_layer_position_size_increase_allowed",
        "repair_layer_entry_authorized",
    )
    if any(governance.get(key) is not False for key in forbidden):
        raise ValueError("repair layer evidence cannot authorize ranking or entry")
    if (governance.get(
            "historical_repair_layer_scenario_context_allowed") is not True
            or governance.get(
                "requires_next_session_auction_middle_board_loss_and_theme_confirmation")
            is not True):
        raise ValueError("repair layer evidence must remain confirmation-dependent")
    summaries = evidence.get("summaries", {})
    for period in ("development", "validation"):
        if not all(key in summaries.get(period, {}) for key in (
                "SYNCHRONIZED_3", "PARTIAL_0_TO_2",
                "MIDDLE_CONFIRMED", "MIDDLE_UNCONFIRMED")):
            raise ValueError("repair layer comparison summaries missing")
    latest = evidence.get("latest_repair_layer_context", {})
    return {
        "status": "BOUND_DIAGNOSTIC_ONLY",
        "source_version": evidence["version"],
        "repair_sessions": evidence.get("scope", {}).get("repair_sessions"),
        "validation_sessions": evidence.get("split", {}).get(
            "validation_sessions"),
        "latest_signal_date": latest.get("signal_date"),
        "latest_first_board_confirmed": latest.get(
            "first_board_layer_confirmed"),
        "latest_middle_board_confirmed": latest.get(
            "middle_board_layer_confirmed"),
        "latest_height_confirmed": latest.get("height_layer_confirmed"),
        "latest_layer_score": latest.get("repair_layer_score"),
        "latest_layer_class": latest.get("repair_layer_class"),
        "development_middle_confirmed_relapse_rate": summaries[
            "development"]["MIDDLE_CONFIRMED"].get("next_relapse_rate"),
        "development_middle_unconfirmed_relapse_rate": summaries[
            "development"]["MIDDLE_UNCONFIRMED"].get("next_relapse_rate"),
        "validation_middle_confirmed_relapse_rate": summaries[
            "validation"]["MIDDLE_CONFIRMED"].get("next_relapse_rate"),
        "validation_middle_unconfirmed_relapse_rate": summaries[
            "validation"]["MIDDLE_UNCONFIRMED"].get("next_relapse_rate"),
        "validation_synchronized_sessions": summaries[
            "validation"]["SYNCHRONIZED_3"].get("sessions"),
        "synchronized_stable": False,
        "middle_confirmation_stable": False,
        "historical_scenario_context_allowed": True,
        "calibrated_probability_allowed": False,
        "direct_emotion_gate_allowed": False,
        "direct_stock_reranking_allowed": False,
        "position_size_increase_allowed": False,
        "requires_next_session_confirmation": True,
        "entry_authorized": False,
        "interpretation": evidence.get("interpretation"),
    }


def _high_to_low_ecology_summary(evidence):
    """Bind aggregate rotation ecology without selecting a supplement stock."""
    if evidence is None:
        return {
            "status": "UNAVAILABLE", "source_version": None,
            "window_active": None, "ecology_score": None,
            "historical_scenario_allowed": False,
            "calibrated_probability_allowed": False,
            "stock_selection_allowed": False,
            "same_theme_mapping_allowed": False,
            "entry_authorized": False,
        }
    required = {
        "version", "status", "data_contract", "window_definition",
        "frozen_development_thresholds", "scope", "split", "summaries",
        "session_clustered_healthy_minus_collapse_masked",
        "healthy_high_to_low_lower_adverse_stable_across_time",
        "latest_market_context", "governance", "interpretation",
    }
    if required - set(evidence):
        raise ValueError("high-to-low ecology evidence fields missing")
    contract = evidence.get("data_contract", {})
    governance = evidence.get("governance", {})
    if (evidence.get("version") != "v189-high-to-low-ecology-audit-1"
            or evidence.get("status") != "DIAGNOSTIC_ONLY"):
        raise ValueError("high-to-low ecology evidence version or status invalid")
    if (contract.get("same_day_window_features_available_at_close") is not True
            or contract.get(
                "low_strength_threshold_frozen_from_development_only") is not True
            or contract.get("next_session_fields_used_only_as_labels") is not True
            or contract.get("specific_theme_alignment_available") is not False
            or contract.get("stock_level_supplement_identity_available") is not False
            or contract.get("actual_stock_entries") != 0
            or contract.get("complete_stock_trades") != 0):
        raise ValueError("high-to-low ecology data contract is not fail-closed")
    if evidence.get(
            "healthy_high_to_low_lower_adverse_stable_across_time") is not False:
        raise ValueError("unstable high-to-low ecology cannot be promoted")
    forbidden = (
        "calibrated_high_to_low_window_probability_allowed",
        "high_to_low_window_direct_emotion_gate_allowed",
        "high_to_low_window_direct_stock_reranking_allowed",
        "high_to_low_stock_selection_allowed",
        "same_theme_supplement_mapping_allowed",
        "position_size_increase_allowed", "entry_authorized",
    )
    if any(governance.get(key) is not False for key in forbidden):
        raise ValueError("high-to-low ecology cannot authorize ranking or entry")
    if (governance.get("historical_high_to_low_market_scenario_allowed") is not True
            or governance.get(
                "requires_specific_theme_candidate_auction_and_open_confirmation")
            is not True):
        raise ValueError("high-to-low ecology must remain confirmation-dependent")
    summaries = evidence.get("summaries", {})
    for period in ("development", "validation"):
        if not all(key in summaries.get(period, {}) for key in (
                "HEALTHY_ROTATION_WINDOW", "COLLAPSE_MASKED_LOW_STRENGTH")):
            raise ValueError("high-to-low ecology summaries missing")
    latest = evidence.get("latest_market_context", {})
    return {
        "status": "BOUND_DIAGNOSTIC_ONLY",
        "source_version": evidence["version"],
        "high_to_low_windows": evidence.get("scope", {}).get(
            "high_to_low_windows"),
        "validation_sessions": evidence.get("split", {}).get(
            "validation_sessions"),
        "latest_signal_date": latest.get("signal_date"),
        "height_risk_increased": latest.get("height_risk_increased"),
        "low_layer_strong": latest.get("low_layer_strong"),
        "ecology_score": latest.get("ecology_score"),
        "window_active": latest.get("high_to_low_window_active"),
        "ecology_class": latest.get("ecology_class"),
        "development_healthy_adverse_rate": summaries[
            "development"]["HEALTHY_ROTATION_WINDOW"].get(
                "next_adverse_state_rate"),
        "development_fragile_adverse_rate": summaries[
            "development"]["COLLAPSE_MASKED_LOW_STRENGTH"].get(
                "next_adverse_state_rate"),
        "validation_healthy_adverse_rate": summaries[
            "validation"]["HEALTHY_ROTATION_WINDOW"].get(
                "next_adverse_state_rate"),
        "validation_fragile_adverse_rate": summaries[
            "validation"]["COLLAPSE_MASKED_LOW_STRENGTH"].get(
                "next_adverse_state_rate"),
        "stable_across_time": False,
        "historical_scenario_allowed": True,
        "calibrated_probability_allowed": False,
        "direct_emotion_gate_allowed": False,
        "direct_stock_reranking_allowed": False,
        "stock_selection_allowed": False,
        "same_theme_mapping_allowed": False,
        "position_size_increase_allowed": False,
        "requires_specific_theme_and_live_confirmation": True,
        "entry_authorized": False,
        "interpretation": evidence.get("interpretation"),
    }


def _index_emotion_divergence_summary(evidence, signal_date):
    """Bind broad-index context without allowing it to override relay ecology."""
    if evidence is None:
        return {
            "status": "UNAVAILABLE", "source_version": None,
            "direct_emotion_gate_allowed": False,
            "direct_stock_reranking_allowed": False,
            "entry_authorized": False,
        }
    required = {
        "version", "status", "data_quality", "latest_context",
        "weekly_clustered_comparisons", "replication_gate", "governance",
        "interpretation",
    }
    if required - set(evidence):
        raise ValueError("index-emotion divergence evidence fields missing")
    if (evidence.get("version") != "v203-index-emotion-divergence-audit-1"
            or evidence.get("status") != "DIAGNOSTIC_ONLY"):
        raise ValueError("index-emotion divergence version or status invalid")
    governance = evidence.get("governance", {})
    if governance.get("divergence_may_be_displayed_as_context") is not True:
        raise ValueError("index-emotion divergence context display is not authorized")
    forbidden = (
        "broad_index_direction_equals_short_term_emotion",
        "index_up_may_override_short_loss_effect",
        "divergence_direct_emotion_gate_allowed",
        "divergence_direct_stock_reranking_allowed",
        "divergence_entry_authorized",
    )
    if any(governance.get(key) is not False for key in forbidden):
        raise ValueError("index-emotion divergence cannot authorize a gate or entry")
    latest = evidence.get("latest_context", {})
    if str(latest.get("date")) != str(signal_date):
        raise ValueError("index-emotion latest context does not match signal date")
    comparison = evidence.get("weekly_clustered_comparisons", {}).get(
        "full", {}).get("next_adverse_state", {})
    gate = evidence.get("replication_gate", {}).get("next_adverse_state", {})
    return {
        "status": "BOUND_DIAGNOSTIC_ONLY",
        "source_version": evidence["version"],
        "signal_date": latest.get("date"),
        "sh_a_return_pct": latest.get("sh_a_return_pct"),
        "sz_a_return_pct": latest.get("sz_a_return_pct"),
        "previous_limit_median_return": latest.get(
            "previous_limit_median_return"),
        "divergence_category": latest.get("divergence_category"),
        "emotion_state": latest.get("emotion_state"),
        "divergence_next_adverse_difference": comparison.get(
            "observed_mean_difference_a_minus_b"),
        "divergence_next_adverse_ci95": comparison.get(
            "weekly_cluster_bootstrap_ci95"),
        "replication_gate_passed": bool(gate.get("replication_gate_passed")),
        "index_status": evidence.get("data_quality", {}).get("index_status"),
        "missing_emotion_sessions": evidence.get("data_quality", {}).get(
            "emotion_rows_missing_on_index_calendar"),
        "broad_index_may_override_short_ecology": False,
        "direct_emotion_gate_allowed": False,
        "direct_stock_reranking_allowed": False,
        "entry_authorized": False,
        "interpretation": evidence.get("interpretation"),
    }


def _list_value(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _clean_value(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value.item() if hasattr(value, "item") else value


def _text_value(value, default):
    value = _clean_value(value)
    text = str(value).strip() if value is not None else ""
    return text or default


def _regime(row):
    active = [name for name, field in STATE_FLAGS.items() if _float(row, field) == 1]
    return active[0] if len(active) == 1 else "UNAVAILABLE"


def _mainboard_code(value):
    code = str(value).strip().zfill(6)
    return code.startswith(("000", "001", "002", "003", "600", "601", "603", "605"))


def _previous_limit_metrics(previous_limit, closing):
    """Keep prior-limit mean, median, tail and promotion as distinct facts."""
    if previous_limit is None:
        return {
            "status": "UNAVAILABLE", "observation_count": 0,
            "mean_return": None, "median_return": None, "tail_10_return": None,
            "negative_count": None, "negative_rate": None,
            "large_loss_count": None, "large_loss_rate": None,
            "promotion_count": None, "promotion_rate": None,
            "level_feedback": {},
        }
    required = {"代码", "涨跌幅", "昨日连板数"}
    if required - set(previous_limit):
        raise ValueError("previous-limit evidence fields missing")
    frame = previous_limit.copy()
    frame["代码"] = frame["代码"].astype(str).str.zfill(6)
    frame = frame[frame["代码"].map(_mainboard_code)].copy()
    frame["涨跌幅"] = pd.to_numeric(frame["涨跌幅"], errors="coerce") / 100
    frame["昨日连板数"] = pd.to_numeric(frame["昨日连板数"], errors="coerce")
    frame = frame.dropna(subset=["涨跌幅", "昨日连板数"])
    if frame["代码"].duplicated().any():
        raise ValueError("previous-limit evidence contains duplicate codes")
    current_map = dict(zip(closing.code.astype(str).str.zfill(6),
                           pd.to_numeric(closing.boards, errors="coerce")))
    promoted = frame.apply(
        lambda row: current_map.get(row["代码"]) == row["昨日连板数"] + 1, axis=1)
    values = frame["涨跌幅"]
    levels = {}
    for level, group in frame.groupby("昨日连板数"):
        level_promoted = promoted.loc[group.index]
        levels[str(int(level))] = {
            "observation_count": int(len(group)),
            "mean_return": float(group["涨跌幅"].mean()),
            "median_return": float(group["涨跌幅"].median()),
            "promotion_count": int(level_promoted.sum()),
            "promotion_rate": float(level_promoted.mean()),
        }
    return {
        "status": "VERIFIED_DELAYED_PROVIDER_POOL",
        "observation_count": int(len(frame)),
        "mean_return": float(values.mean()) if len(values) else None,
        "median_return": float(values.median()) if len(values) else None,
        "tail_10_return": float(values.quantile(.10)) if len(values) else None,
        "negative_count": int(values.lt(0).sum()),
        "negative_rate": float(values.lt(0).mean()) if len(values) else None,
        "large_loss_count": int(values.le(-.07).sum()),
        "large_loss_rate": float(values.le(-.07).mean()) if len(values) else None,
        "promotion_count": int(promoted.sum()),
        "promotion_rate": float(promoted.mean()) if len(promoted) else None,
        "level_feedback": levels,
    }


def _failed_limit_entry_metrics(failed):
    """Infer same-day P&L from an ordinary 10% limit fill to failed-board close."""
    raw_changes = (failed["change_pct"] if "change_pct" in failed
                   else pd.Series(index=failed.index, dtype=float))
    changes = pd.to_numeric(raw_changes, errors="coerce") / 100
    returns = ((1 + changes) / 1.10 - 1).dropna()
    return {
        "status": "INFERRED_10PCT_LIMIT_FILL_TO_CLOSE",
        "observation_count": int(len(returns)),
        "mean_return": float(returns.mean()) if len(returns) else None,
        "median_return": float(returns.median()) if len(returns) else None,
        "tail_10_return": float(returns.quantile(.10)) if len(returns) else None,
        "large_loss_count": int(returns.le(-.07).sum()),
        "large_loss_rate": float(returns.le(-.07).mean()) if len(returns) else None,
        "loss_5_count": int(returns.le(-.05).sum()),
        "loss_5_rate": float(returns.le(-.05).mean()) if len(returns) else None,
    }


VERIFIED_THEME_STATUSES = {
    "verified_same_day_group",
    "verified_explicit_source_roster",
    "verified_specific_point_in_time_schema",
    "verified_membership_and_driver",
}


def _strict_boolean_series(series, field_name):
    mapping = {True: True, False: False, 1: True, 0: False,
               "true": True, "false": False, "1": True, "0": False,
               "True": True, "False": False}
    normalized = series.map(mapping)
    if normalized.isna().any():
        raise ValueError(f"{field_name} contains non-boolean values")
    return normalized.astype(bool)


def _candidate(row, risk_map=None, theme_map=None, competition_map=None,
               cross_theme_map=None, abnormal_path_map=None, symbolic_name_map=None):
    risks = (risk_map or {}).get(str(row.code).zfill(6), [])
    code = str(row.code).zfill(6)
    theme_row = (theme_map or {}).get(code)
    theme_status = "unavailable_real_specific_theme"
    theme_role = "UNVERIFIED"
    theme_evidence = {"status": "UNAVAILABLE", "theme_group_id": None,
                      "theme": None, "role": theme_role, "evidence_id": None,
                      "source_tier": None, "support_ex_self": None,
                      "same_height_rivals": None, "ladder_verified": None}
    if theme_row is not None:
        theme_status = _text_value(theme_row.get("theme_status"), "unavailable")
        theme_role = _text_value(theme_row.get("theme_role"), "UNVERIFIED")
        theme_evidence = {
            "status": "VERIFIED" if theme_status in VERIFIED_THEME_STATUSES else "INFERRED_OR_LIMITED",
            "theme_group_id": _clean_value(theme_row.get("theme_group_id")),
            "theme": _clean_value(theme_row.get("theme")), "role": theme_role,
            "evidence_id": _clean_value(theme_row.get("primary_evidence_id")),
            "source_tier": _clean_value(theme_row.get("source_tier")),
            "source_claim_status": _clean_value(theme_row.get("source_claim_status")),
            "evidence_available_at": _clean_value(theme_row.get("evidence_available_at")),
            "evidence_time_basis": _clean_value(theme_row.get("evidence_time_basis")),
            "identity_tags": _clean_value(theme_row.get("identity_tags")),
            "capacity_rank_in_theme": _float(theme_row, "capacity_rank_in_theme"),
            "business_branch": _clean_value(theme_row.get("business_branch")),
            "membership_claim_status": _clean_value(
                theme_row.get("membership_claim_status")),
            "membership_verified": (
                bool(theme_row.get("membership_verified"))
                if pd.notna(theme_row.get("membership_verified")) else None),
            "driver_claim_status": _clean_value(theme_row.get("driver_claim_status")),
            "driver_verified": (
                bool(theme_row.get("driver_verified"))
                if pd.notna(theme_row.get("driver_verified")) else None),
            "formal_mapping_eligible": (
                bool(theme_row.get("formal_high_low_mapping_eligible"))
                if pd.notna(theme_row.get("formal_high_low_mapping_eligible"))
                else theme_status in VERIFIED_THEME_STATUSES),
            "support_ex_self": _float(theme_row, "theme_support_ex_self"),
            "same_height_rivals": _float(theme_row, "same_height_rivals"),
            "ladder_verified": bool(theme_row.get("theme_ladder_verified"))
            if pd.notna(theme_row.get("theme_ladder_verified")) else None,
        }
    competition_row = (competition_map or {}).get(code)
    competition_identity = "unavailable"
    competition_evidence = {"status": "UNAVAILABLE", "source": None,
                            "observed_at": None, "available_at": None,
                            "same_board_peer_count": None, "higher_board_peer_count": None,
                            "same_theme_peer_count": None,
                            "same_theme_board_peer_count": None,
                            "higher_theme_peer_count": None,
                            "events": []}
    if competition_row is not None:
        competition_identity = _text_value(competition_row.get("competition_identity"), "unavailable")
        competition_evidence = {
            "status": _text_value(competition_row.get("evidence_status"), "VERIFIED"),
            "source": _clean_value(competition_row.get("source")),
            "observed_at": _clean_value(competition_row.get("observed_at")),
            "available_at": _clean_value(competition_row.get("available_at")),
            "same_board_peer_count": _float(competition_row, "same_board_peer_count"),
            "higher_board_peer_count": _float(competition_row, "higher_board_peer_count"),
            "same_theme_peer_count": _float(competition_row, "same_theme_peer_count"),
            "same_theme_board_peer_count": _float(
                competition_row, "same_theme_board_peer_count"),
            "higher_theme_peer_count": _float(competition_row, "higher_theme_peer_count"),
            "events": _list_value(competition_row.get("events", [])),
        }
    cross_row = (cross_theme_map or {}).get(code)
    cross_theme_race = {
        "status": "UNAVAILABLE", "lane": None, "theme_evidence_confidence": None,
        "peer_evidence_scope": None, "events": [],
        "first_competing_theme_count": None,
        "first_competing_candidate_count": None,
        "first_own_theme_first_count": None, "first_own_theme_relay_count": None,
        "first_existing_relay_backed": None, "first_isolated_theme_first": None,
        "relay_same_height_theme_count": None,
        "relay_same_height_candidate_count": None,
        "relay_higher_theme_count": None, "relay_higher_candidate_count": None,
        "relay_own_theme_first_support_count": None,
        "relay_own_theme_middle_support_count": None,
        "relay_own_theme_lower_support_count": None,
        "relay_same_theme_same_height_rival_count": None,
        "relay_global_height_gap": None, "model_eligible": False,
        "winner_claimed": False, "entry_authorized": False,
    }
    if cross_row is not None:
        cross_theme_race = {
            "status": _text_value(cross_row.get("cross_theme_race_status"), "UNAVAILABLE"),
            "lane": _clean_value(cross_row.get("race_lane")),
            "theme_evidence_confidence": _clean_value(
                cross_row.get("theme_evidence_confidence")),
            "peer_evidence_scope": _clean_value(cross_row.get("race_peer_evidence_scope")),
            "events": _list_value(cross_row.get("cross_theme_race_events", [])),
            "first_competing_theme_count": _float(
                cross_row, "first_cross_theme_competing_theme_count"),
            "first_competing_candidate_count": _float(
                cross_row, "first_cross_theme_competing_candidate_count"),
            "first_own_theme_first_count": _float(
                cross_row, "first_own_theme_first_count"),
            "first_own_theme_relay_count": _float(
                cross_row, "first_own_theme_relay_count"),
            "first_existing_relay_backed": (
                _truthy(cross_row.get("first_existing_relay_backed"))
                if pd.notna(cross_row.get("first_existing_relay_backed")) else None),
            "first_isolated_theme_first": (
                _truthy(cross_row.get("first_isolated_theme_first"))
                if pd.notna(cross_row.get("first_isolated_theme_first")) else None),
            "relay_same_height_theme_count": _float(
                cross_row, "relay_cross_theme_same_height_theme_count"),
            "relay_same_height_candidate_count": _float(
                cross_row, "relay_cross_theme_same_height_candidate_count"),
            "relay_higher_theme_count": _float(
                cross_row, "relay_cross_theme_higher_theme_count"),
            "relay_higher_candidate_count": _float(
                cross_row, "relay_cross_theme_higher_candidate_count"),
            "relay_own_theme_first_support_count": _float(
                cross_row, "relay_own_theme_first_support_count"),
            "relay_own_theme_middle_support_count": _float(
                cross_row, "relay_own_theme_middle_support_count"),
            "relay_own_theme_lower_support_count": _float(
                cross_row, "relay_own_theme_lower_support_count"),
            "relay_same_theme_same_height_rival_count": _float(
                cross_row, "relay_same_theme_same_height_rival_count"),
            "relay_global_height_gap": _float(cross_row, "relay_global_height_gap"),
            "model_eligible": _truthy(
                cross_row.get("eligible_for_cross_theme_race_training")),
            "winner_claimed": _truthy(cross_row.get("winner_claimed")),
            "entry_authorized": _truthy(cross_row.get("entry_authorized")),
        }
    abnormal_path = (abnormal_path_map or {}).get(code, {
        "diagnostic": {
            "status": "UNAVAILABLE", "rule_id": None, "benchmark_code": None,
            "distance_100_pp": None, "distance_200_pp": None,
            "scenario_limit_boards_to_100": None, "scenario_status": "UNAVAILABLE",
            "available_at": None,
        },
        "path_role_status": "NOT_ELIGIBLE",
        "blocking_reasons": ["CANDIDATE_RULE_DISTANCE_EVIDENCE_UNAVAILABLE"],
        "calibrated_probability": None, "buy_signal": False,
        "candidate_reranking_allowed": False,
    })
    symbolic_clusters = (symbolic_name_map or {}).get(code, [])
    return {
        "code": code, "name": row.get("name"), "boards": int(row.boards),
        "board_group": row.board_group, "industry_provider": row.get("industry_provider"),
        "turnover": _float(row, "turnover"), "amount": _float(row, "amount"),
        "first_seal_time": row.get("first_seal_time"),
        "failed_seal_count": int(row.failed_seal_count) if pd.notna(row.failed_seal_count) else None,
        "security_status": row.security_status, "eligibility_status": row.eligibility_status,
        "theme_status": theme_status, "theme_role": theme_role,
        "theme_evidence": theme_evidence,
        "active_hard_risks": risks,
        # These fields are deliberately explicit.  An absent auction/theme/order
        # record is an unknown condition, never an implicit pass or a zero.
        "competition_identity": competition_identity,
        "competition_evidence": competition_evidence,
        "cross_theme_race": cross_theme_race,
        "abnormal_path": abnormal_path,
        "symbolic_name_clusters": symbolic_clusters,
        "symbolic_name_status": (
            "INFERRED_SYMBOLIC_CLUSTER_NOT_REAL_THEME"
            if symbolic_clusters else "NO_QUALIFYING_SYMBOLIC_CLUSTER"
        ),
        "continuation_logic": [], "opposition_evidence": [],
        "competitor_invalidation": [], "risk_factors": [],
        "auction_conditions": [], "opening_conditions": [], "cancel_conditions": [],
        "auction_evidence_status": "UNAVAILABLE",
        "opening_evidence_status": "UNAVAILABLE",
        "execution_state": "DATA_LIMITED",
        "rank_probability": None, "probability_status": "UNAVAILABLE",
        "entry_permission": "NO_ENTER",
    }


def _height_potential(candidates, maximum_board):
    """Return a transparent EOD ranking, never a calibrated probability."""
    if not candidates:
        return None
    identity_points = {
        "height_anchor": 18, "unique_survivor": 16, "turnover_core": 14,
        "pioneer": 11, "capacity_core": 9, "supplement": 3, "unavailable": 0,
    }
    ranked = []
    for row in candidates:
        score, support, opposition = row["boards"] * 12.0, [], []
        if row["boards"] == maximum_board:
            score += 12
            support.append("当前处于最高板层级，具有高度先发优势")
        identity = str(row.get("competition_identity", "unavailable")).lower()
        score += identity_points.get(identity, 0)
        if identity != "unavailable":
            support.append(f"竞争身份已记录为{identity}")
        else:
            opposition.append("竞争身份未完整核验")
        theme = row.get("theme_evidence", {})
        if theme.get("status") == "VERIFIED":
            score += 8
            support.append("真实细分题材成员身份有时点化证据")
        else:
            opposition.append("真实细分题材证据不完整")
        if theme.get("ladder_verified") is True:
            score += 7
            support.append("题材梯队已核验")
        support_count = theme.get("support_ex_self")
        if support_count is not None:
            score += min(max(float(support_count), 0), 5)
        if row.get("active_hard_risks"):
            score -= 25
            opposition.append("存在有效硬风险事件")
        failed = row.get("failed_seal_count")
        if failed is not None and failed <= 1:
            score += 4
            support.append("封板反复程度较低")
        elif failed is not None and failed >= 2:
            score -= 5
            opposition.append("反复炸板削弱高度竞争力")
        turnover = row.get("turnover")
        if turnover is not None and 4 <= turnover <= 20:
            score += 4
            support.append("换手处于可观察区间")
        ranked.append({
            "code": row["code"], "name": row["name"], "boards": row["boards"],
            "research_score_not_probability": round(score, 4),
            "evidence_status": "INFERRED_EOD_RANKING_NOT_PROBABILITY",
            "supporting_evidence": support or ["仅有板位基础信息"],
            "opposition_evidence": opposition or ["次日竞价与竞争淘汰尚未发生"],
            "same_as_current_anchor": row["boards"] == maximum_board,
        })
    return sorted(ranked, key=lambda row: (-row["research_score_not_probability"],
                                            -row["boards"], row["code"]))[0]


def _current_market_core(candidates):
    """Select a verified EOD market core without turning it into a buy signal."""
    identity_priority = {
        "height_anchor": 5, "unique_survivor": 4, "pioneer": 3,
        "turnover_core": 2, "capacity_core": 1,
    }
    eligible = []
    for row in candidates:
        identity = str(row.get("competition_identity", "unavailable")).lower()
        theme = row.get("theme_evidence") or {}
        race = row.get("cross_theme_race") or {}
        if not (
            row.get("boards", 0) >= 2
            and identity in identity_priority
            and theme.get("status") == "VERIFIED"
            and theme.get("driver_verified") is True
            and theme.get("formal_mapping_eligible") is not False
            and race.get("model_eligible") is True
            and race.get("entry_authorized") is False
            and row.get("eligibility_status") == "VERIFIED_ELIGIBLE"
            and not row.get("active_hard_risks")
        ):
            continue
        support = theme.get("support_ex_self")
        failed = row.get("failed_seal_count")
        eligible.append((
            -identity_priority[identity], -int(row["boards"]),
            -(float(support) if support is not None else -1),
            failed if failed is not None else 999, row["code"], row,
        ))
    if not eligible:
        return None
    row = sorted(eligible)[0][-1]
    return {
        "code": row["code"], "name": row["name"], "boards": row["boards"],
        "competition_identity": row["competition_identity"],
        "theme": row["theme_evidence"].get("theme"),
        "evidence_status": "VERIFIED_EOD_CORE_IDENTITY_NOT_BUY_SIGNAL",
        "buy_signal": False,
    }


def _validated_group_primary(candidates, threshold, group):
    """Return at most one calibrated group primary; otherwise explain cash."""
    status = {
        "group": group, "candidate_count": len(candidates),
        "validated_candidate_count": 0, "threshold": threshold,
        "status": "CASH_NO_VALIDATED_THRESHOLD" if threshold is None
        else "CASH_NO_VALIDATED_CANDIDATE",
        "not_a_buy_signal": True,
    }
    if threshold is None:
        return None, status
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        status["status"] = "CASH_INVALID_VALIDATED_THRESHOLD"
        return None, status
    valid = []
    for row in candidates:
        probability = row.get("rank_probability")
        try:
            probability = float(probability)
        except (TypeError, ValueError):
            continue
        theme = row.get("theme_evidence") or {}
        race = row.get("cross_theme_race") or {}
        if not (
            row.get("probability_status") == "VALIDATED_WALK_FORWARD_CALIBRATED"
            and probability >= threshold
            and row.get("entry_permission") == "CONDITIONAL_ONLY"
            and theme.get("status") == "VERIFIED"
            and theme.get("driver_verified") is True
            and theme.get("formal_mapping_eligible") is not False
            and str(row.get("competition_identity", "unavailable")).lower() != "unavailable"
            and race.get("model_eligible") is True
            and race.get("entry_authorized") is False
            and row.get("eligibility_status") == "VERIFIED_ELIGIBLE"
            and not row.get("active_hard_risks")
        ):
            continue
        valid.append((probability, int(row["boards"]), row["code"], row))
    status["validated_candidate_count"] = len(valid)
    if not valid:
        return None, status
    probability, _, _, row = sorted(valid, key=lambda item: (-item[0], -item[1], item[2]))[0]
    status["status"] = "FROZEN_CONDITIONAL_PRIMARY_REQUIRES_NEXT_SESSION_CONFIRMATION"
    return {
        "code": row["code"], "name": row["name"], "boards": row["boards"],
        "calibrated_probability": probability, "threshold": threshold,
        "evidence_status": "VALIDATED_WALK_FORWARD_CALIBRATED",
        "buy_signal": False, "requires_auction_opening_confirmation": True,
    }, status


def _explicit_high_attention_exclusions(candidates, maximum_board,
                                        highest_potential=None, challenger=None):
    """Explain why attention proxies are not group primaries or entries."""
    role_codes = {
        row.get("code") for row in (highest_potential, challenger)
        if isinstance(row, dict) and row.get("code")
    }
    exclusions = []
    for row in candidates:
        if not (row.get("boards", 0) >= 3 or row.get("boards") == maximum_board
                or row.get("code") in role_codes or row.get("active_hard_risks")):
            continue
        reasons = []
        theme = row.get("theme_evidence") or {}
        race = row.get("cross_theme_race") or {}
        first_seal = str(row.get("first_seal_time", "")).split(".")[0].zfill(6)
        if row.get("active_hard_risks"):
            reasons.append("ACTIVE_HARD_RISK")
        if first_seal == "092500" and row.get("failed_seal_count") == 0:
            reasons.append("ONE_PRICE_PROXY_BUYABILITY_NOT_PROVEN")
        if theme.get("status") != "VERIFIED":
            reasons.append("SPECIFIC_THEME_NOT_VERIFIED")
        elif theme.get("driver_verified") is not True:
            reasons.append("ACTIVE_DRIVER_NOT_VERIFIED")
        if str(row.get("competition_identity", "unavailable")).lower() == "unavailable":
            reasons.append("COMPETITION_IDENTITY_UNAVAILABLE")
        if race.get("model_eligible") is not True:
            reasons.append("CROSS_THEME_RACE_NOT_MODEL_ELIGIBLE")
        if row.get("eligibility_status") != "VERIFIED_ELIGIBLE":
            reasons.append("SECURITY_ELIGIBILITY_NOT_VERIFIED")
        if row.get("probability_status") != "VALIDATED_WALK_FORWARD_CALIBRATED":
            reasons.append("CALIBRATED_CONTINUATION_PROBABILITY_UNAVAILABLE")
        if not reasons:
            reasons.append("NEXT_SESSION_AUCTION_OPENING_CONFIRMATION_UNAVAILABLE")
        exclusions.append({
            "code": row["code"], "name": row["name"], "boards": row["boards"],
            "attention_basis": "HEIGHT_OR_RESEARCH_ROLE_PROXY_NOT_VERIFIED_POPULARITY",
            "exclusion_reasons": reasons, "buy_signal": False,
        })
    return sorted(exclusions, key=lambda row: (-row["boards"], row["code"]))[:10]


def _high_low_supplement_map(candidates, maximum_board):
    """Record every 3+ board high and keep incomplete evidence non-actionable."""
    high_identities = {"height_anchor", "unique_survivor", "pioneer", "capacity_core"}
    low_identities = {"unique_survivor", "pioneer", "capacity_core", "turnover_core"}
    identity_points = {"unique_survivor": 16, "pioneer": 14,
                       "capacity_core": 11, "turnover_core": 9}
    highs = [row for row in candidates if row["boards"] >= 3]
    highs.sort(key=lambda row: (-row["boards"], row["code"]))
    mappings = []
    for high in highs:
        high_theme = high.get("theme_evidence", {})
        blockers = []
        high_identity = str(high.get("competition_identity", "unavailable")).lower()
        core_identity_verified = (
            high["boards"] == maximum_board or high_identity in high_identities
        )
        if not core_identity_verified:
            blockers.append("HIGH_POSITION_CORE_IDENTITY_NOT_VERIFIED")
        if high_theme.get("status") != "VERIFIED" or not high_theme.get("theme_group_id"):
            blockers.append(
                "HIGH_CORE_ACTIVE_THEME_NOT_FULLY_VERIFIED"
                if high_theme.get("membership_verified") is True
                else "HIGH_CORE_SPECIFIC_THEME_NOT_VERIFIED"
            )
        if high_theme.get("theme_group_id") and high_theme.get("driver_verified") is False:
            blockers.append("ACTIVE_DRIVER_NOT_VERIFIED")
        if (high_theme.get("status") == "VERIFIED"
                and high_theme.get("formal_mapping_eligible") is False):
            blockers.append("EXPECTATION_OR_FORMAL_MAPPING_GATE_NOT_PASSED")
        membership_lows = []
        if (high_theme.get("theme_group_id")
                and high_theme.get("membership_verified") is True
                and high_theme.get("driver_verified") is False):
            for low in candidates:
                low_theme = low.get("theme_evidence", {})
                first_seal = str(low.get("first_seal_time", "")).split(".")[0].zfill(6)
                one_price_proxy = first_seal == "092500" and low.get("failed_seal_count") == 0
                if not (
                    low["code"] != high["code"]
                    and low["boards"] <= 2 and low["boards"] < high["boards"]
                    and low_theme.get("theme_group_id") == high_theme.get("theme_group_id")
                    and low_theme.get("business_branch") == high_theme.get("business_branch")
                    and low_theme.get("membership_verified") is True
                    and low_theme.get("driver_verified") is False
                    and not low.get("active_hard_risks") and not one_price_proxy
                ):
                    continue
                tags = {tag for tag in str(low_theme.get("identity_tags") or "").split("|") if tag}
                score = float(low["boards"] * 4)
                if "FIRST_SEAL_PIONEER" in tags:
                    score += 8
                if "CAPACITY_CORE_BY_CLOSE_AMOUNT" in tags:
                    score += 6
                if low.get("theme_role") in {"THEME_HEIGHT_LEADER", "SAME_HEIGHT_CORE"}:
                    score += 4
                failed = low.get("failed_seal_count")
                if failed is not None and failed <= 1:
                    score += 3
                elif failed is not None and failed >= 2:
                    score -= 5
                membership_lows.append({
                    "code": low["code"], "name": low["name"], "boards": low["boards"],
                    "theme": low_theme.get("theme"),
                    "business_branch": low_theme.get("business_branch"),
                    "theme_role": low.get("theme_role"),
                    "research_score_not_probability": round(score, 4),
                    "evidence_status": "VERIFIED_MEMBERSHIP_ACTIVE_DRIVER_NOT_VERIFIED",
                    "not_a_formal_supplement_candidate": True, "buy_signal": False,
                })
        membership_lows.sort(key=lambda row: (
            -row["research_score_not_probability"], -row["boards"], row["code"]
        ))
        membership_lows = membership_lows[:3]
        narrative_lows = []
        if (high_theme.get("theme_group_id")
                and high_theme.get("membership_verified") is not True):
            for low in candidates:
                low_theme = low.get("theme_evidence", {})
                first_seal = str(low.get("first_seal_time", "")).split(".")[0].zfill(6)
                one_price_proxy = first_seal == "092500" and low.get("failed_seal_count") == 0
                if not (
                    low["code"] != high["code"]
                    and low["boards"] <= 2 and low["boards"] < high["boards"]
                    and low_theme.get("theme_group_id") == high_theme.get("theme_group_id")
                    and (low_theme.get("source_claim_status") == "secondary_market_narrative"
                         or low_theme.get("driver_claim_status") == "secondary_market_narrative")
                    and not low.get("active_hard_risks") and not one_price_proxy
                ):
                    continue
                tags = {tag for tag in str(low_theme.get("identity_tags") or "").split("|") if tag}
                score = float(low["boards"] * 4)
                if "FIRST_SEAL_PIONEER" in tags:
                    score += 8
                if "CAPACITY_CORE_BY_CLOSE_AMOUNT" in tags:
                    score += 6
                if low.get("theme_role") in {"THEME_HEIGHT_LEADER", "SAME_HEIGHT_CORE"}:
                    score += 4
                failed = low.get("failed_seal_count")
                if failed is not None and failed <= 1:
                    score += 3
                elif failed is not None and failed >= 2:
                    score -= 5
                turnover = low.get("turnover")
                if turnover is not None and 4 <= turnover <= 20:
                    score += 2
                narrative_lows.append({
                    "code": low["code"], "name": low["name"], "boards": low["boards"],
                    "theme": low_theme.get("theme"), "theme_role": low.get("theme_role"),
                    "research_score_not_probability": round(score, 4),
                    "evidence_status": "SECONDARY_NARRATIVE_ONLY_NOT_VERIFIED",
                    "not_a_formal_supplement_candidate": True,
                    "buy_signal": False,
                })
        narrative_lows.sort(key=lambda row: (
            -row["research_score_not_probability"], -row["boards"], row["code"]
        ))
        narrative_lows = narrative_lows[:3]
        lows = []
        if not blockers:
            for low in candidates:
                low_theme = low.get("theme_evidence", {})
                identity = str(low.get("competition_identity", "unavailable")).lower()
                first_seal = str(low.get("first_seal_time", "")).split(".")[0].zfill(6)
                one_price_proxy = first_seal == "092500" and low.get("failed_seal_count") == 0
                if not (
                    low["code"] != high["code"]
                    and low["boards"] <= 2 and low["boards"] < high["boards"]
                    and low_theme.get("status") == "VERIFIED"
                    and low_theme.get("theme_group_id") == high_theme.get("theme_group_id")
                    and low_theme.get("formal_mapping_eligible") is not False
                    and identity in low_identities
                    and not low.get("active_hard_risks")
                    and not one_price_proxy
                ):
                    continue
                score = float(identity_points[identity] + low["boards"] * 5)
                if low_theme.get("ladder_verified") is True:
                    score += 5
                support = low_theme.get("support_ex_self")
                if support is not None:
                    score += min(max(float(support), 0), 5)
                failed = low.get("failed_seal_count")
                if failed is not None and failed <= 1:
                    score += 3
                elif failed is not None and failed >= 2:
                    score -= 5
                turnover = low.get("turnover")
                if turnover is not None and 4 <= turnover <= 20:
                    score += 2
                lows.append((score, low))
        lows.sort(key=lambda item: (-item[0], -item[1]["boards"], item[1]["code"]))
        if not lows and not blockers:
            blockers.append("NO_ELIGIBLE_SAME_THEME_LOW_CORE")
        selected = None
        if lows:
            score, low = lows[0]
            selected = {
                "code": low["code"], "name": low["name"], "boards": low["boards"],
                "theme": low["theme_evidence"].get("theme"),
                "competition_identity": low.get("competition_identity"),
                "research_score_not_probability": round(score, 4),
                "evidence_status": "INFERRED_EOD_MAPPING_REQUIRES_NEXT_SESSION_CONFIRMATION",
                "buy_signal": False,
            }
        mappings.append({
            "high_core": {
                "code": high["code"], "name": high["name"], "boards": high["boards"],
                "theme": high_theme.get("theme"),
                "theme_group_id": high_theme.get("theme_group_id"),
                "competition_identity": high_identity,
                "core_identity_verified": core_identity_verified,
            },
            "mapping_status": (
                "MAPPED_RESEARCH_ONLY" if selected
                else "VERIFIED_MEMBERSHIP_ONLY_DRIVER_UNVERIFIED" if membership_lows
                else "NARRATIVE_ALTERNATIVES_ONLY_NOT_VERIFIED" if narrative_lows
                else "NO_ELIGIBLE_MAPPING"
            ),
            "supplement_candidate": selected,
            "verified_membership_only_low_alternatives": membership_lows,
            "narrative_only_low_alternatives": narrative_lows,
            "blocking_reasons": blockers,
            "capital_migration_hypothesis": (
                "only if high-position risk rises while the verified specific-theme ladder and "
                "profit ecology remain alive"
            ),
            "next_session_confirmation": [
                "高标风险上升但题材未整体坍塌",
                "低位候选竞价和开盘强于同题材低位对手且有真实换手",
                "题材中位与容量反馈没有同步恶化",
            ],
            "invalidation": [
                "高标与题材梯队同步崩塌",
                "低位候选被同题材竞争者淘汰",
                "一字无成交路径或超过最高接受价",
            ],
            "buy_signal": False,
        })
    mapped_from_top = [
        item for item in mappings
        if item["high_core"]["boards"] == maximum_board
        and item["supplement_candidate"] is not None
    ]
    primary = None
    if mapped_from_top:
        primary = sorted(
            (item["supplement_candidate"] for item in mapped_from_top),
            key=lambda row: (-row["research_score_not_probability"], -row["boards"], row["code"]),
        )[0]
    return mappings, primary


def _anchor_premium_review(candidate, regime, money_state, broken_state, height_state):
    first_seal = str(candidate.get("first_seal_time", "")).split(".")[0].zfill(6)
    one_price_proxy = first_seal == "092500" and candidate.get("failed_seal_count") == 0
    if candidate.get("active_hard_risks") or regime in {"大分歧", "退潮"} or broken_state == "ADVERSE":
        space = "LOW"
        conclusion = "负反馈或硬风险使买入后的剩余溢价不足以覆盖隔夜风险"
    elif money_state == "SUPPORTIVE" and height_state == "SUPPORTIVE":
        space = "MEDIUM"
        conclusion = "环境支持继续竞争，但仍需次日价格和承接确认，不能盘后直接买入"
    else:
        space = "UNAVAILABLE"
        conclusion = "市场证据相互冲突或不完整，无法在盘后确认实际溢价空间"
    buyability = "UNBUYABLE" if one_price_proxy else "ANCHOR_ONLY"
    return {
        "code": candidate["code"], "name": candidate["name"], "boards": candidate["boards"],
        "premium_space": space, "premium_evidence_status": "INFERRED_EOD_CONDITIONAL",
        "buyability": buyability,
        "one_price_proxy": one_price_proxy,
        "conclusion": conclusion,
        "upgrade_to_conditional_buy": [
            "次日竞价价格未透支且实际可成交",
            "第一次分歧缩量、重新站回VWAP或换手回封",
            "同题材核心与梯队同步支持，竞争者未完成卡位",
        ],
        "immediate_cancel": [
            "高开超过计划上限后跌破VWAP并放量",
            "两次冲板不封或同题材核心破板",
            "中高位负反馈扩散或出现新的硬风险",
        ],
        "next_day_execution": {
            "high_open": "高开5%—7%只观察换手回封；高于7%默认取消，除非盘前已冻结例外且板块同步超预期",
            "flat_open": "平开至高开5%观察首次分歧缩量承接，收回VWAP并由题材核心同步确认后才升级",
            "low_open": "低开不猜弱转强；只有快速收回昨收与VWAP、量能改善且竞争者转弱才继续观察，否则取消",
            "sector_break": "同题材高度股、容量核心或中位梯队同步走弱，立即取消个股买入计划",
            "holding": "计划持有1—3个交易日；晋级次日不及预期、跌破VWAP放量或板块负反馈扩散时优先退出",
            "price_rule": "最高接受价必须在次日盘前按昨收和失效位计算；超过上限即使随后涨停也不追",
        },
    }


def build(snapshot, snapshot_manifest, market, selective_policy, asof, announcements=None,
          theme_roles=None, competition=None, previous_limit=None,
          cross_theme_races=None, abnormal_paths=None, height_anchor_evidence=None,
          opening_gap_evidence=None, opening_gap_emotion_evidence=None,
          weak_to_strong_evidence=None, first_board_supply_evidence=None,
          promotion_payoff_evidence=None, repair_followthrough_evidence=None,
          repair_layer_evidence=None, high_to_low_ecology_evidence=None,
          index_emotion_divergence_evidence=None,
          previous_symbolic_name_evidence=None):
    signal_date = snapshot_manifest["signal_date"]
    mainboard_mask = _strict_boolean_series(snapshot.is_mainboard, "is_mainboard")
    rows = snapshot[(snapshot.signal_date == signal_date) & mainboard_mask].copy()
    closing = rows[rows.pool_assignment.eq("CLOSING_LIMIT_UP")].copy()
    failed = rows[rows.pool_assignment.eq("FAILED_BOARD_OBSERVATION")].copy()
    if len(closing) != snapshot_manifest["mainboard_closing_limit_rows"]:
        raise ValueError("closing pool differs from snapshot manifest")
    if len(failed) != snapshot_manifest["mainboard_failed_board_rows"]:
        raise ValueError("failed pool differs from snapshot manifest")
    if set(closing.code.astype(str)) & set(failed.code.astype(str)):
        raise ValueError("closing and failed pools overlap")
    symbolic_name_evidence = symbolic_name_theme.build(
        snapshot, snapshot_manifest, asof, previous_symbolic_name_evidence)
    symbolic_name_map = {
        str(row["code"]).zfill(6): row.get("clusters", [])
        for row in symbolic_name_evidence["candidate_records"]
    }
    day = market[market.date.eq(signal_date)]
    if len(day) != 1 or day.iloc[0].get("data_status") != "verified":
        raise ValueError("verified market-emotion row unavailable")
    emotion = day.iloc[0]
    regime = _regime(emotion)
    premium = _float(emotion, "yesterday_premium")
    promotion = _float(emotion, "promotion_overall")
    previous_tail = _float(emotion, "previous_limit_tail_loss")
    failed_tail = _float(emotion, "failed_tail_loss")
    failed_rate = _float(emotion, "failed_rate")
    seal_rate = _float(emotion, "seal_rate")
    space = _float(emotion, "space_feedback")
    max_board = _integer(emotion, "max_board")
    ladder_gaps = _integer(emotion, "ladder_gap_count")
    down_limits = _integer(emotion, "down_limit_count")
    one_word_share = _float(emotion, "one_word_share")
    crowding_risk = _float(emotion, "crowding_risk")
    current_limit_count = _integer(emotion, "limit_count")
    prior_verified = market[
        (market.date.astype(str) < str(signal_date))
        & market.data_status.eq("verified")
    ].sort_values("date")
    prior_failed_rate = None
    prior_limit_count = None
    if len(prior_verified):
        prior_row = prior_verified.iloc[-1]
        prior_failed_rate = _float(prior_row, "failed_rate")
        prior_limit_count = _integer(prior_row, "limit_count")
    limit_count_change = (
        current_limit_count - prior_limit_count
        if current_limit_count is not None and prior_limit_count is not None else None
    )
    failed_rate_change = (
        failed_rate - prior_failed_rate
        if failed_rate is not None and prior_failed_rate is not None else None
    )
    breadth_quality = (
        "FRAGILE_EXPANSION" if limit_count_change is not None
        and limit_count_change > 0 and failed_rate_change is not None
        and failed_rate_change > 0 else
        "CLEAN_EXPANSION" if limit_count_change is not None
        and limit_count_change > 0 and failed_rate_change is not None
        and failed_rate_change <= 0 else
        "NO_EXPANSION" if limit_count_change is not None and limit_count_change <= 0
        else "UNAVAILABLE"
    )
    previous_metrics = _previous_limit_metrics(previous_limit, closing)
    if previous_metrics["status"] == "UNAVAILABLE":
        previous_metrics.update({
            "median_return": premium,
            "tail_10_return": previous_tail,
            "promotion_rate": promotion,
            "status": "LEGACY_MARKET_ROW_MEDIAN_ONLY",
        })
    failed_entry_metrics = _failed_limit_entry_metrics(failed)
    previous_mean = previous_metrics["mean_return"]
    previous_median = previous_metrics["median_return"]
    previous_tail = previous_metrics["tail_10_return"]
    promotion = previous_metrics["promotion_rate"]
    theme_map = {}
    theme_known = 0
    theme_verified = 0
    theme_membership_verified_driver_unverified = 0
    theme_observed_secondary = 0
    theme_rumor = 0
    if theme_roles is not None:
        required_theme = {"code", "signal_date", "theme_status", "theme_role", "theme_group_id",
                          "theme", "primary_evidence_id", "source_tier"}
        if required_theme - set(theme_roles):
            raise ValueError("theme role evidence fields missing")
        theme_frame = theme_roles.copy()
        theme_frame["code"] = theme_frame.code.astype(str).str.zfill(6)
        theme_frame["signal_date"] = pd.to_datetime(theme_frame.signal_date, errors="coerce").dt.strftime("%Y-%m-%d")
        if not bool(theme_frame.signal_date.eq(signal_date).any()):
            raise ValueError("theme role evidence has no matching signal_date")
        theme_frame = theme_frame[theme_frame.signal_date.eq(signal_date)].copy()
        if theme_frame.code.duplicated().any():
            raise ValueError("theme role evidence contains duplicate candidate codes")
        cutoff = _local_naive_timestamp(asof)
        if "evidence_available_at" in theme_frame:
            available = theme_frame.evidence_available_at.map(_local_naive_timestamp)
            if available.isna().any() or available.gt(cutoff).any():
                raise ValueError("theme role evidence availability time is invalid or after asof")
        if "evidence_published_at" in theme_frame:
            raw_published = theme_frame.evidence_published_at.fillna("").astype(str).str.strip()
            has_published = raw_published.ne("")
            published = raw_published[has_published].map(_local_naive_timestamp)
            if published.isna().any() or published.gt(cutoff).any():
                raise ValueError("theme role evidence publication time is invalid or after asof")
            if not has_published.all() and "evidence_available_at" not in theme_frame:
                raise ValueError("theme role evidence without publication time needs availability time")
        theme_map = {str(row.code).zfill(6): row.to_dict() for _, row in theme_frame.iterrows()}
        candidate_codes = set(closing.code.astype(str).str.zfill(6))
        theme_known = len(candidate_codes & set(theme_map))
        theme_verified = sum(
            row.get("theme_status") in VERIFIED_THEME_STATUSES
            for code, row in theme_map.items() if code in candidate_codes
        )
        theme_membership_verified_driver_unverified = sum(
            row.get("theme_status") == "verified_membership_driver_unverified"
            for code, row in theme_map.items() if code in candidate_codes
        )
        theme_observed_secondary = sum(
            row.get("theme_status") == "observed_explicit_secondary_roster"
            for code, row in theme_map.items() if code in candidate_codes
        )
        theme_rumor = sum(
            row.get("theme_status") == "rumor_explicit_roster"
            for code, row in theme_map.items() if code in candidate_codes
        )
    competition_map = {}
    competition_known = 0
    if competition is not None:
        required_competition = {
            "code", "signal_date", "competition_identity", "source", "observed_at",
            "available_at", "data_status", "universe_complete",
        }
        if required_competition - set(competition):
            raise ValueError("competition evidence fields missing")
        competition_frame = competition.copy()
        competition_frame["code"] = competition_frame.code.astype(str).str.zfill(6)
        competition_frame["signal_date"] = pd.to_datetime(
            competition_frame.signal_date, errors="coerce").dt.strftime("%Y-%m-%d")
        if not bool(competition_frame.signal_date.eq(signal_date).any()):
            raise ValueError("competition evidence has no matching signal_date")
        competition_frame = competition_frame[competition_frame.signal_date.eq(signal_date)].copy()
        if competition_frame.code.duplicated().any():
            raise ValueError("competition evidence contains duplicate candidate codes")
        cutoff = _local_naive_timestamp(asof)
        for field in ("observed_at", "available_at"):
            parsed = competition_frame[field].map(_local_naive_timestamp)
            if parsed.isna().any():
                raise ValueError("competition evidence timestamps are invalid")
            competition_frame[field] = parsed
        valid_identity = competition_frame.competition_identity.astype(str).str.lower().isin(COMPETITION_IDENTITIES)
        valid = (competition_frame.data_status.astype(str).str.lower().eq("verified")
                 & competition_frame.source.astype(str).str.strip().ne("")
                 & competition_frame.universe_complete.map(_truthy)
                 & valid_identity
                 & competition_frame.observed_at.le(competition_frame.available_at)
                 & competition_frame.available_at.le(cutoff)
                 & competition_frame.observed_at.dt.strftime("%Y-%m-%d").eq(signal_date))
        if not bool(valid.all()):
            raise ValueError("competition evidence is not verified and complete at asof")
        competition_frame["competition_identity"] = competition_frame.competition_identity.astype(str).str.lower()
        for field in ("observed_at", "available_at"):
            competition_frame[field] = competition_frame[field].map(
                lambda value: value.isoformat() + "+08:00")
        competition_map = {str(row.code).zfill(6): row.to_dict() for _, row in competition_frame.iterrows()}
        candidate_codes = set(closing.code.astype(str).str.zfill(6))
        competition_known = len(candidate_codes & set(competition_map))
    cross_theme_map = {}
    cross_theme_attached = 0
    cross_theme_known = 0
    cross_theme_model_eligible = 0
    if cross_theme_races is not None:
        required_cross = {
            "signal_date", "code", "race_lane", "cross_theme_race_status",
            "eligible_for_cross_theme_race_training", "winner_claimed",
            "entry_authorized", "race_asof", "theme_evidence_confidence",
            "race_peer_evidence_scope",
        }
        if required_cross - set(cross_theme_races):
            raise ValueError("cross-theme race evidence fields missing")
        cross_frame = cross_theme_races.copy()
        cross_frame["code"] = cross_frame.code.astype(str).str.zfill(6)
        cross_frame["signal_date"] = pd.to_datetime(
            cross_frame.signal_date, errors="coerce").dt.strftime("%Y-%m-%d")
        cross_frame = cross_frame[cross_frame.signal_date.eq(signal_date)].copy()
        if cross_frame.empty or cross_frame.code.duplicated().any():
            raise ValueError("cross-theme race evidence must be unique and nonempty")
        cutoff = _local_naive_timestamp(asof)
        race_asof = cross_frame.race_asof.map(_local_naive_timestamp)
        if race_asof.isna().any() or race_asof.gt(cutoff).any():
            raise ValueError("cross-theme race evidence is after asof")
        candidate_codes = set(closing.code.astype(str).str.zfill(6))
        if not set(cross_frame.code).issubset(candidate_codes):
            raise ValueError("cross-theme race evidence contains non-candidate codes")
        eligible = _strict_boolean_series(
            cross_frame.eligible_for_cross_theme_race_training,
            "eligible_for_cross_theme_race_training")
        winner = _strict_boolean_series(cross_frame.winner_claimed, "winner_claimed")
        entry = _strict_boolean_series(cross_frame.entry_authorized, "entry_authorized")
        if winner.any() or entry.any():
            raise ValueError("descriptive cross-theme evidence cannot claim winner or entry")
        if (eligible & ~cross_frame.cross_theme_race_status.eq(
                "MODEL_ELIGIBLE_COMPLETE_SCOPE")).any():
            raise ValueError("cross-theme training eligibility conflicts with status")
        board_map = closing.assign(
            code=closing.code.astype(str).str.zfill(6)).set_index("code").boards.to_dict()
        expected_lane = cross_frame.code.map(
            lambda code: "FIRST_BOARD_NEW_CYCLE_RACE"
            if int(board_map[code]) == 1 else "RELAY_HEIGHT_RACE")
        if not cross_frame.race_lane.eq(expected_lane).all():
            raise ValueError("cross-theme race lane conflicts with board height")
        cross_frame["eligible_for_cross_theme_race_training"] = eligible
        cross_frame["winner_claimed"] = winner
        cross_frame["entry_authorized"] = entry
        cross_theme_map = {
            str(row.code).zfill(6): row.to_dict() for _, row in cross_frame.iterrows()}
        cross_theme_attached = len(candidate_codes & set(cross_theme_map))
        cross_theme_known = int((
            ~cross_frame.cross_theme_race_status.eq("UNKNOWN_THEME_NO_RACE_INFERENCE")
        ).sum())
        cross_theme_model_eligible = int(eligible.sum())
    abnormal_path_map = {}
    abnormal_path_summary = {
        "status": "UNAVAILABLE", "closing_candidate_count": len(closing),
        "evidence_attached_count": 0, "verified_distance_count": 0,
        "path_eligible_count": 0, "rule_distance_closest": None,
        "hundred_percent_abnormal_path_potential": None,
        "plain_price_doubling_is_sufficient": False,
        "distance_is_positive_ranking_factor": False,
        "calibrated_probability": None, "buy_signal": False,
        "candidate_reranking_allowed": False,
    }
    if abnormal_paths is not None:
        required_top = {
            "status", "asof", "signal_date", "closing_candidate_count",
            "evidence_attached_count", "verified_distance_count", "path_eligible_count",
            "rule_distance_closest", "hundred_percent_abnormal_path_potential",
            "plain_price_doubling_is_sufficient", "distance_is_positive_ranking_factor",
            "calibrated_probability", "buy_signal", "candidate_reranking_allowed",
            "candidate_records",
        }
        if required_top - set(abnormal_paths):
            raise ValueError("abnormal-path role artifact fields missing")
        cutoff = _local_naive_timestamp(asof)
        path_asof = _local_naive_timestamp(abnormal_paths.get("asof"))
        if (abnormal_paths.get("status") != "DIAGNOSTIC_ONLY"
                or abnormal_paths.get("signal_date") != signal_date
                or pd.isna(path_asof) or path_asof > cutoff):
            raise ValueError("abnormal-path role artifact is invalid or after asof")
        candidate_codes = set(closing.code.astype(str).str.zfill(6))
        records = abnormal_paths.get("candidate_records")
        if not isinstance(records, list) or len(records) != len(closing):
            raise ValueError("abnormal-path artifact must cover every closing candidate")
        record_codes = [str(row.get("code", "")).zfill(6) for row in records]
        if set(record_codes) != candidate_codes or len(record_codes) != len(set(record_codes)):
            raise ValueError("abnormal-path candidate coverage differs from closing pool")
        if (abnormal_paths.get("buy_signal") is not False
                or abnormal_paths.get("candidate_reranking_allowed") is not False
                or abnormal_paths.get("calibrated_probability") is not None
                or abnormal_paths.get("plain_price_doubling_is_sufficient") is not False
                or abnormal_paths.get("distance_is_positive_ranking_factor") is not False):
            raise ValueError("abnormal-path diagnostic cannot authorize ranking or entry")
        for role_key in ("rule_distance_closest", "hundred_percent_abnormal_path_potential"):
            role = abnormal_paths.get(role_key)
            if role is not None and (
                    role.get("code") not in candidate_codes
                    or role.get("buy_signal") is not False
                    or role.get("candidate_reranking_allowed") is not False
                    or role.get("calibrated_probability") is not None):
                raise ValueError("abnormal-path role contains an invalid claim")
        abnormal_path_map = {
            str(row["code"]).zfill(6): row for row in records
        }
        abnormal_path_summary = {
            key: abnormal_paths.get(key) for key in abnormal_path_summary
        }
    height_anchor_buyability = _height_anchor_buyability_summary(height_anchor_evidence)
    opening_gap_payoff = _opening_gap_payoff_summary(opening_gap_evidence)
    opening_gap_emotion = _opening_gap_emotion_summary(opening_gap_emotion_evidence)
    weak_to_strong_hindsight = _weak_to_strong_hindsight_summary(
        weak_to_strong_evidence)
    first_board_supply = _first_board_supply_summary(first_board_supply_evidence)
    promotion_payoff = _promotion_payoff_summary(promotion_payoff_evidence)
    repair_followthrough = _repair_followthrough_summary(
        repair_followthrough_evidence)
    repair_layer = _repair_layer_summary(repair_layer_evidence)
    high_to_low_ecology = _high_to_low_ecology_summary(
        high_to_low_ecology_evidence)
    index_emotion_divergence = _index_emotion_divergence_summary(
        index_emotion_divergence_evidence, signal_date)
    money_state = (
        "ADVERSE" if previous_median is not None and previous_median <= 0
        and promotion is not None and promotion < .25 else
        "SUPPORTIVE" if previous_median is not None and previous_median > 0
        and promotion is not None and promotion >= .25 else "MIXED"
    )
    profit_distribution_state = (
        "BROAD_POSITIVE" if previous_mean is not None and previous_mean > 0
        and previous_median is not None and previous_median > 0 else
        "NARROW_POSITIVE" if previous_mean is not None and previous_mean > 0
        and previous_median is not None and previous_median <= 0 else
        "BROAD_NEGATIVE" if previous_mean is not None and previous_mean <= 0
        and previous_median is not None and previous_median <= 0 else
        "MEDIAN_POSITIVE_MEAN_NONPOSITIVE" if previous_mean is not None
        and previous_mean <= 0 and previous_median is not None and previous_median > 0
        else "UNAVAILABLE"
    )
    profit_concentration_gap = (
        previous_mean - previous_median
        if previous_mean is not None and previous_median is not None else None
    )
    money_impact = {
        "ADVERSE": "negative median feedback and weak promotion prohibit treating a high close as next-day edge",
        "SUPPORTIVE": "positive median feedback and adequate promotion support risk appetite but do not prove candidate persistence",
        "MIXED": "premium and promotion evidence disagree or are incomplete; narrow candidate exposure and require open confirmation",
    }[money_state]
    broken_adverse = ((previous_tail is not None and previous_tail <= -.07)
                      or (failed_entry_metrics["large_loss_rate"] is not None
                          and failed_entry_metrics["large_loss_rate"] >= .10))
    broken_supportive = (previous_tail is not None and previous_tail > -.05
                         and failed_entry_metrics["large_loss_rate"] is not None
                         and failed_entry_metrics["large_loss_rate"] < .10
                         and (down_limits is None or down_limits <= 2))
    broken_state = "ADVERSE" if broken_adverse else "SUPPORTIVE" if broken_supportive else "MIXED"
    broken_impact = {
        "ADVERSE": "tail loss or failed-board large-loss frequency is elevated; reduce or cancel relay exposure",
        "SUPPORTIVE": "tail loss is contained, but failed-board mark-to-close loss still requires strict fill and exit discipline",
        "MIXED": "loss evidence is neither clearly benign nor severe; do not infer safety from the absence of limit-downs",
    }[broken_state]
    height_state = (
        "ADVERSE" if space == 0 or (ladder_gaps is not None and ladder_gaps >= 2) else
        "SUPPORTIVE" if space is not None and space >= .5
        and (ladder_gaps is None or ladder_gaps <= 1) else "MIXED"
    )
    height_impact = {
        "ADVERSE": "the prior space leader did not advance or the ladder is broken; current height names are emotion anchors, not automatic entries",
        "SUPPORTIVE": "space feedback and ladder continuity are constructive, subject to candidate-level tradability and theme confirmation",
        "MIXED": "height evidence is incomplete or conflicting; avoid using board count alone as a buy reason",
    }[height_state]
    dimensions = {
        "money_effect": {
            "state": money_state,
            "evidence": {"previous_limit_mean_return": previous_mean,
                         "previous_limit_median_return": previous_median,
                         "profit_distribution_state": profit_distribution_state,
                         "mean_median_gap": profit_concentration_gap,
                         "all_board_promotion_rate": promotion,
                         "observation_count": previous_metrics["observation_count"],
                         "source_status": previous_metrics["status"]},
            "impact": (
                money_impact + "; positive mean with non-positive median means profits are "
                "concentrated in a minority and cannot be treated as broad relay support"
                if profit_distribution_state == "NARROW_POSITIVE" else money_impact
            ),
        },
        "broken_loss_effect": {
            "state": broken_state,
            "evidence": {"previous_limit_tail_loss": previous_tail,
                         "failed_limit_entry_mean_return": failed_entry_metrics["mean_return"],
                         "failed_limit_entry_median_return": failed_entry_metrics["median_return"],
                         "failed_limit_entry_tail_10_return": failed_entry_metrics["tail_10_return"],
                         "failed_limit_entry_large_loss_count": failed_entry_metrics["large_loss_count"],
                         "failed_limit_entry_large_loss_rate": failed_entry_metrics["large_loss_rate"],
                         "failed_limit_entry_observation_count": failed_entry_metrics["observation_count"],
                         "failed_limit_entry_source_status": failed_entry_metrics["status"],
                         "down_limit_count": down_limits},
            "impact": broken_impact,
        },
        "height_feedback": {
            "state": height_state,
            "evidence": {"maximum_board": max_board, "space_feedback": space,
                         "ladder_gap_count": ladder_gaps,
                         "one_word_share": one_word_share,
                         "crowding_risk_proxy": crowding_risk,
                         "one_word_share_direct_risk_gate_allowed": False},
            "impact": (
                height_impact
                + "; aggregate one-price-board share describes crowding and buyability, "
                "but does not monotonically predict next-session deterioration"
            ),
        },
        "theme_divergence": {
            "state": "MIXED" if theme_verified else "INFERRED" if theme_known else "UNAVAILABLE",
            "evidence": {"failed_board_rate": failed_rate,
                         "specific_theme_membership": theme_verified or None,
                         "verified_membership_driver_unverified":
                         theme_membership_verified_driver_unverified or None,
                         "theme_candidate_coverage": theme_known or None,
                         "observed_secondary_membership": theme_observed_secondary or None,
                         "rumor_membership": theme_rumor or None,
                         "same_theme_support": "VERIFIED_ROSTER" if theme_verified else
                         "SECONDARY_NARRATIVE_ONLY" if theme_observed_secondary else None},
            "impact": (
                "verified business membership can establish a same-chain roster, but only a "
                "separately verified active driver and frozen expectation can authorize formal "
                "competition or high-low mapping; secondary narratives remain descriptive only"
            ),
        },
        "repair_quality": {
            "state": "ADVERSE" if regime == "退潮"
            and seal_rate is not None and seal_rate < .6 else "MIXED",
            "evidence": {"market_regime": regime, "seal_rate": seal_rate,
                         "breadth_quality": breadth_quality,
                         "limit_count_change": limit_count_change,
                         "failed_board_rate_change": failed_rate_change,
                         "breadth_quality_direct_state_change_allowed": False,
                         "b1_b3_market_path_scenario": (
                             "ELEVATED_ADVERSE_PATH_RESEARCH_WARNING"
                             if breadth_quality == "FRAGILE_EXPANSION" else
                             "NO_ELEVATED_WARNING_FROM_BREADTH_QUALITY"
                             if breadth_quality == "CLEAN_EXPANSION" else
                             "UNAVAILABLE"),
                         "b1_b3_market_path_probability": None,
                         "candidate_holding_period_inference_allowed": False,
                         "opening_follow_through": None},
            "impact": (
                "closing limit-up breadth expanded while failed-board pressure also rose; "
                "treat the repair as fragile and require stricter next-session confirmation"
                if breadth_quality == "FRAGILE_EXPANSION" else
                "breadth expansion did not worsen the failed-board rate, but no next-session "
                "auction/opening evidence exists, so repair still cannot be confirmed"
                if breadth_quality == "CLEAN_EXPANSION" else
                "no next-session auction/opening evidence exists, so repair cannot be confirmed"
            ),
        },
    }
    risk_map = {}
    announcement_summary = {"status": "UNAVAILABLE", "rows": 0, "covered_candidate_codes": 0,
                            "active_hard_events": 0, "active_hard_candidate_codes": 0}
    if announcements is not None:
        required = {"code", "published_at", "risk_type", "title", "risk_hard", "data_status"}
        if required - set(announcements):
            raise ValueError("announcement evidence fields missing")
        notices = announcements.copy()
        notices["code"] = notices.code.astype(str).str.zfill(6)
        published = pd.to_datetime(notices.published_at, errors="coerce")
        cutoff = pd.Timestamp(asof)
        if cutoff.tzinfo is not None:
            cutoff = cutoff.tz_localize(None)
        # Date-only announcement indexes are conservatively normalized to
        # 23:59:59.  For an after-close review that artificial timestamp must
        # not hide an item already received by the collector.  Receipt time is
        # admissible only as availability evidence; it does not upgrade the
        # provider's publication time to an exact timestamp.
        received = (pd.to_datetime(notices.get("source_timestamp"), errors="coerce")
                    if "source_timestamp" in notices else
                    pd.Series(pd.NaT, index=notices.index))
        date_only = (notices.get("published_at_status", pd.Series("", index=notices.index))
                     .eq("date_only_conservative_end_of_day"))
        known_by_cutoff = published.notna() & (
            published.le(cutoff) | (date_only & received.notna() & received.le(cutoff))
        )
        effective = known_by_cutoff & notices.data_status.eq("verified_index_date_only")
        hard = notices[effective & pd.to_numeric(notices.risk_hard, errors="coerce").eq(1)].copy()
        for code, group in hard.groupby("code"):
            risk_map[code] = group[["published_at", "risk_type", "title"]].to_dict("records")
        closing_codes = set(closing.code.astype(str).str.zfill(6))
        announcement_summary = {
            "status": "VERIFIED_INDEX_DATE_ONLY",
            "rows": int(len(notices)),
            "covered_candidate_codes": int(len(set(notices.code) & closing_codes)),
            "active_hard_events": int(len(hard)),
            "active_hard_candidate_codes": int(hard.code.nunique()),
            "ordinary_abnormal_notice_is_mechanical_veto": False,
        }
    first = sorted((_candidate(row, risk_map, theme_map, competition_map, cross_theme_map,
                               abnormal_path_map, symbolic_name_map)
                    for _, row in closing[closing.boards.eq(1)].iterrows()),
                   key=lambda row: row["code"])
    relay = sorted((_candidate(row, risk_map, theme_map, competition_map, cross_theme_map,
                               abnormal_path_map, symbolic_name_map)
                    for _, row in closing[closing.boards.ge(2)].iterrows()),
                   key=lambda row: (-row["boards"], row["code"]))
    highest = closing[closing.boards.eq(closing.boards.max())]
    current_maximum_board = int(closing.boards.max()) if len(closing) else 0
    anchors = [{"code": str(row.code).zfill(6), "name": row.get("name"), "boards": int(row.boards),
                "role": "EMOTION_ANCHOR_NOT_BUY_CANDIDATE",
                "active_hard_risks": risk_map.get(str(row.code).zfill(6), [])}
               for _, row in highest.iterrows()]
    candidate_map = {row["code"]: row for row in [*relay, *first]}
    highest_potential = _height_potential(relay if relay else first, current_maximum_board)
    lower_height_pool = [row for row in relay if row["boards"] < current_maximum_board]
    if lower_height_pool:
        next_level = max(row["boards"] for row in lower_height_pool)
        lower_height_pool = [row for row in lower_height_pool if row["boards"] == next_level]
    next_height_challenger = _height_potential(
        lower_height_pool if lower_height_pool else first, current_maximum_board)
    anchor_premium_reviews = [
        _anchor_premium_review(candidate_map[row["code"]], regime, money_state,
                               broken_state, height_state)
        for row in anchors if row["code"] in candidate_map
    ]
    potential_premium_review = (
        _anchor_premium_review(candidate_map[highest_potential["code"]], regime, money_state,
                               broken_state, height_state)
        if highest_potential and highest_potential["code"] in candidate_map else None)
    challenger_premium_review = (
        _anchor_premium_review(candidate_map[next_height_challenger["code"]], regime, money_state,
                               broken_state, height_state)
        if next_height_challenger and next_height_challenger["code"] in candidate_map else None)
    high_low_mappings, high_low_primary = _high_low_supplement_map(
        [*relay, *first], current_maximum_board)
    current_market_core = _current_market_core([*relay, *first])
    validated_threshold = selective_policy.get("operational_threshold")
    first_primary, first_primary_status = _validated_group_primary(
        first, validated_threshold, "FIRST_BOARD")
    relay_primary, relay_primary_status = _validated_group_primary(
        relay, validated_threshold, "RELAY")
    group_primaries = [
        ("FIRST_BOARD", first_primary), ("RELAY", relay_primary)
    ]
    group_primaries = [(group, row) for group, row in group_primaries if row is not None]
    if group_primaries:
        comprehensive_group, comprehensive_primary = sorted(
            group_primaries,
            key=lambda item: (-item[1]["calibrated_probability"],
                              -item[1]["boards"], item[1]["code"]),
        )[0]
        comprehensive_primary = dict(comprehensive_primary)
        comprehensive_primary["source_group"] = comprehensive_group
        comprehensive_primary_status = (
            "FROZEN_UNIQUE_CONDITIONAL_PRIMARY_REQUIRES_NEXT_SESSION_CONFIRMATION")
    else:
        comprehensive_primary = None
        comprehensive_primary_status = "CASH_NO_VALIDATED_GROUP_PRIMARY"
    explicit_exclusions = _explicit_high_attention_exclusions(
        [*relay, *first], current_maximum_board, highest_potential,
        next_height_challenger)
    reasons = []
    if snapshot_manifest.get("full_market_completeness") != "verified":
        reasons.append("FULL_MARKET_POOL_COMPLETENESS_NOT_INDEPENDENTLY_VERIFIED")
    if snapshot_manifest.get("security_unavailable_or_ineligible_closing_rows", 0):
        reasons.append("POINT_IN_TIME_SECURITY_STATUS_INCOMPLETE")
    if selective_policy.get("operational_threshold") is None:
        reasons.append("NO_VALIDATED_FULL_POOL_SELECTION_THRESHOLD")
    if regime in {"大分歧", "退潮"}:
        reasons.append("EMOTION_REGIME_ADVERSE")
    if dimensions["broken_loss_effect"]["state"] == "ADVERSE":
        reasons.append("BROKEN_BOARD_TAIL_LOSS_ADVERSE")
    if not theme_verified:
        reasons.append("REAL_SPECIFIC_THEME_EVIDENCE_UNAVAILABLE")
        if theme_membership_verified_driver_unverified:
            reasons.append("BUSINESS_MEMBERSHIP_VERIFIED_BUT_ACTIVE_DRIVER_NOT_VERIFIED")
        if theme_observed_secondary:
            reasons.append("SECONDARY_THEME_NARRATIVE_NOT_BUSINESS_OR_CATALYST_VERIFIED")
    elif theme_verified < len(closing):
        reasons.append("REAL_SPECIFIC_THEME_EVIDENCE_INCOMPLETE")
    if not competition_known:
        reasons.append("LEADER_COMPETITION_EVIDENCE_UNAVAILABLE")
    elif competition_known < len(closing):
        reasons.append("LEADER_COMPETITION_EVIDENCE_INCOMPLETE")
    if not cross_theme_attached:
        reasons.append("CROSS_THEME_RACE_EVIDENCE_UNAVAILABLE")
    elif (cross_theme_attached < len(closing)
          or cross_theme_known < len(closing)
          or cross_theme_model_eligible < len(closing)):
        reasons.append("CROSS_THEME_RACE_INCOMPLETE_OR_UNVERIFIED")
    if abnormal_path_summary["verified_distance_count"] < len(closing):
        reasons.append("SEVERE_ABNORMAL_RULE_DISTANCE_INCOMPLETE")
    reasons.extend(["NEXT_SESSION_AUCTION_AND_OPENING_EVIDENCE_UNAVAILABLE",
                    "ACTUAL_EXECUTION_AND_COMPLETE_TRADE_MODEL_UNAVAILABLE"])
    theme_status = ("VERIFIED_FULL_CANDIDATE_ROLES" if theme_verified == len(closing) and len(closing)
                    else "VERIFIED_CANDIDATE_ROLES" if theme_verified
                    else "VERIFIED_MEMBERSHIP_ACTIVE_DRIVER_UNVERIFIED"
                    if theme_membership_verified_driver_unverified
                    else "OBSERVED_SECONDARY_NARRATIVE_ONLY" if theme_observed_secondary
                    else "RUMOR_ONLY" if theme_rumor else "UNAVAILABLE")
    readiness = {
        "decision_mode": "CONDITIONAL_ONLY",
        "entry_plan_status": "UNAVAILABLE_UNTIL_AUCTION_CONFIRMATION",
        "auction_evidence_status": "UNAVAILABLE",
        "opening_evidence_status": "UNAVAILABLE",
        "theme_evidence_status": theme_status,
        "actual_fill_status": "UNAVAILABLE",
        "exit_evidence_status": "UNAVAILABLE",
        "conditional_only": True,
        "unknown_fields": [
            "auction_match_and_cancellation",
            "opening_vwap_and_reclaim", "competitor_elimination", "entry_fill", "t_plus_one_exit",
        ],
        "permission_rule": "NO_ENTER_UNTIL_REQUIRED_FIELDS_ARE_VERIFIED",
    }
    if theme_verified < len(closing):
        readiness["unknown_fields"].insert(0, "specific_theme_membership_for_unverified_candidates")
    if competition_known < len(closing):
        readiness["unknown_fields"].insert(0, "competition_identity_for_unverified_candidates")
    if cross_theme_model_eligible < len(closing):
        readiness["unknown_fields"].insert(
            0, "cross_theme_first_board_and_relay_race_for_unverified_candidates")
    return {
        "status": "CASH", "asof": asof, "signal_date": signal_date,
        "decision_scope": "all_observed_mainboard_nonst_candidate_status_pending_full_verification",
        "market": {
            "emotion_regime": regime, "limit_count_mainboard": _integer(emotion, "limit_count"),
            "consecutive_count": _integer(emotion, "consecutive_count"),
            "failed_board_count": _integer(emotion, "failed_count"),
            "down_limit_count": down_limits, "seal_rate": seal_rate, "failed_rate": failed_rate,
            "metric_status": "VERIFIED_MARKET_ROW_WITH_MISSING_FIELDS" if any(
                value is None for value in (premium, promotion, previous_tail, failed_tail,
                                            failed_rate, seal_rate, space, max_board,
                                            ladder_gaps, down_limits)) else "VERIFIED_COMPLETE_MARKET_ROW",
            "emotion_score": _float(emotion, "emotion_score"),
            "adverse_next_probability_research_only": _float(emotion, "emotion_adverse_next_probability"),
            "forecast_status": emotion.get("emotion_forecast_status"),
            "previous_limit_feedback": previous_metrics,
            "failed_limit_entry_feedback": failed_entry_metrics,
        },
        "emotion_dimensions": dimensions,
        "announcement_summary": announcement_summary,
        "theme_summary": {
            "status": theme_status, "known_candidate_count": theme_known,
            "verified_candidate_count": theme_verified,
            "verified_membership_driver_unverified_count":
            theme_membership_verified_driver_unverified,
            "observed_secondary_candidate_count": theme_observed_secondary,
            "rumor_candidate_count": theme_rumor,
            "closing_candidate_count": len(closing),
            "coverage": theme_known / len(closing) if len(closing) else None,
            "positive_coverage": theme_verified / len(closing) if len(closing) else None,
            "source": "point_in_time_theme_roles" if theme_roles is not None else None,
        },
        "symbolic_name_summary": symbolic_name_evidence,
        "competition_summary": {
            "status": "VERIFIED_COMPLETE" if competition_known == len(closing) and len(closing)
            else "VERIFIED_PARTIAL" if competition_known else "UNAVAILABLE",
            "known_candidate_count": competition_known,
            "closing_candidate_count": len(closing),
            "coverage": competition_known / len(closing) if len(closing) else None,
            "source": "point_in_time_competition_snapshot" if competition is not None else None,
        },
        "cross_theme_race_summary": {
            "status": (
                "MODEL_ELIGIBLE_COMPLETE_SCOPE"
                if cross_theme_model_eligible == len(closing) and len(closing)
                else "DESCRIPTIVE_INCOMPLETE_OR_UNVERIFIED"
                if cross_theme_attached else "UNAVAILABLE"
            ),
            "attached_candidate_count": cross_theme_attached,
            "known_theme_candidate_count": cross_theme_known,
            "model_eligible_candidate_count": cross_theme_model_eligible,
            "closing_candidate_count": len(closing),
            "first_board_and_relay_use_separate_lanes": True,
            "common_winner_score_forbidden": True,
            "source": "confidence_aware_cross_theme_races"
            if cross_theme_races is not None else None,
        },
        "abnormal_path_summary": abnormal_path_summary,
        "height_anchor_buyability_summary": height_anchor_buyability,
        "opening_gap_payoff_summary": opening_gap_payoff,
        "opening_gap_emotion_summary": opening_gap_emotion,
        "weak_to_strong_hindsight_summary": weak_to_strong_hindsight,
        "first_board_supply_summary": first_board_supply,
        "promotion_payoff_summary": promotion_payoff,
        "repair_followthrough_summary": repair_followthrough,
        "repair_layer_summary": repair_layer,
        "high_to_low_ecology_summary": high_to_low_ecology,
        "index_emotion_divergence_summary": index_emotion_divergence,
        "pool_summary": {
            "closing_limit_mainboard": len(closing), "first_board": len(first), "relay": len(relay),
            "failed_board_mainboard": len(failed),
            "board_level_counts": snapshot_manifest["mainboard_board_level_counts"],
            "verified_eligible_closing": snapshot_manifest["verified_eligible_closing_rows"],
            "unverified_or_ineligible_closing": snapshot_manifest[
                "security_unavailable_or_ineligible_closing_rows"],
        },
        "first_board_candidates": first, "relay_candidates": relay,
        "failed_board_codes": sorted(failed.code.astype(str).str.zfill(6).tolist()),
        "roles": {
            "emotion_anchors": anchors,
            "height_anchor_buyability_evidence": height_anchor_buyability,
            "opening_gap_payoff_evidence": opening_gap_payoff,
            "opening_gap_emotion_evidence": opening_gap_emotion,
            "weak_to_strong_hindsight_evidence": weak_to_strong_hindsight,
            "first_board_supply_evidence": first_board_supply,
            "promotion_payoff_evidence": promotion_payoff,
            "repair_followthrough_evidence": repair_followthrough,
            "repair_layer_evidence": repair_layer,
            "high_to_low_ecology_evidence": high_to_low_ecology,
            "index_emotion_divergence_evidence": index_emotion_divergence,
            "current_market_core": current_market_core,
            "current_market_core_status": (
                "VERIFIED_EOD_CORE_IDENTITY_NOT_BUY_SIGNAL"
                if current_market_core else "NO_FULLY_VERIFIED_CURRENT_MARKET_CORE"),
            "highest_leader_potential": highest_potential,
            "highest_leader_potential_status": (
                "INFERRED_EOD_RANKING_REQUIRES_NEXT_SESSION_CONFIRMATION"
                if highest_potential else "NO_CLOSING_LIMIT_CANDIDATE"),
            "next_height_challenger": next_height_challenger,
            "next_height_challenger_status": (
                "INFERRED_IF_CURRENT_ANCHOR_FAILS_NOT_PROBABILITY"
                if next_height_challenger else "NO_LOWER_HEIGHT_CHALLENGER"),
            "anchor_premium_reviews": anchor_premium_reviews,
            "potential_anchor_premium_review": potential_premium_review,
            "challenger_premium_review": challenger_premium_review,
            "high_low_supplement_mappings": high_low_mappings,
            "high_low_mapping_coverage": {
                "identified_high_cores": sum(
                    item["high_core"]["core_identity_verified"] for item in high_low_mappings),
                "high_position_records": len(high_low_mappings),
                "unverified_core_identity_records": sum(
                    not item["high_core"]["core_identity_verified"] for item in high_low_mappings),
                "mapped_high_cores": sum(
                    item["supplement_candidate"] is not None for item in high_low_mappings),
                "unmapped_high_cores": sum(
                    item["supplement_candidate"] is None for item in high_low_mappings),
                "narrative_alternative_records": sum(
                    bool(item["narrative_only_low_alternatives"])
                    for item in high_low_mappings),
                "verified_membership_only_records": sum(
                    bool(item.get("verified_membership_only_low_alternatives"))
                    for item in high_low_mappings),
            },
            "high_low_supplement_candidate": high_low_primary,
            "high_low_supplement_status": (
                "EOD_RESEARCH_MAPPING_REQUIRES_NEXT_SESSION_CONFIRMATION"
                if high_low_primary else "NO_VERIFIED_MAPPING"
            ),
            "first_board_primary_candidate": first_primary,
            "first_board_primary_status": first_primary_status,
            "relay_primary_candidate": relay_primary,
            "relay_primary_status": relay_primary_status,
            "comprehensive_primary_candidate": comprehensive_primary,
            "comprehensive_primary_status": comprehensive_primary_status,
            "explicit_high_attention_exclusions": explicit_exclusions,
            "hundred_percent_abnormal_path_potential": abnormal_path_summary[
                "hundred_percent_abnormal_path_potential"],
            "hundred_percent_abnormal_path_status": (
                "DIAGNOSTIC_PATH_ONLY_NOT_BUY_REASON"
                if abnormal_path_summary["hundred_percent_abnormal_path_potential"]
                else "NO_VERIFIED_PATH_POTENTIAL"),
            "abnormal_rule_distance_closest": abnormal_path_summary[
                "rule_distance_closest"],
            "current_best_buy_opportunity": None,
            "current_best_buy_opportunity_status": "UNAVAILABLE_AUCTION_OPENING_AND_EXECUTION_EVIDENCE",
        },
        "selection": {
            "validated_threshold": validated_threshold,
            "conditional_primary": None, "maximum_acceptable_price": None,
            "position_shares": 0, "holding_plan": None,
        },
        "decision_readiness": readiness,
        "blocking_reasons": sorted(set(reasons)),
        "next_session_required_checks": [
            "09:20/09:24:50/09:25 auction match and cancellation evidence",
            "09:30/09:35/10:00 VWAP, pullback volume, reclaim and reseal evidence",
            "verified specific-theme roster and competitor elimination",
            "announcements and abnormal-move distance available before decision",
        ],
        "one_primary_maximum": True, "automatic_backup_allowed": False,
        "order_submitted": False, "position_created": False,
        "guarantee_of_premium": False, "direct_recommendation": False,
    }


def attach_decision_plan(report, plan=None):
    """Attach independently authored, timestamp-matched research; never fabricate it."""
    from daily_decision_contract_v136 import audit as audit_plan
    if plan is None:
        report["emotion_selection_plan"] = {
            "status": "UNAVAILABLE", "issues": ["FROZEN_EMOTION_SELECTION_PLAN_NOT_PROVIDED"],
            "entry_permission": "NO_ENTER"}
        return report
    result = audit_plan(plan)
    issues = list(result["issues"])
    if plan.get("report_date") != report["signal_date"] or plan.get("asof") != report["asof"]:
        issues.append("PLAN_REVIEW_TIME_MISMATCH")
    pool = {str(c["code"]).zfill(6): c for group in ("first_board_candidates", "relay_candidates")
            for c in report[group]}
    for candidate in plan.get("focus_candidates", []):
        code = str(candidate.get("code", "")).zfill(6)
        if code not in pool:
            issues.append("PLAN_CANDIDATE_OUTSIDE_REVIEW_POOL")
        elif candidate.get("boards") != pool[code].get("boards"):
            issues.append("PLAN_CANDIDATE_BOARD_MISMATCH")
    summary = plan.get("universe_summary", {})
    if (summary.get("closing_limit_count") != report["pool_summary"]["closing_limit_mainboard"]
            or summary.get("first_board_count") != report["pool_summary"]["first_board"]):
        issues.append("PLAN_REVIEW_POOL_COUNT_MISMATCH")
    result["issues"] = sorted(set(issues))
    if issues:
        result["status"] = "INVALID_REPORT"
    report["emotion_selection_plan"] = result
    # Keep the supplied evidence for audit, including invalid plans; do not promote selections.
    report["supplied_decision_plan"] = plan
    return report


def markdown(report):
    market, pool = report["market"], report["pool_summary"]
    symbolic = report.get("symbolic_name_summary", {})
    symbolic_top = symbolic.get("top_cluster")
    first_board_upgrade = report.get("first_board_screening_summary", {})
    theme_matrix_upgrade = report.get("theme_screening_matrix_summary", {})
    potential = report["roles"].get("highest_leader_potential")
    potential_review = report["roles"].get("potential_anchor_premium_review")
    challenger = report["roles"].get("next_height_challenger")
    challenger_review = report["roles"].get("challenger_premium_review")
    anchor_reviews = report["roles"].get("anchor_premium_reviews", [])
    high_low_mappings = report["roles"].get("high_low_supplement_mappings", [])
    high_low_primary = report["roles"].get("high_low_supplement_candidate")
    current_core = report["roles"].get("current_market_core")
    first_primary = report["roles"].get("first_board_primary_candidate")
    relay_primary = report["roles"].get("relay_primary_candidate")
    comprehensive_primary = report["roles"].get("comprehensive_primary_candidate")
    exclusions = report["roles"].get("explicit_high_attention_exclusions", [])
    height_anchor_evidence = report["roles"].get("height_anchor_buyability_evidence", {})
    opening_gap_evidence = report["roles"].get("opening_gap_payoff_evidence", {})
    opening_gap_emotion = report["roles"].get("opening_gap_emotion_evidence", {})
    weak_to_strong = report["roles"].get(
        "weak_to_strong_hindsight_evidence", {})
    first_board_supply = report["roles"].get("first_board_supply_evidence", {})
    promotion_payoff = report["roles"].get("promotion_payoff_evidence", {})
    repair_followthrough = report["roles"].get(
        "repair_followthrough_evidence", {})
    repair_layer = report["roles"].get("repair_layer_evidence", {})
    high_to_low_ecology = report["roles"].get(
        "high_to_low_ecology_evidence", {})
    index_emotion_divergence = report["roles"].get(
        "index_emotion_divergence_evidence", {})
    mapping_lines = []
    for item in high_low_mappings:
        prefix = f"- {item['high_core']['name']}({item['high_core']['code']}) → "
        if item["supplement_candidate"]:
            low = item["supplement_candidate"]
            mapping_lines.append(
                prefix + f"{low['name']}({low['code']})，仅为同细分题材资金迁移研究映射，"
                "次日仍需确认。"
            )
        elif item.get("verified_membership_only_low_alternatives"):
            alternatives = "、".join(
                f"{row['name']}({row['code']})"
                for row in item["verified_membership_only_low_alternatives"]
            )
            mapping_lines.append(
                prefix + "同业务支线已核验备选（当日驱动未核验，不是补涨候选）="
                + alternatives + "；阻断=" + "、".join(item["blocking_reasons"])
            )
        elif item.get("narrative_only_low_alternatives"):
            alternatives = "、".join(
                f"{row['name']}({row['code']})" for row in item["narrative_only_low_alternatives"]
            )
            mapping_lines.append(
                prefix + "正式映射无；二手叙事备选（不是补涨候选）=" + alternatives
                + "；阻断=" + "、".join(item["blocking_reasons"])
            )
        else:
            mapping_lines.append(
                prefix + "无合格低位映射；" + "、".join(item["blocking_reasons"])
            )
    lines = [
        f"# {report['signal_date']} 龙头战法盘后复盘（证据版）", "",
        f"结论：`{report['status']}`。当前不产生买入候选。", "",
        "情绪—选股预案审核：" + report.get("emotion_selection_plan", {}).get("status", "UNAVAILABLE")
        + "。结构审核不代表预测有效或买点已确认。", "",
        "## 市场与情绪", "",
        f"情绪状态：{market['emotion_regime']}；主板涨停{pool['closing_limit_mainboard']}只，"
        f"首板{pool['first_board']}只、连板{pool['relay']}只、炸板{pool['failed_board_mainboard']}只、"
        f"跌停{_display_number(market['down_limit_count'])}只。封板率{_display_percent(market['seal_rate'])}，"
        f"炸板率{_display_percent(market['failed_rate'])}。", "",
        f"昨日涨停平均反馈{_display_percent(report['emotion_dimensions']['money_effect']['evidence']['previous_limit_mean_return'], 2)}，"
        f"中位反馈{_display_percent(report['emotion_dimensions']['money_effect']['evidence']['previous_limit_median_return'], 2)}，"
        f"晋级率{_display_percent(report['emotion_dimensions']['money_effect']['evidence']['all_board_promotion_rate'], 1)}；"
        f"尾部反馈{_display_percent(report['emotion_dimensions']['broken_loss_effect']['evidence']['previous_limit_tail_loss'], 2)}。", "",
        "指数层与短线接力生态层：" + (
            f"上证A股/深证A股涨跌为"
            f"{_display_percent(index_emotion_divergence.get('sh_a_return_pct', 0) / 100, 2)}/"
            f"{_display_percent(index_emotion_divergence.get('sz_a_return_pct', 0) / 100, 2)}，"
            f"昨日涨停中位反馈"
            f"{_display_percent(index_emotion_divergence.get('previous_limit_median_return'), 2)}，"
            f"分类={index_emotion_divergence.get('divergence_category')}。"
            "指数翻红不能覆盖涨停中位亏损、炸板与断板负反馈；"
            f"背离复制闸门={'通过' if index_emotion_divergence.get('replication_gate_passed') else '未通过'}，"
            "所以只作情境提示，不作硬否决、个股排序或买点。"
            if index_emotion_divergence.get("status") == "BOUND_DIAGNOSTIC_ONLY"
            else "研究证据未绑定；不得用指数颜色替代短线生态判断。"
        ), "",
        "晋级身份与买后赔率分离：" + (
            f"历史验证{promotion_payoff.get('validation_sessions')}日中，次日晋级率约"
            f"{_display_percent(promotion_payoff.get('promotion_rate'), 1)}，"
            f"完整分母可买B+2正收益路径约"
            f"{_display_percent(promotion_payoff.get('full_buyable_b2_positive_rate'), 1)}；"
            f"已晋级股票中仍有"
            f"{_display_percent(promotion_payoff.get('promoted_without_buyable_b2_positive_rate'), 1)}"
            "未形成可买且B+2为正的路径。晋级率只描述身份延续，不能冒充交易胜率或买点。"
            if promotion_payoff.get("status") == "BOUND_DIAGNOSTIC_ONLY"
            else "研究证据未绑定；晋级率不能代替可成交、B+2/B+3收益和大面风险。"
        ), "",
        "修复日次日延续校验：" + (
            f"当前修复质量{_display_number(repair_followthrough.get('latest_quality_score'))}/4，"
            f"类别{repair_followthrough.get('latest_quality_class')}；最近验证段宽修复次日再退潮"
            f"{_display_percent(repair_followthrough.get('validation_broad_relapse_rate'), 1)}，"
            f"窄修复{_display_percent(repair_followthrough.get('validation_narrow_relapse_rate'), 1)}。"
            "但发现段差异弱，且两段差异区间均跨零；这里只支持主情景，不给校准概率、仓位或买点。"
            "次日仍须由竞价、中位晋级和炸板亏钱效应共同确认。"
            if repair_followthrough.get("status") == "BOUND_DIAGNOSTIC_ONLY"
            else "研究证据未绑定；不能把收盘修复直接外推为次日一致或可买。"
        ), "",
        "修复层级传导：" + (
            f"低位={'确认' if repair_layer.get('latest_first_board_confirmed') else '未确认'}、"
            f"中位={'确认' if repair_layer.get('latest_middle_board_confirmed') else '未确认'}、"
            f"高度={'确认' if repair_layer.get('latest_height_confirmed') else '未确认'}，"
            f"共{_display_number(repair_layer.get('latest_layer_score'))}/3层，"
            f"类别{repair_layer.get('latest_layer_class')}。开发段中位确认/未确认的次日复退潮率为"
            f"{_display_percent(repair_layer.get('development_middle_confirmed_relapse_rate'), 1)}/"
            f"{_display_percent(repair_layer.get('development_middle_unconfirmed_relapse_rate'), 1)}，"
            f"验证段变为{_display_percent(repair_layer.get('validation_middle_confirmed_relapse_rate'), 1)}/"
            f"{_display_percent(repair_layer.get('validation_middle_unconfirmed_relapse_rate'), 1)}；"
            f"三层同步在验证段仅{_display_number(repair_layer.get('validation_synchronized_sessions'))}日。"
            "方向未稳定，中位板只能作为次日确认对象，不能成为硬闸门、个股加分或加仓理由。"
            if repair_layer.get("status") == "BOUND_DIAGNOSTIC_ONLY"
            else "研究证据未绑定；不能从高标独强或首板扩张推断修复已经传导。"
        ), "",
        "高切低市场窗口：" + (
            f"高度风险增加={'是' if high_to_low_ecology.get('height_risk_increased') else '否'}，"
            f"低位层偏强={'是' if high_to_low_ecology.get('low_layer_strong') else '否'}，"
            f"生态{_display_number(high_to_low_ecology.get('ecology_score'))}/4，"
            f"窗口={'激活' if high_to_low_ecology.get('window_active') else '未激活'}。"
            f"开发段健康/脆弱窗口次日逆风率为"
            f"{_display_percent(high_to_low_ecology.get('development_healthy_adverse_rate'), 1)}/"
            f"{_display_percent(high_to_low_ecology.get('development_fragile_adverse_rate'), 1)}，"
            f"验证段为{_display_percent(high_to_low_ecology.get('validation_healthy_adverse_rate'), 1)}/"
            f"{_display_percent(high_to_low_ecology.get('validation_fragile_adverse_rate'), 1)}。"
            "跨期不稳定，且历史数据没有真实细分题材和个股补涨身份；该窗口不能自动选补涨股、加仓或买入。"
            if high_to_low_ecology.get("status") == "BOUND_DIAGNOSTIC_ONLY"
            else "研究证据未绑定；不得因低位活跃或高标走弱机械认定高切低。"
        ), "",
        f"赚钱分布：{report['emotion_dimensions']['money_effect']['evidence']['profit_distribution_state']}；"
        f"均值-中位差{_display_percent(report['emotion_dimensions']['money_effect']['evidence']['mean_median_gap'], 2)}。"
        "平均为正但中位不为正时，只表示少数强股抬高均值，不升级为普遍接力友好。", "",
        f"涨停扩张质量：{report['emotion_dimensions']['repair_quality']['evidence']['breadth_quality']}；"
        f"涨停家数变化{_display_number(report['emotion_dimensions']['repair_quality']['evidence']['limit_count_change'])}，"
        f"炸板率变化{_display_percent(report['emotion_dimensions']['repair_quality']['evidence']['failed_board_rate_change'], 2)}。"
        "涨停增加但炸板率同步上升时，按脆弱扩张处理，不定义为全面修复。", "",
        "首板供给分母校验：" + (
            f"历史验证{first_board_supply.get('validation_sessions')}日中，"
            f"每天至少一只首板晋级的比例为"
            f"{_display_percent(first_board_supply.get('validation_any_promotion_rate'), 1)}，"
            f"几乎恒真；首板≤{_display_number(first_board_supply.get('low_supply_max'))}只/"
            f">{_display_number(first_board_supply.get('high_supply_strictly_above'))}只时，"
            f"单票晋级率分别约"
            f"{_display_percent(first_board_supply.get('validation_low_supply_candidate_rate'), 1)}/"
            f"{_display_percent(first_board_supply.get('validation_high_supply_candidate_rate'), 1)}。"
            "首板多只说明供给和广度，不能升级单票赔率、情绪许可或买点。"
            if first_board_supply.get("status") == "BOUND_DIAGNOSTIC_ONLY"
            else "研究证据未绑定；首板家数不能代替单票质量与题材结构。"
        ), "",
        f"一致拥挤代理：一字占比"
        f"{_display_percent(report['emotion_dimensions']['height_feedback']['evidence']['one_word_share'], 1)}，"
        f"拥挤风险代理{_display_percent(report['emotion_dimensions']['height_feedback']['evidence']['crowding_risk_proxy'], 1)}。"
        "该信息用于判断拥挤与可买性，不单独预测次日必分歧；个股一字买不到仍按执行约束处理。", "",
        f"未来B+1至B+3市场路径："
        f"{report['emotion_dimensions']['repair_quality']['evidence']['b1_b3_market_path_scenario']}。"
        "当前只有历史情景差异，没有校准概率；不得由市场路径直接推导个股持有三天。", "",
        f"按涨停价成交至炸板收盘的推断反馈：平均"
        f"{_display_percent(report['emotion_dimensions']['broken_loss_effect']['evidence']['failed_limit_entry_mean_return'], 2)}，"
        f"中位{_display_percent(report['emotion_dimensions']['broken_loss_effect']['evidence']['failed_limit_entry_median_return'], 2)}，"
        f"大于等于7%亏损"
        f"{_display_number(report['emotion_dimensions']['broken_loss_effect']['evidence']['failed_limit_entry_large_loss_count'])}/"
        f"{_display_number(report['emotion_dimensions']['broken_loss_effect']['evidence']['failed_limit_entry_observation_count'])}；"
        "该项为普通10%涨停价成交代理，不是实际成交记录。", "",
        f"真实题材证据：{report['theme_summary']['status']}；覆盖"
        f"{report['theme_summary']['known_candidate_count']}/{report['theme_summary']['closing_candidate_count']}只，"
        f"业务+驱动强验证覆盖{report['theme_summary']['verified_candidate_count']}/"
        f"{report['theme_summary']['closing_candidate_count']}只，"
        f"仅业务归属已核验{report['theme_summary'].get('verified_membership_driver_unverified_count', 0)}只。", "",
        (
            f"名称/符号共振：`DIAGNOSTIC_ONLY`；识别{symbolic.get('cluster_count', 0)}组。"
            + (
                f"首位“{symbolic_top['token']}”覆盖{symbolic_top['member_count']}只，"
                f"其中连板{symbolic_top['relay_member_count']}只、首板"
                f"{symbolic_top['first_board_member_count']}只、炸板观察"
                f"{symbolic_top['failed_board_member_count']}只，板位"
                f"{symbolic_top['distinct_board_levels']}、跨"
                f"{symbolic_top['distinct_industry_count']}个行业，分类"
                f"{symbolic_top['classification']}。"
                if symbolic_top else "当日没有达到冻结门槛的名称聚集。"
            )
            + "名称匹配是已归档池内事实，‘符号题材’属于推断；不得替代产业题材、候选排序或买点。"
        ), "",
        (
            "首板V218旁路："
            f"{first_board_upgrade.get('status', 'UNAVAILABLE')}；挑战模型升级="
            f"{first_board_upgrade.get('challenger_promoted', False)}；"
            f"盘后冻结观察={((first_board_upgrade.get('current_primary_close_quality_watch') or {}).get('name') or '无')}。"
            "该观察不是买入信号，次日确认失败不顺延替补。"
        ), "",
        (
            "题材V219矩阵："
            f"{theme_matrix_upgrade.get('status', 'UNAVAILABLE')}；"
            f"真实题材证据可用={theme_matrix_upgrade.get('real_theme_evidence_available', False)}；"
            "真实题材、二手名单、行业代理和字辈叙事分通道，禁止跨通道总分比较。"
        ), "",
        f"龙头竞争证据：{report['competition_summary']['status']}；覆盖"
        f"{report['competition_summary']['known_candidate_count']}/"
        f"{report['competition_summary']['closing_candidate_count']}只。", "",
        f"跨题材卡位证据：{report['cross_theme_race_summary']['status']}；已绑定"
        f"{report['cross_theme_race_summary']['attached_candidate_count']}/"
        f"{report['cross_theme_race_summary']['closing_candidate_count']}只，其中题材已知"
        f"{report['cross_theme_race_summary']['known_theme_candidate_count']}只、可训练"
        f"{report['cross_theme_race_summary']['model_eligible_candidate_count']}只。"
        "首板只进入新周期/低位注意力赛道，连板只进入高度/梯队资金赛道，禁止共用胜者分。", "",
        f"严重异常路径证据：{report['abnormal_path_summary']['status']}；规则距离已核验"
        f"{report['abnormal_path_summary']['verified_distance_count']}/"
        f"{report['abnormal_path_summary']['closing_candidate_count']}只，符合完整路径角色"
        f"{report['abnormal_path_summary']['path_eligible_count']}只。接近阈值只是算术位置，"
        "不增加候选分、不代表命中概率，也不是买入理由。", "",
        f"公告索引覆盖{report['announcement_summary']['covered_candidate_codes']}只候选，"
        f"有效硬风险事件{report['announcement_summary']['active_hard_events']}条、"
        f"涉及{report['announcement_summary']['active_hard_candidate_codes']}只；普通异动公告不作机械否决。", "",
        "## 最高标、情绪锚与买入价值", "",
        "- 最高标可买性校验：" + (
            f"纠正标签验证{height_anchor_evidence.get('validation_sessions')}个交易日，"
            f"板位基线日内首选成功率{_display_percent(height_anchor_evidence.get('boards_only_daily_top1_success_rate'), 1)}，"
            f"加入身份、情绪和量价后的挑战模型为"
            f"{_display_percent(height_anchor_evidence.get('challenger_daily_top1_success_rate'), 1)}；"
            "未证明增益，最高标不得自动填入连板主候选或当前买点。"
            if height_anchor_evidence.get("status") == "BOUND_DIAGNOSTIC_ONLY"
            else "研究证据未绑定；最高标只能作为情绪角色，不能推导买点。"
        ),
        "- 情绪锚：" + "、".join(
            f"{x['name']}({x['code']}) {x['boards']}板，硬风险{len(x['active_hard_risks'])}条"
            for x in report["roles"]["emotion_anchors"]),
        "- 已有情绪锚溢价审查：" + ("；".join(
            f"{x['name']}({x['code']})={x['premium_space']}/{x['buyability']}，{x['conclusion']}"
            for x in anchor_reviews) if anchor_reviews else "无。"),
        "- 最可能成为/保持下一最高标：" + (
            f"{potential['name']}({potential['code']})，当前{potential['boards']}板，"
            f"研究分{potential['research_score_not_probability']:.1f}（非概率，"
            f"{potential['evidence_status']}）。" if potential else "无。"),
        "- 潜在最高标溢价与可买性：" + (
            f"{potential_review['premium_space']}/{potential_review['buyability']}，"
            f"{potential_review['conclusion']}" if potential_review else "不可判断。"),
        "- 当前锚失败后的高度接班研究首位：" + (
            f"{challenger['name']}({challenger['code']})，当前{challenger['boards']}板，"
            f"仅在当前锚断板且自身主动晋级时成立；"
            f"溢价/可买性={challenger_review['premium_space']}/{challenger_review['buyability']}。"
            if challenger and challenger_review else "无。"),
        "- 当前买入机会最好：无；盘后只能形成次日条件预案，保持现金。", "",
        "- 100%严重异常路径潜力：" + (
            f"{report['roles']['hundred_percent_abnormal_path_potential']['name']}"
            f"({report['roles']['hundred_percent_abnormal_path_potential']['code']})，"
            f"距100%阈值{report['roles']['hundred_percent_abnormal_path_potential']['distance_100_pp']:.2f}个百分点，"
            f"反事实连续涨停路径{report['roles']['hundred_percent_abnormal_path_potential']['scenario_limit_boards_to_100']}步；"
            "仅诊断，不构成概率或买点。"
            if report['roles']['hundred_percent_abnormal_path_potential'] else
            "无；规则、基准、重置、题材路径或身份证据不完整时不强选。"),
        "- 规则距离最近：" + (
            f"{report['roles']['abnormal_rule_distance_closest']['name']}"
            f"({report['roles']['abnormal_rule_distance_closest']['code']})，"
            f"距100%阈值{report['roles']['abnormal_rule_distance_closest']['distance_100_pp']:.2f}个百分点；"
            "它可以与路径潜力角色不同，且距离最近本身不提高交易价值。"
            if report['roles']['abnormal_rule_distance_closest'] else
            "不可用；没有候选完成规则距离全链核验。"), "",
        "- 角色边界：入选最高标候选不等于可以买；只有溢价空间、可成交性、竞价承接和板块同步全部确认后才能升级。", "",
        "## 固定角色与分组主候选", "",
        "- 当前市场核心：" + (
            f"{current_core['name']}({current_core['code']})，{current_core['boards']}板，"
            f"身份={current_core['competition_identity']}；这是收盘身份，不是买点。"
            if current_core else "无；最高板不会自动顶替市场核心。"),
        "- 连板主候选：" + (
            f"{relay_primary['name']}({relay_primary['code']})，校准晋级概率"
            f"{relay_primary['calibrated_probability']:.1%}；仅冻结观察，待次日确认。"
            if relay_primary else "无；缺少达到冻结阈值的样本外校准候选。"),
        "- 首板主候选：" + (
            f"{first_primary['name']}({first_primary['code']})，校准晋级概率"
            f"{first_primary['calibrated_probability']:.1%}；仅冻结观察，待次日确认。"
            if first_primary else "无；缺少达到冻结阈值的样本外校准候选。"),
        "- 综合唯一首选：" + (
            f"{comprehensive_primary['name']}({comprehensive_primary['code']})，来源="
            f"{comprehensive_primary['source_group']}；仍不是盘后直接买入指令。"
            if comprehensive_primary else "空仓；首板与连板均无验证合格主候选，不强选。"),
        "- 当前买入机会最好：无；竞价、开盘、可成交价格和风险收益尚未核验。", "",
        "### 明确排除的高关注代理", "",
        *(
            [f"- {row['name']}({row['code']}) {row['boards']}板："
             + "、".join(row["exclusion_reasons"])
             for row in exclusions]
            or ["- 无需列示；当前没有达到高度/研究角色代理条件的股票。"]
        ), "",
        "## 高位到低位补涨映射", "",
        *(mapping_lines or ["- 当前没有可核验的高位核心映射。"]),
        "- 高切低主候选：" + (
            f"{high_low_primary['name']}({high_low_primary['code']})；不是买点，"
            "只有高标风险上升但题材未崩、其主动胜出且可成交时才继续。"
            if high_low_primary else "无，不为完成输出强行挑选。"), "",
        "## 潜在最高标次日买入预案", "",
        "- 开盘强度历史校验：" + (
            f"验证{opening_gap_evidence.get('validation_sessions')}个交易日；"
            f">7%开盘中一字不可买占"
            f"{_display_percent(opening_gap_evidence.get('above_seven_one_price_rate'), 1)}，"
            f"可触板路径B+1大面代理在5%—7%/高于7%分别为"
            f"{_display_percent(opening_gap_evidence.get('five_to_seven_big_loss_rate'), 1)}/"
            f"{_display_percent(opening_gap_evidence.get('above_seven_big_loss_rate'), 1)}。"
            "收益优势区间未稳定通过，因此0%—5%或5%—7%都不是自动买点；"
            "高于7%继续作默认风险标记，等待真实承接、回封和题材同步。"
            if opening_gap_evidence.get("status") == "BOUND_DIAGNOSTIC_ONLY"
            else "研究证据未绑定；沿用条件判断，开盘涨幅本身不构成买点。"
        ),
        "- 情绪×高开交互校验：" + (
            f"验证{opening_gap_emotion.get('validation_sessions')}个交易日；"
            "启动/升温/修复、加速/一致、分歧/退潮三组中的收益差异未在前后时段稳定复现。"
            "因此环境标签既不能豁免>7%高开风险，也不能把逆风高开机械取消；"
            "仍以真实承接、回封、题材同步和价格上限为准。"
            if opening_gap_emotion.get("status") == "BOUND_DIAGNOSTIC_ONLY"
            else "研究证据未绑定；情绪标签不能自行覆盖开盘执行规则。"
        ),
        "- 低开弱转强分母校验：" + (
            f"验证{weak_to_strong.get('validation_sessions')}个交易日；"
            f"全部低开候选的可买B+2正收益路径仅"
            f"{_display_percent(weak_to_strong.get('full_low_open_success_rate'), 1)}，"
            f"若事后只保留后来触板者则变为"
            f"{_display_percent(weak_to_strong.get('later_touch_conditioned_rate'), 1)}，"
            f"表面胜率被放大约{weak_to_strong.get('conditioning_inflation_multiple'):.1f}倍。"
            "后来触板是09:30之后的未来信息；低开既不能直接买，也不能机械剔除，"
            "必须等收回昨收与VWAP、量能改善、题材同步后再判断。"
            if weak_to_strong.get("status") == "BOUND_DIAGNOSTIC_ONLY"
            else "研究证据未绑定；低开不能用后来走势倒推09:30买点。"
        ),
        *( [
            f"- 高开：{potential_review['next_day_execution']['high_open']}",
            f"- 平开：{potential_review['next_day_execution']['flat_open']}",
            f"- 低开：{potential_review['next_day_execution']['low_open']}",
            f"- 板块走弱：{potential_review['next_day_execution']['sector_break']}",
            f"- 价格上限：{potential_review['next_day_execution']['price_rule']}",
            f"- 持有与退出：{potential_review['next_day_execution']['holding']}",
        ] if potential_review else ["- 当前没有可制定预案的潜在最高标。"]), "",
        "## 禁止买入原因", "",
        *["- " + reason for reason in report["blocking_reasons"]], "",
        "## 决策可执行性", "",
        f"模式：`{report['decision_readiness']['decision_mode']}`；入场计划："
        f"`{report['decision_readiness']['entry_plan_status']}`；实际成交："
        f"`{report['decision_readiness']['actual_fill_status']}`；退出证据："
        f"`{report['decision_readiness']['exit_evidence_status']}`。", "",
        "缺失字段不会按0处理：" + "、".join(report["decision_readiness"]["unknown_fields"]) + "。", "",
        "## 连板池", "",
        "|代码|名称|板位|真实题材|符号标签|跨题材赛道|卡位证据|竞争身份|证券状态|硬风险|", "|---|---|---:|---|---|---|---|---|---|---:|",
        *[f"|{x['code']}|{x['name']}|{x['boards']}|{x['theme_status']}|{','.join(c['token'] for c in x['symbolic_name_clusters']) or '无'}|{x['cross_theme_race']['lane'] or '不可用'}|{x['cross_theme_race']['status']}|{x['competition_identity']}|{x['eligibility_status']}|{len(x['active_hard_risks'])}|"
          for x in report["relay_candidates"]], "", "## 首板池", "",
        "|代码|名称|真实题材|符号标签|跨题材赛道|卡位证据|竞争身份|证券状态|硬风险|", "|---|---|---|---|---|---|---|---|---:|",
        *[f"|{x['code']}|{x['name']}|{x['theme_status']}|{','.join(c['token'] for c in x['symbolic_name_clusters']) or '无'}|{x['cross_theme_race']['lane'] or '不可用'}|{x['cross_theme_race']['status']}|{x['competition_identity']}|{x['eligibility_status']}|{len(x['active_hard_risks'])}|"
          for x in report["first_board_candidates"]], "",
        "本报告没有实时竞价、开盘成交或真实退出证据；题材证据仅在绑定的时点化来源通过校验后显示，不构成买卖建议。",
    ]
    return "\n".join(lines) + "\n"


def run(snapshot_path, snapshot_manifest_path, market_path, policy_path, output_dir, asof,
        announcements_path=None, theme_roles_path=None, competition_path=None,
        previous_limit_path=None, decision_plan_path=None, cross_theme_races_path=None,
        abnormal_paths_path=None, height_anchor_evidence_path=None,
        opening_gap_evidence_path=None, opening_gap_emotion_evidence_path=None,
        weak_to_strong_evidence_path=None, first_board_supply_evidence_path=None,
        promotion_payoff_evidence_path=None, repair_followthrough_evidence_path=None,
        repair_layer_evidence_path=None, high_to_low_ecology_evidence_path=None,
        index_emotion_divergence_evidence_path=None,
        previous_symbolic_name_evidence_path=None,
        first_board_screening_evidence_path=None,
        theme_screening_matrix_evidence_path=None,
        manual_official_update_path=None, official_archive_path=None,
        official_registered_path=None):
    paths = list(map(Path, (snapshot_path, snapshot_manifest_path, market_path, policy_path)))
    announcement_path = Path(announcements_path) if announcements_path else None
    if announcement_path:
        paths.append(announcement_path)
    theme_roles_path = Path(theme_roles_path) if theme_roles_path else None
    if theme_roles_path:
        paths.append(theme_roles_path)
    competition_path = Path(competition_path) if competition_path else None
    if competition_path:
        paths.append(competition_path)
    previous_limit_path = Path(previous_limit_path) if previous_limit_path else None
    if previous_limit_path:
        paths.append(previous_limit_path)
    decision_plan_path = Path(decision_plan_path) if decision_plan_path else None
    if decision_plan_path:
        paths.append(decision_plan_path)
    cross_theme_races_path = Path(cross_theme_races_path) if cross_theme_races_path else None
    if cross_theme_races_path:
        paths.append(cross_theme_races_path)
    abnormal_paths_path = Path(abnormal_paths_path) if abnormal_paths_path else None
    if abnormal_paths_path:
        paths.append(abnormal_paths_path)
    height_anchor_evidence_path = (Path(height_anchor_evidence_path)
                                   if height_anchor_evidence_path else None)
    if height_anchor_evidence_path:
        paths.append(height_anchor_evidence_path)
    opening_gap_evidence_path = (Path(opening_gap_evidence_path)
                                 if opening_gap_evidence_path else None)
    if opening_gap_evidence_path:
        paths.append(opening_gap_evidence_path)
    opening_gap_emotion_evidence_path = (Path(opening_gap_emotion_evidence_path)
                                         if opening_gap_emotion_evidence_path else None)
    if opening_gap_emotion_evidence_path:
        paths.append(opening_gap_emotion_evidence_path)
    weak_to_strong_evidence_path = (Path(weak_to_strong_evidence_path)
                                    if weak_to_strong_evidence_path else None)
    if weak_to_strong_evidence_path:
        paths.append(weak_to_strong_evidence_path)
    first_board_supply_evidence_path = (Path(first_board_supply_evidence_path)
                                        if first_board_supply_evidence_path else None)
    if first_board_supply_evidence_path:
        paths.append(first_board_supply_evidence_path)
    promotion_payoff_evidence_path = (Path(promotion_payoff_evidence_path)
                                      if promotion_payoff_evidence_path else None)
    if promotion_payoff_evidence_path:
        paths.append(promotion_payoff_evidence_path)
    repair_followthrough_evidence_path = (Path(repair_followthrough_evidence_path)
                                          if repair_followthrough_evidence_path else None)
    if repair_followthrough_evidence_path:
        paths.append(repair_followthrough_evidence_path)
    repair_layer_evidence_path = (Path(repair_layer_evidence_path)
                                  if repair_layer_evidence_path else None)
    if repair_layer_evidence_path:
        paths.append(repair_layer_evidence_path)
    high_to_low_ecology_evidence_path = (Path(high_to_low_ecology_evidence_path)
                                         if high_to_low_ecology_evidence_path else None)
    if high_to_low_ecology_evidence_path:
        paths.append(high_to_low_ecology_evidence_path)
    index_emotion_divergence_evidence_path = (
        Path(index_emotion_divergence_evidence_path)
        if index_emotion_divergence_evidence_path else None)
    if index_emotion_divergence_evidence_path:
        paths.append(index_emotion_divergence_evidence_path)
    previous_symbolic_name_evidence_path = (
        Path(previous_symbolic_name_evidence_path)
        if previous_symbolic_name_evidence_path else None)
    if previous_symbolic_name_evidence_path:
        paths.append(previous_symbolic_name_evidence_path)
    first_board_screening_evidence_path = (
        Path(first_board_screening_evidence_path)
        if first_board_screening_evidence_path else None)
    if first_board_screening_evidence_path:
        paths.append(first_board_screening_evidence_path)
    theme_screening_matrix_evidence_path = (
        Path(theme_screening_matrix_evidence_path)
        if theme_screening_matrix_evidence_path else None)
    if theme_screening_matrix_evidence_path:
        paths.append(theme_screening_matrix_evidence_path)
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable daily review")
    report = build(pd.read_csv(paths[0], dtype={"code": str}),
                   json.loads(paths[1].read_text(encoding="utf-8")),
                   pd.read_csv(paths[2]), json.loads(paths[3].read_text(encoding="utf-8")), asof,
                   pd.read_csv(announcement_path, dtype={"code": str}) if announcement_path else None,
                   pd.read_csv(theme_roles_path, dtype={"code": str}) if theme_roles_path else None,
                   pd.read_csv(competition_path, dtype={"code": str}) if competition_path else None,
                   pd.read_csv(previous_limit_path, dtype={"代码": str}) if previous_limit_path else None,
                   pd.read_csv(cross_theme_races_path, dtype={"code": str})
                   if cross_theme_races_path else None,
                   json.loads(abnormal_paths_path.read_text(encoding="utf-8"))
                   if abnormal_paths_path else None,
                   json.loads(height_anchor_evidence_path.read_text(encoding="utf-8"))
                   if height_anchor_evidence_path else None,
                   json.loads(opening_gap_evidence_path.read_text(encoding="utf-8"))
                   if opening_gap_evidence_path else None,
                   json.loads(opening_gap_emotion_evidence_path.read_text(encoding="utf-8"))
                   if opening_gap_emotion_evidence_path else None,
                   json.loads(weak_to_strong_evidence_path.read_text(encoding="utf-8"))
                   if weak_to_strong_evidence_path else None,
                   json.loads(first_board_supply_evidence_path.read_text(encoding="utf-8"))
                   if first_board_supply_evidence_path else None,
                   json.loads(promotion_payoff_evidence_path.read_text(encoding="utf-8"))
                   if promotion_payoff_evidence_path else None,
                   json.loads(repair_followthrough_evidence_path.read_text(encoding="utf-8"))
                   if repair_followthrough_evidence_path else None,
                   json.loads(repair_layer_evidence_path.read_text(encoding="utf-8"))
                   if repair_layer_evidence_path else None,
                   json.loads(high_to_low_ecology_evidence_path.read_text(encoding="utf-8"))
                   if high_to_low_ecology_evidence_path else None,
                   json.loads(index_emotion_divergence_evidence_path.read_text(
                       encoding="utf-8"))
                   if index_emotion_divergence_evidence_path else None,
                   json.loads(previous_symbolic_name_evidence_path.read_text(
                       encoding="utf-8"))
                   if previous_symbolic_name_evidence_path else None)
    attach_decision_plan(report, json.loads(decision_plan_path.read_text(encoding="utf-8"))
                         if decision_plan_path else None)
    report["first_board_screening_summary"] = (
        json.loads(first_board_screening_evidence_path.read_text(encoding="utf-8"))
        if first_board_screening_evidence_path else {
            "status": "UNAVAILABLE", "challenger_promoted": False,
            "current_primary_close_quality_watch": None,
        })
    report["theme_screening_matrix_summary"] = (
        json.loads(theme_screening_matrix_evidence_path.read_text(encoding="utf-8"))
        if theme_screening_matrix_evidence_path else {
            "status": "UNAVAILABLE", "real_theme_evidence_available": False,
        })
    import forecast_review_bridge_v277 as expectation_bridge
    official_text = ""
    official_inputs = (manual_official_update_path, official_archive_path, official_registered_path)
    if any(official_inputs):
        if not all(official_inputs):
            raise ValueError("official appendix requires update, archive and registered forecast together")
        import official_review_bridge_v284 as official_bridge
        update, frozen = official_bridge.authenticate(*official_inputs)
        official_bridge.attach(report, update, frozen)
        official_text = official_bridge.markdown(report[official_bridge.KEY])
        paths.extend(map(Path, (manual_official_update_path, official_registered_path,
                               Path(official_archive_path) / "report.json",
                               official_bridge.__file__)))
    report["source_hashes"] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in paths + [Path(__file__),
                                                    Path(symbolic_name_theme.__file__),
                                                    Path(expectation_bridge.__file__)]}
    expectation_bridge.attach(report)
    target.mkdir(parents=True, exist_ok=False)
    with (target / "review.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    (target / "review.md").write_text(markdown(report) + official_text + expectation_bridge.markdown(report["next_session_expectation"]), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--snapshot-manifest", required=True)
    parser.add_argument("--market", required=True)
    parser.add_argument("--selective-policy", required=True)
    parser.add_argument("--announcements")
    parser.add_argument("--theme-roles", help="optional point-in-time candidate_theme_roles.csv")
    parser.add_argument("--competition", help="optional verified point-in-time competition snapshot CSV")
    parser.add_argument("--previous-limit", help="optional archived previous-limit pool for exact mean/median/promotion")
    parser.add_argument("--decision-plan", help="versioned closing plan with evidence-linked next-session outlook")
    parser.add_argument("--cross-theme-races",
                        help="optional V170 confidence-aware first-board/relay race CSV")
    parser.add_argument("--abnormal-paths",
                        help="optional V171 fail-closed severe-abnormal path role JSON")
    parser.add_argument("--height-anchor-evidence",
                        help="optional V181 corrected-label height-anchor buyability JSON")
    parser.add_argument("--opening-gap-evidence",
                        help="optional V182 opening-gap payoff tradeoff JSON")
    parser.add_argument("--opening-gap-emotion-evidence",
                        help="optional V183 opening-gap by emotion interaction JSON")
    parser.add_argument("--weak-to-strong-evidence",
                        help="optional V184 low-open hindsight-denominator audit JSON")
    parser.add_argument("--first-board-supply-evidence",
                        help="optional V185 first-board supply base-rate audit JSON")
    parser.add_argument("--promotion-payoff-evidence",
                        help="optional V186 promotion versus B+2/B+3 payoff audit JSON")
    parser.add_argument("--repair-followthrough-evidence",
                        help="optional V187 repair-day next-session follow-through audit JSON")
    parser.add_argument("--repair-layer-evidence",
                        help="optional V188 low/middle/height repair transmission audit JSON")
    parser.add_argument("--high-to-low-ecology-evidence",
                        help="optional V189 aggregate high-to-low ecology audit JSON")
    parser.add_argument("--index-emotion-divergence-evidence",
                        help="optional V203 broad-index versus relay-ecology audit JSON")
    parser.add_argument("--previous-symbolic-name-evidence",
                        help="optional prior-session V217 report for expansion comparison")
    parser.add_argument("--first-board-screening-evidence",
                        help="optional V218 first-board challenger report JSON")
    parser.add_argument("--theme-screening-matrix-evidence",
                        help="optional V219 evidence-tiered theme matrix report JSON")
    parser.add_argument("--manual-official-update", help="V283 post-freeze manual update JSON")
    parser.add_argument("--official-archive", help="matching hash-bound official disclosure archive directory")
    parser.add_argument("--official-registered", help="original immutable forecast registry JSON")
    parser.add_argument("--asof", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.snapshot, args.snapshot_manifest, args.market, args.selective_policy,
                         args.output, args.asof, args.announcements, args.theme_roles,
                         args.competition, args.previous_limit, args.decision_plan,
                         args.cross_theme_races, args.abnormal_paths,
                         args.height_anchor_evidence, args.opening_gap_evidence,
                         args.opening_gap_emotion_evidence,
                         args.weak_to_strong_evidence,
                         args.first_board_supply_evidence,
                         args.promotion_payoff_evidence,
                         args.repair_followthrough_evidence,
                         args.repair_layer_evidence,
                         args.high_to_low_ecology_evidence,
                         args.index_emotion_divergence_evidence,
                         args.previous_symbolic_name_evidence,
                         args.first_board_screening_evidence,
                         args.theme_screening_matrix_evidence,
                         args.manual_official_update, args.official_archive,
                         args.official_registered),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
