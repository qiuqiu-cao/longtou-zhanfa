import json
import unittest

import pandas as pd

from abnormal_path_role_v171 import build as build_abnormal_paths
from current_daily_review import (
    _current_market_core, _explicit_high_attention_exclusions,
    _validated_group_primary, build, markdown,
)


class CurrentDailyReviewTests(unittest.TestCase):
    def height_anchor_evidence(self):
        return {
            "version": "v181-height-anchor-buyability-1",
            "status": "DIAGNOSTIC_ONLY",
            "target_contract": {
                "no_touch_rows_retained_as_negative": True,
                "next_session_one_price_rows_retained_as_negative": True,
                "actual_fills": 0,
                "complete_trades": 0,
            },
            "split": {"validation_sessions": 150, "validation_previously_inspected": True},
            "models": {"boards_only": {}, "identity_emotion_price_volume": {}},
            "daily_top1_within_height_anchors": {
                "baseline_success_rate": .18,
                "challenger_success_rate": .18,
            },
            "height_anchor_buyability_uplift_supported": False,
            "governance": {
                "height_anchor_can_auto_fill_relay_primary": False,
                "height_anchor_can_auto_fill_current_buy": False,
                "candidate_probability_claim_allowed": False,
                "direct_stock_reranking_allowed": False,
                "entry_authorized": False,
            },
            "interpretation": "height identity is not buyability",
        }

    def opening_gap_evidence(self):
        return {
            "version": "v182-opening-gap-payoff-tradeoff-1",
            "status": "DIAGNOSTIC_ONLY",
            "feature_contract": {
                "prior_close_model_use_allowed": False,
                "auction_09_25_equivalence_claim_allowed": False,
                "entry_open_gap_available_not_before": (
                    "after_opening_call_auction_match_approximately_09:25"),
                "continuous_auction_begins_at": "09:30",
            },
            "target_contract": {
                "no_touch_rows_retained_as_negative": True,
                "next_session_one_price_rows_retained_as_negative": True,
                "actual_fills": 0, "complete_trades": 0,
            },
            "split": {"validation_sessions": 151},
            "summaries": {"ALL": {"validation": {
                "GT_5_LE_7": {
                    "one_price_unbuyable_rate": 0.0,
                    "conditional_b1_big_loss_rate": .3417,
                    "combined_buyable_b2_positive_rate": .2446,
                },
                "GT_7": {
                    "one_price_unbuyable_rate": .3191,
                    "conditional_b1_big_loss_rate": .3947,
                    "combined_buyable_b2_positive_rate": .2267,
                },
            }}},
            "directional_replication": {
                "five_to_seven_vs_above_seven_both_ci_confirmed": False,
            },
            "governance": {
                "opening_gap_direct_entry_allowed": False,
                "opening_gap_direct_stock_reranking_allowed": False,
                "above_seven_default_risk_flag_allowed": True,
                "zero_to_five_automatic_buy_allowed": False,
                "five_to_seven_automatic_buy_allowed": False,
                "requires_intraday_absorption_reseal_and_theme_confirmation": True,
            },
            "interpretation": "opening strength is a tradeoff",
        }

    def opening_gap_emotion_evidence(self):
        return {
            "version": "v183-opening-gap-emotion-interaction-1",
            "status": "DIAGNOSTIC_ONLY",
            "input_contract": {
                "v182_corrected_observations_required": True,
                "official_open_available_only_after_opening_call_match": True,
                "continuous_auction_begins_at_09_30": True,
                "actual_fills": 0, "complete_trades": 0,
            },
            "scope": {"regime_sessions": {
                "SUPPORTIVE": 159, "HOT": 110, "ADVERSE": 234}},
            "split": {"validation_sessions": 151},
            "temporal_consistency": {},
            "difference_in_differences": {},
            "opening_gap_emotion_interaction_supported": False,
            "governance": {
                "emotion_can_override_above_seven_risk_flag": False,
                "opening_gap_emotion_interaction_factor_allowed": False,
                "supportive_regime_automatic_high_open_buy_allowed": False,
                "adverse_regime_automatic_high_open_cancel_allowed": False,
                "direct_stock_reranking_allowed": False,
                "entry_authorized": False,
            },
            "interpretation": "regime interaction is unstable",
        }

    def weak_to_strong_evidence(self):
        return {
            "version": "v184-weak-to-strong-hindsight-audit-1",
            "status": "DIAGNOSTIC_ONLY",
            "source_contract": {
                "v182_version": "v182-opening-gap-payoff-tradeoff-1",
                "no_touch_rows_retained_as_negative": True,
                "next_session_one_price_rows_retained_as_negative": True,
                "actual_fills": 0, "complete_trades": 0,
            },
            "split": {"validation_sessions": 151},
            "denominator_contract": {
                "later_touch_available_at_09_30": False,
                "conditional_result_may_not_be_presented_as_opening_win_rate": True,
            },
            "hindsight_inflation": {"ALL": {"validation_low_open": {
                "full_opening_denominator_success_rate": .036,
                "later_touch_conditioned_b2_positive_rate": .381,
                "conditioning_inflation_multiple": 10.57,
            }}},
            "low_open_conditional_payoff_edge_supported": False,
            "governance": {
                "later_touch_may_be_used_as_09_30_feature": False,
                "low_open_automatic_weak_to_strong_buy_allowed": False,
                "low_open_automatic_reject_allowed": False,
                "requires_live_reclaim_vwap_volume_and_theme_confirmation": True,
                "direct_stock_reranking_allowed": False,
                "entry_authorized": False,
            },
            "interpretation": "later touch is future information",
        }

    def first_board_supply_evidence(self):
        return {
            "version": "v185-first-board-supply-base-rate-audit-1",
            "status": "DIAGNOSTIC_ONLY",
            "target_contract": {
                "source_no_touch_rows_retained_as_negative": True,
                "missing_promotion_labels_filled_as_zero": False,
                "actual_fills": 0, "complete_trades": 0,
            },
            "split": {"validation_sessions": 151},
            "frozen_supply_thresholds_from_development": {
                "low_supply_max": 22.0, "high_supply_strictly_above": 49.0,
            },
            "summaries": {"validation": {
                "ALL": {"daily_any_promotion_rate": .993},
                "LOW_SUPPLY": {"candidate_promotion_rate": .255},
                "HIGH_SUPPLY": {"candidate_promotion_rate": .188},
            }},
            "date_clustered_high_minus_low": {},
            "daily_any_promotion_is_discriminative": False,
            "high_first_board_supply_positive_edge_supported": False,
            "governance": {
                "any_first_board_promoted_may_rank_a_candidate": False,
                "first_board_supply_positive_ranking_factor_allowed": False,
                "high_supply_automatic_emotion_upgrade_allowed": False,
                "high_supply_automatic_reject_allowed": False,
                "requires_candidate_theme_identity_and_execution_confirmation": True,
                "direct_stock_reranking_allowed": False,
                "entry_authorized": False,
            },
            "interpretation": "breadth is not candidate quality",
        }

    def promotion_payoff_evidence(self):
        return {
            "version": "v186-promotion-payoff-separation-audit-1",
            "status": "DIAGNOSTIC_ONLY",
            "target_contract": {
                "no_touch_rows_retained_as_negative": True,
                "one_price_rows_retained_as_negative": True,
                "touchable_unknown_b2_excluded_from_b2_denominator": True,
                "unknown_touch_rows_excluded_not_filled": True,
                "missing_promotion_labels_filled_as_zero": False,
                "actual_fills": 0, "complete_trades": 0,
            },
            "split": {"validation_sessions": 151},
            "summaries": {
                "ALL": {"validation": {
                    "promotion_rate": .238,
                    "full_denominator_buyable_b2_positive_rate": .102,
                    "promoted_without_buyable_b2_positive_rate": .629,
                    "promoted_and_buyable_b2_b3_both_positive_rate": .314,
                }},
                "CONSECUTIVE_BOARD": {"validation": {
                    "promoted_without_buyable_b2_positive_rate": .625,
                }},
            },
            "date_clustered_identity_minus_payoff": {},
            "promotion_is_buyable_b2_payoff_equivalent": False,
            "promotion_is_buyable_b2_b3_sustained_equivalent": False,
            "governance": {
                "promotion_probability_may_be_presented_as_trade_win_probability": False,
                "promotion_probability_may_replace_b2_b3_payoff_target": False,
                "board_level_promotion_rate_may_directly_rank_entries": False,
                "requires_separate_buyability_payoff_and_big_loss_models": True,
                "direct_stock_reranking_allowed": False,
                "entry_authorized": False,
            },
            "interpretation": "identity is not payoff",
        }

    def repair_followthrough_evidence(self):
        return {
            "version": "v187-repair-day-followthrough-audit-1",
            "status": "DIAGNOSTIC_ONLY",
            "data_contract": {
                "signal_features_available_at_repair_day_close": True,
                "next_session_fields_used_only_as_labels": True,
                "sparse_signal_days_excluded": True,
                "missing_quality_fields_filled_as_zero": False,
                "actual_stock_entries": 0, "complete_stock_trades": 0,
            },
            "repair_definition": {},
            "scope": {"repair_sessions": 131},
            "split": {"validation_sessions": 40},
            "summaries": {
                "development": {
                    "BROAD_REPAIR": {"relapse_rate": .431},
                    "NARROW_REPAIR": {"relapse_rate": .500},
                },
                "validation": {
                    "BROAD_REPAIR": {"relapse_rate": .250},
                    "NARROW_REPAIR": {"relapse_rate": .500},
                },
            },
            "session_clustered_broad_minus_narrow": {
                "development": {"metrics": {"relapse_rate": {
                    "estimate": -.069, "ci95": [-.272, .126]}}},
                "validation": {"metrics": {"relapse_rate": {
                    "estimate": -.250, "ci95": [-.521, .042]}}},
            },
            "broad_repair_lower_relapse_stable_across_time": False,
            "latest_repair_context": {
                "signal_date": "2026-09-16", "quality_score": 3,
                "quality_class": "BROAD_REPAIR",
            },
            "governance": {
                "historical_repair_scenario_context_allowed": True,
                "calibrated_next_relapse_probability_allowed": False,
                "repair_score_direct_emotion_gate_allowed": False,
                "repair_score_direct_stock_reranking_allowed": False,
                "repair_score_entry_authorized": False,
                "requires_next_session_auction_middle_board_and_loss_confirmation": True,
            },
            "interpretation": "repair quality requires next-session confirmation",
        }

    def repair_layer_evidence(self):
        return {
            "version": "v188-repair-layer-transmission-audit-1",
            "status": "DIAGNOSTIC_ONLY",
            "data_contract": {
                "same_day_layers_available_at_repair_close": True,
                "thresholds_frozen_from_development_only": True,
                "next_session_fields_used_only_as_labels": True,
                "missing_layer_fields_filled_as_zero": False,
                "actual_stock_entries": 0, "complete_stock_trades": 0,
            },
            "layer_definition": {},
            "frozen_development_thresholds": {},
            "scope": {"repair_sessions": 131},
            "split": {"validation_sessions": 40},
            "summaries": {
                "development": {
                    "SYNCHRONIZED_3": {"sessions": 24,
                                        "next_relapse_rate": .292},
                    "PARTIAL_0_TO_2": {"sessions": 67,
                                        "next_relapse_rate": .522},
                    "MIDDLE_CONFIRMED": {"sessions": 46,
                                         "next_relapse_rate": .348},
                    "MIDDLE_UNCONFIRMED": {"sessions": 45,
                                           "next_relapse_rate": .578},
                },
                "validation": {
                    "SYNCHRONIZED_3": {"sessions": 3,
                                        "next_relapse_rate": .333},
                    "PARTIAL_0_TO_2": {"sessions": 37,
                                        "next_relapse_rate": .405},
                    "MIDDLE_CONFIRMED": {"sessions": 12,
                                         "next_relapse_rate": .417},
                    "MIDDLE_UNCONFIRMED": {"sessions": 28,
                                           "next_relapse_rate": .393},
                },
            },
            "session_clustered_comparisons": {},
            "synchronized_repair_lower_relapse_stable_across_time": False,
            "middle_confirmation_lower_relapse_stable_across_time": False,
            "latest_repair_layer_context": {
                "signal_date": "2026-09-16",
                "first_board_layer_confirmed": True,
                "middle_board_layer_confirmed": False,
                "height_layer_confirmed": True,
                "repair_layer_score": 2,
                "repair_layer_class": "PARTIAL_0_TO_2",
            },
            "governance": {
                "historical_repair_layer_scenario_context_allowed": True,
                "calibrated_layer_followthrough_probability_allowed": False,
                "repair_layer_score_direct_emotion_gate_allowed": False,
                "middle_board_confirmation_positive_ranking_factor_allowed": False,
                "repair_layer_direct_stock_reranking_allowed": False,
                "repair_layer_position_size_increase_allowed": False,
                "repair_layer_entry_authorized": False,
                "requires_next_session_auction_middle_board_loss_and_theme_confirmation": True,
            },
            "interpretation": "repair layers are descriptive only",
        }

    def high_to_low_ecology_evidence(self):
        return {
            "version": "v189-high-to-low-ecology-audit-1",
            "status": "DIAGNOSTIC_ONLY",
            "data_contract": {
                "same_day_window_features_available_at_close": True,
                "low_strength_threshold_frozen_from_development_only": True,
                "next_session_fields_used_only_as_labels": True,
                "specific_theme_alignment_available": False,
                "stock_level_supplement_identity_available": False,
                "actual_stock_entries": 0, "complete_stock_trades": 0,
            },
            "window_definition": {},
            "frozen_development_thresholds": {},
            "scope": {"high_to_low_windows": 106},
            "split": {"validation_sessions": 203},
            "summaries": {
                "development": {
                    "HEALTHY_ROTATION_WINDOW": {
                        "sessions": 31, "next_adverse_state_rate": .516},
                    "COLLAPSE_MASKED_LOW_STRENGTH": {
                        "sessions": 47, "next_adverse_state_rate": .489},
                },
                "validation": {
                    "HEALTHY_ROTATION_WINDOW": {
                        "sessions": 12, "next_adverse_state_rate": .333},
                    "COLLAPSE_MASKED_LOW_STRENGTH": {
                        "sessions": 16, "next_adverse_state_rate": .688},
                },
            },
            "session_clustered_healthy_minus_collapse_masked": {},
            "healthy_high_to_low_lower_adverse_stable_across_time": False,
            "latest_market_context": {
                "signal_date": "2026-09-16",
                "height_risk_increased": False,
                "low_layer_strong": True,
                "ecology_score": 3,
                "high_to_low_window_active": False,
                "ecology_class": "NOT_ACTIVE",
            },
            "governance": {
                "historical_high_to_low_market_scenario_allowed": True,
                "calibrated_high_to_low_window_probability_allowed": False,
                "high_to_low_window_direct_emotion_gate_allowed": False,
                "high_to_low_window_direct_stock_reranking_allowed": False,
                "high_to_low_stock_selection_allowed": False,
                "same_theme_supplement_mapping_allowed": False,
                "position_size_increase_allowed": False,
                "entry_authorized": False,
                "requires_specific_theme_candidate_auction_and_open_confirmation": True,
            },
            "interpretation": "market window does not identify a supplement stock",
        }

    def snapshot(self):
        return pd.DataFrame([
            {"signal_date": "2026-09-04", "code": "000001", "name": "FIRST", "boards": 1,
             "board_group": "FIRST_BOARD", "pool_assignment": "CLOSING_LIMIT_UP", "is_mainboard": True,
             "industry_provider": "A", "turnover": 5, "amount": 100, "first_seal_time": 93000,
             "failed_seal_count": 0, "security_status": "verified", "eligibility_status": "VERIFIED_ELIGIBLE"},
            {"signal_date": "2026-09-04", "code": "600001", "name": "HIGH", "boards": 3,
             "board_group": "CONSECUTIVE_BOARD", "pool_assignment": "CLOSING_LIMIT_UP", "is_mainboard": True,
             "industry_provider": "B", "turnover": 8, "amount": 200, "first_seal_time": 94000,
             "failed_seal_count": 1, "security_status": "unavailable",
             "eligibility_status": "SECURITY_STATUS_UNAVAILABLE_OR_INELIGIBLE"},
            {"signal_date": "2026-09-04", "code": "000002", "name": "FAILED", "boards": 0,
             "board_group": "FAILED_BOARD", "pool_assignment": "FAILED_BOARD_OBSERVATION", "is_mainboard": True,
             "industry_provider": "A", "turnover": 9, "amount": 80, "first_seal_time": 95000,
             "failed_seal_count": 2, "change_pct": 0.0,
             "security_status": "verified", "eligibility_status": "VERIFIED_ELIGIBLE"},
        ])

    def manifest(self):
        return {"signal_date": "2026-09-04", "mainboard_closing_limit_rows": 2,
                "mainboard_failed_board_rows": 1, "mainboard_board_level_counts": {"1": 1, "3": 1},
                "verified_eligible_closing_rows": 1,
                "security_unavailable_or_ineligible_closing_rows": 1,
                "full_market_completeness": "not_independently_verified"}

    def market(self):
        return pd.DataFrame([{"date": "2026-09-04", "data_status": "verified", "state_retreat": 1,
                              "state_start": 0, "state_warming": 0, "state_acceleration": 0,
                              "state_consensus": 0, "state_divergence": 0, "state_repair": 0,
                              "yesterday_premium": -.02, "promotion_overall": 0,
                              "previous_limit_tail_loss": -.08, "failed_tail_loss": -.05,
                              "failed_rate": .5, "seal_rate": .5, "space_feedback": 0,
                              "max_board": 3, "ladder_gap_count": 2, "down_limit_count": 8,
                              "limit_count": 2, "consecutive_count": 1, "failed_count": 1,
                              "emotion_score": .2, "emotion_adverse_next_probability": .5,
                              "emotion_forecast_status": "USABLE"}])

    def test_complete_groups_and_cash(self):
        result = build(self.snapshot(), self.manifest(), self.market(),
                       {"operational_threshold": None}, "2026-09-05T20:00:00+08:00")
        self.assertEqual(result["status"], "CASH")
        self.assertEqual(len(result["first_board_candidates"]), 1)
        self.assertEqual(len(result["relay_candidates"]), 1)
        self.assertEqual(result["selection"]["position_shares"], 0)
        self.assertIsNone(result["roles"]["current_market_core"])
        self.assertIsNone(result["roles"]["first_board_primary_candidate"])
        self.assertIsNone(result["roles"]["relay_primary_candidate"])
        self.assertIsNone(result["roles"]["comprehensive_primary_candidate"])
        self.assertEqual(result["roles"]["comprehensive_primary_status"],
                         "CASH_NO_VALIDATED_GROUP_PRIMARY")
        self.assertIn("symbolic_name_summary", result)
        self.assertEqual(result["symbolic_name_summary"]["status"], "DIAGNOSTIC_ONLY")
        self.assertFalse(result["symbolic_name_summary"]["governance"]["direct_entry_allowed"])
        self.assertIn("名称/符号共振", markdown(result))

    def test_abnormal_path_roles_are_bound_but_never_authorize_entry(self):
        abnormal = build_abnormal_paths(
            self.snapshot(), None, "2026-09-05T19:00:00+08:00")
        result = build(
            self.snapshot(), self.manifest(), self.market(),
            {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
            abnormal_paths=abnormal,
        )
        self.assertEqual(result["abnormal_path_summary"]["verified_distance_count"], 0)
        self.assertIsNone(result["roles"]["hundred_percent_abnormal_path_potential"])
        self.assertIsNone(result["roles"]["abnormal_rule_distance_closest"])
        self.assertIn("SEVERE_ABNORMAL_RULE_DISTANCE_INCOMPLETE",
                      result["blocking_reasons"])
        self.assertIn("接近阈值只是算术位置", markdown(result))

    def test_abnormal_path_future_asof_is_rejected(self):
        abnormal = build_abnormal_paths(
            self.snapshot(), None, "2026-09-05T20:01:00+08:00")
        with self.assertRaisesRegex(ValueError, "after asof"):
            build(
                self.snapshot(), self.manifest(), self.market(),
                {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
                abnormal_paths=abnormal,
            )

    def test_corrected_height_anchor_evidence_is_bound_without_promoting_a_buy(self):
        result = build(
            self.snapshot(), self.manifest(), self.market(),
            {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
            height_anchor_evidence=self.height_anchor_evidence(),
        )
        evidence = result["roles"]["height_anchor_buyability_evidence"]
        self.assertEqual(evidence["status"], "BOUND_DIAGNOSTIC_ONLY")
        self.assertEqual(evidence["validation_sessions"], 150)
        self.assertFalse(evidence["challenger_uplift_supported"])
        self.assertFalse(evidence["candidate_probability_claim_allowed"])
        self.assertFalse(evidence["entry_authorized"])
        self.assertIsNone(result["roles"]["relay_primary_candidate"])
        self.assertIn("未证明增益", markdown(result))

    def test_height_anchor_evidence_that_authorizes_entry_is_rejected(self):
        evidence = self.height_anchor_evidence()
        evidence["governance"]["entry_authorized"] = True
        with self.assertRaisesRegex(ValueError, "cannot authorize"):
            build(
                self.snapshot(), self.manifest(), self.market(),
                {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
                height_anchor_evidence=evidence,
            )

    def test_height_anchor_evidence_must_keep_no_touch_rows_negative(self):
        evidence = self.height_anchor_evidence()
        evidence["target_contract"]["no_touch_rows_retained_as_negative"] = False
        with self.assertRaisesRegex(ValueError, "target contract"):
            build(
                self.snapshot(), self.manifest(), self.market(),
                {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
                height_anchor_evidence=evidence,
            )

    def test_opening_gap_evidence_is_bound_only_as_opening_risk_context(self):
        result = build(
            self.snapshot(), self.manifest(), self.market(),
            {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
            opening_gap_evidence=self.opening_gap_evidence(),
        )
        evidence = result["roles"]["opening_gap_payoff_evidence"]
        self.assertEqual(evidence["status"], "BOUND_DIAGNOSTIC_ONLY")
        self.assertFalse(evidence["prior_close_model_use_allowed"])
        self.assertFalse(evidence["opening_gap_direct_entry_allowed"])
        self.assertTrue(evidence["above_seven_default_risk_flag_allowed"])
        self.assertIn("收益优势区间未稳定通过", markdown(result))

    def test_opening_gap_evidence_cannot_leak_into_prior_close(self):
        evidence = self.opening_gap_evidence()
        evidence["feature_contract"]["prior_close_model_use_allowed"] = True
        with self.assertRaisesRegex(ValueError, "leaks"):
            build(
                self.snapshot(), self.manifest(), self.market(),
                {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
                opening_gap_evidence=evidence,
            )

    def test_opening_gap_evidence_cannot_authorize_entry(self):
        evidence = self.opening_gap_evidence()
        evidence["governance"]["opening_gap_direct_entry_allowed"] = True
        with self.assertRaisesRegex(ValueError, "cannot authorize"):
            build(
                self.snapshot(), self.manifest(), self.market(),
                {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
                opening_gap_evidence=evidence,
            )

    def test_opening_gap_emotion_evidence_is_bound_without_override(self):
        result = build(
            self.snapshot(), self.manifest(), self.market(),
            {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
            opening_gap_emotion_evidence=self.opening_gap_emotion_evidence(),
        )
        evidence = result["roles"]["opening_gap_emotion_evidence"]
        self.assertEqual(evidence["status"], "BOUND_DIAGNOSTIC_ONLY")
        self.assertFalse(evidence["interaction_supported"])
        self.assertFalse(evidence["emotion_can_override_above_seven_risk_flag"])
        self.assertFalse(evidence["entry_authorized"])
        self.assertIn("环境标签既不能豁免", markdown(result))

    def test_opening_gap_emotion_evidence_cannot_authorize_override(self):
        evidence = self.opening_gap_emotion_evidence()
        evidence["governance"]["emotion_can_override_above_seven_risk_flag"] = True
        with self.assertRaisesRegex(ValueError, "cannot authorize"):
            build(
                self.snapshot(), self.manifest(), self.market(),
                {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
                opening_gap_emotion_evidence=evidence,
            )

    def test_opening_gap_emotion_evidence_must_not_claim_supported_interaction(self):
        evidence = self.opening_gap_emotion_evidence()
        evidence["opening_gap_emotion_interaction_supported"] = True
        with self.assertRaisesRegex(ValueError, "cannot be promoted"):
            build(
                self.snapshot(), self.manifest(), self.market(),
                {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
                opening_gap_emotion_evidence=evidence,
            )

    def test_weak_to_strong_evidence_is_bound_without_future_leakage(self):
        result = build(
            self.snapshot(), self.manifest(), self.market(),
            {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
            weak_to_strong_evidence=self.weak_to_strong_evidence(),
        )
        evidence = result["roles"]["weak_to_strong_hindsight_evidence"]
        self.assertEqual(evidence["status"], "BOUND_DIAGNOSTIC_ONLY")
        self.assertAlmostEqual(evidence["conditioning_inflation_multiple"], 10.57)
        self.assertFalse(evidence["later_touch_may_be_used_as_09_30_feature"])
        self.assertFalse(evidence["entry_authorized"])
        self.assertIn("表面胜率被放大约10.6倍", markdown(result))

    def test_weak_to_strong_evidence_cannot_use_later_touch_at_open(self):
        evidence = self.weak_to_strong_evidence()
        evidence["governance"]["later_touch_may_be_used_as_09_30_feature"] = True
        with self.assertRaisesRegex(ValueError, "cannot authorize"):
            build(
                self.snapshot(), self.manifest(), self.market(),
                {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
                weak_to_strong_evidence=evidence,
            )

    def test_weak_to_strong_evidence_cannot_authorize_entry(self):
        evidence = self.weak_to_strong_evidence()
        evidence["governance"]["entry_authorized"] = True
        with self.assertRaisesRegex(ValueError, "cannot authorize"):
            build(
                self.snapshot(), self.manifest(), self.market(),
                {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
                weak_to_strong_evidence=evidence,
            )

    def test_first_board_supply_is_bound_as_base_rate_warning(self):
        result = build(
            self.snapshot(), self.manifest(), self.market(),
            {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
            first_board_supply_evidence=self.first_board_supply_evidence(),
        )
        evidence = result["roles"]["first_board_supply_evidence"]
        self.assertEqual(evidence["status"], "BOUND_DIAGNOSTIC_ONLY")
        self.assertFalse(evidence[
            "first_board_supply_positive_ranking_factor_allowed"])
        self.assertFalse(evidence["entry_authorized"])
        self.assertIn("几乎恒真", markdown(result))
        self.assertIn("首板多只说明供给和广度", markdown(result))

    def test_first_board_supply_cannot_authorize_emotion_upgrade(self):
        evidence = self.first_board_supply_evidence()
        evidence["governance"]["high_supply_automatic_emotion_upgrade_allowed"] = True
        with self.assertRaisesRegex(ValueError, "cannot authorize"):
            build(
                self.snapshot(), self.manifest(), self.market(),
                {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
                first_board_supply_evidence=evidence,
            )

    def test_promotion_payoff_evidence_keeps_identity_and_trade_target_separate(self):
        result = build(
            self.snapshot(), self.manifest(), self.market(),
            {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
            promotion_payoff_evidence=self.promotion_payoff_evidence(),
        )
        evidence = result["roles"]["promotion_payoff_evidence"]
        self.assertEqual(evidence["status"], "BOUND_DIAGNOSTIC_ONLY")
        self.assertFalse(evidence[
            "promotion_probability_may_be_presented_as_trade_win_probability"])
        self.assertFalse(evidence["entry_authorized"])
        rendered = markdown(result)
        self.assertIn("晋级身份与买后赔率分离", rendered)
        self.assertIn("晋级率只描述身份延续", rendered)

    def test_promotion_payoff_evidence_cannot_replace_payoff_target(self):
        evidence = self.promotion_payoff_evidence()
        evidence["governance"][
            "promotion_probability_may_replace_b2_b3_payoff_target"] = True
        with self.assertRaisesRegex(ValueError, "cannot authorize"):
            build(
                self.snapshot(), self.manifest(), self.market(),
                {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
                promotion_payoff_evidence=evidence,
            )

    def test_repair_followthrough_is_bound_as_scenario_only(self):
        result = build(
            self.snapshot(), self.manifest(), self.market(),
            {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
            repair_followthrough_evidence=self.repair_followthrough_evidence(),
        )
        evidence = result["roles"]["repair_followthrough_evidence"]
        self.assertEqual(evidence["status"], "BOUND_DIAGNOSTIC_ONLY")
        self.assertEqual(evidence["latest_quality_score"], 3)
        self.assertFalse(evidence["lower_relapse_stable_across_time"])
        self.assertFalse(evidence["calibrated_probability_allowed"])
        self.assertFalse(evidence["direct_emotion_gate_allowed"])
        self.assertFalse(evidence["entry_authorized"])
        rendered = markdown(result)
        self.assertIn("修复日次日延续校验", rendered)
        self.assertIn("不给校准概率、仓位或买点", rendered)

    def test_repair_followthrough_cannot_authorize_emotion_gate_or_entry(self):
        for key in ("repair_score_direct_emotion_gate_allowed",
                    "repair_score_entry_authorized"):
            evidence = self.repair_followthrough_evidence()
            evidence["governance"][key] = True
            with self.assertRaisesRegex(ValueError, "cannot authorize"):
                build(
                    self.snapshot(), self.manifest(), self.market(),
                    {"operational_threshold": None},
                    "2026-09-05T20:00:00+08:00",
                    repair_followthrough_evidence=evidence,
                )

    def test_repair_followthrough_stability_claim_is_rejected(self):
        evidence = self.repair_followthrough_evidence()
        evidence["broad_repair_lower_relapse_stable_across_time"] = True
        with self.assertRaisesRegex(ValueError, "cannot be promoted"):
            build(
                self.snapshot(), self.manifest(), self.market(),
                {"operational_threshold": None},
                "2026-09-05T20:00:00+08:00",
                repair_followthrough_evidence=evidence,
            )

    def test_repair_layer_evidence_is_bound_without_promoting_middle_board(self):
        result = build(
            self.snapshot(), self.manifest(), self.market(),
            {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
            repair_layer_evidence=self.repair_layer_evidence(),
        )
        evidence = result["roles"]["repair_layer_evidence"]
        self.assertEqual(evidence["status"], "BOUND_DIAGNOSTIC_ONLY")
        self.assertEqual(evidence["latest_layer_score"], 2)
        self.assertFalse(evidence["latest_middle_board_confirmed"])
        self.assertFalse(evidence["middle_confirmation_stable"])
        self.assertFalse(evidence["direct_emotion_gate_allowed"])
        self.assertFalse(evidence["position_size_increase_allowed"])
        self.assertFalse(evidence["entry_authorized"])
        rendered = markdown(result)
        self.assertIn("修复层级传导", rendered)
        self.assertIn("中位=未确认", rendered)

    def test_repair_layer_evidence_cannot_authorize_ranking_position_or_entry(self):
        for key in ("middle_board_confirmation_positive_ranking_factor_allowed",
                    "repair_layer_position_size_increase_allowed",
                    "repair_layer_entry_authorized"):
            evidence = self.repair_layer_evidence()
            evidence["governance"][key] = True
            with self.assertRaisesRegex(ValueError, "cannot authorize"):
                build(
                    self.snapshot(), self.manifest(), self.market(),
                    {"operational_threshold": None},
                    "2026-09-05T20:00:00+08:00",
                    repair_layer_evidence=evidence,
                )

    def test_repair_layer_stability_claim_is_rejected(self):
        evidence = self.repair_layer_evidence()
        evidence["middle_confirmation_lower_relapse_stable_across_time"] = True
        with self.assertRaisesRegex(ValueError, "cannot be promoted"):
            build(
                self.snapshot(), self.manifest(), self.market(),
                {"operational_threshold": None},
                "2026-09-05T20:00:00+08:00",
                repair_layer_evidence=evidence,
            )

    def test_high_to_low_ecology_is_bound_without_selecting_a_stock(self):
        result = build(
            self.snapshot(), self.manifest(), self.market(),
            {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
            high_to_low_ecology_evidence=self.high_to_low_ecology_evidence(),
        )
        evidence = result["roles"]["high_to_low_ecology_evidence"]
        self.assertEqual(evidence["status"], "BOUND_DIAGNOSTIC_ONLY")
        self.assertFalse(evidence["window_active"])
        self.assertTrue(evidence["low_layer_strong"])
        self.assertFalse(evidence["stock_selection_allowed"])
        self.assertFalse(evidence["same_theme_mapping_allowed"])
        self.assertFalse(evidence["entry_authorized"])
        rendered = markdown(result)
        self.assertIn("高切低市场窗口", rendered)
        self.assertIn("窗口=未激活", rendered)

    def test_high_to_low_ecology_cannot_authorize_mapping_position_or_entry(self):
        for key in ("high_to_low_stock_selection_allowed",
                    "same_theme_supplement_mapping_allowed",
                    "position_size_increase_allowed", "entry_authorized"):
            evidence = self.high_to_low_ecology_evidence()
            evidence["governance"][key] = True
            with self.assertRaisesRegex(ValueError, "cannot authorize"):
                build(
                    self.snapshot(), self.manifest(), self.market(),
                    {"operational_threshold": None},
                    "2026-09-05T20:00:00+08:00",
                    high_to_low_ecology_evidence=evidence,
                )

    def test_high_to_low_stability_claim_is_rejected(self):
        evidence = self.high_to_low_ecology_evidence()
        evidence["healthy_high_to_low_lower_adverse_stable_across_time"] = True
        with self.assertRaisesRegex(ValueError, "cannot be promoted"):
            build(
                self.snapshot(), self.manifest(), self.market(),
                {"operational_threshold": None},
                "2026-09-05T20:00:00+08:00",
                high_to_low_ecology_evidence=evidence,
            )

    def test_five_emotion_dimensions_are_present(self):
        result = build(self.snapshot(), self.manifest(), self.market(),
                       {"operational_threshold": None}, "2026-09-05T20:00:00+08:00")
        self.assertEqual(set(result["emotion_dimensions"]), {
            "money_effect", "broken_loss_effect", "height_feedback", "theme_divergence", "repair_quality"})
        self.assertEqual(result["market"]["emotion_regime"], "退潮")

    def test_positive_median_is_not_described_as_negative(self):
        previous = pd.DataFrame([
            {"代码": "000001", "涨跌幅": 4.0, "昨日连板数": 1},
            {"代码": "600001", "涨跌幅": 2.0, "昨日连板数": 2},
        ])
        result = build(self.snapshot(), self.manifest(), self.market(),
                       {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
                       previous_limit=previous)
        block = result["emotion_dimensions"]["money_effect"]
        self.assertNotIn("negative median", block["impact"])
        self.assertIn(block["state"], {"MIXED", "SUPPORTIVE"})

    def test_previous_limit_mean_median_and_promotion_are_not_conflated(self):
        previous = pd.DataFrame([
            {"代码": "000001", "涨跌幅": 10.0, "昨日连板数": 1},
            {"代码": "600001", "涨跌幅": -8.0, "昨日连板数": 2},
            {"代码": "600002", "涨跌幅": -2.0, "昨日连板数": 1},
        ])
        result = build(self.snapshot(), self.manifest(), self.market(),
                       {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
                       previous_limit=previous)
        evidence = result["emotion_dimensions"]["money_effect"]["evidence"]
        self.assertAlmostEqual(evidence["previous_limit_mean_return"], 0.0)
        self.assertAlmostEqual(evidence["previous_limit_median_return"], -0.02)
        self.assertAlmostEqual(evidence["all_board_promotion_rate"], 1 / 3)
        rendered = markdown(result)
        self.assertIn("平均反馈0.00%", rendered)
        self.assertIn("中位反馈-2.00%", rendered)

    def test_positive_mean_negative_median_is_narrow_profit_not_supportive(self):
        previous = pd.DataFrame([
            {"代码": "000001", "涨跌幅": 10.0, "昨日连板数": 1},
            {"代码": "600001", "涨跌幅": -1.0, "昨日连板数": 2},
            {"代码": "600002", "涨跌幅": -2.0, "昨日连板数": 1},
        ])
        result = build(self.snapshot(), self.manifest(), self.market(),
                       {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
                       previous_limit=previous)
        block = result["emotion_dimensions"]["money_effect"]
        self.assertEqual(block["state"], "MIXED")
        self.assertEqual(block["evidence"]["profit_distribution_state"],
                         "NARROW_POSITIVE")
        self.assertAlmostEqual(block["evidence"]["mean_median_gap"], .07 / 3 + .01)
        self.assertIn("concentrated in a minority", block["impact"])
        self.assertIn("赚钱分布：NARROW_POSITIVE", markdown(result))

    def test_failed_board_loss_is_measured_from_limit_fill_not_close_pct(self):
        result = build(self.snapshot(), self.manifest(), self.market(),
                       {"operational_threshold": None}, "2026-09-05T20:00:00+08:00")
        evidence = result["emotion_dimensions"]["broken_loss_effect"]["evidence"]
        # A failed board that closes at +0% loses 9.09% from a 10% limit fill.
        self.assertAlmostEqual(evidence["failed_limit_entry_mean_return"], -1 / 11)
        self.assertEqual(evidence["failed_limit_entry_large_loss_count"], 1)
        self.assertEqual(evidence["failed_limit_entry_observation_count"], 1)
        self.assertIn("普通10%涨停价成交代理", markdown(result))

    def test_breadth_expansion_with_rising_failed_rate_is_fragile(self):
        market = pd.concat([
            pd.DataFrame([{
                **self.market().iloc[0].to_dict(), "date": "2026-09-03",
                "limit_count": 1, "failed_rate": .20,
            }]),
            self.market(),
        ], ignore_index=True)
        result = build(self.snapshot(), self.manifest(), market,
                       {"operational_threshold": None},
                       "2026-09-05T20:00:00+08:00")
        block = result["emotion_dimensions"]["repair_quality"]
        self.assertEqual(block["state"], "ADVERSE")
        self.assertEqual(block["evidence"]["breadth_quality"],
                         "FRAGILE_EXPANSION")
        self.assertEqual(block["evidence"]["limit_count_change"], 1)
        self.assertAlmostEqual(block["evidence"]["failed_board_rate_change"], .30)
        self.assertEqual(block["evidence"]["b1_b3_market_path_scenario"],
                         "ELEVATED_ADVERSE_PATH_RESEARCH_WARNING")
        self.assertIsNone(block["evidence"]["b1_b3_market_path_probability"])
        self.assertFalse(block["evidence"][
            "candidate_holding_period_inference_allowed"])
        self.assertIn("涨停扩张质量：FRAGILE_EXPANSION", markdown(result))
        self.assertIn("不得由市场路径直接推导个股持有三天", markdown(result))

    def test_one_word_share_is_crowding_context_not_direct_risk_gate(self):
        market = self.market().copy()
        market.loc[0, "one_word_share"] = .40
        market.loc[0, "crowding_risk"] = .70
        result = build(self.snapshot(), self.manifest(), market,
                       {"operational_threshold": None},
                       "2026-09-05T20:00:00+08:00")
        evidence = result["emotion_dimensions"]["height_feedback"]["evidence"]
        self.assertAlmostEqual(evidence["one_word_share"], .40)
        self.assertFalse(evidence["one_word_share_direct_risk_gate_allowed"])
        rendered = markdown(result)
        self.assertIn("一致拥挤代理：一字占比40.0%", rendered)
        self.assertIn("不单独预测次日必分歧", rendered)

    def test_highest_board_is_anchor_not_buy(self):
        result = build(self.snapshot(), self.manifest(), self.market(),
                       {"operational_threshold": None}, "2026-09-05T20:00:00+08:00")
        self.assertEqual(result["roles"]["emotion_anchors"][0]["code"], "600001")
        self.assertEqual(result["roles"]["highest_leader_potential"]["code"], "600001")
        self.assertEqual(result["roles"]["highest_leader_potential"]["evidence_status"],
                         "INFERRED_EOD_RANKING_NOT_PROBABILITY")
        self.assertEqual(result["roles"]["next_height_challenger"]["code"], "000001")
        self.assertEqual(result["roles"]["anchor_premium_reviews"][0]["premium_space"], "LOW")
        self.assertIsNone(result["roles"]["current_best_buy_opportunity"])
        rendered = markdown(result)
        self.assertIn("最可能成为/保持下一最高标", rendered)
        self.assertIn("当前锚失败后的高度接班研究首位", rendered)
        self.assertIn("入选最高标候选不等于可以买", rendered)
        self.assertIn("最高板不会自动顶替市场核心", rendered)
        self.assertIn("综合唯一首选：空仓", rendered)

    def test_verified_market_core_is_descriptive_and_requires_full_evidence(self):
        candidate = {
            "code": "600001", "name": "CORE", "boards": 3,
            "competition_identity": "height_anchor",
            "theme_evidence": {"status": "VERIFIED", "driver_verified": True,
                               "formal_mapping_eligible": True,
                               "theme": "specific", "support_ex_self": 2},
            "cross_theme_race": {"model_eligible": True, "entry_authorized": False},
            "eligibility_status": "VERIFIED_ELIGIBLE", "active_hard_risks": [],
            "failed_seal_count": 1,
        }
        core = _current_market_core([candidate])
        self.assertEqual(core["code"], "600001")
        self.assertFalse(core["buy_signal"])
        candidate["theme_evidence"]["driver_verified"] = False
        self.assertIsNone(_current_market_core([candidate]))

    def test_group_primary_requires_calibrated_probability_and_all_gates(self):
        candidate = {
            "code": "000001", "name": "FIRST", "boards": 1,
            "rank_probability": .76,
            "probability_status": "VALIDATED_WALK_FORWARD_CALIBRATED",
            "entry_permission": "CONDITIONAL_ONLY",
            "competition_identity": "pioneer",
            "theme_evidence": {"status": "VERIFIED", "driver_verified": True,
                               "formal_mapping_eligible": True},
            "cross_theme_race": {"model_eligible": True, "entry_authorized": False},
            "eligibility_status": "VERIFIED_ELIGIBLE", "active_hard_risks": [],
        }
        primary, status = _validated_group_primary([candidate], .70, "FIRST_BOARD")
        self.assertEqual(primary["code"], "000001")
        self.assertFalse(primary["buy_signal"])
        self.assertEqual(status["validated_candidate_count"], 1)
        candidate["probability_status"] = "UNAVAILABLE"
        primary, status = _validated_group_primary([candidate], .70, "FIRST_BOARD")
        self.assertIsNone(primary)
        self.assertEqual(status["status"], "CASH_NO_VALIDATED_CANDIDATE")
        primary, status = _validated_group_primary([candidate], None, "FIRST_BOARD")
        self.assertIsNone(primary)
        self.assertEqual(status["status"], "CASH_NO_VALIDATED_THRESHOLD")

    def test_high_attention_exclusion_explains_why_height_is_not_a_pick(self):
        candidate = {
            "code": "600001", "name": "HIGH", "boards": 4,
            "first_seal_time": 92500, "failed_seal_count": 0,
            "theme_evidence": {"status": "INFERRED_OR_LIMITED",
                               "driver_verified": None},
            "competition_identity": "unavailable",
            "cross_theme_race": {"model_eligible": False},
            "eligibility_status": "SECURITY_STATUS_UNAVAILABLE_OR_INELIGIBLE",
            "probability_status": "UNAVAILABLE", "active_hard_risks": [],
        }
        exclusions = _explicit_high_attention_exclusions([candidate], 4)
        self.assertEqual(exclusions[0]["code"], "600001")
        self.assertIn("ONE_PRICE_PROXY_BUYABILITY_NOT_PROVEN",
                      exclusions[0]["exclusion_reasons"])
        self.assertIn("CALIBRATED_CONTINUATION_PROBABILITY_UNAVAILABLE",
                      exclusions[0]["exclusion_reasons"])

    def test_pool_overlap_is_rejected(self):
        frame = self.snapshot()
        frame.loc[2, "code"] = "000001"
        with self.assertRaisesRegex(ValueError, "overlap"):
            build(frame, self.manifest(), self.market(), {"operational_threshold": None},
                  "2026-09-05T20:00:00+08:00")

    def test_string_false_never_enters_mainboard_pool(self):
        frame = self.snapshot()
        frame["is_mainboard"] = frame.is_mainboard.astype(object)
        frame.loc[1, "code"] = "688001"
        frame.loc[1, "is_mainboard"] = "False"
        frame.loc[[0, 2], "is_mainboard"] = "True"
        manifest = self.manifest()
        manifest.update(mainboard_closing_limit_rows=1,
                        mainboard_board_level_counts={"1": 1},
                        verified_eligible_closing_rows=1,
                        security_unavailable_or_ineligible_closing_rows=0)
        result = build(frame, manifest, self.market(), {"operational_threshold": None},
                       "2026-09-05T20:00:00+08:00")
        codes = {row["code"] for row in result["first_board_candidates"] + result["relay_candidates"]}
        self.assertNotIn("688001", codes)

    def test_active_hard_notice_is_attached_but_ordinary_notice_is_not(self):
        notices = pd.DataFrame([
            {"code": "600001", "published_at": "2026-09-04 23:59:59", "risk_type": "风险提示_风险公告",
             "title": "risk", "risk_hard": 1, "data_status": "verified_index_date_only"},
            {"code": "000001", "published_at": "2026-09-04 23:59:59", "risk_type": "监管背景_普通异常波动",
             "title": "ordinary", "risk_hard": 0, "data_status": "verified_index_date_only"},
        ])
        result = build(self.snapshot(), self.manifest(), self.market(),
                       {"operational_threshold": None}, "2026-09-05T20:00:00+08:00", notices)
        high = result["relay_candidates"][0]
        first = result["first_board_candidates"][0]
        self.assertEqual(len(high["active_hard_risks"]), 1)
        self.assertEqual(first["active_hard_risks"], [])
        self.assertFalse(result["announcement_summary"]["ordinary_abnormal_notice_is_mechanical_veto"])

    def test_date_only_notice_received_before_after_close_cutoff_is_effective(self):
        notices = pd.DataFrame([{
            "code": "600001", "published_at": "2026-09-04 23:59:59",
            "published_at_status": "date_only_conservative_end_of_day",
            "source_timestamp": "2026-09-04 20:39:45",
            "risk_type": "风险提示_风险公告", "title": "risk", "risk_hard": 1,
            "data_status": "verified_index_date_only",
        }])
        result = build(self.snapshot(), self.manifest(), self.market(),
                       {"operational_threshold": None}, "2026-09-04T20:50:00+08:00", notices)
        self.assertEqual(len(result["relay_candidates"][0]["active_hard_risks"]), 1)

    def test_date_only_notice_received_after_cutoff_is_not_future_leaked(self):
        notices = pd.DataFrame([{
            "code": "600001", "published_at": "2026-09-04 23:59:59",
            "published_at_status": "date_only_conservative_end_of_day",
            "source_timestamp": "2026-09-04 20:39:45",
            "risk_type": "风险提示_风险公告", "title": "risk", "risk_hard": 1,
            "data_status": "verified_index_date_only",
        }])
        result = build(self.snapshot(), self.manifest(), self.market(),
                       {"operational_threshold": None}, "2026-09-04T20:30:00+08:00", notices)
        self.assertEqual(result["relay_candidates"][0]["active_hard_risks"], [])

    def test_unknown_execution_fields_are_explicit_and_not_zero(self):
        result = build(self.snapshot(), self.manifest(), self.market(),
                       {"operational_threshold": None}, "2026-09-05T20:00:00+08:00")
        self.assertEqual(result["decision_readiness"]["entry_plan_status"],
                         "UNAVAILABLE_UNTIL_AUCTION_CONFIRMATION")
        self.assertEqual(result["decision_readiness"]["actual_fill_status"], "UNAVAILABLE")
        self.assertIn("entry_fill", result["decision_readiness"]["unknown_fields"])
        self.assertEqual(result["relay_candidates"][0]["execution_state"], "DATA_LIMITED")
        self.assertEqual(result["relay_candidates"][0]["auction_conditions"], [])

    def test_markdown_handles_missing_market_values(self):
        market = self.market().copy()
        for field in ("yesterday_premium", "previous_limit_tail_loss", "failed_rate", "seal_rate",
                      "down_limit_count", "max_board", "ladder_gap_count", "limit_count",
                      "consecutive_count", "failed_count"):
            market.loc[0, field] = None
        result = build(self.snapshot(), self.manifest(), market,
                       {"operational_threshold": None}, "2026-09-05T20:00:00+08:00")
        rendered = markdown(result)
        self.assertIn("不可用", rendered)
        self.assertNotIn("nan", rendered.lower())

    def test_verified_theme_roles_are_bound_without_promoting_to_entry(self):
        theme_roles = pd.DataFrame([{
            "signal_date": "2026-09-04", "code": "000001", "theme_status": "verified_same_day_group",
            "theme_role": "PIONEER", "theme_group_id": "theme.alpha", "theme": "具体催化",
            "primary_evidence_id": "ev-1", "source_tier": "A", "theme_support_ex_self": 2,
            "same_height_rivals": 0, "theme_ladder_verified": True,
        }])
        result = build(self.snapshot(), self.manifest(), self.market(),
                       {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
                       theme_roles=theme_roles)
        first = result["first_board_candidates"][0]
        self.assertEqual(first["theme_status"], "verified_same_day_group")
        self.assertEqual(first["theme_role"], "PIONEER")
        self.assertEqual(first["theme_evidence"]["evidence_id"], "ev-1")
        self.assertEqual(result["theme_summary"]["verified_candidate_count"], 1)
        self.assertIn("REAL_SPECIFIC_THEME_EVIDENCE_INCOMPLETE", result["blocking_reasons"])
        self.assertIsNone(result["roles"]["current_best_buy_opportunity"])

    def test_source_bound_theme_status_is_verified(self):
        theme_roles = pd.DataFrame([{
            "signal_date": "2026-09-04", "code": "000001",
            "theme_status": "verified_explicit_source_roster", "theme_role": "THEME_SUPPORT",
            "theme_group_id": "theme.alpha", "theme": "具体催化",
            "primary_evidence_id": "https://example.test/a", "source_tier": "B",
            "theme_support_ex_self": 2, "same_height_rivals": 0,
            "theme_ladder_verified": True,
            "evidence_published_at": "2026-09-04T16:00:00+08:00",
        }])
        result = build(self.snapshot(), self.manifest(), self.market(),
                       {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
                       theme_roles=theme_roles)
        self.assertEqual(result["first_board_candidates"][0]["theme_evidence"]["status"], "VERIFIED")
        self.assertEqual(result["theme_summary"]["verified_candidate_count"], 1)

    def test_secondary_theme_roster_is_visible_but_not_verified_or_mapped(self):
        theme_roles = pd.DataFrame([{
            "signal_date": "2026-09-04", "code": "600001",
            "theme_status": "observed_explicit_secondary_roster",
            "theme_role": "THEME_HEIGHT_LEADER", "theme_group_id": "theme.alpha",
            "theme": "市场二手叙事", "primary_evidence_id": "https://example.test/a",
            "source_tier": "C_PUBLIC_REVIEW_EXPLICIT_ROSTER",
            "source_claim_status": "secondary_market_narrative",
            "theme_support_ex_self": 1, "same_height_rivals": 0,
            "theme_ladder_verified": True, "evidence_published_at": "",
            "evidence_available_at": "2026-09-05T19:00:00+08:00",
            "evidence_time_basis": "retrieval_time_only_publication_time_unavailable",
        }])
        result = build(self.snapshot(), self.manifest(), self.market(),
                       {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
                       theme_roles=theme_roles)
        high = result["relay_candidates"][0]
        self.assertEqual(high["theme_evidence"]["status"], "INFERRED_OR_LIMITED")
        self.assertEqual(result["theme_summary"]["observed_secondary_candidate_count"], 1)
        self.assertEqual(result["theme_summary"]["verified_candidate_count"], 0)
        self.assertEqual(result["decision_readiness"]["theme_evidence_status"],
                         "OBSERVED_SECONDARY_NARRATIVE_ONLY")
        self.assertIn("SECONDARY_THEME_NARRATIVE_NOT_BUSINESS_OR_CATALYST_VERIFIED",
                      result["blocking_reasons"])
        mapping = result["roles"]["high_low_supplement_mappings"][0]
        self.assertIn("HIGH_CORE_SPECIFIC_THEME_NOT_VERIFIED", mapping["blocking_reasons"])

    def test_all_three_plus_board_rows_get_mapping_records_and_secondary_alternatives_stay_nonactionable(self):
        snapshot = self.snapshot().copy()
        extra = snapshot.iloc[[1]].copy()
        extra.loc[:, "code"] = "600002"
        extra.loc[:, "name"] = "SECOND_HIGH"
        extra.loc[:, "boards"] = 4
        snapshot.loc[snapshot.code.eq("600001"), "boards"] = 6
        low = snapshot.iloc[[0]].copy()
        low.loc[:, "code"] = "000003"
        low.loc[:, "name"] = "PCB_LOW"
        snapshot = pd.concat([snapshot, extra, low], ignore_index=True)
        manifest = self.manifest() | {
            "mainboard_closing_limit_rows": 4,
            "mainboard_board_level_counts": {"1": 2, "4": 1, "6": 1},
            "verified_eligible_closing_rows": 2,
            "security_unavailable_or_ineligible_closing_rows": 2,
        }
        theme_roles = pd.DataFrame([
            {"signal_date": "2026-09-04", "code": "600002",
             "theme_status": "observed_explicit_secondary_roster",
             "theme_role": "THEME_HEIGHT_LEADER", "theme_group_id": "pcb",
             "theme": "PCB板", "primary_evidence_id": "https://example.test/review",
             "source_tier": "C_PUBLIC_REVIEW_EXPLICIT_ROSTER",
             "source_claim_status": "secondary_market_narrative",
             "theme_support_ex_self": 1, "same_height_rivals": 0,
             "theme_ladder_verified": True, "evidence_published_at": "",
             "evidence_available_at": "2026-09-05T19:00:00+08:00",
             "evidence_time_basis": "retrieval_time_only_publication_time_unavailable",
             "identity_tags": "THEME_HEIGHT_LEADER", "capacity_rank_in_theme": 1},
            {"signal_date": "2026-09-04", "code": "000003",
             "theme_status": "observed_explicit_secondary_roster",
             "theme_role": "THEME_SUPPORT", "theme_group_id": "pcb",
             "theme": "PCB板", "primary_evidence_id": "https://example.test/review",
             "source_tier": "C_PUBLIC_REVIEW_EXPLICIT_ROSTER",
             "source_claim_status": "secondary_market_narrative",
             "theme_support_ex_self": 1, "same_height_rivals": 0,
             "theme_ladder_verified": True, "evidence_published_at": "",
             "evidence_available_at": "2026-09-05T19:00:00+08:00",
             "evidence_time_basis": "retrieval_time_only_publication_time_unavailable",
             "identity_tags": "THEME_SUPPORT|FIRST_SEAL_PIONEER",
             "capacity_rank_in_theme": 2},
        ])
        result = build(snapshot, manifest, self.market(), {"operational_threshold": None},
                       "2026-09-05T20:00:00+08:00", theme_roles=theme_roles)
        mappings = result["roles"]["high_low_supplement_mappings"]
        self.assertEqual([item["high_core"]["code"] for item in mappings],
                         ["600001", "600002"])
        second = mappings[1]
        self.assertEqual(second["mapping_status"],
                         "NARRATIVE_ALTERNATIVES_ONLY_NOT_VERIFIED")
        self.assertIsNone(second["supplement_candidate"])
        self.assertEqual(second["narrative_only_low_alternatives"][0]["code"], "000003")
        self.assertFalse(second["narrative_only_low_alternatives"][0]["buy_signal"])
        self.assertEqual(result["roles"]["high_low_mapping_coverage"]["high_position_records"], 2)
        self.assertIn("二手叙事备选（不是补涨候选）", markdown(result))

    def test_verified_business_membership_without_active_driver_stays_nonactionable(self):
        theme_roles = pd.DataFrame([
            {"signal_date": "2026-09-04", "code": "600001",
             "theme_status": "verified_membership_driver_unverified",
             "theme_role": "THEME_HEIGHT_LEADER", "theme_group_id": "pcb_direct",
             "theme": "PCB直接制造", "business_branch": "direct_pcb_manufacturing",
             "primary_evidence_id": "https://example.test/high.pdf",
             "source_tier": "A_PRIMARY_EXCHANGE_FILING",
             "source_claim_status": "secondary_market_narrative",
             "membership_claim_status": "verified_primary", "membership_verified": True,
             "driver_claim_status": "secondary_market_narrative", "driver_verified": False,
             "formal_high_low_mapping_eligible": False,
             "theme_support_ex_self": 1, "same_height_rivals": 0,
             "theme_ladder_verified": False, "identity_tags": "THEME_HEIGHT_LEADER",
             "capacity_rank_in_theme": 1,
             "evidence_available_at": "2026-09-05T19:00:00+08:00"},
            {"signal_date": "2026-09-04", "code": "000001",
             "theme_status": "verified_membership_driver_unverified",
             "theme_role": "THEME_SUPPORT", "theme_group_id": "pcb_direct",
             "theme": "PCB直接制造", "business_branch": "direct_pcb_manufacturing",
             "primary_evidence_id": "https://example.test/low.pdf",
             "source_tier": "A_PRIMARY_EXCHANGE_FILING",
             "source_claim_status": "secondary_market_narrative",
             "membership_claim_status": "verified_primary", "membership_verified": True,
             "driver_claim_status": "secondary_market_narrative", "driver_verified": False,
             "formal_high_low_mapping_eligible": False,
             "theme_support_ex_self": 1, "same_height_rivals": 0,
             "theme_ladder_verified": False,
             "identity_tags": "THEME_SUPPORT|FIRST_SEAL_PIONEER",
             "capacity_rank_in_theme": 2,
             "evidence_available_at": "2026-09-05T19:00:00+08:00"},
        ])
        result = build(self.snapshot(), self.manifest(), self.market(),
                       {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
                       theme_roles=theme_roles)
        self.assertEqual(result["theme_summary"][
            "verified_membership_driver_unverified_count"], 2)
        self.assertEqual(result["theme_summary"]["verified_candidate_count"], 0)
        mapping = result["roles"]["high_low_supplement_mappings"][0]
        self.assertEqual(mapping["mapping_status"],
                         "VERIFIED_MEMBERSHIP_ONLY_DRIVER_UNVERIFIED")
        self.assertIsNone(mapping["supplement_candidate"])
        self.assertEqual(mapping["verified_membership_only_low_alternatives"][0]["code"],
                         "000001")
        self.assertFalse(mapping["verified_membership_only_low_alternatives"][0]["buy_signal"])
        self.assertIn("ACTIVE_DRIVER_NOT_VERIFIED", mapping["blocking_reasons"])
        self.assertIn("同业务支线已核验备选", markdown(result))

    def test_future_theme_publication_is_rejected(self):
        theme_roles = pd.DataFrame([{
            "signal_date": "2026-09-04", "code": "000001",
            "theme_status": "verified_explicit_source_roster", "theme_role": "THEME_SUPPORT",
            "theme_group_id": "theme.alpha", "theme": "具体催化",
            "primary_evidence_id": "https://example.test/a", "source_tier": "B",
            "theme_support_ex_self": 1, "same_height_rivals": 0,
            "theme_ladder_verified": True,
            "evidence_published_at": "2026-09-06T16:00:00+08:00",
        }])
        with self.assertRaisesRegex(ValueError, "after asof"):
            build(self.snapshot(), self.manifest(), self.market(),
                  {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
                  theme_roles=theme_roles)

    def test_verified_competition_snapshot_is_bound_without_promoting_to_entry(self):
        competition = pd.DataFrame([{
            "signal_date": "2026-09-04", "code": "600001",
            "competition_identity": "height_anchor", "source": "verified snapshot",
            "observed_at": "2026-09-04T14:59:00+08:00",
            "available_at": "2026-09-04T15:01:00+08:00",
            "data_status": "verified", "universe_complete": True,
            "same_board_peer_count": 0, "higher_board_peer_count": 0,
            "events": '[{"type":"UNIQUE_HEIGHT"}]',
        }])
        result = build(self.snapshot(), self.manifest(), self.market(),
                       {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
                       competition=competition)
        high = result["relay_candidates"][0]
        self.assertEqual(high["competition_identity"], "height_anchor")
        self.assertEqual(high["competition_evidence"]["events"][0]["type"], "UNIQUE_HEIGHT")
        self.assertIsNone(high["competition_evidence"]["same_theme_peer_count"])
        self.assertEqual(result["competition_summary"]["status"], "VERIFIED_PARTIAL")
        self.assertIn("LEADER_COMPETITION_EVIDENCE_INCOMPLETE", result["blocking_reasons"])
        self.assertIsNone(result["roles"]["current_best_buy_opportunity"])
        json.dumps(result, ensure_ascii=False, allow_nan=False)

    def test_every_high_core_has_explicit_high_low_mapping_result(self):
        theme_roles = pd.DataFrame([
            {"signal_date": "2026-09-04", "code": "600001",
             "theme_status": "verified_same_day_group", "theme_role": "HEIGHT_ANCHOR",
             "theme_group_id": "theme.alpha", "theme": "具体细分A",
             "primary_evidence_id": "ev-high", "source_tier": "A",
             "theme_support_ex_self": 1, "same_height_rivals": 0,
             "theme_ladder_verified": True},
            {"signal_date": "2026-09-04", "code": "000001",
             "theme_status": "verified_same_day_group", "theme_role": "PIONEER",
             "theme_group_id": "theme.alpha", "theme": "具体细分A",
             "primary_evidence_id": "ev-low", "source_tier": "A",
             "theme_support_ex_self": 1, "same_height_rivals": 0,
             "theme_ladder_verified": True},
        ])
        competition = pd.DataFrame([
            {"signal_date": "2026-09-04", "code": "600001",
             "competition_identity": "height_anchor", "source": "snapshot",
             "observed_at": "2026-09-04T14:59:00+08:00",
             "available_at": "2026-09-04T15:01:00+08:00",
             "data_status": "verified", "universe_complete": True},
            {"signal_date": "2026-09-04", "code": "000001",
             "competition_identity": "pioneer", "source": "snapshot",
             "observed_at": "2026-09-04T14:59:00+08:00",
             "available_at": "2026-09-04T15:01:00+08:00",
             "data_status": "verified", "universe_complete": True},
        ])
        result = build(self.snapshot(), self.manifest(), self.market(),
                       {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
                       theme_roles=theme_roles, competition=competition)
        mappings = result["roles"]["high_low_supplement_mappings"]
        self.assertEqual(len(mappings), 1)
        self.assertEqual(mappings[0]["high_core"]["code"], "600001")
        self.assertEqual(mappings[0]["supplement_candidate"]["code"], "000001")
        self.assertFalse(mappings[0]["buy_signal"])
        self.assertEqual(result["roles"]["high_low_supplement_candidate"]["code"], "000001")
        self.assertIn("高位到低位补涨映射", markdown(result))

    def test_unverified_high_theme_yields_explicit_no_mapping(self):
        result = build(self.snapshot(), self.manifest(), self.market(),
                       {"operational_threshold": None}, "2026-09-05T20:00:00+08:00")
        mapping = result["roles"]["high_low_supplement_mappings"][0]
        self.assertEqual(mapping["mapping_status"], "NO_ELIGIBLE_MAPPING")
        self.assertIn("HIGH_CORE_SPECIFIC_THEME_NOT_VERIFIED", mapping["blocking_reasons"])
        self.assertIsNone(result["roles"]["high_low_supplement_candidate"])

    def test_string_false_competition_completeness_is_rejected(self):
        competition = pd.DataFrame([{
            "signal_date": "2026-09-04", "code": "600001",
            "competition_identity": "height_anchor", "source": "snapshot",
            "observed_at": "2026-09-04T14:59:00+08:00",
            "available_at": "2026-09-04T15:01:00+08:00",
            "data_status": "verified", "universe_complete": "False",
        }])
        with self.assertRaisesRegex(ValueError, "not verified and complete"):
            build(self.snapshot(), self.manifest(), self.market(),
                  {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
                  competition=competition)

    def test_confidence_aware_cross_theme_races_are_bound_without_reranking(self):
        races = pd.DataFrame([
            {"signal_date": "2026-09-04", "code": "000001",
             "race_lane": "FIRST_BOARD_NEW_CYCLE_RACE",
             "cross_theme_race_status": "DESCRIPTIVE_SECONDARY_NARRATIVE_ONLY",
             "eligible_for_cross_theme_race_training": False,
             "winner_claimed": False, "entry_authorized": False,
             "race_asof": "2026-09-05T19:00:00+08:00",
             "theme_evidence_confidence": "SECONDARY_NARRATIVE",
             "race_peer_evidence_scope": "SAME_CONFIDENCE_ONLY",
             "first_cross_theme_competing_theme_count": 3,
             "first_cross_theme_competing_candidate_count": 7,
             "first_own_theme_first_count": 2, "first_own_theme_relay_count": 1,
             "first_existing_relay_backed": True, "first_isolated_theme_first": False,
             "cross_theme_race_events":
             '[{"type":"FIRST_BOARD_CROSS_THEME_NEW_CYCLE_RACE"}]'},
            {"signal_date": "2026-09-04", "code": "600001",
             "race_lane": "RELAY_HEIGHT_RACE",
             "cross_theme_race_status": "UNKNOWN_THEME_NO_RACE_INFERENCE",
             "eligible_for_cross_theme_race_training": False,
             "winner_claimed": False, "entry_authorized": False,
             "race_asof": "2026-09-05T19:00:00+08:00",
             "theme_evidence_confidence": "UNKNOWN",
             "race_peer_evidence_scope": "SAME_CONFIDENCE_ONLY",
             "cross_theme_race_events": "[]"},
        ])
        result = build(self.snapshot(), self.manifest(), self.market(),
                       {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
                       cross_theme_races=races)
        first = result["first_board_candidates"][0]["cross_theme_race"]
        relay = result["relay_candidates"][0]["cross_theme_race"]
        self.assertEqual(first["lane"], "FIRST_BOARD_NEW_CYCLE_RACE")
        self.assertEqual(first["first_competing_theme_count"], 3)
        self.assertIsNone(first["relay_higher_theme_count"])
        self.assertEqual(relay["lane"], "RELAY_HEIGHT_RACE")
        self.assertEqual(result["cross_theme_race_summary"]["attached_candidate_count"], 2)
        self.assertEqual(result["cross_theme_race_summary"]["known_theme_candidate_count"], 1)
        self.assertEqual(result["cross_theme_race_summary"]["model_eligible_candidate_count"], 0)
        self.assertIn("CROSS_THEME_RACE_INCOMPLETE_OR_UNVERIFIED",
                      result["blocking_reasons"])
        self.assertIsNone(result["roles"]["current_best_buy_opportunity"])
        self.assertIn("首板只进入新周期/低位注意力赛道", markdown(result))

    def test_cross_theme_race_cannot_claim_a_winner_or_entry(self):
        races = pd.DataFrame([{
            "signal_date": "2026-09-04", "code": "000001",
            "race_lane": "FIRST_BOARD_NEW_CYCLE_RACE",
            "cross_theme_race_status": "DESCRIPTIVE_SECONDARY_NARRATIVE_ONLY",
            "eligible_for_cross_theme_race_training": False,
            "winner_claimed": True, "entry_authorized": False,
            "race_asof": "2026-09-05T19:00:00+08:00",
            "theme_evidence_confidence": "SECONDARY_NARRATIVE",
            "race_peer_evidence_scope": "SAME_CONFIDENCE_ONLY",
        }])
        with self.assertRaisesRegex(ValueError, "cannot claim winner or entry"):
            build(self.snapshot(), self.manifest(), self.market(),
                  {"operational_threshold": None}, "2026-09-05T20:00:00+08:00",
                  cross_theme_races=races)


if __name__ == "__main__":
    unittest.main()
