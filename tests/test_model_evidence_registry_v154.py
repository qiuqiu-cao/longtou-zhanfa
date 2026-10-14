import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_evidence_registry_v154 import build


class ModelEvidenceRegistryV154Tests(unittest.TestCase):
    def _root(self, no_touch=True):
        holder = tempfile.TemporaryDirectory()
        root = Path(holder.name)
        payloads = {
            "results/v152/full_pool_outcomes_b123_touchfixed_cutoff_20251226_v2/manifest.json": {
                "target_contract": {"no_touch_rows_retained_as_negative": no_touch}},
            "results/v152/group_rank_walkforward_touchfixed_20260917_v4/report.json": {
                "target_coverage": {"no_touch_rows_retained_as_negative": no_touch}},
            "results/v153/group_rank_governance_touchfixed_20260917_v1.json": {
                "first_board_probability_improvement_claim_allowed": False,
                "relay_probability_improvement_claim_allowed": False,
                "complete_trade_probability_claim_allowed": False},
            "results/current_acceptance_report_20260917_v154.json": {
                "paper_trade_ready": False, "deployment_allowed": False},
        }
        for relative, payload in payloads.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
        return holder, root

    def test_claims_fail_closed(self):
        holder, root = self._root()
        self.addCleanup(holder.cleanup)
        report = build(root)
        permissions = report["claim_permissions"]
        self.assertTrue(permissions["next_session_aggregate_emotion_risk_warning"])
        self.assertFalse(permissions["emotion_can_directly_rerank_stocks"])
        self.assertFalse(permissions["combined_leader_probability_improvement"])
        self.assertFalse(permissions["severe_abnormal_calibrated_hit_probability"])
        self.assertFalse(permissions["severe_abnormal_candidate_reranking"])
        self.assertFalse(permissions["severe_abnormal_closest_distance_role_display"])
        self.assertFalse(permissions["severe_abnormal_path_potential_present"])
        self.assertFalse(permissions["ladder_gap_positive_ranking_factor_use"])
        self.assertFalse(permissions["first_open_delayed_price_discovery_factor_use"])
        self.assertFalse(permissions["first_open_automatic_healthy_divergence_claim"])
        self.assertFalse(permissions["daily_six_role_contract_complete"])
        self.assertFalse(permissions["daily_group_primary_or_cash_contract_complete"])
        self.assertFalse(permissions["daily_role_contract_authorizes_entry"])
        self.assertFalse(permissions["anchor_destination_historical_scenario_rates"])
        self.assertFalse(permissions["anchor_destination_prior_close_factor_use"])
        self.assertFalse(permissions["anchor_destination_automatic_high_low_entry"])
        self.assertFalse(permissions["intraday_level1_net_flow_direct_model_use"])
        self.assertFalse(permissions["intraday_minute_sequence_causal_feedback_claim"])
        self.assertFalse(permissions["intraday_flow_or_sequence_entry_authorized"])
        self.assertFalse(permissions["profit_concentration_historical_scenario_use"])
        self.assertFalse(permissions["profit_concentration_direct_stock_reranking"])
        self.assertFalse(permissions["profit_concentration_entry_authorized"])
        self.assertFalse(permissions["breadth_quality_historical_scenario_use"])
        self.assertFalse(permissions["breadth_quality_direct_emotion_gate"])
        self.assertFalse(permissions["breadth_quality_direct_stock_reranking"])
        self.assertFalse(permissions["breadth_quality_entry_authorized"])
        self.assertFalse(permissions["consensus_crowding_direct_emotion_gate"])
        self.assertFalse(permissions["consensus_crowding_direct_stock_reranking"])
        self.assertFalse(permissions["consensus_crowding_entry_authorized"])
        self.assertFalse(permissions["breadth_path_historical_scenario_use"])
        self.assertFalse(permissions["breadth_path_two_to_three_session_probability"])
        self.assertFalse(permissions["breadth_path_candidate_holding_inference"])
        self.assertFalse(permissions["breadth_path_entry_authorized"])
        self.assertFalse(permissions["height_anchor_buyability_uplift_supported"])
        self.assertFalse(permissions["height_anchor_candidate_probability_claim"])
        self.assertFalse(permissions["height_anchor_auto_relay_primary"])
        self.assertFalse(permissions["height_anchor_entry_authorized"])
        self.assertFalse(permissions["opening_gap_prior_close_model_use"])
        self.assertFalse(permissions["opening_gap_direct_stock_reranking"])
        self.assertFalse(permissions["opening_gap_direct_entry_authorized"])
        self.assertFalse(permissions["opening_gap_above_seven_risk_flag"])
        self.assertFalse(permissions["opening_gap_emotion_interaction_factor"])
        self.assertFalse(permissions["emotion_can_override_above_seven_risk_flag"])
        self.assertFalse(permissions["opening_gap_emotion_direct_reranking"])
        self.assertFalse(permissions["opening_gap_emotion_entry_authorized"])
        self.assertFalse(permissions["weak_to_strong_later_touch_as_open_feature"])
        self.assertFalse(permissions["low_open_automatic_weak_to_strong_buy"])
        self.assertFalse(permissions["low_open_direct_reranking"])
        self.assertFalse(permissions["low_open_entry_authorized"])
        self.assertFalse(permissions["low_open_requires_live_confirmation"])
        self.assertFalse(permissions["first_board_supply_any_success_candidate_rank"])
        self.assertFalse(permissions["first_board_supply_positive_ranking_factor"])
        self.assertFalse(permissions["first_board_supply_automatic_emotion_upgrade"])
        self.assertFalse(permissions["first_board_supply_direct_reranking"])
        self.assertFalse(permissions["first_board_supply_entry_authorized"])
        self.assertFalse(permissions["promotion_probability_as_trade_win_probability"])
        self.assertFalse(permissions["promotion_probability_replaces_b2_b3_target"])
        self.assertFalse(permissions["board_level_promotion_direct_entry_ranking"])
        self.assertFalse(permissions["promotion_payoff_direct_stock_reranking"])
        self.assertFalse(permissions["promotion_payoff_entry_authorized"])
        self.assertFalse(permissions[
            "repair_followthrough_historical_scenario_context"])
        self.assertFalse(permissions[
            "repair_followthrough_calibrated_probability"])
        self.assertFalse(permissions["repair_score_direct_emotion_gate"])
        self.assertFalse(permissions["repair_score_direct_stock_reranking"])
        self.assertFalse(permissions["repair_score_entry_authorized"])
        self.assertFalse(permissions[
            "repair_layer_historical_scenario_context"])
        self.assertFalse(permissions["repair_layer_calibrated_probability"])
        self.assertFalse(permissions["repair_layer_direct_emotion_gate"])
        self.assertFalse(permissions[
            "middle_board_confirmation_positive_ranking_factor"])
        self.assertFalse(permissions["repair_layer_direct_stock_reranking"])
        self.assertFalse(permissions["repair_layer_position_size_increase"])
        self.assertFalse(permissions["repair_layer_entry_authorized"])
        self.assertFalse(permissions["high_to_low_historical_market_scenario"])
        self.assertFalse(permissions[
            "high_to_low_calibrated_window_probability"])
        self.assertFalse(permissions["high_to_low_direct_emotion_gate"])
        self.assertFalse(permissions["high_to_low_direct_stock_reranking"])
        self.assertFalse(permissions["high_to_low_stock_selection"])
        self.assertFalse(permissions[
            "high_to_low_same_theme_supplement_mapping"])
        self.assertFalse(permissions["high_to_low_position_size_increase"])
        self.assertFalse(permissions["high_to_low_entry_authorized"])
        self.assertFalse(permissions["two_to_three_day_emotion_path_probability"])
        self.assertFalse(permissions["emotion_dynamic_path_factors_promoted"])
        self.assertFalse(permissions["point_in_time_theme_factor_training"])
        self.assertFalse(permissions["theme_catalyst_expectation_backlog_complete"])
        self.assertFalse(permissions["theme_catalyst_partial_evidence_factor_use"])
        self.assertFalse(permissions["short_term_hypothesis_direct_model_use"])
        self.assertFalse(permissions["sparse_review_dates_as_trading_calendar"])
        self.assertFalse(permissions["missing_pool_session_may_shift_labels_forward"])
        self.assertFalse(permissions["first_board_missing_bar_is_unfilled_trade"])
        self.assertFalse(permissions["first_board_execution_evidence_complete"])
        self.assertFalse(permissions["fast_seal_positive_ranking_factor_use"])
        self.assertFalse(permissions["fast_board_control_intent_inference"])
        self.assertFalse(permissions[
            "fast_board_structure_positive_ranking_factor_use"])
        self.assertFalse(permissions["cumulative_turnover_factor_use"])
        self.assertFalse(permissions["failed_seal_path_factor_use"])
        self.assertFalse(permissions[
            "first_board_profit_ranker_direct_stock_reranking"])
        self.assertFalse(permissions["first_board_profit_ranker_entry_authorized"])
        self.assertFalse(permissions["first_board_dual_target_promoted"])
        self.assertFalse(permissions["first_board_dual_target_entry_authorized"])
        self.assertFalse(permissions["five_factor_complete_historical_rows_available"])
        self.assertFalse(permissions["announcement_soft_risk_factor_history_complete"])
        self.assertFalse(permissions["competitor_elimination_formal_factor_use"])
        self.assertFalse(permissions["same_height_five_factor_uplift_validated"])
        self.assertFalse(permissions[
            "same_height_fast_seal_or_zero_open_champion_allowed"])
        self.assertFalse(permissions["theme_driver_formal_factor_use"])
        self.assertFalse(permissions["severe_abnormal_verified_label_calibration"])
        self.assertFalse(permissions["theme_evidence_availability_as_strength_factor"])
        self.assertFalse(permissions["theme_roster_direct_profit_factor_use"])
        self.assertFalse(permissions["theme_profit_coverage_direct_stock_reranking"])
        self.assertFalse(permissions["theme_profit_coverage_entry_authorized"])
        self.assertFalse(permissions["win_rate_alone_sufficient_for_strategy"])
        self.assertFalse(permissions["payoff_tail_ranker_promotion"])
        self.assertFalse(permissions["payoff_tail_entry_authorized"])
        self.assertFalse(permissions["big_loss_veto_model_promoted"])
        self.assertFalse(permissions["big_loss_veto_policy_promoted"])
        self.assertFalse(permissions["big_loss_veto_entry_authorized"])
        self.assertFalse(permissions["risk_probability_as_position_size"])
        self.assertFalse(permissions[
            "simple_profit_risk_probability_multiplication"])
        self.assertFalse(permissions["opening_gap_as_prior_close_feature"])
        self.assertFalse(permissions["opening_gap_filter_with_same_open_fill"])
        self.assertFalse(permissions["opening_0935_entry_authorized"])
        self.assertFalse(permissions["opening_causality_entry_authorized"])
        self.assertFalse(permissions["same_day_stop_as_executed_fill"])
        self.assertFalse(permissions["B_plus_1_down_limit_queue_assumed_filled"])
        self.assertFalse(permissions["fixed_stop_loss_guarantee"])
        self.assertFalse(permissions["t1_exit_audit_entry_authorized"])
        self.assertFalse(permissions["same_open_conditional_exit_fill"])
        self.assertFalse(permissions["hindsight_open_close_exit_oracle"])
        self.assertFalse(permissions["historical_open_proxy_is_broker_exit"])
        self.assertFalse(permissions["post_open_exit_rule_authorized"])
        self.assertFalse(permissions["exit_timing_policy_promoted"])
        self.assertFalse(permissions["index_emotion_divergence_context_display"])
        self.assertFalse(permissions["broad_index_equals_short_term_emotion"])
        self.assertFalse(permissions["index_up_overrides_short_loss_effect"])
        self.assertFalse(permissions["index_emotion_divergence_direct_gate"])
        self.assertFalse(permissions[
            "index_emotion_divergence_stock_reranking"])
        self.assertFalse(permissions[
            "index_emotion_divergence_entry_authorized"])
        self.assertFalse(permissions["secondary_theme_narrative_factor_use"])
        self.assertFalse(permissions["verified_membership_without_driver_factor_use"])
        self.assertFalse(permissions[
            "verified_membership_without_driver_high_low_mapping"])
        self.assertFalse(permissions["cross_theme_race_factor_training"])
        self.assertFalse(permissions["cross_theme_race_winner_or_entry_claim"])
        self.assertFalse(permissions["complete_trade_win_rate_claim"])

    def test_rejects_missing_no_touch_negative_fix(self):
        holder, root = self._root(no_touch=False)
        self.addCleanup(holder.cleanup)
        with self.assertRaisesRegex(ValueError, "no-touch"):
            build(root)

    def test_missing_corrected_authority_is_an_error(self):
        holder, root = self._root()
        self.addCleanup(holder.cleanup)
        (root / "results/current_acceptance_report_20260917_v154.json").unlink()
        with self.assertRaises(FileNotFoundError):
            build(root)


if __name__ == "__main__":
    unittest.main()
