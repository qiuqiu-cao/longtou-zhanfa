import unittest

import pandas as pd

from short_term_knowledge_audit_v158 import REQUIRED, audit


class ShortTermKnowledgeAuditV158Tests(unittest.TestCase):
    def frame(self):
        rows = []
        for number in range(1, 75):
            rows.append({
                "hypothesis_id": f"H{number:03d}", "experience": "经验",
                "source_url": "https://example.com/source", "observable_proxy": "x;y",
                "expected_direction": "条件性", "counterexample": "存在明确反例",
                "test_status": "待回测", "notes": "不得直接用于生产",
            })
        return pd.DataFrame(rows)

    def test_complete_library_is_ready_for_testing_not_production(self):
        report = audit(self.frame())
        self.assertEqual(report["status"], "HYPOTHESIS_LIBRARY_READY_FOR_TESTING")
        self.assertEqual(report["hypothesis_count"], 74)
        self.assertFalse(report["direct_model_use_allowed"])

    def test_duplicate_hypothesis_is_rejected(self):
        frame = self.frame()
        frame.loc[1, "hypothesis_id"] = "H001"
        with self.assertRaisesRegex(ValueError, "unique ids"):
            audit(frame)

    def test_missing_category_is_reported(self):
        report = audit(self.frame().query("hypothesis_id != 'H014'"))
        self.assertEqual(report["status"], "INVALID_LIBRARY")
        self.assertIn("execution_and_cost", report["missing_category_hypotheses"])

    def test_new_blind_spot_category_is_required(self):
        report = audit(self.frame().query("hypothesis_id != 'H029'"))
        self.assertEqual(report["status"], "INVALID_LIBRARY")
        self.assertIn(
            "post_close_disclosure_and_flow_proxies",
            report["missing_category_hypotheses"],
        )

    def test_membership_driver_separation_category_is_required(self):
        report = audit(self.frame().query("hypothesis_id != 'H041'"))
        self.assertEqual(report["status"], "INVALID_LIBRARY")
        self.assertIn(
            "theme_membership_driver_and_expectation_separation",
            report["missing_category_hypotheses"],
        )

    def test_confidence_aware_cross_theme_category_is_required(self):
        report = audit(self.frame().query("hypothesis_id != 'H046'"))
        self.assertEqual(report["status"], "INVALID_LIBRARY")
        self.assertIn(
            "confidence_aware_cross_theme_races",
            report["missing_category_hypotheses"],
        )

    def test_delayed_price_discovery_category_is_required(self):
        report = audit(self.frame().query("hypothesis_id != 'H049'"))
        self.assertEqual(report["status"], "INVALID_LIBRARY")
        self.assertIn(
            "delayed_price_discovery_and_ladder_gaps",
            report["missing_category_hypotheses"],
        )

    def test_intraday_causality_category_is_required(self):
        report = audit(self.frame().query("hypothesis_id != 'H053'"))
        self.assertEqual(report["status"], "INVALID_LIBRARY")
        self.assertIn(
            "rotation_flow_and_intraday_causality",
            report["missing_category_hypotheses"],
        )

    def test_profit_concentration_category_is_required(self):
        report = audit(self.frame().query("hypothesis_id != 'H054'"))
        self.assertEqual(report["status"], "INVALID_LIBRARY")
        self.assertIn(
            "profit_concentration_and_distribution",
            report["missing_category_hypotheses"],
        )

    def test_breadth_expansion_quality_category_is_required(self):
        report = audit(self.frame().query("hypothesis_id != 'H055'"))
        self.assertEqual(report["status"], "INVALID_LIBRARY")
        self.assertIn(
            "breadth_expansion_quality",
            report["missing_category_hypotheses"],
        )

    def test_height_anchor_identity_vs_buyability_category_is_required(self):
        report = audit(self.frame().query("hypothesis_id != 'H056'"))
        self.assertEqual(report["status"], "INVALID_LIBRARY")
        self.assertIn(
            "height_anchor_identity_vs_buyability",
            report["missing_category_hypotheses"],
        )

    def test_opening_gap_strength_payoff_tradeoff_category_is_required(self):
        report = audit(self.frame().query("hypothesis_id != 'H057'"))
        self.assertEqual(report["status"], "INVALID_LIBRARY")
        self.assertIn(
            "opening_gap_strength_payoff_tradeoff",
            report["missing_category_hypotheses"],
        )

    def test_opening_gap_emotion_interaction_category_is_required(self):
        report = audit(self.frame().query("hypothesis_id != 'H058'"))
        self.assertEqual(report["status"], "INVALID_LIBRARY")
        self.assertIn(
            "opening_gap_emotion_interaction",
            report["missing_category_hypotheses"],
        )

    def test_weak_to_strong_hindsight_category_is_required(self):
        report = audit(self.frame().query("hypothesis_id != 'H059'"))
        self.assertEqual(
            report["missing_category_hypotheses"][
                "weak_to_strong_hindsight_conditioning"],
            ["H059"],
        )

    def test_first_board_supply_base_rate_category_is_required(self):
        report = audit(self.frame().query("hypothesis_id != 'H060'"))
        self.assertEqual(
            report["missing_category_hypotheses"]["first_board_supply_base_rate"],
            ["H060"],
        )

    def test_repair_day_followthrough_quality_category_is_required(self):
        report = audit(self.frame().query("hypothesis_id != 'H061'"))
        self.assertEqual(
            report["missing_category_hypotheses"][
                "repair_day_followthrough_quality"],
            ["H061"],
        )

    def test_repair_layer_transmission_category_is_required(self):
        report = audit(self.frame().query("hypothesis_id != 'H062'"))
        self.assertEqual(
            report["missing_category_hypotheses"]["repair_layer_transmission"],
            ["H062"],
        )

    def test_high_to_low_ecology_window_category_is_required(self):
        report = audit(self.frame().query("hypothesis_id != 'H063'"))
        self.assertEqual(
            report["missing_category_hypotheses"]["high_to_low_ecology_window"],
            ["H063"],
        )

    def test_first_relay_lane_attention_category_is_required(self):
        report = audit(self.frame().query("hypothesis_id != 'H064'"))
        self.assertEqual(
            report["missing_category_hypotheses"][
                "first_relay_lane_attention_competition"],
            ["H064"],
        )

    def test_severe_abnormal_reset_evidence_category_is_required(self):
        report = audit(self.frame().query("hypothesis_id != 'H065'"))
        self.assertEqual(
            report["missing_category_hypotheses"][
                "severe_abnormal_reset_evidence_hierarchy"],
            ["H065"],
        )

    def test_fast_board_structure_category_is_required(self):
        report = audit(self.frame().query("hypothesis_id != 'H066'"))
        self.assertEqual(
            report["missing_category_hypotheses"][
                "fast_board_structure_vs_control_intent"],
            ["H066"],
        )

    def test_trading_calendar_category_is_required(self):
        report = audit(self.frame().query("hypothesis_id != 'H067'"))
        self.assertEqual(
            report["missing_category_hypotheses"][
                "trading_calendar_vs_observation_dates"],
            ["H067"],
        )

    def test_theme_attention_coverage_bias_category_is_required(self):
        report = audit(self.frame().query("hypothesis_id != 'H068'"))
        self.assertEqual(
            report["missing_category_hypotheses"][
                "theme_evidence_attention_and_coverage_bias"],
            ["H068"],
        )

    def test_win_rate_expectancy_tail_risk_category_is_required(self):
        report = audit(self.frame().query("hypothesis_id != 'H069'"))
        self.assertEqual(
            report["missing_category_hypotheses"][
                "win_rate_expectancy_and_tail_risk"],
            ["H069"],
        )

    def test_selective_tail_risk_veto_category_is_required(self):
        report = audit(self.frame().query("hypothesis_id != 'H070'"))
        self.assertEqual(
            report["missing_category_hypotheses"][
                "selective_tail_risk_veto_and_coverage"],
            ["H070"],
        )

    def test_opening_information_execution_causality_category_is_required(self):
        report = audit(self.frame().query("hypothesis_id != 'H071'"))
        self.assertEqual(
            report["missing_category_hypotheses"][
                "opening_information_and_execution_causality"],
            ["H071"],
        )

    def test_t1_unexitable_loss_category_is_required(self):
        report = audit(self.frame().query("hypothesis_id != 'H072'"))
        self.assertEqual(
            report["missing_category_hypotheses"][
                "t1_unexitable_loss_and_down_limit_queue"],
            ["H072"],
        )

    def test_exit_information_timing_category_is_required(self):
        report = audit(self.frame().query("hypothesis_id != 'H073'"))
        self.assertEqual(
            report["missing_category_hypotheses"][
                "exit_information_and_timing_causality"],
            ["H073"],
        )

    def test_broad_index_vs_short_term_ecology_category_is_required(self):
        report = audit(self.frame().query("hypothesis_id != 'H074'"))
        self.assertEqual(
            report["missing_category_hypotheses"][
                "broad_index_vs_short_term_ecology"],
            ["H074"],
        )


if __name__ == "__main__":
    unittest.main()
