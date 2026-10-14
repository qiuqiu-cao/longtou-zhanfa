import unittest

from synthesize_emotion_policy_evidence_v122 import synthesize


class EmotionPolicySynthesisTests(unittest.TestCase):
    def test_aggregate_pass_does_not_promote_candidate_veto(self):
        aggregate = {"status": "DIAGNOSTIC_ONLY", "economic_gate_passed": True,
                     "direction_pass_count": 4, "strong_ci_pass_count": 3}
        board = {"status": "DIAGNOSTIC_ONLY", "complete_trade_count": 0,
                 "board_groups": {
                     "FIRST_BOARD": {"risk_overlay_proxy_gate_passed": False,
                                     "direction_pass_count": 7, "strong_ci_pass_count": 0},
                     "CONSECUTIVE_BOARD": {"risk_overlay_proxy_gate_passed": False,
                                           "direction_pass_count": 1, "strong_ci_pass_count": 0},
                 }}
        report = synthesize(aggregate, board)
        policy = report["policy_governance"]
        self.assertTrue(policy["aggregate_risk_warning_allowed"])
        self.assertFalse(policy["blanket_relay_veto_promoted_to_live"])
        self.assertFalse(policy["claim_emotion_veto_improves_relay_returns"])
        self.assertFalse(report["deployment_authorized"])

    def test_both_board_groups_are_required(self):
        aggregate = {"status": "DIAGNOSTIC_ONLY"}
        board = {"status": "DIAGNOSTIC_ONLY", "board_groups": {"FIRST_BOARD": {}}}
        with self.assertRaisesRegex(ValueError, "both required"):
            synthesize(aggregate, board)


if __name__ == "__main__":
    unittest.main()
