import unittest

import pandas as pd

from build_emotion_decision_card_v92 import build, markdown


class EmotionDecisionCardTests(unittest.TestCase):
    def inputs(self):
        market = pd.DataFrame([{"date": "2026-09-09", "emotion_state": "大分歧",
            "limit_count": 48, "consecutive_count": 15, "failed_count": 29,
            "down_limit_count": 7, "seal_rate": .62, "failed_rate": .38,
            "promotion_mid": .28, "yesterday_premium": 0, "previous_limit_tail_loss": -.05,
            "max_board": 5, "ladder_gap_count": 1}])
        baseline = {"latest_prediction": {"signal_date": "2026-09-09",
            "probabilities": {"大分歧": .36, "退潮": .19, "修复": .24, "加速": .08, "一致": .01},
            "group_probabilities": {"RISK_OFF": .55, "REPAIR": .24, "RISK_ON": .21}}}
        ensemble = {"promotion_gate_passed": False, "latest_research_prediction": {
            "signal_date": "2026-09-09", "probabilities": {"RISK_OFF": .49, "REPAIR": .31, "RISK_ON": .20},
            "top_class": "RISK_OFF"}}
        policy = {"risk_off_threshold": .50, "status": "FROZEN_PAPER_VALIDATION_ONLY",
            "forward_gate": {"status": "NOT_STARTED"},
            "current_shadow_application": {"signal_date": "2026-09-09", "triggered": False}}
        review = {"signal_date": "2026-09-09", "status": "CASH"}
        return market, baseline, ensemble, policy, review

    def test_below_threshold_is_lean_not_veto(self):
        card = build(*self.inputs())
        self.assertEqual(card["stance"], "RISK_OFF_LEAN_BELOW_VETO")
        self.assertFalse(card["paper_risk_veto"]["triggered"])
        self.assertFalse(card["trade_effect"]["direct_buy_authorized"])
        self.assertIn("风险关闭", markdown(card))

    def test_mismatched_dates_rejected(self):
        inputs = list(self.inputs())
        inputs[-1] = {"signal_date": "2026-09-08", "status": "CASH"}
        with self.assertRaisesRegex(ValueError, "do not match"):
            build(*inputs)

    def test_frozen_policy_can_score_later_signal_date(self):
        market, baseline, ensemble, policy, review = self.inputs()
        policy["current_shadow_application"]["signal_date"] = "2026-09-08"
        card = build(market, baseline, ensemble, policy, review)
        self.assertTrue(card["paper_risk_veto"]["current_trigger_recomputed_from_frozen_threshold"])
        self.assertFalse(card["paper_risk_veto"]["triggered"])

    def test_conditional_recovery_is_distinct_from_buy_signal(self):
        recovery = {"promotion_gate_passed": True, "latest_research_prediction": {
            "signal_date": "2026-09-09", "recovery_probability": .53,
            "continued_risk_off_probability": .47, "top_class": "RECOVERY",
            "status": "RESEARCH_ONLY_NOT_FORWARD_VALIDATED"}}
        governance = {"uncertainty_stability_gate_passed": True,
                      "effective_historical_gate_passed": True}
        card = build(*self.inputs(), recovery=recovery, recovery_governance=governance)
        self.assertEqual(card["interpretation"]["conditional_transition"],
                         "RECOVERY_EDGE_BUT_NOT_TRADE_SIGNAL")
        self.assertFalse(card["trade_effect"]["direct_buy_authorized"])

    def test_failed_repair_quality_model_is_visible_but_not_actionable(self):
        quality = {"promotion_gate_passed": False, "latest_research_prediction": {
            "signal_date": "2026-09-09",
            "probabilities": {"CONTINUED_RISK_OFF": .34,
                              "WEAK_OR_NARROW_REPAIR": .40,
                              "EFFECTIVE_REPAIR": .26},
            "top_class": "WEAK_OR_NARROW_REPAIR",
            "status": "RESEARCH_ONLY_NOT_FORWARD_VALIDATED",
        }}
        card = build(*self.inputs(), repair_quality=quality)
        split = card["forecasts"]["repair_quality_split_challenger"]
        self.assertFalse(split["historical_gate_passed"])
        self.assertFalse(split["decision_authorized"])
        self.assertIn("REJECTED", card["interpretation"]["repair_quality_split"])
        self.assertIn("不参与仓位", markdown(card))

    def test_aggregate_damage_evidence_does_not_claim_relay_return_benefit(self):
        evidence = {
            "evidence_summary": {"aggregate_market_damage_gate_passed": True,
                "first_board_return_proxy_gate_passed": False,
                "relay_return_proxy_gate_passed": False, "complete_trade_count": 0},
            "policy_governance": {"blanket_relay_veto_promoted_to_live": False},
        }
        card = build(*self.inputs(), policy_evidence=evidence)
        scope = card["trade_effect"]["economic_evidence_scope"]
        self.assertTrue(scope["aggregate_market_damage_gate_passed"])
        self.assertFalse(scope["relay_return_proxy_gate_passed"])
        self.assertFalse(scope["blanket_relay_veto_promoted_to_live"])


if __name__ == "__main__":
    unittest.main()
