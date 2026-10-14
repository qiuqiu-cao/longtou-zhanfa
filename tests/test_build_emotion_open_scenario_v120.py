import unittest

from build_emotion_open_scenario_v120 import build, classify_925, classify_935


class EmotionOpenScenarioTests(unittest.TestCase):
    def test_missing_fields_never_confirm_repair(self):
        self.assertEqual(classify_925({})["state"], "DATA_LIMITED")
        self.assertEqual(classify_935({})["state"], "DATA_LIMITED")

    def test_known_adverse_condition_can_close_gate_with_other_fields_missing(self):
        result = classify_925({"down_limit_indication_count": 10})
        self.assertEqual(result["state"], "CONTINUED_RISK_OFF")
        self.assertEqual(result["emotion_gate"], "CLOSED")

    def test_effective_repair_requires_all_six_0935_checks(self):
        values = {
            "previous_limit_return_median": .001,
            "early_failed_board_rate": .20,
            "down_limit_count": 2,
            "relay_positive_rate": .60,
            "verified_core_above_vwap_rate": .70,
            "verified_theme_sync_rate": .60,
        }
        self.assertEqual(classify_935(values)["state"], "EFFECTIVE_REPAIR_CONFIRMED_AT_0935")
        values["verified_theme_sync_rate"] = .49
        self.assertEqual(classify_935(values)["state"], "WEAK_OR_NARROW_REPAIR")

    def test_plan_keeps_direct_buy_disabled(self):
        card = {"signal_date": "2026-09-11", "paper_risk_veto": {"triggered": True},
                "forecasts": {"three_regime_nested_ensemble": {
                    "RISK_OFF": .50, "REPAIR": .28, "RISK_ON": .22}}}
        frozen = {"signal_date": "2026-09-11", "target_date": "2026-09-14", "outcome": None}
        plan = build(card, frozen)
        self.assertEqual(plan["preopen_default"], "RISK_OFF")
        self.assertFalse(plan["direct_buy_authorized"])
        self.assertEqual(plan["decision_effect"]["unattended_user_default"], "CASH_NO_PRESET_BLIND_ORDER")


if __name__ == "__main__":
    unittest.main()
