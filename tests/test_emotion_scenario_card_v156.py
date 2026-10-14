import unittest

import pandas as pd

from emotion_scenario_card_v156 import build


class EmotionScenarioCardV156Tests(unittest.TestCase):
    def market(self):
        return pd.DataFrame([{
            "date": "2026-09-16", "emotion_state": "修复", "limit_count": 80,
            "consecutive_count": 11, "max_board": 6, "failed_count": 10,
            "down_limit_count": 4, "failed_rate": .111, "seal_rate": .889,
            "yesterday_premium": .0096, "previous_limit_tail_loss": -.033,
            "promotion_mid": .2, "ladder_gap_count": 2, "middle_density": .91,
            "data_status": "verified", "source_timestamp": "2026-09-16 15:00:00",
        }])

    def risk(self, band="MIXED"):
        return {
            "status": "DIAGNOSTIC_ONLY",
            "outer_evaluation": {"risk_warning_gate_passed": True},
            "latest_research_prediction": {
                "signal_date": "2026-09-16", "next_session_risk_off_probability": .44,
                "risk_band": band,
            },
        }

    def damage(self, passed=False, band="UNAVAILABLE"):
        return {
            "status": "DIAGNOSTIC_ONLY",
            "aggregate_gate": {"passed": passed},
            "latest_research_prediction": {
                "signal_date": "2026-09-16", "damage_band": band,
                "components": {
                    "down_limit_spread": {
                        "probability": .22, "validated_for_aggregate": True,
                        "outcome": "down_limit_count", "damage_direction": "HIGH"},
                    "premium_loss": {
                        "probability": .39, "validated_for_aggregate": False,
                        "outcome": "yesterday_premium", "damage_direction": "LOW"},
                },
            },
        }

    def test_unready_damage_vector_cannot_raise_confidence(self):
        card = build(self.market(), self.risk(), self.damage())
        forecast = card["next_session_forecast"]
        self.assertEqual(forecast["confidence"], "V144_ONLY_DAMAGE_VECTOR_NOT_READY")
        self.assertEqual(forecast["damage_band"], "UNAVAILABLE")
        self.assertFalse(card["leader_selection_policy"]["emotion_direct_stock_reranking_allowed"])

    def test_aligned_high_risk_can_be_corroborated(self):
        card = build(self.market(), self.risk("HIGH"),
                     self.damage(True, "HIGH_DAMAGE_RISK"))
        self.assertEqual(card["next_session_forecast"]["confidence"], "CORROBORATED_HIGH")

    def test_unverified_market_is_rejected(self):
        market = self.market()
        market.loc[0, "data_status"] = "inferred"
        with self.assertRaisesRegex(ValueError, "aligned governed"):
            build(market, self.risk(), self.damage())


if __name__ == "__main__":
    unittest.main()
