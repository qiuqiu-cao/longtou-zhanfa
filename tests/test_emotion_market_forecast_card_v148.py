import copy
import unittest

import pandas as pd

from emotion_market_forecast_card_v148 import build


def market(state="修复"):
    return pd.DataFrame([{
        "date": "2026-09-16", "emotion_state": state, "limit_count": 80,
        "consecutive_count": 11, "max_board": 6, "failed_count": 10,
        "down_limit_count": 4, "failed_rate": .1111, "seal_rate": .8889,
        "yesterday_premium": .031, "previous_limit_tail_loss": -.033,
        "promotion_mid": .2, "ladder_gap_count": 2, "middle_density": .667,
        "data_status": "verified", "source": "fixture",
        "source_timestamp": "2026-09-16T15:00:00+08:00",
    }])


def report(band="MIXED", probability=.44):
    return {
        "status": "DIAGNOSTIC_ONLY",
        "outer_evaluation": {"risk_warning_gate_passed": True},
        "latest_research_prediction": {
            "signal_date": "2026-09-16", "risk_band": band,
            "next_session_risk_off_probability": probability,
        },
        "policy": {"individual_stock_ranking_use_allowed": False},
    }


class EmotionMarketForecastCardTests(unittest.TestCase):
    def test_mixed_repair_has_explicit_uncertainty_and_no_three_day_claim(self):
        card = build(market(), report())
        self.assertEqual(card["next_session"]["primary_path"], "MIXED_ROTATION_OR_DIVERGENCE")
        self.assertFalse(card["leader_selection_effect"]["stock_reranking_allowed"])
        self.assertIsNone(card["two_to_three_session_outlook"]["probability"])

    def test_high_consensus_maps_to_divergence_risk_not_certainty(self):
        card = build(market("一致"), report("HIGH", .70))
        self.assertEqual(card["next_session"]["consistency_vs_divergence"], "偏分歧")
        self.assertFalse(card["entry_authorized"])

    def test_low_never_adds_stock_score(self):
        card = build(market("升温"), report("LOW", .20))
        self.assertIn("WARMING_OR_ACCELERATION", card["next_session"]["primary_path"])
        self.assertFalse(card["leader_selection_effect"]["positive_score_bonus_allowed"])

    def test_invalid_date_or_failed_gate_is_rejected(self):
        item = report()
        item["latest_research_prediction"]["signal_date"] = "2026-09-15"
        with self.assertRaises(ValueError):
            build(market(), item)
        item = report()
        item["outer_evaluation"]["risk_warning_gate_passed"] = False
        with self.assertRaises(ValueError):
            build(market(), item)


if __name__ == "__main__":
    unittest.main()
