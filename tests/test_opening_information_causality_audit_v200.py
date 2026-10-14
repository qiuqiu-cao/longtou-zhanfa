import unittest

import pandas as pd

from opening_information_causality_audit_v200 import (
    audit, band_summary, opening_gap_band,
)


class OpeningInformationCausalityAuditV200Test(unittest.TestCase):
    def test_gap_bands_use_percent_boundaries(self):
        values = pd.Series([-.01, 0, .03, .06, .08])
        self.assertEqual(
            opening_gap_band(values).tolist(),
            ["LE_0", "LE_0", "GT_0_LE_5", "GT_5_LE_7", "GT_7"],
        )

    def test_band_summary_does_not_treat_missing_target_as_safe(self):
        frame = pd.DataFrame([
            {"signal_date": "2026-01-05", "opening_gap_band": "LE_0",
             "big_loss_target_available": 1, "big_loss_target": 1,
             "net_return_B_plus_1": -.1, "net_return_B_plus_2": -.2,
             "risk_emotion_balanced_probability": .8},
            {"signal_date": "2026-01-06", "opening_gap_band": "LE_0",
             "big_loss_target_available": 0, "big_loss_target": None,
             "net_return_B_plus_1": None, "net_return_B_plus_2": None,
             "risk_emotion_balanced_probability": .1},
        ])
        result = band_summary(frame)["LE_0"]
        self.assertEqual(result["rows"], 1)
        self.assertEqual(result["big_loss_rate"], 1)

    def test_official_open_feature_cannot_filter_same_open_fill(self):
        predictions = pd.DataFrame([{
            "signal_date": "2026-01-05", "code": "000001", "name": "A",
            "feature_available_at": "2026-01-05T15:05:00+08:00",
            "fill_status": "FILLED_OPEN_PROXY", "big_loss_target": 1,
            "big_loss_target_available": 1, "big_loss_B_plus_1": 1,
            "net_return_B_plus_1": -.1, "net_return_B_plus_2": -.2,
            "risk_base_probability": .2, "risk_emotion_probability": .3,
            "risk_emotion_balanced_probability": .4,
        }])
        outcomes = pd.DataFrame([{
            "signal_date": "2026-01-05", "code": "000001",
            "buy_date_B": "2026-01-06", "entry_rule": "B_OPEN_AUCTION_PROXY",
            "fill_status": "FILLED_OPEN_PROXY", "B_open_gap": .06,
            "B_open_upper_limit": 0, "B_one_price_upper_limit": 0,
        }])
        top1 = predictions[["signal_date", "code"]].copy()
        top1["ranker"] = "profit_emotion_probability"
        _, report = audit(predictions, outcomes, top1, {
            "technical_complete_sequences": 0, "full_complete_sequences": 0})
        self.assertFalse(report["availability_contract"][
            "opening_gap_can_filter_same_official_open_fill"])
        self.assertFalse(report["governance"]["09_35_entry_authorized"])
        self.assertEqual(report["scope"]["frozen_top1_big_loss_cases"], 1)


if __name__ == "__main__":
    unittest.main()
