import unittest

import pandas as pd

from first_board_profit_link_audit_v126 import audit


class FirstBoardProfitLinkAuditV126Test(unittest.TestCase):
    def test_unfilled_top1_is_not_replaced(self):
        predictions = pd.DataFrame([
            {"signal_date": "2026-01-05", "code": "000001", "name": "A",
             "promotion_label": 1, "first_seal_minute": 10,
             "speed_probability": .9, "base_quality_probability": .9,
             "emotion_quality_probability": .9},
            {"signal_date": "2026-01-05", "code": "000002", "name": "B",
             "promotion_label": 0, "first_seal_minute": 20,
             "speed_probability": .1, "base_quality_probability": .1,
             "emotion_quality_probability": .1},
        ])
        outcomes = pd.DataFrame([
            {"signal_date": "2026-01-05", "code": "000001",
             "fill_status": "UNVERIFIABLE_LIMIT_QUEUE", "net_return_B_plus_1": None,
             "net_return_B_plus_2": None, "net_return_B_plus_3": None,
             "big_loss_B_plus_1": None},
            {"signal_date": "2026-01-05", "code": "000002",
             "fill_status": "FILLED_OPEN_PROXY", "net_return_B_plus_1": .1,
             "net_return_B_plus_2": .1, "net_return_B_plus_3": .1,
             "big_loss_B_plus_1": 0},
        ])
        selected, metrics = audit(predictions, outcomes)
        self.assertTrue((selected.code == "000001").all())
        self.assertEqual(metrics["base_quality"]["filled_trades"], 0)

    def test_missing_bar_is_not_in_fill_denominator(self):
        predictions = pd.DataFrame([
            {"signal_date": "2026-01-05", "code": "000001", "name": "A",
             "promotion_label": 0, "first_seal_minute": 10,
             "speed_probability": .9, "base_quality_probability": .9,
             "emotion_quality_probability": .9},
            {"signal_date": "2026-01-06", "code": "000001", "name": "A",
             "promotion_label": 1, "first_seal_minute": 10,
             "speed_probability": .9, "base_quality_probability": .9,
             "emotion_quality_probability": .9},
        ])
        outcomes = pd.DataFrame([
            {"signal_date": "2026-01-05", "code": "000001",
             "fill_status": "MISSING_B_BAR", "net_return_B_plus_1": None,
             "net_return_B_plus_2": None, "net_return_B_plus_3": None,
             "big_loss_B_plus_1": None},
            {"signal_date": "2026-01-06", "code": "000001",
             "fill_status": "FILLED_OPEN_PROXY", "net_return_B_plus_1": .02,
             "net_return_B_plus_2": .03, "net_return_B_plus_3": .04,
             "big_loss_B_plus_1": 0},
        ])
        _, metrics = audit(predictions, outcomes)
        result = metrics["base_quality"]
        self.assertEqual(result["missing_execution_evidence_sessions"], 1)
        self.assertEqual(result["execution_observed_sessions"], 1)
        self.assertEqual(result["fill_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
