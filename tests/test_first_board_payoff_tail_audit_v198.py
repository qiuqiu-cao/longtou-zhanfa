import unittest

import pandas as pd

from first_board_payoff_tail_audit_v198 import audit, trade_metrics


class FirstBoardPayoffTailAuditV198Test(unittest.TestCase):
    def test_high_win_rate_can_still_have_negative_expectancy(self):
        frame = pd.DataFrame([
            {"signal_date": "2026-01-05", "fill_status": "FILLED_OPEN_PROXY",
             "net_return_B_plus_2": .05, "big_loss_B_plus_1": 0},
            {"signal_date": "2026-01-06", "fill_status": "FILLED_OPEN_PROXY",
             "net_return_B_plus_2": .05, "big_loss_B_plus_1": 0},
            {"signal_date": "2026-01-07", "fill_status": "FILLED_OPEN_PROXY",
             "net_return_B_plus_2": -.20, "big_loss_B_plus_1": 1},
        ])
        result = trade_metrics(frame)
        self.assertAlmostEqual(result["win_rate"], 2 / 3)
        self.assertLess(result["mean_net_return"], 0)
        self.assertLess(result["payoff_ratio"], 1)
        self.assertLess(result["profit_factor"], 1)

    def test_queue_is_not_replaced_or_counted_as_trade(self):
        frame = pd.DataFrame([
            {"signal_date": "2026-01-05", "fill_status": "UNVERIFIABLE_LIMIT_QUEUE",
             "net_return_B_plus_2": None, "big_loss_B_plus_1": None},
            {"signal_date": "2026-01-06", "fill_status": "FILLED_OPEN_PROXY",
             "net_return_B_plus_2": .05, "big_loss_B_plus_1": 0},
        ])
        result = trade_metrics(frame)
        self.assertEqual(result["selected_sessions"], 2)
        self.assertEqual(result["unverifiable_limit_queue_sessions"], 1)
        self.assertEqual(result["filled_trades"], 1)

    def test_audit_detects_higher_hit_negative_expectancy(self):
        rows = []
        for ranker, returns in {
            "profit_emotion_probability": [.05, .05, -.20],
            "base_quality_probability": [.05, -.05, -.05],
        }.items():
            for index, value in enumerate(returns):
                rows.append({
                    "ranker": ranker, "signal_date": f"2026-01-0{index + 5}",
                    "code": f"{ranker}-{index}", "fill_status": "FILLED_OPEN_PROXY",
                    "net_return_B_plus_2": value, "big_loss_B_plus_1": value <= -.07,
                })
        report = audit(pd.DataFrame(rows), bootstrap_repetitions=20)
        self.assertTrue(report["diagnosis"][
            "emotion_ranker_higher_hit_but_negative_expectancy"])
        self.assertFalse(report["governance"]["ranker_promotion_allowed"])


if __name__ == "__main__":
    unittest.main()
