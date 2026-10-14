import unittest

import pandas as pd

from profit_concentration_emotion_v177 import build


class ProfitConcentrationEmotionV177Tests(unittest.TestCase):
    def market(self):
        return pd.DataFrame([
            {"date": "2024-01-02", "data_status": "verified",
             "previous_limit_mean_return": .03, "previous_limit_median_return": -.01,
             "state_divergence": 0, "state_retreat": 0, "failed_rate": .10,
             "promotion_mid": .50, "down_limit_count": 1},
            {"date": "2024-01-03", "data_status": "verified",
             "previous_limit_mean_return": .04, "previous_limit_median_return": .02,
             "state_divergence": 1, "state_retreat": 0, "failed_rate": .40,
             "promotion_mid": .10, "down_limit_count": 5},
            {"date": "2024-01-04", "data_status": "verified",
             "previous_limit_mean_return": -.02, "previous_limit_median_return": -.03,
             "state_divergence": 0, "state_retreat": 0, "failed_rate": .20,
             "promotion_mid": .30, "down_limit_count": 2},
            {"date": "2024-01-05", "data_status": "verified",
             "previous_limit_mean_return": .02, "previous_limit_median_return": .01,
             "state_divergence": 0, "state_retreat": 1, "failed_rate": .50,
             "promotion_mid": .05, "down_limit_count": 6},
        ])

    def test_mean_positive_median_negative_is_narrow_not_supportive(self):
        rows, report = build(self.market(), "2024-01-10")
        first = rows.set_index("signal_date").loc["2024-01-02"]
        self.assertEqual(first.profit_state, "NARROW_POSITIVE")
        self.assertEqual(first.next_adverse_state, 1)
        self.assertFalse(report["governance"][
            "mean_premium_alone_sufficient_for_supportive_emotion"])
        self.assertTrue(report["governance"]["median_and_distribution_required"])

    def test_next_session_fields_are_shifted_not_same_day(self):
        rows, _ = build(self.market(), "2024-01-10")
        first = rows.set_index("signal_date").loc["2024-01-02"]
        self.assertAlmostEqual(first.next_failed_rate, .40)
        self.assertAlmostEqual(first.next_promotion_mid, .10)
        self.assertEqual(first.next_down_limit_count, 5)

    def test_last_row_without_next_session_is_not_filled_as_negative(self):
        rows, _ = build(self.market(), "2024-01-10")
        self.assertNotIn("2024-01-05", set(rows.signal_date))

    def test_duplicate_market_date_is_rejected(self):
        frame = pd.concat([self.market(), self.market().iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "unique"):
            build(frame, "2024-01-10")


if __name__ == "__main__":
    unittest.main()
