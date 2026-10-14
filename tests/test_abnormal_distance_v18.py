import unittest

import pandas as pd

from abnormal_distance_v18 import interval_deviation, window_summary, reconstruct, project_limit_path


class DistanceTests(unittest.TestCase):
    def test_compound_not_sum_and_not_relative_ratio(self):
        self.assertAlmostEqual(interval_deviation([10, 10], [1, 1]), 18.99)

    def test_suffix_exposes_hidden_rally_after_loss(self):
        summary = window_summary([-50] + [10]*8, [0]*9, 10)
        self.assertGreater(summary["maximum_suffix_deviation_pct"], 100)
        self.assertLess(summary["trailing_deviation_pct"], 100)

    def test_exact_threshold(self):
        self.assertEqual(interval_deviation([100], [0]), 100)

    def test_nonfinite_rejected(self):
        for bad in [float("nan"), float("inf"), -100]:
            with self.assertRaises(ValueError):
                interval_deviation([bad], [0])

    def fixture(self):
        dates = pd.bdate_range("2026-07-01", periods=35).strftime("%Y-%m-%d").tolist()
        return dates, pd.DataFrame([dict(date=d, stock_return_pct=1, benchmark_return_pct=0,
                                       eligible_session=1, reset_before=0, data_status="verified") for d in dates])

    def test_unknown_left_history_and_known_reset(self):
        dates, frame = self.fixture()
        result = reconstruct(frame, dates)
        self.assertFalse(result.iloc[0].coverage_complete)
        self.assertTrue(result.iloc[29].coverage_complete)
        frame.loc[0, "reset_before"] = 1
        self.assertTrue(reconstruct(frame, dates).iloc[0].coverage_complete)

    def test_missing_session_does_not_bridge_window(self):
        dates, frame = self.fixture()
        result = reconstruct(frame.drop(index=31), dates)
        self.assertEqual(result.iloc[31].indicator_status, "unavailable")
        self.assertFalse(result.iloc[32].coverage_complete)

    def test_no_limit_day_is_not_silently_zero(self):
        dates, frame = self.fixture()
        frame.loc[31, "eligible_session"] = 0
        self.assertEqual(reconstruct(frame, dates).iloc[31].indicator_status, "unavailable")

    def test_future_rows_do_not_change_past(self):
        dates, frame = self.fixture()
        left = reconstruct(frame.iloc[:30], dates[:30])
        right = reconstruct(frame, dates).iloc[:30].reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right)

    def test_projection_is_not_probability_and_post_breach_flagged(self):
        result = project_limit_path([0]*24 + [10]*6, [0]*30, [10, 10, 10], [0, 0, 0])
        self.assertFalse(result.iloc[0].scenario_threshold_hit)
        self.assertTrue(result.iloc[1].scenario_threshold_hit)
        self.assertTrue(result.iloc[2].after_prior_breach)
        self.assertNotIn("target_probability", result)


if __name__ == "__main__":
    unittest.main()
