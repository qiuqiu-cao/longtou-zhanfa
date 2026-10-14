import unittest
import numpy as np
import pandas as pd

from research.experience_stock_character import historical_features, paired_interval, wilson


class RecordedHistoryTests(unittest.TestCase):
    sessions = ["2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27", "2026-08-28", "2026-08-31", "2026-09-01", "2026-09-02"]

    def fixture(self):
        return pd.DataFrame([
            {"date": "2026-08-24", "target_date": "2026-08-25", "code": "000001", "label_valid": True, "target": 1.},
            {"date": "2026-08-25", "target_date": "2026-08-26", "code": "000001", "label_valid": True, "target": 0.},
            {"date": "2026-08-26", "target_date": "2026-08-27", "code": "000001", "label_valid": False, "target": np.nan},
            {"date": "2026-08-27", "target_date": "2026-08-28", "code": "000001", "label_valid": True, "target": 1.},
            {"date": "2026-08-31", "target_date": "2026-09-01", "code": "000001", "label_valid": True, "target": 1.},
            {"date": "2026-08-24", "target_date": "2026-08-25", "code": "000002", "label_valid": True, "target": 1.},
        ])

    def test_same_day_and_future_outcomes_excluded(self):
        rows = pd.DataFrame([{"date": "2026-08-28", "code": "000001"}])
        original = self.fixture()
        changed = original.copy()
        changed.loc[changed.target_date.ge("2026-08-28"), "target"] = 0.
        a = historical_features(rows, original, self.sessions)
        b = historical_features(rows, changed, self.sessions)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(a.prior_known_n.item(), 2)

    def test_unknown_outcome_not_failure_and_other_code_not_used(self):
        rows = pd.DataFrame([{"date": "2026-08-28", "code": "000001"}])
        feature = historical_features(rows, self.fixture(), self.sessions).iloc[0]
        self.assertEqual(feature.prior_recorded_n, 3)
        self.assertEqual(feature.prior_unknown_n, 1)
        self.assertEqual(feature.prior_successes, 1)
        self.assertAlmostEqual(feature.prior_relay_wilson_lower, wilson(1, 2)[0])

    def test_evaluation_freeze_excludes_newly_realized_september_history(self):
        rows = pd.DataFrame([{"date": "2026-09-02", "code": "000001"}])
        feature = historical_features(rows, self.fixture(), self.sessions, frozen_at="2026-09-01").iloc[0]
        self.assertEqual(feature.prior_known_n, 3)
        self.assertEqual(feature.prior_max_target_date, "2026-08-28")

    def test_zero_history_is_unknown_wilson_statistic(self):
        rows = pd.DataFrame([{"date": "2026-08-28", "code": "000003"}])
        feature = historical_features(rows, self.fixture(), self.sessions).iloc[0]
        self.assertTrue(np.isnan(feature.prior_relay_wilson_lower))
        self.assertEqual(feature.prior_known_n, 0)
        self.assertEqual(feature.log_prior_relay_known_n, 0)

    def test_calendar_window_uses_sessions_not_weekdays(self):
        sessions = pd.bdate_range("2025-01-01", periods=270).strftime("%Y-%m-%d").tolist()
        history = pd.DataFrame([
            {"date": sessions[0], "target_date": sessions[1], "code": "000001", "label_valid": True, "target": 0.},
            {"date": sessions[20], "target_date": sessions[21], "code": "000001", "label_valid": True, "target": 1.},
        ])
        f = historical_features(pd.DataFrame([{"date": sessions[269], "code": "000001"}]), history, sessions).iloc[0]
        self.assertEqual(f.prior_known_n, 1)
        self.assertEqual(f.prior_window_start, sessions[19])

    def test_bootstrap_retains_unknown_official_slots(self):
        daily = pd.DataFrame({"date": [self.sessions[0], self.sessions[2]], "difference": [1., -1.]})
        r = paired_interval(daily, self.sessions)
        self.assertEqual(r["official_calendar_slots"], 8)
        self.assertEqual(r["unknown_slots"], 6)
        self.assertLessEqual(r["interval95"][0], 0)
        self.assertGreaterEqual(r["interval95"][1], 0)


if __name__ == "__main__":
    unittest.main()
