import unittest

import pandas as pd

from high_to_low_ecology_audit_v189 import build, prepare


class HighToLowEcologyAuditV189Tests(unittest.TestCase):
    def frame(self):
        rows = []
        dates = pd.bdate_range("2023-01-02", periods=500)
        for index, date in enumerate(dates):
            healthy = index % 4 == 0
            window = index % 2 == 0
            adverse_next_source = (index - 1) % 4 != 0
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "data_status": "verified",
                "state_divergence": int(adverse_next_source),
                "state_retreat": 0,
                "max_board": 4,
                "space_feedback": 0 if window else 1,
                "promotion_1to2": .40 if window else .05,
                "promotion_mid": .40 if healthy else .10,
                "previous_limit_median_return": .02 if healthy else -.01,
                "failed_rate": .15 if healthy else .45,
                "down_limit_count": 2 if healthy else 10,
            })
        return pd.DataFrame(rows)

    def test_next_session_fields_are_labels(self):
        prepared = prepare(self.frame())
        first = prepared.iloc[0]
        self.assertEqual(first.date, pd.Timestamp("2023-01-02"))
        self.assertEqual(first.next_date, pd.Timestamp("2023-01-03"))

    def test_development_threshold_is_not_changed_by_validation(self):
        first, _ = build(self.frame(), bootstrap_draws=100)
        changed = self.frame()
        changed.loc[400:, "promotion_1to2"] = .99
        second, _ = build(changed, bootstrap_draws=100)
        self.assertEqual(first["frozen_development_thresholds"],
                         second["frozen_development_thresholds"])

    def test_market_window_does_not_authorize_stock_selection(self):
        report, observations = build(self.frame(), bootstrap_draws=100)
        self.assertEqual(report["status"], "DIAGNOSTIC_ONLY")
        self.assertGreater(len(observations), 100)
        governance = report["governance"]
        self.assertTrue(governance[
            "historical_high_to_low_market_scenario_allowed"])
        self.assertFalse(governance["high_to_low_stock_selection_allowed"])
        self.assertFalse(governance["same_theme_supplement_mapping_allowed"])
        self.assertFalse(governance["entry_authorized"])
        self.assertEqual(report["latest_market_context"]["signal_date"],
                         "2024-11-29")

    def test_missing_ecology_field_is_excluded_not_zero_filled(self):
        frame = self.frame()
        frame.loc[0, "failed_rate"] = None
        prepared = prepare(frame)
        self.assertNotIn(pd.Timestamp("2023-01-02"), set(prepared.date))


if __name__ == "__main__":
    unittest.main()
