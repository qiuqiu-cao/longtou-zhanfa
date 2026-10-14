import unittest
from duration_reference_v41 import reference


class DurationReferenceTests(unittest.TestCase):
    def setUp(self):
        self.report = dict(training_exclusive_cutoff="2025-06-16",per_height={"3":dict(episodes=100,curve=[],restricted_mean_future_limit_sessions=1.2)})
        self.now = "2026-09-04T22:00:00+08:00"

    def test_not_holding_forecast(self):
        result = reference(3,self.report,self.now,self.now)
        self.assertEqual(result["historical_episodes"],100)
        self.assertIsNone(result["expected_holding_sessions"])
        self.assertIsNone(result["target_100_probability"])

    def test_missing_height_not_zero(self):
        self.assertEqual(reference(None,self.report,self.now,self.now)["evidence_state"],"UNKNOWN_BOARD_HEIGHT")

    def test_unseen_height(self):
        self.assertEqual(reference(9,self.report,self.now,self.now)["evidence_state"],"NO_HEIGHT_REFERENCE")

    def test_report_received_future(self):
        with self.assertRaises(ValueError):
            reference(3,self.report,self.now,"2026-09-05T00:00:00+08:00")

    def test_training_future(self):
        with self.assertRaises(ValueError):
            reference(3,self.report,"2024-01-01T00:00:00+08:00","2024-01-01T00:00:00+08:00")
