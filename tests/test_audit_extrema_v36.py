import unittest
from audit_extrema_v36 import classify


class ExtremaTests(unittest.TestCase):
    def row(self, differences):
        return dict(code="000017", entry_date="2024-01-15", observation={},
                    differences=differences, limit_price_proxy=10)

    def test_missed_touch(self):
        r = classify(self.row(dict(high=dict(minute=9.98, daily=10))))
        self.assertEqual(r["state"], "MINUTE_RANGE_NARROWER")
        self.assertTrue(r["daily_limit_touch_not_in_minute"])
        self.assertFalse(r["exact_touch_time_usable"])

    def test_low_inward(self):
        r = classify(self.row(dict(low=dict(minute=9, daily=8.9))))
        self.assertEqual(r["state"], "MINUTE_RANGE_NARROWER")
        self.assertFalse(r["stop_fill_usable"])

    def test_outward_not_inward(self):
        r = classify(self.row(dict(high=dict(minute=10, daily=9.9))))
        self.assertEqual(r["state"], "OTHER_DISAGREEMENT")

    def test_open_mismatch_not_inward(self):
        r = classify(self.row(dict(open=dict(minute=9, daily=9.1))))
        self.assertEqual(r["state"], "OTHER_DISAGREEMENT")

    def test_match_not_execution(self):
        r = classify(self.row({}))
        self.assertEqual(r["state"], "MATCH_ONLY")
        self.assertFalse(r["stop_fill_usable"])

    def test_missing_not_match(self):
        self.assertEqual(classify(dict(code="000017", entry_date="2024-01-15"))["state"], "UNAVAILABLE")
