import unittest

import pandas as pd

from scan_target_events_v18 import scan_one


class ScanTests(unittest.TestCase):
    def fixture(self):
        dates = pd.bdate_range("2026-01-01", periods=40).strftime("%Y-%m-%d")
        closes = [10.0] * 32 + [10 * 1.1**n for n in range(1, 9)]
        bars = pd.DataFrame(dict(date=dates, close=closes, pre_close=[10]+closes[:-1]))
        benchmark = pd.DataFrame(dict(date=dates, close=100.0))
        return bars, benchmark

    def test_compound_crossing(self):
        bars, benchmark = self.fixture()
        result = scan_one(bars, benchmark)
        self.assertEqual(result.cross_100_proxy.sum(), 1)
        self.assertGreater(result.iloc[-1].max_suffix_10d_pct, 100)

    def test_missing_session_does_not_bridge(self):
        bars, benchmark = self.fixture()
        result = scan_one(bars.drop(index=36), benchmark)
        self.assertTrue(pd.isna(result.iloc[-1].max_suffix_10d_pct))

    def test_future_append_does_not_change_past(self):
        bars, benchmark = self.fixture()
        left = scan_one(bars.iloc[:35], benchmark.iloc[:35])
        right = scan_one(bars, benchmark).iloc[:35]
        pd.testing.assert_frame_equal(left, right)


if __name__ == "__main__":
    unittest.main()
