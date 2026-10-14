import copy
import unittest

import pandas as pd

from audit_historical_5min_v33 import inspect


class HistoricalMinuteAuditTests(unittest.TestCase):
    def setUp(self):
        day = "2024-01-17"
        times = [t.strftime("%Y%m%d%H%M%S") + "000"
                 for start, end in (("09:35", "11:30"), ("13:05", "15:00"))
                 for t in pd.date_range(day + " " + start, day + " " + end, freq="5min")]
        self.raw = dict(code="sz.000017", requested_date=day, error_code="0",
                        fields=["date", "time", "code", "open", "high", "low", "close", "volume", "amount", "adjustflag"],
                        rows=[[day, t, "sz.000017", "8.72", "8.72", "8.72", "8.72", "100", "872", "3"] for t in times])
        self.daily = dict(open=8.72, high=8.72, low=8.72, close=8.72)

    def test_match_does_not_establish_execution(self):
        result = inspect(self.raw, self.daily)
        self.assertEqual(result["status"], "DAILY_OHLC_MATCH_ONLY")
        self.assertEqual(result["rows"], 48)
        self.assertFalse(result["fill_verified"])
        self.assertFalse(result["auction_verified"])
        self.assertIsNone(result["exact_seal_time"])

    def test_incomplete_duplicate_and_reordered_bars(self):
        for change in (lambda r: r.pop(), lambda r: r.__setitem__(1, r[0]), lambda r: r.reverse()):
            raw = copy.deepcopy(self.raw)
            change(raw["rows"])
            self.assertEqual(inspect(raw, self.daily)["status"], "INVALID_IDENTITY_OR_COVERAGE")

    def test_identity_and_adjustment(self):
        for index, value in ((0, "2024-01-16"), (2, "sz.000759"), (9, "1")):
            raw = copy.deepcopy(self.raw)
            raw["rows"][0][index] = value
            self.assertEqual(inspect(raw, self.daily)["status"], "INVALID_IDENTITY_OR_COVERAGE")

    def test_invalid_prices_and_volume(self):
        for index, value in ((3, "nan"), (4, "8"), (5, "9"), (6, "0"), (7, "-1")):
            raw = copy.deepcopy(self.raw)
            raw["rows"][0][index] = value
            self.assertEqual(inspect(raw, self.daily)["status"], "INVALID_OHLCV")

    def test_wider_daily_range_is_consistent_and_preserved(self):
        daily = dict(self.daily, low=8.49)
        result = inspect(self.raw, daily)
        self.assertEqual(result["status"], "DAILY_OHLC_RANGE_CONSISTENT")
        self.assertEqual(result["differences"]["low"], dict(minute=8.72, daily=8.49))
        self.assertFalse(result["fill_verified"])

    def test_endpoint_difference_is_a_contradiction(self):
        daily = dict(self.daily, close=8.70)
        result = inspect(self.raw, daily)
        self.assertEqual(result["status"], "DAILY_OHLC_CONTRADICTION")
        self.assertEqual(result["differences"]["close"], dict(minute=8.72, daily=8.70))

    def test_empty(self):
        self.raw["rows"] = []
        self.assertEqual(inspect(self.raw, self.daily)["status"], "EMPTY")
