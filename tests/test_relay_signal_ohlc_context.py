import unittest

from relay_signal_ohlc_context import qualify, cents
from archive_long_interval_v290 import daily_parse


class SignalOhlcTests(unittest.TestCase):
    def setUp(self):
        self.day = "2026-08-03"
        self.code = "600001"
        self.sessions = {self.day}

    def bar(self, **kwargs):
        return {"date": self.day, "code": self.code, "open": 10., "high": 10.,
            "low": 10., "close": 10., "available_at": "2026-10-02T11:00:00+00:00", **kwargs}

    def result(self, *bars):
        return qualify(self.day, self.code, 10., list(bars), self.sessions)

    def test_signal_one_price_is_only_retrospective(self):
        r = self.result(self.bar())
        self.assertIs(r["actual_one_word_daily"], True)
        self.assertTrue(r["diagnostic_eligible"])
        self.assertFalse(r["strict_original_time_eligible"])
        self.assertIn("fill", r["does_not_certify"])

    def test_exact_cent_spread_is_not_one_word(self):
        r = self.result(self.bar(open=9.99, low=9.99))
        self.assertIs(r["actual_one_word_daily"], False)

    def test_one_cent_above_pool_price_stays_unknown(self):
        r = self.result(self.bar(high=10.01))
        self.assertIsNone(r["actual_one_word_daily"])
        self.assertEqual(r["status"], "POOL_CLOSE_NOT_DAILY_HIGH")

    def test_conflicting_sources_do_not_pick_a_winner(self):
        r = self.result(self.bar(), self.bar(open=9.99, low=9.99))
        self.assertEqual(r["status"], "SOURCE_OHLC_CONFLICT")
        self.assertIsNone(r["actual_one_word_daily"])

    def test_target_day_and_other_stock_rejected(self):
        for extra in [{"date": "2026-08-04"}, {"code": "600002"}]:
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "substitution"):
                self.result(self.bar(**extra))

    def test_missing_and_missing_receipt_preserved(self):
        self.assertIsNone(self.result()["actual_one_word_daily"])
        self.assertIsNone(self.result(self.bar(available_at=None))["earliest_actual_receipt"])

    def test_official_date_and_subcent_required(self):
        with self.assertRaisesRegex(ValueError, "official"):
            qualify("2026-08-02", self.code, 10., [], self.sessions)
        with self.assertRaisesRegex(ValueError, "exact-cent"):
            cents(10.001)

    def test_adjusted_data_never_substitutes_for_day(self):
        with self.assertRaisesRegex(ValueError, "unadjusted"):
            daily_parse({"data": {"sh600001": {"qfqday": [[self.day, 10, 10, 10, 10, 1]]}}}, "sh600001")


if __name__ == "__main__":
    unittest.main()
