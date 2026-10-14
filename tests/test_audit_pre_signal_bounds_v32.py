import unittest
import pandas as pd
from audit_pre_signal_bounds_v32 import bounds_at


class BoundTests(unittest.TestCase):
    def fixture(self):
        dates = pd.bdate_range("2024-01-01", periods=32).strftime("%Y-%m-%d").tolist()
        index = pd.DataFrame(dict(date=dates, close=100))
        bars = pd.DataFrame(dict(code="000001", date=dates, close=10, pre_close=10,
                                 status_verified=True, st=False, new_listing=False))
        return bars, index, "000001", dates[-2]

    def test_complete_flat_below_but_not_trade_permission(self):
        result = bounds_at(*self.fixture())
        self.assertTrue(result["current_below_both_conditional"])
        self.assertFalse(result["prior_events_ruled_out"])
        self.assertFalse(result["trading_permission"])

    def test_future_data_does_not_change_bound(self):
        bars, index, code, day = self.fixture()
        first = bounds_at(bars, index, code, day)
        bars.loc[31, "close"] = 100000
        self.assertEqual(first, bounds_at(bars, index, code, day))

    def test_missing_stock_date_unknown(self):
        bars, index, code, day = self.fixture()
        self.assertIsNone(bounds_at(bars.drop(index=10), index, code, day)["current_below_both_conditional"])

    def test_missing_index_price_unknown(self):
        bars, index, code, day = self.fixture()
        index.loc[10, "close"] = None
        self.assertEqual(bounds_at(bars, index, code, day)["status"], "DATA_LIMITED")

    def test_above_bound_not_confirmed_trigger(self):
        bars, index, code, day = self.fixture()
        bars["close"] = 11
        result = bounds_at(bars, index, code, day)
        self.assertFalse(result["current_below_both_conditional"])
        self.assertEqual(result["interpretation"], "cannot_decide_actual_trigger_without_reset_history")

    def test_status_unknown_not_certified(self):
        bars, index, code, day = self.fixture()
        bars.loc[10, "status_verified"] = False
        self.assertEqual(bounds_at(bars, index, code, day)["status"], "DATA_LIMITED")


if __name__ == "__main__":
    unittest.main()
