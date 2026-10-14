import unittest

import pandas as pd

from t1_unexitable_loss_audit_v201 import audit, rounded_limit


class T1UnexitableLossAuditV201Test(unittest.TestCase):
    def test_round_half_up_lower_limit(self):
        self.assertEqual(rounded_limit(10.05, -1), 9.05)

    def test_same_day_breach_is_not_executable_and_down_limit_queue_is_not_fill(self):
        outcomes = pd.DataFrame([
            {"signal_date": "2026-01-05", "code": "000001", "name": "A",
             "buy_date_B": "2026-01-06", "fill_status": "FILLED_OPEN_PROXY",
             "entry_price_gross": 10.0, "exit_date_B_plus_1": "2026-01-07",
             "net_return_B_plus_1": -.08},
            {"signal_date": "2026-01-05", "code": "000002", "name": "B",
             "buy_date_B": "2026-01-06", "fill_status": "FILLED_OPEN_PROXY",
             "entry_price_gross": 10.0, "exit_date_B_plus_1": "2026-01-07",
             "net_return_B_plus_1": -.12},
            {"signal_date": "2026-01-05", "code": "000003", "name": "C",
             "buy_date_B": "2026-01-06", "fill_status": "FILLED_OPEN_PROXY",
             "entry_price_gross": 10.0, "exit_date_B_plus_1": "2026-01-07",
             "net_return_B_plus_1": -.09},
        ])
        bars = pd.DataFrame([
            {"date": "2026-01-06", "code": "000001", "open": 10, "high": 10.2,
             "low": 9.2, "close": 9.8, "pre_close": 9.5},
            {"date": "2026-01-07", "code": "000001", "open": 9.5, "high": 9.8,
             "low": 9.4, "close": 9.6, "pre_close": 9.8},
            {"date": "2026-01-06", "code": "000002", "open": 10, "high": 10.1,
             "low": 9.6, "close": 10, "pre_close": 9.5},
            {"date": "2026-01-07", "code": "000002", "open": 9, "high": 9,
             "low": 9, "close": 9, "pre_close": 10},
            {"date": "2026-01-06", "code": "000003", "open": 10, "high": 10.1,
             "low": 9.7, "close": 10, "pre_close": 9.5},
            {"date": "2026-01-07", "code": "000003", "open": 9, "high": 9.4,
             "low": 9, "close": 9.3, "pre_close": 10},
        ])
        top1 = pd.DataFrame([{"signal_date": "2026-01-05", "code": "000001",
                              "ranker": "profit_emotion_probability"}])
        rows, report = audit(outcomes, bars, top1)
        indexed = rows.set_index("code")
        self.assertEqual(indexed.loc["000001", "B_intraday_breach_minus_7pct"], 1)
        self.assertFalse(indexed.loc["000001", "same_day_stop_executable"])
        self.assertEqual(indexed.loc["000002", "B_plus_1_exit_status"],
                         "ONE_PRICE_DOWN_LIMIT_UNSELLABLE")
        self.assertEqual(indexed.loc["000003", "B_plus_1_exit_status"],
                         "OPEN_DOWN_LIMIT_QUEUE_UNVERIFIABLE")
        self.assertTrue(pd.isna(indexed.loc[
            "000003", "B_plus_1_open_exit_net_return"]))
        self.assertFalse(report["governance"][
            "B_plus_1_down_limit_queue_may_be_assumed_filled"])
        self.assertFalse(report["governance"]["entry_authorized"])


if __name__ == "__main__":
    unittest.main()
