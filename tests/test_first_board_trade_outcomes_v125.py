import unittest

import pandas as pd

from first_board_trade_outcomes_v125 import build_outcomes, limit_price, net_return


class FirstBoardTradeOutcomesV125Test(unittest.TestCase):
    def test_limit_price_half_up(self):
        self.assertEqual(limit_price(7.05), 7.76)

    def test_upper_limit_open_is_not_assumed_filled(self):
        observations = pd.DataFrame([{ "signal_date": "2026-01-05", "code": "000001",
                                      "name": "A", "promotion_label": 1}])
        bars = pd.DataFrame([{ "date": "2026-01-06", "code": "000001", "open": 11,
                              "high": 11, "low": 10.5, "close": 10.8, "pre_close": 10,
                              "st": False, "new_listing": False,
                              "status_verified": True, "universe_complete": True}])
        calendar = pd.DataFrame({"date": ["2026-01-05", "2026-01-06"]})
        result = build_outcomes(observations, bars, calendar)
        self.assertEqual(result.iloc[0].fill_status, "UNVERIFIABLE_LIMIT_QUEUE")

    def test_open_proxy_net_return_uses_costs(self):
        self.assertLess(net_return(10, 10), 0)


if __name__ == "__main__":
    unittest.main()
