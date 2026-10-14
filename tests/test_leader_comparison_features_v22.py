import unittest
import pandas as pd
from leader_comparison_features_v22 import build


class ComparisonTests(unittest.TestCase):
    def fixture(self):
        obs = pd.DataFrame([dict(code=c, signal_date="2026-01-05", episode_id=c, boards=b, start_known=k,
                                 future_max_board=99) for c,b,k in [("000001",4,True),("000002",3,True),("000003",8,False)]])
        bars = pd.DataFrame([dict(code=c,date="2026-01-05",open=11,high=11,low=11,close=11,pre_close=10,
                                  amount=100,float_cap=1000,turnover=2) for c in obs.code])
        return obs, bars

    def test_unknown_height_does_not_become_top(self):
        result = build(*self.fixture())
        self.assertEqual(result.iloc[0].known_pool_max_boards, 4)
        self.assertEqual(result.iloc[0].unknown_height_pool_count, 1)
        self.assertTrue(pd.isna(result.iloc[2].sole_known_top_proxy))
        self.assertNotIn("future_max_board", result)

    def test_one_price_is_only_proxy(self):
        result = build(*self.fixture())
        self.assertTrue(result.one_price_day_proxy.eq(1).all())
        self.assertNotIn("buyable", result)

    def test_missing_float_cap_not_zero_ratio(self):
        obs, bars = self.fixture()
        bars["float_cap"] = 0
        self.assertTrue(build(obs,bars).amount_to_float_cap.isna().all())

    def test_first_board_and_unknown_preserved(self):
        obs, bars = self.fixture()
        obs.loc[0, "boards"] = 1
        obs.loc[2, "boards"] = float("nan")
        result = build(obs, bars, full_limit_pool=True)
        self.assertEqual(result.board_group.tolist(), ["FIRST_BOARD", "CONSECUTIVE_BOARD", "UNKNOWN_LEFT_CENSORED"])
        self.assertTrue(pd.isna(result.iloc[2].boards_for_comparison))
        self.assertEqual(len(result), 3)

    def test_future_prices_and_outcomes_do_not_enter_features(self):
        obs, bars = self.fixture()
        obs.loc[0, "boards"] = 1
        before = build(obs, bars, full_limit_pool=True)
        future = bars.copy()
        future["date"] = "2026-01-06"
        future["close"] = 999
        obs["future_max_board"] = -999
        after = build(obs, pd.concat([bars, future]), full_limit_pool=True)
        pd.testing.assert_frame_equal(before, after)

    def test_known_missing_height_rejected(self):
        obs, bars = self.fixture()
        obs.loc[0, "boards"] = float("nan")
        with self.assertRaises(ValueError):
            build(obs, bars, full_limit_pool=True)


if __name__ == "__main__":
    unittest.main()
