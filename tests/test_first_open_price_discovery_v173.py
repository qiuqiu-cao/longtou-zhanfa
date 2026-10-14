import unittest

import pandas as pd

from first_open_price_discovery_v173 import build, identify


class FirstOpenPriceDiscoveryTests(unittest.TestCase):
    def fixture(self):
        dates = pd.date_range("2025-01-02", periods=20, freq="B").strftime("%Y-%m-%d")
        calendar = pd.DataFrame({"date": dates, "data_status": "verified"})
        features = []
        outcomes = []

        def add(date, code, boards, one_word, streak, turnover=.1):
            features.append({
                "date": date, "code": code, "boards": boards,
                "one_word_proxy": one_word, "one_word_streak": streak,
                "turnover": turnover, "bomb_count": 0,
                "first_seal_minute": 10, "last_seal_minute": 10,
            })
            outcomes.append({
                "signal_date": date, "code": code,
                "promotion_LABEL_ONLY": 1 if code != "600002" else 0,
                "next_session_touched_limit_LABEL_ONLY": 1,
                "b2_net_close_mark_LABEL_ONLY": .03 if code != "600002" else -.08,
                "b2_big_loss_mark_LABEL_ONLY": 0 if code != "600002" else 1,
            })

        add(dates[0], "600001", 2, 1, 1)
        add(dates[1], "600001", 3, 0, 0, .2)
        add(dates[0], "600002", 2, 1, 1)
        add(dates[1], "600002", 3, 1, 2)
        add(dates[2], "600002", 4, 0, 0, .3)
        add(dates[2], "600003", 2, 0, 0, .12)
        return pd.DataFrame(features), pd.DataFrame(outcomes), calendar

    def test_identifies_one_day_and_long_streak_first_open(self):
        features, _, calendar = self.fixture()
        rows = identify(features, calendar, "2025-02-28")
        by_code = {row.code: row.path_group for row in rows.itertuples()}
        self.assertEqual(by_code["600001"], "FIRST_OPEN_AFTER_1_ONE_PRICE")
        self.assertEqual(by_code["600002"], "FIRST_OPEN_AFTER_2PLUS_ONE_PRICE")
        self.assertEqual(by_code["600003"], "ORDINARY_TURNOVER_RELAY")

    def test_previous_session_must_be_consecutive_board(self):
        features, _, calendar = self.fixture()
        features.loc[features.code.eq("600001") & features.one_word_proxy.eq(0), "boards"] = 4
        rows = identify(features, calendar, "2025-02-28")
        item = rows[rows.code.eq("600001")].iloc[0]
        self.assertEqual(item.path_group, "ORDINARY_TURNOVER_RELAY")

    def test_outcomes_and_governance_remain_separate(self):
        report = build(*self.fixture(), cutoff_exclusive="2025-02-28")[1]
        self.assertFalse(report["governance"]["first_open_is_automatic_healthy_divergence"])
        self.assertFalse(report["governance"]["turnover_is_monotonic_positive_factor"])
        self.assertFalse(report["governance"]["entry_authorized"])
        self.assertIn("conditional daily opportunity proxies", report["limitations"][3])

    def test_missing_feature_fails_closed(self):
        features, outcomes, calendar = self.fixture()
        with self.assertRaisesRegex(ValueError, "feature fields missing"):
            build(features.drop(columns=["one_word_streak"]), outcomes, calendar,
                  cutoff_exclusive="2025-02-28")


if __name__ == "__main__":
    unittest.main()
