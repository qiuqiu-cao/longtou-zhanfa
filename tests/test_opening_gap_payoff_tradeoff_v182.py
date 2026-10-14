import unittest

import pandas as pd

from opening_gap_payoff_tradeoff_v182 import build


class OpeningGapPayoffTradeoffV182Tests(unittest.TestCase):
    def fixture(self):
        features, outcomes, bars = [], [], []
        dates = pd.bdate_range("2024-01-02", periods=220)
        gaps = [-1.0, 3.0, 6.0, 8.0]
        for index, signal in enumerate(dates):
            entry = signal + pd.offsets.BDay(1)
            for lane, gap in enumerate(gaps):
                code = f"{index * 4 + lane + 1:06d}"
                features.append({
                    "code": code, "signal_date": signal.strftime("%Y-%m-%d"),
                    "boards": 1 if lane < 2 else 2,
                    "board_group": "FIRST_BOARD" if lane < 2 else "CONSECUTIVE_BOARD",
                })
                touched = lane >= 1
                one_price = lane == 3 and index % 2 == 0
                b2 = .02 if lane == 2 else -.03
                outcomes.append({
                    "code": code, "signal_date": signal.strftime("%Y-%m-%d"),
                    "entry_date_LABEL_ONLY": entry.strftime("%Y-%m-%d"),
                    "next_session_touched_limit_LABEL_ONLY": touched,
                    "next_session_one_price_limit_LABEL_ONLY": one_price,
                    "b1_big_loss_mark_LABEL_ONLY": bool(lane == 3),
                    "b2_net_close_mark_LABEL_ONLY": b2 if touched and not one_price else None,
                    "actual_entry_verified": False,
                    "complete_trade_profit_LABEL_ONLY": None,
                })
                bars.append({
                    "code": code, "date": entry.strftime("%Y-%m-%d"),
                    "pre_close": 10.0, "open": 10.0 * (1 + gap / 100),
                })
        return pd.DataFrame(features), pd.DataFrame(outcomes), pd.DataFrame(bars)

    def test_no_touch_and_one_price_are_negative(self):
        rows, report = build(*self.fixture(), cutoff_exclusive="2026-01-01",
                             bootstrap_draws=30)
        self.assertTrue(report["target_contract"]["no_touch_rows_retained_as_negative"])
        self.assertTrue(report["target_contract"][
            "next_session_one_price_rows_retained_as_negative"])
        self.assertFalse(rows.loc[rows.opening_gap_band.eq("LE_0"), "target"].any())
        self.assertFalse(rows.loc[rows.one_price, "target"].any())

    def test_opening_gap_is_not_prior_close_or_direct_entry_factor(self):
        _, report = build(*self.fixture(), cutoff_exclusive="2026-01-01",
                          bootstrap_draws=30)
        self.assertFalse(report["feature_contract"]["prior_close_model_use_allowed"])
        self.assertFalse(report["feature_contract"]["auction_09_25_equivalence_claim_allowed"])
        self.assertEqual(
            report["feature_contract"]["entry_open_gap_available_not_before"],
            "after_opening_call_auction_match_approximately_09:25",
        )
        self.assertEqual(
            report["feature_contract"]["continuous_auction_begins_at"], "09:30")
        self.assertFalse(report["governance"]["opening_gap_direct_entry_allowed"])
        self.assertTrue(report["governance"][
            "requires_intraday_absorption_reseal_and_theme_confirmation"])
        # One-price rows may carry a future loss label but are not touchable entries;
        # they must not contaminate conditional filled-path loss rates.
        dev = report["summaries"]["ALL"]["development"]
        self.assertEqual(dev["GT_7"]["conditional_b1_big_loss_rate"], 1.0)

    def test_actual_entry_or_complete_trade_claim_is_rejected(self):
        features, outcomes, bars = self.fixture()
        outcomes.loc[0, "actual_entry_verified"] = True
        with self.assertRaisesRegex(ValueError, "verified entries"):
            build(features, outcomes, bars, bootstrap_draws=10)
        features, outcomes, bars = self.fixture()
        outcomes.loc[0, "complete_trade_profit_LABEL_ONLY"] = .1
        with self.assertRaisesRegex(ValueError, "complete trades"):
            build(features, outcomes, bars, bootstrap_draws=10)

    def test_duplicate_bar_is_rejected(self):
        features, outcomes, bars = self.fixture()
        bars = pd.concat([bars, bars.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "keys must be unique"):
            build(features, outcomes, bars, bootstrap_draws=10)


if __name__ == "__main__":
    unittest.main()
