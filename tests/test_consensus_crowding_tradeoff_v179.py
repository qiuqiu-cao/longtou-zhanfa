import unittest

import numpy as np
import pandas as pd

from consensus_crowding_tradeoff_v179 import build


class ConsensusCrowdingTradeoffV179Tests(unittest.TestCase):
    def frame(self, n=240):
        index = np.arange(n)
        return pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=n).strftime("%Y-%m-%d"),
            "data_status": "verified",
            "one_word_share": .05 + (index % 20) / 40,
            "state_consensus": (index % 31 == 0).astype(int),
            "state_divergence": (index % 7 == 0).astype(int),
            "state_retreat": (index % 11 == 0).astype(int),
            "failed_rate": .2 + (index % 9) / 20,
            "promotion_mid": .2 + (index % 8) / 10,
            "previous_limit_median_return": -.02 + (index % 10) / 200,
        })

    def test_thresholds_are_frozen_on_development_dates(self):
        rows, report = build(self.frame(), "2026-01-01")
        split = report["split_date"]
        development = rows[rows.signal_date.lt(split)]
        self.assertAlmostEqual(
            report["development_thresholds"]["low_one_word_share_q30"],
            development.signal_one_word_share.quantile(.30))
        self.assertAlmostEqual(
            report["development_thresholds"]["high_one_word_share_q70"],
            development.signal_one_word_share.quantile(.70))

    def test_targets_use_immediately_next_verified_session(self):
        rows, _ = build(self.frame(), "2026-01-01")
        first = rows.iloc[0]
        source = self.frame()
        target = source.loc[source.date.eq(first.next_session)].iloc[0]
        self.assertAlmostEqual(first.next_failed_rate, target.failed_rate)
        self.assertAlmostEqual(first.next_promotion_mid, target.promotion_mid)

    def test_unverified_adjacent_session_removes_signal(self):
        frame = self.frame()
        removed_date = frame.loc[100, "date"]
        prior_date = frame.loc[99, "date"]
        next_date = frame.loc[101, "date"]
        frame.loc[100, "data_status"] = "sparse"
        rows, _ = build(frame, "2026-01-01")
        self.assertNotIn(removed_date, set(rows.signal_date))
        self.assertNotIn(prior_date, set(rows.signal_date))
        self.assertNotIn(next_date, set(rows.signal_date))

    def test_never_authorizes_stock_ranking_or_entry(self):
        _, report = build(self.frame(), "2026-01-01")
        self.assertFalse(report["governance"]["direct_stock_reranking_allowed"])
        self.assertFalse(report["governance"]["entry_authorized"])
        duplicate = pd.concat([self.frame(), self.frame().iloc[[0]]])
        with self.assertRaisesRegex(ValueError, "valid and unique"):
            build(duplicate, "2026-01-01")


if __name__ == "__main__":
    unittest.main()
