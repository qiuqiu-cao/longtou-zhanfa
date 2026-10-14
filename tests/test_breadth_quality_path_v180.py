import unittest

import numpy as np
import pandas as pd

from breadth_quality_path_v180 import build


class BreadthQualityPathV180Tests(unittest.TestCase):
    def frame(self, n=220):
        index = np.arange(n)
        return pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=n).strftime("%Y-%m-%d"),
            "data_status": "verified",
            "limit_count": 20 + (index % 5) * 4 + index // 5,
            "failed_rate": .2 + (index % 7) / 20,
            "state_divergence": (index % 9 == 0).astype(int),
            "state_retreat": (index % 13 == 0).astype(int),
            "promotion_mid": .2 + (index % 6) / 10,
            "previous_limit_median_return": -.01 + (index % 8) / 200,
            "down_limit_count": index % 10,
        })

    def test_three_future_sessions_are_shifted_not_current(self):
        rows, report = build(self.frame(), "2026-01-01")
        row = rows.iloc[0]
        source = self.frame()
        expected = []
        for session in (row.session_b1, row.session_b2, row.session_b3):
            item = source.loc[source.date.eq(session)].iloc[0]
            expected.append(bool(item.state_divergence or item.state_retreat))
        self.assertEqual(bool(row.adverse_b1), expected[0])
        self.assertEqual(bool(row.adverse_any_b1_b3), any(expected))
        self.assertTrue(report["path_contract"][
            "b1_b2_b3_are_market_sessions_not_candidate_trade_returns"])

    def test_requires_three_verified_future_sessions(self):
        frame = self.frame()
        frame.loc[100, "data_status"] = "sparse"
        rows, _ = build(frame, "2026-01-01")
        affected = set(frame.loc[97:101, "date"])
        self.assertTrue(affected.isdisjoint(set(rows.signal_date)))

    def test_split_comparisons_are_separate(self):
        _, report = build(self.frame(), "2026-01-01")
        self.assertIn("adverse_any_b1_b2", report["discovery_comparisons"])
        self.assertIn("adverse_any_b1_b3", report["validation_comparisons"])

    def test_never_turns_market_path_into_holding_or_entry(self):
        _, report = build(self.frame(), "2026-01-01")
        governance = report["governance"]
        self.assertFalse(governance["two_to_three_session_probability_claim_allowed"])
        self.assertFalse(governance["candidate_holding_period_inference_allowed"])
        self.assertFalse(governance["direct_stock_reranking_allowed"])
        self.assertFalse(governance["entry_authorized"])


if __name__ == "__main__":
    unittest.main()
