import unittest

import pandas as pd

from ladder_gap_tradeoff_v172 import build


class LadderGapTradeoffTests(unittest.TestCase):
    def fixture(self):
        candidates, outcomes, market = [], [], []
        for index in range(12):
            date = f"2025-01-{index + 2:02d}"
            code = f"600{index:03d}"
            damaged = index < 4
            healthy_gap = 4 <= index < 8
            gap = 2 if damaged or healthy_gap else 0
            candidates.append({
                "code": code, "signal_date": date, "boards": 4,
                "valid_identity": True,
            })
            outcomes.append({
                "code": code, "signal_date": date,
                "promotion_LABEL_ONLY": 0 if damaged else 1,
                "next_session_touched_limit_LABEL_ONLY": 0 if damaged else 1,
                "b2_net_close_mark_LABEL_ONLY": -.1 if damaged else .1,
                "b2_big_loss_mark_LABEL_ONLY": 1 if damaged else 0,
            })
            market.append({
                "date": date, "data_status": "verified", "max_board": 4,
                "ladder_gap_count": gap,
                "promotion_mid": .1 if damaged else .5,
                "previous_limit_tail_loss": -.08 if damaged else -.02,
            })
        return pd.DataFrame(candidates), pd.DataFrame(outcomes), pd.DataFrame(market)

    def test_gap_is_conditioned_on_middle_damage(self):
        report = build(*self.fixture())
        self.assertIn("GAP_2PLUS_DAMAGED", report["group_summaries"])
        self.assertIn("GAP_2PLUS_HEALTHY", report["group_summaries"])
        comparison = report["date_clustered_comparisons"][
            "promotion_damaged_vs_healthy_gap"]
        self.assertLess(comparison["estimate"], 0)
        self.assertTrue(report["ladder_gap_positive_ranking_factor_allowed"])

    def test_non_anchor_rows_are_excluded(self):
        candidates, outcomes, market = self.fixture()
        extra = candidates.iloc[[0]].copy()
        extra["code"] = "601999"
        extra["boards"] = 2
        candidates = pd.concat([candidates, extra], ignore_index=True)
        extra_outcome = outcomes.iloc[[0]].copy()
        extra_outcome["code"] = "601999"
        outcomes = pd.concat([outcomes, extra_outcome], ignore_index=True)
        report = build(candidates, outcomes, market)
        self.assertEqual(report["height_anchor_observations"], 12)

    def test_b2_is_explicitly_proxy_not_executable(self):
        report = build(*self.fixture())
        self.assertTrue(any("conditional" in item.lower()
                            for item in report["limitations"]))
        self.assertIn("not executable", report["interpretation"])

    def test_missing_fields_fail_closed(self):
        candidates, outcomes, market = self.fixture()
        with self.assertRaisesRegex(ValueError, "market fields missing"):
            build(candidates, outcomes, market.drop(columns=["promotion_mid"]))


if __name__ == "__main__":
    unittest.main()
