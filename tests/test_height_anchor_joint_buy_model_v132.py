import unittest

import numpy as np
import pandas as pd

from height_anchor_joint_buy_model_v132 import choose_threshold, daily_pick, joint_score


class HeightAnchorJointBuyModelV132Test(unittest.TestCase):
    def test_winner_missing_outcome_is_not_borrowed_from_runner_up(self):
        from height_anchor_transition_v130 import _daily_pick
        rows = pd.DataFrame({
            "signal_date": ["2026-01-05"] * 2,
            "code": ["000001", "000002"],
            "joint_score_not_calibrated": [.9, .8],
            "b2_positive_target": [np.nan, 1.0],
            "b2_net_close_mark_LABEL_ONLY": [np.nan, .12],
        })
        for picked in (daily_pick(rows), _daily_pick(rows, "joint_score_not_calibrated")):
            self.assertEqual(picked.iloc[0].code, "000001")
            self.assertTrue(pd.isna(picked.iloc[0].b2_positive_target))
            self.assertTrue(pd.isna(picked.iloc[0].b2_net_close_mark_LABEL_ONLY))

    def test_joint_score_is_conjunctive(self):
        high = joint_score(pd.Series([.8]), pd.Series([.8]), pd.Series([.8]), pd.Series([.1])).iloc[0]
        weak_profit = joint_score(pd.Series([.8]), pd.Series([.8]), pd.Series([.1]), pd.Series([.1])).iloc[0]
        self.assertGreater(high, weak_profit)
        self.assertTrue(np.isfinite(high))

    def test_daily_pick_never_substitutes_second_name(self):
        rows = pd.DataFrame([
            {"signal_date": "2026-01-05", "code": "000001", "joint_score_not_calibrated": .9,
             "execution_class": "QUEUE_UNVERIFIABLE"},
            {"signal_date": "2026-01-05", "code": "000002", "joint_score_not_calibrated": .8,
             "execution_class": "LIMIT_PRICE_PATH_PROXY"},
        ])
        picked = daily_pick(rows)
        self.assertEqual(picked.iloc[0].code, "000001")

    def test_threshold_stays_closed_with_too_few_fills(self):
        rows = pd.DataFrame({
            "execution_class": ["LIMIT_PRICE_PATH_PROXY"] * 2,
            "b2_positive_target": [1, 1], "b2_large_loss_target": [0, 0],
            "joint_score_not_calibrated": [.7, .8],
        })
        result = choose_threshold(rows, min_fills=3)
        self.assertIsNone(result["threshold"])
        self.assertEqual(result["status"], "INSUFFICIENT_PRIOR_FILLS")

    def test_threshold_cannot_claim_70_when_large_loss_gate_fails(self):
        rows = pd.DataFrame({
            "execution_class": ["LIMIT_PRICE_PATH_PROXY"] * 4,
            "b2_positive_target": [1, 1, 1, 0], "b2_large_loss_target": [0, 0, 0, 1],
            "joint_score_not_calibrated": [.9, .8, .7, .6],
        })
        result = choose_threshold(rows, min_fills=4, target_positive=.7, max_large_loss=.1)
        self.assertIsNone(result["threshold"])
        self.assertEqual(result["status"], "NO_THRESHOLD_MEETS_CHALLENGE")


if __name__ == "__main__":
    unittest.main()
