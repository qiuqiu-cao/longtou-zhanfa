import unittest

import pandas as pd

from full_pool_selective_policy_experiment import (
    MINIMUM_SELECTIONS, best_with_minimum, metrics, threshold_frontier,
)


class FullPoolSelectivePolicyExperimentTests(unittest.TestCase):
    def top1(self, n=40, successes=28):
        dates = pd.bdate_range("2025-01-02", periods=n).strftime("%Y-%m-%d")
        return pd.DataFrame([
            {"signal_date": dates[i], "code": f"{i:06d}",
             "boards": 1 if i % 2 else 2,
             "board_group": "FIRST_BOARD" if i % 2 else "CONSECUTIVE_BOARD",
             "model_score_NOT_LIVE": .4 if i < 35 else .2,
             "emotion_active": True,
             "opportunity_proxy_label": int(i < successes)}
            for i in range(n)
        ])

    def test_metrics_keeps_first_and_relay_separate(self):
        result = metrics(self.top1())
        self.assertEqual(result["first_board_selections"], 20)
        self.assertEqual(result["consecutive_board_selections"], 20)

    def test_frontier_uses_fixed_grid(self):
        result = threshold_frontier(self.top1())
        self.assertEqual(result[0]["threshold"], .1)
        self.assertEqual(result[-1]["threshold"], .45)

    def test_gate_requires_precision_and_minimum_sample(self):
        frontier = threshold_frontier(self.top1(n=40, successes=28))
        point = next(row for row in frontier if row["threshold"] == .4)
        self.assertGreaterEqual(point["selections"], MINIMUM_SELECTIONS)
        self.assertTrue(point["proxy_gate_pass"])

    def test_inactive_days_do_not_generate_signals(self):
        rows = self.top1()
        rows["emotion_active"] = False
        result = threshold_frontier(rows)
        self.assertTrue(all(row["selections"] == 0 for row in result))

    def test_high_precision_tiny_sample_fails(self):
        rows = self.top1(n=10, successes=10)
        result = threshold_frontier(rows)
        self.assertTrue(all(not row["proxy_gate_pass"] for row in result))

    def test_best_does_not_use_below_minimum(self):
        frontier = [
            {"minimum_sample_pass": False, "proxy_precision": 1., "selections": 2, "threshold": .4},
            {"minimum_sample_pass": True, "proxy_precision": .5, "selections": 30, "threshold": .2},
        ]
        self.assertEqual(best_with_minimum(frontier)["threshold"], .2)


if __name__ == "__main__":
    unittest.main()
