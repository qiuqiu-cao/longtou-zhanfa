import unittest

import pandas as pd

from relay_joint_policy_v8 import execution_metrics, passes, wilson


class RelayJointPolicyV8Tests(unittest.TestCase):
    def test_wilson_interval_is_ordered(self):
        low, high = wilson(7, 10)
        self.assertLess(low, 0.7)
        self.assertGreater(high, 0.7)

    def test_joint_gate_requires_loss_and_frequency_together(self):
        metrics = {
            "executed": 31, "executed_promotion_precision": 0.71,
            "large_loss_label_coverage": 1.0, "large_loss_rate": 0.11,
            "signals_per_active_week": 4.0,
        }
        self.assertFalse(passes(metrics))
        metrics["large_loss_rate"] = 0.09
        self.assertTrue(passes(metrics))

    def test_blocked_sell_counts_for_large_loss_but_not_fake_net_return(self):
        frame = pd.DataFrame([
            {"execution_status": "sell_blocked_one_word_down", "promoted_next_day": 0.0,
             "large_loss": 1.0, "net_return": None},
        ])
        metrics = execution_metrics(frame, active_dates=5)
        self.assertEqual(metrics["executed"], 1)
        self.assertEqual(metrics["large_loss_rate"], 1.0)
        self.assertEqual(metrics["unresolved_sell_blocked"], 1)


if __name__ == "__main__":
    unittest.main()
