import unittest

import numpy as np
import pandas as pd

from emotion_model_lab_v7 import add_causal_dynamics, canonical_json, chronological_folds, evaluate


class EmotionModelLabV7Tests(unittest.TestCase):
    def test_causal_dynamics_do_not_change_when_future_is_mutated(self):
        frame = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=6),
            **{name: np.arange(6, dtype=float) for name in [
                "breadth_health", "relay_appetite", "ladder_health", "loss_effect_health",
                "crowding_risk", "regime_fragility", "promotion_mid", "failed_rate",
                "down_limit_count", "yesterday_premium", "previous_limit_tail_loss", "max_board",
            ]},
        })
        original, _ = add_causal_dynamics(frame)
        changed = frame.copy()
        changed.loc[5, "relay_appetite"] = 999
        rebuilt, _ = add_causal_dynamics(changed)
        pd.testing.assert_series_equal(original.loc[:4, "relay_appetite_delta_3d"], rebuilt.loc[:4, "relay_appetite_delta_3d"])

    def test_chronological_folds_are_ordered_and_disjoint(self):
        folds = chronological_folds(np.arange(10), 3)
        self.assertEqual(len(set(np.concatenate(folds))), 10)
        self.assertLess(folds[0].max(), folds[1].min())
        self.assertLess(folds[1].max(), folds[2].min())

    def test_joint_gate_requires_all_and_active_views(self):
        n = 12
        frame = pd.DataFrame({
            "adverse_state_next": [0, 1] * 6,
            "emotion_state": ["升温"] * n,
        })
        p = np.array([0.1, 0.9] * 6)
        baseline = np.full(n, 0.5)
        result = evaluate(frame, "adverse_state_next", p, baseline)
        self.assertTrue(result["all_date_gate"])
        self.assertTrue(result["active_gate"])
        self.assertTrue(result["joint_gate"])

    def test_canonical_lock_payload_is_order_independent(self):
        self.assertEqual(canonical_json({"b": 2, "a": 1}), canonical_json({"a": 1, "b": 2}))


if __name__ == "__main__":
    unittest.main()
