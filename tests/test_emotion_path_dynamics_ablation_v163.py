import unittest

import numpy as np
import pandas as pd

from emotion_path_dynamics_ablation_v163 import (
    DYNAMIC_SOURCES, add_dynamic_features, walk_forward_slices,
)


class EmotionPathDynamicsAblationV163Tests(unittest.TestCase):
    def frame(self, n=30):
        pattern = ["启动"] * 4 + ["大分歧"] * 3 + ["修复"] * 2
        states = (pattern * (n // len(pattern) + 1))[:n]
        frame = pd.DataFrame({
            "date": pd.bdate_range("2026-01-02", periods=n),
            "emotion_state": states,
        })
        for index, column in enumerate(DYNAMIC_SOURCES):
            frame[column] = np.arange(n, dtype=float) + index
        return frame

    def test_future_mutation_does_not_change_past_dynamic_features(self):
        first = self.frame()
        before, columns = add_dynamic_features(first)
        changed = first.copy()
        changed.loc[len(changed) - 1, list(DYNAMIC_SOURCES)] = 1e9
        after, _ = add_dynamic_features(changed)
        pd.testing.assert_frame_equal(
            before.loc[:len(before) - 2, columns],
            after.loc[:len(after) - 2, columns],
        )

    def test_state_age_resets_on_state_change(self):
        result, _ = add_dynamic_features(self.frame(9))
        self.assertEqual(result.dyn_state_age.tolist(), [1, 2, 3, 4, 1, 2, 3, 1, 2])

    def test_walk_forward_has_fixed_three_row_purge(self):
        folds = walk_forward_slices(676)
        self.assertEqual(len(folds), 6)
        for train, test in folds:
            self.assertEqual(test.start - train.stop, 3)
            self.assertEqual(test.stop - test.start, 60)

    def test_insufficient_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            walk_forward_slices(350)


if __name__ == "__main__":
    unittest.main()
