import unittest

import numpy as np
import pandas as pd

from relay_recency_walkforward import weights, target_signature, paired_interval


class RecencyTests(unittest.TestCase):
    def setUp(self):
        self.sessions = [str(d.date()) for d in pd.bdate_range("2024-01-02", periods=70)]
        self.cutoff = "2024-05-01"
        self.frame = pd.DataFrame({"date": [self.sessions[0], self.sessions[0], self.sessions[63]],
            "target_date": [self.sessions[1], self.sessions[1], self.sessions[64]],
            "target": [0., 1., 1.], "label_valid": [True, True, True]})

    def test_equal_day_baseline_is_original_full_group_weight(self):
        w = weights(self.frame, self.cutoff, self.sessions, None)
        np.testing.assert_array_equal(w, [.5, .5, 1.])

    def test_recent_day_weight_doubles_over_one_half_life(self):
        w = weights(self.frame, self.cutoff, self.sessions, 63)
        self.assertAlmostEqual(w.iloc[2] / (w.iloc[0] + w.iloc[1]), 2.)

    def test_total_mass_and_group_members_stay_fixed(self):
        before = self.frame.copy(deep=True)
        for life in [None, 63, 126]:
            w = weights(self.frame, self.cutoff, self.sessions, life)
            self.assertAlmostEqual(w.sum(), 2.)
            self.assertEqual(w.iloc[0], w.iloc[1])
            self.assertTrue(w.gt(0).all())
        pd.testing.assert_frame_equal(before, self.frame)

    def test_future_and_missing_maturity_rejected(self):
        for value in [self.cutoff, None]:
            frame = self.frame.copy()
            frame.loc[0, "target_date"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "future or unresolved"):
                weights(frame, self.cutoff, self.sessions, 63)

    def test_unregistered_weight_parameter_rejected(self):
        with self.assertRaisesRegex(ValueError, "unregistered"):
            weights(self.frame, self.cutoff, self.sessions, 20)

    def test_missing_targets_do_not_create_false_change(self):
        a = {"signal_date": "2026-08-31", "code": "600001", "target_date": float("nan"), "truth": None}
        b = {**a, "target_date": float("nan")}
        self.assertEqual(target_signature([a]), target_signature([b]))

    def test_duplicate_cohort_does_not_change_denominator(self):
        a = {"signal_date": "2026-08-31", "code": "600001", "target_date": None, "truth": None}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            target_signature([a, a])

    def test_undefined_bootstrap_draws_are_not_silently_dropped(self):
        a = {"signal_date": "2024-01-02", "code": "600001", "target_date": "2024-01-03", "truth": None, "prediction": False}
        result = paired_interval([a], [a], self.sessions,
            {"start": "2024-01-02", "end": "2024-01-05"},
            {"draws": 8, "official_session_block": 5, "seed": 7, "lower_tail": .01, "upper_tail": .99})
        self.assertEqual(result["undefined_draws"], 8)
        self.assertIsNone(result["lower"])


if __name__ == "__main__":
    unittest.main()
