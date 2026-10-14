import unittest

import numpy as np
import pandas as pd

from emotion_recovery_uncertainty_v95 import audit


class RecoveryUncertaintyTests(unittest.TestCase):
    def test_strong_deterministic_signal_passes(self):
        y = np.array([True, False] * 100)
        frame = pd.DataFrame({"recovery_actual": y,
                              "challenger_recovery_probability": np.where(y, .9, .1),
                              "prior_recovery_probability": .5})
        result = audit(frame, repetitions=300)
        self.assertTrue(result["stable_gate_passed"])
        self.assertGreater(result["ci95"]["auc"][0], .5)

    def test_random_constant_model_fails(self):
        y = np.array([True, False] * 50)
        frame = pd.DataFrame({"recovery_actual": y,
                              "challenger_recovery_probability": .5,
                              "prior_recovery_probability": .5})
        result = audit(frame, repetitions=200)
        self.assertFalse(result["stable_gate_passed"])


if __name__ == "__main__":
    unittest.main()
