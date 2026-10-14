import unittest

import numpy as np
import pandas as pd

from emotion_regime_ensemble_v87 import choose_weight


class EmotionRegimeEnsembleTests(unittest.TestCase):
    def test_weight_is_selected_only_from_supplied_validation(self):
        y = pd.Series(["RISK_ON", "RISK_OFF", "REPAIR"] * 30)
        challenger = np.tile(np.eye(3), (30, 1))
        markov = np.full((90, 3), 1 / 3)
        weight, grid = choose_weight(y, challenger, markov)
        self.assertEqual(weight, 1.0)
        self.assertEqual(len(grid), 11)

    def test_prefers_markov_when_challenger_is_wrong(self):
        y = pd.Series(["RISK_ON", "RISK_OFF", "REPAIR"] * 20)
        markov = np.tile(np.eye(3), (20, 1))
        challenger = np.roll(markov, 1, axis=1)
        weight, _ = choose_weight(y, challenger, markov)
        self.assertEqual(weight, 0.0)


if __name__ == "__main__":
    unittest.main()
