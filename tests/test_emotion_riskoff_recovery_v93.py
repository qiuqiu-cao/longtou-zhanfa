import unittest

import numpy as np
import pandas as pd

from emotion_riskoff_recovery_v93 import binary_metrics, run
from emotion_three_regime_challenger_v86 import FEATURES


class RiskoffRecoveryTests(unittest.TestCase):
    def frame(self, n=600):
        pattern = ["大分歧", "大分歧", "修复", "退潮", "退潮", "升温"]
        states = pattern * (n // len(pattern))
        rows = []
        for i, state in enumerate(states):
            row = {"date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                   "emotion_state": state}
            for j, feature in enumerate(FEATURES):
                row[feature] = float((i + j) % 13)
            rows.append(row)
        return pd.DataFrame(rows)

    def test_only_riskoff_signal_dates_are_evaluated(self):
        report, predictions = run(self.frame())
        self.assertTrue(set(predictions.current_state).issubset({"大分歧", "退潮"}))
        self.assertEqual(report["split"]["train_n"] + report["split"]["blind_n"],
                         len(self.frame().iloc[:-1].query("emotion_state in ['大分歧','退潮']")))

    def test_perfect_binary_probabilities(self):
        y = np.array([True, False, True, False])
        result = binary_metrics(y, np.array([1., 0., 1., 0.]))
        self.assertEqual(result["auc"], 1.0)
        self.assertEqual(result["brier"], 0.0)


if __name__ == "__main__":
    unittest.main()
