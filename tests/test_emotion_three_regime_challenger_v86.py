import unittest

import numpy as np
import pandas as pd

from emotion_three_regime_challenger_v86 import CLASSES, FEATURES, metrics, prepare, run


class EmotionThreeRegimeChallengerTests(unittest.TestCase):
    def frame(self, n=360):
        states = ["大分歧", "修复", "升温"] * (n // 3)
        rows = []
        for i, state in enumerate(states):
            row = {"date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                   "emotion_state": state}
            for j, feature in enumerate(FEATURES):
                row[feature] = float((i + j) % 7)
            row.update({
                "emotion_next_start_probability": .05,
                "emotion_next_warming_probability": .10,
                "emotion_next_acceleration_probability": .10,
                "emotion_next_consensus_probability": .05,
                "emotion_next_divergence_probability": .25,
                "emotion_next_retreat_probability": .20,
                "emotion_next_repair_probability": .25,
            })
            rows.append(row)
        return pd.DataFrame(rows)

    def test_target_is_next_row(self):
        data = prepare(self.frame())
        self.assertEqual(data.iloc[0].target_state, "修复")
        self.assertEqual(data.iloc[0].target_class, "REPAIR")

    def test_metrics_reward_perfect_probabilities(self):
        y = pd.Series(CLASSES)
        result = metrics(y, np.eye(3))
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["multiclass_brier"], 0.0)

    def test_run_keeps_blind_predictions_separate(self):
        report, predictions = run(self.frame())
        self.assertEqual(report["split"]["train_n"] + report["split"]["blind_n"], 359)
        self.assertEqual(len(predictions), report["split"]["blind_n"])
        self.assertEqual(report["latest_research_prediction"]["status"],
                         "RESEARCH_ONLY_NOT_FORWARD_VALIDATED")


if __name__ == "__main__":
    unittest.main()
