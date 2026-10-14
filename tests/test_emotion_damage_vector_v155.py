import unittest

import numpy as np
import pandas as pd

from emotion_damage_vector_v155 import COMPONENTS, run
from emotion_three_regime_challenger_v86 import FEATURES


class EmotionDamageVectorV155Tests(unittest.TestCase):
    def frame(self, n=620):
        rng = np.random.default_rng(155)
        latent = rng.normal(size=n)
        data = pd.DataFrame({
            "date": pd.bdate_range("2023-01-02", periods=n),
            "emotion_state": np.where(latent > .8, "退潮", np.where(latent < -.8, "升温", "修复")),
        })
        for index, column in enumerate(FEATURES):
            data[column] = latent + rng.normal(scale=.8 + index / 100, size=n)
        data["failed_rate"] = 1 / (1 + np.exp(-latent))
        data["previous_limit_tail_loss"] = -.04 - .025 * latent
        data["down_limit_count"] = np.maximum(0, np.rint(8 + 5 * latent))
        data["yesterday_premium"] = .01 - .008 * latent
        data["promotion_mid"] = np.clip(.4 - .12 * latent, 0, 1)
        return data

    def test_builds_five_component_forecasts_without_stock_authority(self):
        report, predictions = run(self.frame())
        self.assertEqual(set(report["components"]), set(COMPONENTS))
        self.assertEqual(len(predictions), 186)
        self.assertFalse(report["aggregate_gate"]["stock_reranking_allowed"])
        self.assertIn(report["latest_research_prediction"]["damage_band"], {
            "HIGH_DAMAGE_RISK", "LOW_DAMAGE_RISK", "MIXED_DAMAGE_RISK", "UNAVAILABLE"})

    def test_missing_outcome_is_rejected(self):
        frame = self.frame().drop(columns=["failed_rate"])
        with self.assertRaisesRegex(ValueError, "damage outcome fields missing"):
            run(frame)

    def test_future_last_row_is_not_used_as_a_label(self):
        report, predictions = run(self.frame())
        self.assertEqual(report["split"]["development_n"] + report["split"]["evaluation_n"], 619)
        self.assertEqual(len(predictions), report["split"]["evaluation_n"])


if __name__ == "__main__":
    unittest.main()
