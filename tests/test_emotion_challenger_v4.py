import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import emotion_challenger_v4 as challenger


class EmotionChallengerV4Tests(unittest.TestCase):
    def _frame(self, rows: int = 12) -> pd.DataFrame:
        index = np.arange(rows)
        frame = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=rows).strftime("%Y-%m-%d"),
            "emotion_state": np.where(index % 3 == 0, "大分歧", "升温"),
            "loss_effect_health": np.where(index % 4 == 0, 0.2, 0.7),
            "promotion_mid": np.where(index % 5 == 0, 0.1, 0.5),
            "down_limit_count": np.where(index % 5 == 0, 8, 1),
            "emotion_adverse_markov_probability": 0.3,
        })
        for offset, feature in enumerate(challenger.FEATURES):
            if feature not in frame:
                frame[feature] = (index + offset) / (rows + offset + 1)
        return frame

    def test_last_labels_are_right_censored(self):
        labelled = challenger.add_targets(self._frame())
        self.assertTrue(math.isnan(labelled.iloc[-1]["adverse_state_next"]))
        self.assertTrue(math.isnan(labelled.iloc[-1]["relay_damage_next"]))

    def test_missing_next_damage_fields_do_not_become_safe_zero(self):
        frame = self._frame()
        frame.loc[5, ["loss_effect_health", "promotion_mid", "down_limit_count"]] = math.nan
        labelled = challenger.add_targets(frame)
        self.assertTrue(math.isnan(labelled.loc[4, "relay_damage_next"]))

    def test_future_mutation_does_not_change_earlier_prediction(self):
        frame = challenger.add_targets(self._frame(18))
        first, _, _ = challenger.causal_predictions(
            frame, "adverse_state_next", initial_history=6, retrain_every=3,
            l2=1.5, epochs=80, learning_rate=0.03,
        )
        mutated_raw = self._frame(18)
        mutated_raw.loc[14:, "emotion_state"] = "退潮"
        mutated_raw.loc[14:, "loss_effect_health"] = 0.05
        second, _, _ = challenger.causal_predictions(
            challenger.add_targets(mutated_raw), "adverse_state_next",
            initial_history=6, retrain_every=3, l2=1.5, epochs=80, learning_rate=0.03,
        )
        self.assertAlmostEqual(first[13], second[13], places=12)

    def test_latest_row_receives_prediction_without_future_label(self):
        frame = challenger.add_targets(self._frame(18))
        probabilities, _, _ = challenger.causal_predictions(
            frame, "relay_damage_next", initial_history=6, retrain_every=3,
            l2=1.5, epochs=80, learning_rate=0.03,
        )
        self.assertTrue(np.isfinite(probabilities[-1]))
        self.assertTrue(math.isnan(frame.iloc[-1]["relay_damage_next"]))

    def test_lock_is_immutable_and_verifiable(self):
        card = {
            "version": "4.0-research",
            "data_end": "2026-08-31",
            "features": challenger.FEATURES,
            "locked_parameters": challenger.LOCKED_PARAMETERS,
            "integrity": {"input_sha256": "input", "script_sha256": "script"},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            locked = challenger.create_lock(card, output)
            self.assertEqual(challenger.verify_lock(card, output)["status"], "PASS")
            changed = {**card, "locked_parameters": {**challenger.LOCKED_PARAMETERS, "l2": 9.0}}
            with self.assertRaises(RuntimeError):
                challenger.create_lock(changed, output)
            self.assertEqual(challenger.verify_lock(changed, output)["status"], "FAIL")
            self.assertEqual(locked["development_data_end"], "2026-08-31")


if __name__ == "__main__":
    unittest.main()
