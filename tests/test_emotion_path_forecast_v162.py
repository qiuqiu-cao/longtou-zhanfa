import unittest

import numpy as np
import pandas as pd

from emotion_path_forecast_v162 import (
    BASE_CLASSES, PATH_CLASSES, build_targets, markov_path_probabilities,
    path_class, split_rows, transition_matrix,
)
from emotion_three_regime_challenger_v86 import FEATURES


class EmotionPathForecastV162Tests(unittest.TestCase):
    def test_path_definitions_are_mutually_exclusive(self):
        self.assertEqual(path_class(["RISK_OFF", "REPAIR", "RISK_OFF"]),
                         "RISK_OFF_DOMINANT")
        self.assertEqual(path_class(["RISK_ON", "REPAIR", "RISK_ON"]),
                         "RISK_ON_DOMINANT")
        self.assertEqual(path_class(["RISK_OFF", "REPAIR", "RISK_ON"]),
                         "REPAIR_TO_RISK_ON")
        self.assertEqual(path_class(["RISK_ON", "RISK_OFF", "REPAIR"]),
                         "MIXED_ROTATION")

    def test_every_three_state_sequence_has_one_known_path(self):
        import itertools
        values = [path_class(sequence) for sequence in itertools.product(BASE_CLASSES, repeat=3)]
        self.assertEqual(len(values), 27)
        self.assertTrue(set(values) <= set(PATH_CLASSES))

    def test_markov_path_probabilities_sum_to_one(self):
        states = pd.Series(["RISK_ON", "RISK_OFF", "REPAIR"] * 20)
        matrix = transition_matrix(states)
        probabilities = markov_path_probabilities(pd.Series(BASE_CLASSES), matrix)
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1))
        self.assertTrue((probabilities >= 0).all())

    def test_three_session_purge_prevents_target_overlap(self):
        states = ["启动", "大分歧", "修复", "升温"] * 110
        frame = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=len(states)),
            "emotion_state": states,
        })
        for index, feature in enumerate(FEATURES):
            frame[feature] = np.arange(len(frame), dtype=float) + index
        data = build_targets(frame)
        _, train, validation, blind, _, _ = split_rows(data)
        self.assertLess(train.path_end_date.max(), validation.date.min())
        self.assertLess(validation.path_end_date.max(), blind.date.min())

    def test_invalid_sequence_is_rejected(self):
        with self.assertRaises(ValueError):
            path_class(["RISK_ON", "REPAIR"])


if __name__ == "__main__":
    unittest.main()
