import unittest

import numpy as np
import pandas as pd

from intraday_shadow_v9 import FILL, PROMOTION, TWO_MORE, score_shadow


class IntradayShadowV9Tests(unittest.TestCase):
    def predictions(self):
        return pd.DataFrame({
            "date": ["2026-09-01", "2026-09-01", "2026-09-01"],
            "code": ["600001", "600002", "600003"],
            "name": ["A", "B", "C"], "boards": [3, 4, 2],
            "emotion_state": ["升温", "升温", "升温"],
            "valid_identity": [0, 0, 1], "buyable_eod": [1, 1, 1],
            "one_word_proxy": [0, 0, 0],
            PROMOTION: [0.7, 0.6, 0.9], FILL: [0.5, 0.5, 0.9],
            TWO_MORE: [0.8, 0.5, 0.9],
        })

    def test_only_daily_top_height_candidate_is_preregistered(self):
        scored = score_shadow(
            self.predictions(), pd.DataFrame(), "2026-09-01", "2026-09-02", "09:35",
        )
        selected = scored[scored["shadow_observation"]]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["code"], "600001")
        self.assertEqual(selected.iloc[0]["decision"], "SHADOW_ONLY_NEVER_ENTER")
        self.assertEqual(selected.iloc[0]["open_confirmation"], "DATA_LIMITED")
        self.assertTrue(np.isnan(selected.iloc[0]["large_loss_probability"]))

    def test_second_board_is_not_rescued_by_high_probability(self):
        scored = score_shadow(
            self.predictions(), pd.DataFrame(), "2026-09-01", "2026-09-02", "09:35",
        )
        second = scored[scored["code"].eq("600003")].iloc[0]
        self.assertFalse(second["shadow_scope"])
        self.assertEqual(second["open_confirmation"], "OUT_OF_RESEARCH_SCOPE")


if __name__ == "__main__":
    unittest.main()
