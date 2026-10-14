import unittest

import pandas as pd

from first_board_promotion_v124 import build_dataset, enrich_emotion, hhmmss_to_minute


class FirstBoardPromotionV124Test(unittest.TestCase):
    def test_hhmmss_to_minute(self):
        self.assertEqual(hhmmss_to_minute("092500"), 10)
        self.assertEqual(hhmmss_to_minute(93000), 15)
        self.assertAlmostEqual(hhmmss_to_minute("093030"), 15.5)

    def test_only_immediate_next_session_labels(self):
        columns = ["code", "name", "boards", "amount", "float_cap", "turnover", "seal_fund",
                   "first_seal", "last_seal", "failed_seal_count", "industry", "signal_date"]
        day1 = pd.DataFrame([
            ["000001", "A", 1, 8e7, 1e9, 8, 2e7, "093000", "093100", 1, "X", "2026-01-05"]
        ], columns=columns)
        day2 = pd.DataFrame([
            ["000001", "A", 2, 9e7, 1.1e9, 9, 2e7, "093100", "093100", 0, "X", "2026-01-06"]
        ], columns=columns)
        day4 = day2.assign(signal_date="2026-01-08")
        calendar = pd.DataFrame({"date": ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]})
        rows, skipped = build_dataset({"2026-01-05": day1, "2026-01-06": day2,
                                       "2026-01-08": day4}, calendar)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows.iloc[0].promotion_label, 1)
        self.assertTrue(any(item["signal_date"] == "2026-01-06" for item in skipped))

    def test_emotion_is_joined_only_on_same_signal_date(self):
        rows = pd.DataFrame({"signal_date": ["2026-01-05", "2026-01-06"]})
        market = pd.DataFrame({
            "date": ["2026-01-05"], "emotion_state": ["升温"],
            "emotion_score": [0.7], "promotion_1to2": [0.3],
        })
        enriched = enrich_emotion(rows, market)
        self.assertEqual(enriched.loc[0, "emotion_supportive_regime"], 1)
        self.assertTrue(pd.isna(enriched.loc[1, "emotion_score"]))


if __name__ == "__main__":
    unittest.main()
