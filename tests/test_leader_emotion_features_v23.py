import unittest
import pandas as pd
from leader_emotion_features_v23 import FIELDS, join_emotion


class EmotionJoinTests(unittest.TestCase):
    def fixture(self):
        features = pd.DataFrame([dict(code="000017", signal_date="2024-01-17")])
        market = pd.DataFrame([{**{f:0.5 for f in FIELDS}, "date":"2024-01-17", "source":"fixture",
                                "source_timestamp":"2024-01-17 15:00:00", "data_status":"inferred",
                                "emotion_next_start_probability":0.99, "future_label":1}])
        return features, market

    def test_does_not_copy_old_forecast_or_label(self):
        result = join_emotion(*self.fixture())
        self.assertNotIn("market_future_label", result)
        self.assertNotIn("market_emotion_next_start_probability", result)
        self.assertEqual(result.iloc[0].market_field_coverage, 1)

    def test_missing_date_keeps_candidate_without_zero_fill(self):
        features, market = self.fixture()
        features["signal_date"] = "2024-01-18"
        result = join_emotion(features, market)
        self.assertEqual(len(result),1)
        self.assertTrue(pd.isna(result.iloc[0].market_failed_rate))

    def test_future_source_date_is_not_used(self):
        features, market = self.fixture()
        market["source_timestamp"] = "2024-01-18 15:00:00"
        result = join_emotion(features, market)
        self.assertEqual(result.iloc[0].market_field_coverage,0)

    def test_same_day_late_record_is_not_available_at_1505(self):
        features, market = self.fixture()
        market["source_timestamp"] = "2024-01-17 16:00:00"
        self.assertEqual(join_emotion(features, market).iloc[0].market_field_coverage,0)


if __name__ == "__main__":
    unittest.main()
