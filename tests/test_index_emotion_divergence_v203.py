import unittest

import pandas as pd

from index_emotion_divergence_v203 import (
    build_calendar, classify, group_summary, replication_gate,
)


class IndexEmotionDivergenceV203Test(unittest.TestCase):
    def test_fixed_sign_classification(self):
        result = classify(
            pd.Series([1, 1, -1, -1, 1]),
            pd.Series([1, 1, -1, -1, -1]),
            pd.Series([.01, 0, .01, -.01, .01]),
        )
        self.assertEqual(result.tolist(), [
            "BOTH_UP_SHORT_POSITIVE", "BOTH_UP_SHORT_LOSS",
            "BOTH_DOWN_SHORT_POSITIVE", "BOTH_DOWN_SHORT_LOSS",
            "MIXED_INDEX",
        ])

    def test_missing_emotion_session_is_not_skipped_as_next_day(self):
        emotion = pd.DataFrame([
            {"date": "2026-01-05", "emotion_state": "修复",
             "previous_limit_median_return": .01, "failed_rate": .1,
             "down_limit_count": 0, "promotion_mid": .3},
            {"date": "2026-01-07", "emotion_state": "退潮",
             "previous_limit_median_return": -.02, "failed_rate": .4,
             "down_limit_count": 5, "promotion_mid": .1},
        ])
        sh = pd.DataFrame({"date": ["2026-01-05", "2026-01-06", "2026-01-07"],
                           "benchmark_return_pct": [1, 1, -1]})
        sz = sh.copy()
        result = build_calendar(emotion, sh, sz).set_index("date")
        self.assertTrue(pd.isna(result.loc["2026-01-05", "next_adverse_state"]))
        self.assertEqual(result.loc["2026-01-06", "next_adverse_state"], 1)
        self.assertEqual(result.loc["2026-01-06", "emotion_row_available"], 0)

    def test_group_summary_uses_only_known_forward_labels(self):
        frame = pd.DataFrame([
            {"date": "2026-01-05", "divergence_category": "BOTH_UP_SHORT_LOSS",
             "next_adverse_state": 1, "next_supportive_state": 0,
             "next_three_any_adverse": 1,
             "next_previous_limit_median_return": -.02,
             "next_failed_rate": .4, "next_down_limit_count": 5,
             "next_promotion_mid": .1},
            {"date": "2026-01-06", "divergence_category": "BOTH_UP_SHORT_LOSS",
             "next_adverse_state": None, "next_supportive_state": None,
             "next_three_any_adverse": None,
             "next_previous_limit_median_return": None,
             "next_failed_rate": None, "next_down_limit_count": None,
             "next_promotion_mid": None},
        ])
        result = group_summary(frame)["BOTH_UP_SHORT_LOSS"]
        self.assertEqual(result["sessions"], 2)
        self.assertEqual(result["next_adverse_state"]["known"], 1)
        self.assertEqual(result["next_adverse_state"]["mean"], 1)

    def test_same_direction_does_not_pass_with_wide_intervals(self):
        item = {
            "group_a_n": 40, "group_b_n": 40,
            "observed_mean_difference_a_minus_b": .1,
            "weekly_cluster_bootstrap_ci95": [-.1, .3],
        }
        comparisons = {
            "discovery_first_70pct": {"metric": dict(item)},
            "validation_last_30pct": {"metric": dict(item)},
        }
        result = replication_gate(comparisons, ["metric"])["metric"]
        self.assertTrue(result["point_estimate_sign_consistent"])
        self.assertFalse(result["replication_gate_passed"])


if __name__ == "__main__":
    unittest.main()
