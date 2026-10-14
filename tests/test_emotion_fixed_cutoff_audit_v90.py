import unittest

import pandas as pd

from emotion_fixed_cutoff_audit_v90 import run
from emotion_three_regime_challenger_v86 import FEATURES


class EmotionFixedCutoffAuditTests(unittest.TestCase):
    def test_cutoff_is_declared_inner_only(self):
        n = 720
        states = ["大分歧", "修复", "升温"] * (n // 3)
        rows = []
        for i, state in enumerate(states):
            row = {"date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=i), "emotion_state": state}
            for j, feature in enumerate(FEATURES):
                row[feature] = float((i + j) % 11)
            row.update({
                "emotion_next_start_probability": .05, "emotion_next_warming_probability": .10,
                "emotion_next_acceleration_probability": .10, "emotion_next_consensus_probability": .05,
                "emotion_next_divergence_probability": .25, "emotion_next_retreat_probability": .20,
                "emotion_next_repair_probability": .25,
                "promotion_mid": .3, "yesterday_premium": .01, "failed_rate": .3,
                "previous_limit_tail_loss": -.05, "down_limit_count": 5,
            })
            rows.append(row)
        report, evidence = run(pd.DataFrame(rows))
        self.assertFalse(report["nested_time_integrity"]["outer_blind_used_for_selection"])
        self.assertEqual(len(evidence), report["nested_time_integrity"]["outer_blind_n"])
        self.assertEqual(report["allowed_use_if_passed"], "RISK_VETO_RESEARCH_CANDIDATE_ONLY")

    def test_economic_outcome_is_shifted_to_target_session(self):
        n = 720
        states = ["大分歧", "修复", "升温"] * (n // 3)
        rows = []
        for i, state in enumerate(states):
            row = {"date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=i), "emotion_state": state}
            for j, feature in enumerate(FEATURES):
                row[feature] = float((i + j) % 11)
            row.update({
                "emotion_next_start_probability": .05, "emotion_next_warming_probability": .10,
                "emotion_next_acceleration_probability": .10, "emotion_next_consensus_probability": .05,
                "emotion_next_divergence_probability": .25, "emotion_next_retreat_probability": .20,
                "emotion_next_repair_probability": .25,
                "promotion_mid": i, "yesterday_premium": i, "failed_rate": i,
                "previous_limit_tail_loss": i, "down_limit_count": i,
            })
            rows.append(row)
        _, evidence = run(pd.DataFrame(rows))
        first = evidence.iloc[0]
        target_index = (pd.Timestamp(first.target_date) - pd.Timestamp("2023-01-01")).days
        self.assertEqual(first.promotion_mid, target_index)


if __name__ == "__main__":
    unittest.main()
