import unittest

import numpy as np
import pandas as pd

from emotion_repair_quality_v118 import CLASSES, FEATURES, prepare, run


class RepairQualityTests(unittest.TestCase):
    def frame(self, n=620):
        states = ["大分歧", "退潮", "修复", "大分歧", "升温", "退潮", "加速"]
        rows = []
        for index in range(n):
            state = states[index % len(states)]
            row = {
                "date": pd.Timestamp("2023-01-02") + pd.offsets.BDay(index),
                "emotion_state": state,
                "previous_limit_median_return": .02 if index % 3 else -.02,
                "promotion_mid": .4 if index % 4 else .1,
                "failed_rate": .2 if index % 5 else .5,
                "down_limit_count": 2 if index % 7 else 10,
            }
            for offset, field in enumerate(FEATURES):
                row[field] = np.sin(index / (offset + 2)) + (index % 5) / 10
            # Restore target-definition columns if they are also model features.
            row.update({
                "promotion_mid": .4 if index % 4 else .1,
                "failed_rate": .2 if index % 5 else .5,
                "down_limit_count": 2 if index % 7 else 10,
            })
            rows.append(row)
        return pd.DataFrame(rows)

    def test_quality_label_uses_next_session_not_signal_session(self):
        frame = self.frame(20)
        data = prepare(frame)
        row = data.iloc[0]
        next_row = frame.iloc[1]
        score = sum([
            next_row.previous_limit_median_return > 0,
            next_row.promotion_mid >= .25,
            next_row.failed_rate <= .30,
            next_row.down_limit_count <= 5,
        ])
        self.assertEqual(row.target_quality_score, score)

    def test_chronological_split_has_label_embargo_and_no_authority(self):
        report, predictions = run(self.frame())
        self.assertLess(
            report["split"]["train_last_target_date"],
            report["split"]["blind_start_signal_date"],
        )
        self.assertFalse(report["decision_authorized"])
        self.assertEqual(set(CLASSES), set(
            field.removeprefix("challenger_p_")
            for field in predictions if field.startswith("challenger_p_")
        ))

    def test_invalid_quality_threshold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "score must be"):
            prepare(self.frame(20), 1)


if __name__ == "__main__":
    unittest.main()
