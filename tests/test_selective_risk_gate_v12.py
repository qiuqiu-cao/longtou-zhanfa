import unittest

import numpy as np
import pandas as pd

import selective_risk_gate_v12 as risk


class SelectiveRiskGateV12Tests(unittest.TestCase):
    def _frame(self, n_dates: int = 120) -> pd.DataFrame:
        rows = []
        for day, date in enumerate(pd.bdate_range("2025-01-01", periods=n_dates)):
            for rank in range(2):
                rows.append({
                    "date": date, "code": f"600{day % 100:02d}{rank}", "boards": 3,
                    "emotion_state": "升温", "buyable_eod": 1, "one_word_proxy": 0,
                    "promotion_target": int(rank == 0), "large_loss_target": int(day % 5 == 0 and rank == 1),
                    "net_return": 0.01 if rank == 0 else -0.02,
                    risk.P_PROMOTION: 0.7 - 0.1 * rank, risk.P_FILL: 0.6,
                    risk.P_TWO_MORE: 0.5 - 0.2 * rank, risk.P_LOSS: 0.2 + 0.2 * rank,
                })
        return pd.DataFrame(rows)

    def test_daily_top_is_unique_and_uses_joint_score(self):
        daily = risk.prepare_daily_top(self._frame())
        self.assertEqual(len(daily), 120)
        self.assertFalse(daily.duplicated("date").any())
        self.assertTrue(daily["code"].str.endswith("0").all())

    def test_split_has_two_embargo_dates_and_no_overlap(self):
        daily = risk.prepare_daily_top(self._frame())
        development, validation, blind, metadata = risk.chronological_split(daily)
        self.assertLess(development["date"].max(), validation["date"].min())
        self.assertLess(validation["date"].max(), blind["date"].min())
        self.assertNotIn(metadata["embargo_1"], set(development["date"]) | set(validation["date"]))
        self.assertNotIn(metadata["embargo_2"], set(validation["date"]) | set(blind["date"]))

    def test_threshold_selection_does_not_read_blind_labels(self):
        daily = risk.prepare_daily_top(self._frame(180))
        development, validation, blind, _ = risk.chronological_split(daily)
        first, _ = risk.choose_threshold(development, validation)
        blind = blind.copy()
        blind["large_loss_target"] = 1 - blind["large_loss_target"]
        second, _ = risk.choose_threshold(development, validation)
        self.assertEqual(first, second)

    def test_wilson_interval_contains_observed_rate(self):
        low, high = risk.wilson_interval(7, 50)
        self.assertLessEqual(low, 7 / 50)
        self.assertGreaterEqual(high, 7 / 50)


if __name__ == "__main__":
    unittest.main()
