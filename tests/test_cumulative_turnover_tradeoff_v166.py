import unittest

import numpy as np
import pandas as pd

from cumulative_turnover_tradeoff_v166 import (
    assign_tiers,
    build_cumulative_features,
    clustered_brier_delta_ci,
    learn_tier_thresholds,
)


class CumulativeTurnoverTradeoffV166Tests(unittest.TestCase):
    def test_episode_sum_uses_exact_consecutive_market_sessions(self):
        candidates = pd.DataFrame([
            {"code": "1", "signal_date": "2026-01-05", "boards": 3},
        ])
        daily = pd.DataFrame([
            {"code": "1", "date": "2026-01-01", "turnover": 1.0},
            {"code": "1", "date": "2026-01-02", "turnover": 2.0},
            {"code": "1", "date": "2026-01-05", "turnover": 3.0},
        ])
        row = build_cumulative_features(candidates, daily).iloc[0]
        self.assertEqual(row["episode_turnover"], 6.0)
        self.assertAlmostEqual(row["episode_average_turnover"], 2.0)
        self.assertAlmostEqual(row["turnover_acceleration_log"], np.log(1.5))

    def test_missing_trading_session_is_unavailable_not_zero(self):
        candidates = pd.DataFrame([
            {"code": "1", "signal_date": "2026-01-05", "boards": 3},
        ])
        daily = pd.DataFrame([
            {"code": "2", "date": "2026-01-01", "turnover": 9.0},
            {"code": "1", "date": "2026-01-02", "turnover": 2.0},
            {"code": "1", "date": "2026-01-05", "turnover": 3.0},
        ])
        row = build_cumulative_features(candidates, daily).iloc[0]
        self.assertTrue(np.isnan(row["episode_turnover"]))
        self.assertEqual(row["cumulative_turnover_status"], "unavailable_not_filled_zero")

    def test_tiers_are_learned_from_training_data(self):
        train = pd.DataFrame({
            "boards": [2] * 90,
            "episode_turnover": np.arange(1, 91, dtype=float),
        })
        thresholds = learn_tier_thresholds(train)
        test = pd.DataFrame({"boards": [2, 2, 2], "episode_turnover": [1, 45, 90]})
        self.assertEqual(assign_tiers(test, thresholds).tolist(), ["LOW", "MID", "HIGH"])

    def test_cluster_bootstrap_preserves_session_as_unit(self):
        pred = pd.DataFrame({
            "signal_date": ["2026-01-01"] * 2 + ["2026-01-02"] * 2,
            "target": [0, 1, 0, 1],
            "baseline_probability": [0.4, 0.6, 0.4, 0.6],
            "enhanced_probability": [0.2, 0.8, 0.2, 0.8],
        })
        result = clustered_brier_delta_ci(pred, repetitions=100)
        self.assertEqual(result["sessions"], 2)
        self.assertLess(result["ci95_high"], 0)


if __name__ == "__main__":
    unittest.main()
