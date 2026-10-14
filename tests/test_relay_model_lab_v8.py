import unittest

import numpy as np
import pandas as pd

from relay_model_lab_v8 import (
    add_causal_stock_memory,
    add_cross_sectional_ranks,
    derive_fill_target,
)


class RelayModelLabV8Tests(unittest.TestCase):
    def test_cross_section_rank_uses_only_same_date(self):
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02"]),
            "turnover": [1.0, 2.0, 999.0], "turnover_vs_prev": [1, 2, 3],
            "amount_float_cap": [1, 2, 3], "amplitude": [1, 2, 3],
            "relative_strength": [1, 2, 3], "price": [1, 2, 3], "float_cap_log": [1, 2, 3],
        })
        ranked = add_cross_sectional_ranks(frame)
        self.assertEqual(ranked.loc[0, "turnover_rank_pct"], 0.5)
        self.assertEqual(ranked.loc[1, "turnover_rank_pct"], 1.0)
        self.assertEqual(ranked.loc[2, "turnover_rank_pct"], 1.0)

    def test_stock_memory_does_not_use_unmatured_future_labels(self):
        base = {
            "code": "600001", "promotion_target": 1.0, "two_more_target": 1.0,
            "large_loss_target": 0.0, "terminal_board": 5,
        }
        frame = pd.DataFrame([
            {**base, "date": pd.Timestamp("2026-01-01"), "promotion_maturity": pd.Timestamp("2026-01-02"),
             "two_more_maturity": pd.Timestamp("2026-01-10"), "large_loss_maturity": pd.Timestamp("2026-01-03")},
            {**base, "date": pd.Timestamp("2026-01-02"), "promotion_maturity": pd.Timestamp("2026-01-03"),
             "two_more_maturity": pd.NaT, "large_loss_maturity": pd.NaT},
        ])
        result = add_causal_stock_memory(frame)
        self.assertEqual(result.loc[1, "prior_promotion_count"], 1)
        self.assertEqual(result.loc[1, "prior_two_more_count"], 0)
        self.assertEqual(result.loc[1, "prior_large_loss_count"], 0)
        self.assertTrue(np.isnan(result.loc[1, "prior_two_more_rate"]))

    def test_fill_target_uses_trade_bar_not_exit_availability(self):
        frame = pd.DataFrame({
            "trade_touched_limit": [True, True, False, np.nan],
            "trade_one_word_up": [False, True, False, np.nan],
        })
        target = derive_fill_target(frame)
        self.assertEqual(target.iloc[0], 1.0)
        self.assertEqual(target.iloc[1], 0.0)
        self.assertEqual(target.iloc[2], 0.0)
        self.assertTrue(np.isnan(target.iloc[3]))


if __name__ == "__main__":
    unittest.main()
