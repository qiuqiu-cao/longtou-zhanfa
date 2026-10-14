import unittest

import numpy as np
import pandas as pd

from emotion_economic_value_audit_v89 import audit, block_bootstrap_difference


class EmotionEconomicValueAuditTests(unittest.TestCase):
    def test_clear_monotonic_damage_passes_directions(self):
        n = 100
        dates = pd.date_range("2026-01-01", periods=n).strftime("%Y-%m-%d")
        p = np.linspace(.1, .9, n)
        predictions = pd.DataFrame({"target_date": dates, "ensemble_p_RISK_OFF": p})
        market = pd.DataFrame({
            "date": dates, "promotion_mid": 1 - p, "yesterday_premium": .05 - .1 * p,
            "failed_rate": p, "previous_limit_tail_loss": -.1 * p,
            "down_limit_count": 10 * p,
        })
        result = audit(predictions, market)
        self.assertEqual(result["direction_pass_count"], 5)
        self.assertTrue(result["economic_gate_passed"])

    def test_low_coverage_rejected(self):
        n = 100
        dates = pd.date_range("2026-01-01", periods=n).strftime("%Y-%m-%d")
        predictions = pd.DataFrame({"target_date": dates, "ensemble_p_RISK_OFF": .5})
        market = pd.DataFrame({"date": dates, "promotion_mid": np.nan,
                               "yesterday_premium": np.nan, "failed_rate": np.nan,
                               "previous_limit_tail_loss": np.nan, "down_limit_count": np.nan})
        with self.assertRaisesRegex(ValueError, "coverage"):
            audit(predictions, market)

    def test_bootstrap_requires_enough_tail_observations(self):
        result = block_bootstrap_difference(np.arange(5.), np.arange(5.))
        self.assertEqual(result["status"], "INSUFFICIENT")


if __name__ == "__main__":
    unittest.main()
