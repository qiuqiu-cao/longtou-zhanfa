import unittest

import numpy as np
import pandas as pd

from seal_speed_tradeoff_audit_v160 import audit


class SealSpeedTradeoffAuditV160Tests(unittest.TestCase):
    def inputs(self):
        features, outcomes, keys = [], [], []
        for day in range(20):
            date = (pd.Timestamp("2026-01-01") + pd.offsets.BDay(day)).strftime("%Y-%m-%d")
            for index in range(6):
                fast = int(index < 2)
                code = f"{day:03d}{index:03d}"
                features.append({
                    "signal_date": date, "code": code, "fast_seal": fast,
                    "first_seal_minute": 15 + index if fast else 40 + index,
                })
                # Fast boards promote more, but have worse B+2 and loss risk.
                outcomes.append({
                    "signal_date": date, "code": code,
                    "promotion_label": int(fast or index == 2),
                    "fill_status": "FILLED_OPEN_PROXY",
                    "net_return_B_plus_2": -.05 if fast else .01,
                    "big_loss_B_plus_1": 1 if fast else 0,
                })
                keys.append({"signal_date": date, "code": code})
        return pd.DataFrame(features), pd.DataFrame(outcomes), pd.DataFrame(keys)

    def test_speed_can_promote_more_without_becoming_a_buy_factor(self):
        _, report = audit(*self.inputs(), bootstrap_repetitions=100)
        self.assertGreater(report["fast_minus_not_fast"]["promotion_rate"], 0)
        self.assertLess(report["fast_minus_not_fast"]["b2_mean_net_return"], 0)
        self.assertGreater(report["fast_minus_not_fast"]["b1_big_loss_rate"], 0)
        self.assertFalse(report["governance"]["fast_seal_positive_ranking_factor_allowed"])
        self.assertEqual(report["governance"]["allowed_role"], "ATTENTION_OR_IDENTITY_CLUE_ONLY")

    def test_missing_key_coverage_is_rejected(self):
        features, outcomes, keys = self.inputs()
        features = features.iloc[:-1]
        with self.assertRaisesRegex(ValueError, "fully covered"):
            audit(features, outcomes, keys, bootstrap_repetitions=10)

    def test_duplicate_prequential_key_is_rejected(self):
        features, outcomes, keys = self.inputs()
        keys = pd.concat([keys, keys.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "unique"):
            audit(features, outcomes, keys, bootstrap_repetitions=10)

    def test_missing_b_bar_is_unavailable_not_an_unfilled_trade(self):
        features, outcomes, keys = self.inputs()
        mask = outcomes.signal_date.eq(outcomes.signal_date.max())
        outcomes.loc[mask, "fill_status"] = "MISSING_B_BAR"
        outcomes.loc[mask, ["net_return_B_plus_2", "big_loss_B_plus_1"]] = np.nan
        _, report = audit(features, outcomes, keys, bootstrap_repetitions=20)
        self.assertEqual(report["missing_execution_evidence_rows"], int(mask.sum()))
        self.assertEqual(report["complete_execution_sessions"], 19)
        fast = report["groups"]["fast"]
        self.assertEqual(fast["execution_observed_n"], 38)
        self.assertEqual(fast["filled_n"], 38)
        self.assertEqual(fast["fill_rate"], 1.0)
        self.assertFalse(report["governance"]["missing_bar_is_unfilled_trade"])


if __name__ == "__main__":
    unittest.main()
