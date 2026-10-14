import unittest

import numpy as np
import pandas as pd

from emotion_board_group_value_audit_v121 import audit


class EmotionBoardGroupValueAuditTests(unittest.TestCase):
    def inputs(self, sessions=500):
        dates = pd.bdate_range("2024-01-02", periods=sessions).strftime("%Y-%m-%d")
        feature_rows, outcome_rows, market_rows = [], [], []
        for index, date in enumerate(dates):
            probability = (index % 50) / 50
            market_rows.append({"date": date, "emotion_adverse_next_probability": probability,
                                "emotion_forecast_status": "USABLE"})
            for group_index, group in enumerate(("FIRST_BOARD", "CONSECUTIVE_BOARD")):
                code = f"{group_index + 1:06d}"
                feature_rows.append({"code": code, "signal_date": date, "board_group": group,
                                     "feature_status": "historical_reconstruction_not_forward_archive"})
                base = .05 - .10 * probability if group == "FIRST_BOARD" else .01
                outcome_rows.append({
                    "code": code, "signal_date": date,
                    "entry_one_price_LABEL_ONLY": False,
                    "price_path_state": "PRICE_OPPORTUNITY_PROXY_ONLY",
                    **{f"b{h}_net_close_mark_LABEL_ONLY": base - .005 * h for h in (1, 2, 3)},
                })
        return pd.DataFrame(feature_rows), pd.DataFrame(outcome_rows), pd.DataFrame(market_rows)

    def test_first_and_relay_are_reported_separately(self):
        report, rows = audit(*self.inputs())
        self.assertEqual(set(report["board_groups"]), {"FIRST_BOARD", "CONSECUTIVE_BOARD"})
        self.assertFalse(report["deployment_authorized"])
        self.assertEqual(report["complete_trade_count"], 0)
        self.assertTrue(set(rows.risk_group) >= {"HIGH", "LOW"})
        ci = report["board_groups"]["FIRST_BOARD"]["metrics"]["b1_mark"]["ci95"]
        self.assertTrue(all(-1 <= value <= 1 for value in ci))

    def test_one_price_candidates_are_excluded(self):
        features, outcomes, market = self.inputs()
        outcomes["entry_one_price_LABEL_ONLY"] = True
        with self.assertRaisesRegex(ValueError, "insufficient sessions"):
            audit(features, outcomes, market)

    def test_duplicate_candidate_key_is_rejected(self):
        features, outcomes, market = self.inputs()
        features = pd.concat([features, features.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate candidate"):
            audit(features, outcomes, market)


if __name__ == "__main__":
    unittest.main()
