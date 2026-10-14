import unittest

import pandas as pd

from fast_board_structure_tradeoff_v194 import audit, classify_fast_structure


class FastBoardStructureTradeoffV194Tests(unittest.TestCase):
    def inputs(self):
        features, outcomes, keys = [], [], []
        definitions = [
            (1, 1, 0, "FILLED_OPEN_PROXY", 1, -.03, 1),
            (1, 0, 1, "FILLED_OPEN_PROXY", 1, .02, 0),
            (1, 0, 0, "UNVERIFIABLE_LIMIT_QUEUE", 0, None, None),
            (0, 0, 0, "FILLED_OPEN_PROXY", 0, .01, 0),
        ]
        for day in range(20):
            date = (pd.Timestamp("2026-01-01") + pd.offsets.BDay(day)).strftime("%Y-%m-%d")
            for index, definition in enumerate(definitions):
                fast, thin, validated, fill, promoted, b2, loss = definition
                code = f"{day:03d}{index:03d}"
                features.append({
                    "signal_date": date, "code": code, "fast_seal": fast,
                    "thin_control_risk": thin, "validated_fast_seal": validated,
                    "one_price_proxy": int(thin), "turnover": 2 if thin else 8,
                    "amount_to_float_cap": .01 if thin else .08,
                    "failed_seal_count": 0,
                })
                outcomes.append({
                    "signal_date": date, "code": code, "promotion_label": promoted,
                    "fill_status": fill, "net_return_B_plus_2": b2,
                    "big_loss_B_plus_1": loss,
                })
                keys.append({"signal_date": date, "code": code})
        return pd.DataFrame(features), pd.DataFrame(outcomes), pd.DataFrame(keys)

    def test_classification_does_not_claim_control_intent(self):
        self.assertEqual(classify_fast_structure(1, 1, 0),
                         "THIN_LIQUIDITY_OR_ONE_PRICE_PROXY")
        self.assertEqual(classify_fast_structure(1, 0, 1),
                         "TURNOVER_VALIDATED_FAST_PROXY")
        self.assertEqual(classify_fast_structure(0, 0, 0), "NOT_FAST")
        self.assertEqual(classify_fast_structure(1, 1, 1), "INVALID_OVERLAP")

    def test_identity_strength_does_not_authorize_entry(self):
        _, report = audit(*self.inputs(), bootstrap_repetitions=50)
        thin = report["groups"]["THIN_LIQUIDITY_OR_ONE_PRICE_PROXY"]
        self.assertEqual(thin["promotion_rate"], 1.0)
        self.assertLess(thin["b2_mean_net_return"], 0)
        self.assertFalse(report["governance"]["control_intent_inference_allowed"])
        self.assertFalse(report["governance"]["fast_structure_positive_ranking_factor_allowed"])
        self.assertFalse(report["governance"]["stock_entry_authorized"])

    def test_missing_bar_is_not_a_failed_fill(self):
        features, outcomes, keys = self.inputs()
        mask = outcomes.signal_date.eq(outcomes.signal_date.max())
        outcomes.loc[mask, "fill_status"] = "MISSING_B_BAR"
        outcomes.loc[mask, ["net_return_B_plus_2", "big_loss_B_plus_1"]] = None
        _, report = audit(features, outcomes, keys, bootstrap_repetitions=20)
        self.assertEqual(report["missing_execution_evidence_rows"], int(mask.sum()))
        self.assertEqual(report["complete_execution_sessions"], 19)
        self.assertFalse(report["governance"]["missing_bar_is_unfilled_trade"])

    def test_duplicate_keys_are_rejected(self):
        features, outcomes, keys = self.inputs()
        keys = pd.concat([keys, keys.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "unique"):
            audit(features, outcomes, keys, bootstrap_repetitions=10)


if __name__ == "__main__":
    unittest.main()
