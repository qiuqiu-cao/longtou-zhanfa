import tempfile
import unittest
from unittest.mock import patch

import freeze_holiday_review_v280 as builder
from forecast_review_bridge_v277 import audit
from next_session_expectation_contract_v271 import now, register, settle
from test_holiday_forecast_clocks_v280 import holiday_plan


class BuilderTests(unittest.TestCase):
    def test_exact_pool_and_binding_without_trade_authorization(self):
        review, plan = builder.build()
        self.assertEqual(len(plan["candidates"]), 11)
        self.assertIsNone(plan["primary_candidate"])
        # A ranking first is retained, but cannot turn a binary FAIL into PROMOTE.
        self.assertEqual(review["model_nonword_rank_first"], "002242")
        self.assertIsNone(plan["non_one_word_primary_candidate"])
        with tempfile.TemporaryDirectory() as directory:
            _, registered = register(plan, directory, now().isoformat())
            binding = audit(review, registered)
        self.assertEqual(binding["status"], "BOUND")
        self.assertFalse(binding["trade_authorized"])

    def test_source_hash_failure_prevents_plan(self):
        with patch.object(builder, "sha", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "snapshot source hash mismatch"):
                builder.build()

    def test_omitted_model_candidate_prevents_freeze(self):
        original = builder.pd.read_csv
        def incomplete(path, *args, **kwargs):
            frame = original(path, *args, **kwargs)
            if str(path).endswith("relay_predictions_LABEL_ONLY.csv"):
                frame = frame[~((frame.date == builder.SIGNAL) & (frame.code == "000678"))]
            return frame
        with patch.object(builder.pd, "read_csv", side_effect=incomplete):
            with self.assertRaisesRegex(ValueError, "denominator"):
                builder.build()

    def test_prediction_metrics_settle_without_execution_selection(self):
        plan = holiday_plan()
        with tempfile.TemporaryDirectory() as directory:
            _, registered = register(plan, directory, "2026-10-02T16:01:00+08:00")
        source = dict(status="verified", source="TEST_ONLY", observed_at="2026-10-08T15:00:00+08:00",
                      available_at="2026-10-08T15:05:00+08:00")
        outcome = dict(date="2026-10-08", market={**source, "primary": "RISK_ON"},
            limit_pool={**source, "scope": "MAINBOARD_NON_ST_CLOSING_LIMIT_UP", "complete": True,
                        "rows": [{"code": "600001", "boards": 3}]})
        result = settle(registered, outcome, "2026-10-08T15:10:00+08:00")
        self.assertTrue(result["identity_primary_promoted"])
        self.assertTrue(result["non_one_word_primary_promoted"])
        self.assertIsNone(result["primary_candidate_promoted"])
        self.assertFalse(result["trade_win_rate_claim_allowed"])


if __name__ == "__main__":
    unittest.main()
