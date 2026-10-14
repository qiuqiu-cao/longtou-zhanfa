import unittest

import pandas as pd

from public_live_preflight_v17 import evaluate


class PublicLivePreflightTests(unittest.TestCase):
    @staticmethod
    def candidates():
        return pd.DataFrame([{"code": "600001"}, {"code": "000001"}])

    @staticmethod
    def review(primary=None):
        plan = {"conditional_primary": primary, "maximum_acceptable_price": 11,
                "position_shares": 100, "holding_plan": {"time_stop": "B+3"}}
        row = {"code": "600001", "auction_conditions": ["a"],
               "opening_conditions": ["o"], "cancel_conditions": ["c"]}
        return {"signal_date": "2026-09-04", "selection": plan,
                "first_board_candidates": [row],
                "relay_candidates": [{"code": "000001"}]}

    def test_collection_can_be_ready_while_decision_is_cash(self):
        result = evaluate(self.candidates(), self.review(), self.candidates(),
                          "2026-09-07", "2026-09-04")
        self.assertTrue(result["collection_ready"])
        self.assertFalse(result["conditional_plan_ready"])
        self.assertEqual(result["status"], "COLLECTION_READY_DECISION_CASH")

    def test_complete_frozen_plan_is_conditional_not_direct_entry(self):
        result = evaluate(self.candidates(), self.review({"code": "600001"}), self.candidates(),
                          "2026-09-07", "2026-09-04")
        self.assertTrue(result["conditional_plan_ready"])
        self.assertFalse(result["direct_entry_authorized"])

    def test_missing_universe_code_blocks_collection(self):
        result = evaluate(self.candidates(), self.review(), pd.DataFrame([{"code": "600001"}]),
                          "2026-09-07", "2026-09-04")
        self.assertEqual(result["status"], "PREFLIGHT_FAILED")


if __name__ == "__main__":
    unittest.main()
