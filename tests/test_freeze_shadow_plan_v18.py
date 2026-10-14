import unittest

from freeze_shadow_plan_v18 import freeze, next_limit


class ShadowPlanTests(unittest.TestCase):
    def review(self):
        return {
            "signal_date": "2026-09-08", "status": "CASH",
            "first_board_candidates": [],
            "relay_candidates": [{
                "code": "600354", "name": "敦煌种业", "boards": 3,
                "board_group": "CONSECUTIVE_BOARD", "active_hard_risks": [],
                "theme_status": "verified_same_day_group",
            }],
        }

    def test_price_rounding_and_fail_closed_flags(self):
        self.assertEqual(next_limit(10.48), 11.53)
        result = freeze(self.review(), "600354", 10.48, "2026-09-09", 100)
        self.assertEqual(result["selection"]["maximum_acceptable_price"], 11.53)
        self.assertEqual(result["status"], "WATCH_SHADOW_ONLY")
        self.assertFalse(result["shadow_plan"]["direct_entry_authorized"])
        self.assertEqual(result["shadow_plan"]["actual_position_shares"], 0)
        self.assertTrue(result["relay_candidates"][0]["cancel_conditions"])

    def test_hard_risk_rejected(self):
        review = self.review()
        review["relay_candidates"][0]["active_hard_risks"] = [{"type": "risk"}]
        with self.assertRaises(ValueError):
            freeze(review, "600354", 10.48, "2026-09-09")


if __name__ == "__main__":
    unittest.main()
