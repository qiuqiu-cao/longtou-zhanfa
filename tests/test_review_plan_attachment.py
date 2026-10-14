import unittest
from current_daily_review import attach_decision_plan
from tests.test_daily_decision_contract_v136 import closing_report


class AttachmentTest(unittest.TestCase):
    def review(self):
        return dict(signal_date="2026-09-07", asof="2026-09-07T09:35:00+08:00",
                    first_board_candidates=[], relay_candidates=[], status="CASH",
                    pool_summary=dict(closing_limit_mainboard=2, first_board=1))

    def test_missing_is_explicit_and_no_selection_created(self):
        result = attach_decision_plan(self.review())
        self.assertEqual(result["emotion_selection_plan"]["status"], "UNAVAILABLE")
        self.assertEqual(result["status"], "CASH")

    def test_time_and_pool_mismatch_rejected(self):
        plan = closing_report()
        plan["asof"] = "2026-09-07T16:00:00+08:00"
        plan["universe_summary"]["first_board_count"] = 3
        result = attach_decision_plan(self.review(), plan)
        issues = result["emotion_selection_plan"]["issues"]
        self.assertIn("PLAN_REVIEW_TIME_MISMATCH", issues)
        self.assertIn("PLAN_REVIEW_POOL_COUNT_MISMATCH", issues)

    def test_outside_candidate_not_promoted(self):
        plan = closing_report()
        plan["focus_candidates"] = [{"code": "000001", "boards": 1}]
        result = attach_decision_plan(self.review(), plan)
        self.assertIn("PLAN_CANDIDATE_OUTSIDE_REVIEW_POOL", result["emotion_selection_plan"]["issues"])
        self.assertEqual(result["status"], "CASH")
