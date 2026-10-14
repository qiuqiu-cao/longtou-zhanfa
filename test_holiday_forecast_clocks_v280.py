from copy import deepcopy
import tempfile
import unittest
from test_next_session_expectation_v271 import plan
from next_session_expectation_contract_v271 import validate_plan, register
from forecast_review_bridge_v277 import audit, review_digest


def holiday_plan():
    p=plan()
    p.update(signal_date="2026-09-30",target_date="2026-10-08",
             evidence_cutoff="2026-09-30T19:18:24+08:00",knowledge_cutoff="2026-10-02T16:00:00+08:00",
             primary_candidate=None,highest_accept_price=None,
             identity_primary_candidate="600001",non_one_word_primary_candidate="600001")
    p["calendar"]={"status":"verified","source":"TEST_ONLY","sessions":["2026-09-30","2026-10-08"]}
    p["candidates"][0]["actual_one_word_daily"]=False
    p["evidence"]=[{"kind":"MARKET_SNAPSHOT","observed_at":"2026-09-30T15:00:00+08:00",
        "available_at":"2026-09-30T19:18:24+08:00","source":"TEST_ONLY","status":"delayed"},
        {"kind":"RESEARCH_DERIVED","observed_at":"2026-10-02T15:59:00+08:00",
         "available_at":"2026-10-02T15:59:00+08:00","source":"TEST_ONLY","status":"inferred"}]
    return p


class ClockTests(unittest.TestCase):
    def test_holiday_research_can_be_registered_without_backdating(self):
        r=validate_plan(holiday_plan(),"2026-10-02T16:01:00+08:00")
        self.assertEqual(r["forecast_status"],"FORWARD_REGISTERED")
        self.assertEqual(r["execution_state"],"PENDING_OPEN_CONFIRMATION")

    def test_new_knowledge_is_not_allowed_under_old_cutoff(self):
        p=holiday_plan();p.pop("knowledge_cutoff")
        with self.assertRaises(ValueError):validate_plan(p,"2026-10-02T16:01:00+08:00")

    def test_future_market_quote_cannot_be_relabelled_as_holiday_snapshot(self):
        p=holiday_plan();p["evidence"][1]["kind"]="MARKET_SNAPSHOT"
        with self.assertRaises(ValueError):validate_plan(p,"2026-10-02T16:01:00+08:00")
        p=holiday_plan();p["evidence"][1]["available_at"]="2026-10-08T09:25:00+08:00"
        with self.assertRaises(ValueError):validate_plan(p,"2026-10-02T16:01:00+08:00")

    def test_prediction_selection_does_not_need_order_price(self):
        p=holiday_plan();validate_plan(p,"2026-10-02T16:01:00+08:00")
        p["candidates"][0]["actual_one_word_daily"]=True
        with self.assertRaises(ValueError):validate_plan(p,"2026-10-02T16:01:00+08:00")
        p=holiday_plan();p["candidates"][0]["promotion_call"]="FAIL"
        with self.assertRaises(ValueError):validate_plan(p,"2026-10-02T16:01:00+08:00")

    def test_knowledge_clock_bounds_checked(self):
        for value in ["2026-09-30T18:00:00+08:00","2026-10-02T17:00:00+08:00"]:
            p=holiday_plan();p["knowledge_cutoff"]=value
            with self.assertRaises(ValueError):validate_plan(p,"2026-10-02T16:01:00+08:00")

    def test_review_created_during_holiday_can_bind(self):
        r={"asof":"2026-10-02T15:59:00+08:00","signal_date":"2026-09-30","status":"WATCH",
           "relay_candidates":[{"code":"600001","boards":2}],"pool_summary":{"relay":1}}
        p=holiday_plan();p["review_hash"]=review_digest(r)
        with tempfile.TemporaryDirectory() as directory:
            _,registered=register(p,directory,"2026-10-02T16:01:00+08:00")
        self.assertEqual(audit(r,registered)["status"],"BOUND")
        self.assertFalse(audit(r,registered)["trade_authorized"])


if __name__=="__main__":unittest.main()
