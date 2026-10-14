import unittest
from risk_budget_audit import audit


class RiskTests(unittest.TestCase):
    def fixture(self):
        policy=dict(policy_id="r1",source="synthetic",data_status="verified",
                    available_at="2026-09-07T08:00:00+08:00",single_trade_risk_fraction=.01,
                    gross_exposure_fraction=.3,theme_exposure_fraction=.15)
        plan=dict(plan_id="p",frozen_at="2026-09-07T08:01:00+08:00",available_at="2026-09-07T08:01:01+08:00",
                  risk_policy_id="r1",account_equity_at_freeze=100000,existing_gross_exposure_cash=0,
                  existing_theme_exposure_cash=0,stress_loss_fraction=.1,max_price=10,max_shares=1000,
                  theme_id="grain",equity_snapshot_id="e1",exposure_snapshot_id="x1")
        return policy,plan

    def test_single_theme_and_gross_limits(self):
        policy,plan=self.fixture()
        result=audit([plan],policy,"2026-09-07T08:02:00+08:00")
        self.assertEqual(result["status"],"PASS")
        self.assertEqual(result["plans"][0]["max_shares_by_single_risk"],1000)
        plan["max_shares"]=2000
        issues=audit([plan],policy,"2026-09-07T08:02:00+08:00")["plans"][0]["issues"]
        self.assertIn("SINGLE_TRADE_RISK_EXCEEDED",issues)
        self.assertIn("THEME_EXPOSURE_EXCEEDED",issues)

    def test_existing_exposure_and_duplicate_snapshot(self):
        policy,plan=self.fixture()
        plan["existing_gross_exposure_cash"]=25000
        self.assertIn("GROSS_EXPOSURE_EXCEEDED",audit([plan],policy,"2026-09-07T08:02:00+08:00")["plans"][0]["issues"])
        plan["existing_gross_exposure_cash"]=0
        second=dict(plan,plan_id="p2",theme_id="ai")
        result=audit([plan,second],policy,"2026-09-07T08:02:00+08:00")
        self.assertTrue(all("SHARED_EXPOSURE_SNAPSHOT_CANNOT_AGGREGATE_MULTIPLE_PLANS" in r["issues"] for r in result["plans"]))

    def test_missing_policy_is_unavailable(self):
        self.assertEqual(audit([],None,"2026-09-07T08:02:00+08:00")["status"],"UNAVAILABLE")


if __name__=="__main__": unittest.main()
