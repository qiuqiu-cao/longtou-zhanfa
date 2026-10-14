import unittest
from entry_plan_audit import audit, numeric_condition


class PlanAuditTests(unittest.TestCase):
    def fixture(self):
        def at(t): return "2026-09-07T"+t+"+08:00"
        plan=dict(plan_id="p",code="600001",kind="paper",source="synthetic",
                  frozen_at=at("08:00:00"),available_at=at("08:00:01"),entry_start=at("09:25:00"),
                  entry_end=at("10:00:00"),min_price=10,max_price=11,max_shares=100,
                  required_condition_ids=["auction","absorption"])
        confirm=dict(confirmation_id="c",plan_id="p",code="600001",kind="paper",source="synthetic",
                     observed_at=at("09:35:00"),available_at=at("09:35:01"),expires_at=at("09:36:00"),
                     data_status="verified",conditions={"auction":True,"absorption":True},cancelled=False)
        fill=dict(plan_id="p",confirmation_id="c",code="600001",kind="paper",filled_at=at("09:35:02"),
                  side="BUY",shares=100,price=10.5,evidence_id="f")
        plan["condition_rules"]={id:dict(metric=id,unit="ratio",window_seconds=300,definition_id=id+"-v1",
                                         threshold=.5,operator=">=",max_age_seconds=5) for id in ("auction","absorption")}
        confirm["measurements"]={id:dict(metric=id,unit="ratio",window_seconds=300,definition_id=id+"-v1",
                                         value=.6,observed_at=at("09:35:00"),available_at=at("09:35:01"),
                                         source="synthetic",evidence_id=id,data_status="verified") for id in ("auction","absorption")}
        return fill,plan,confirm

    def test_after_confirmation_and_price_bounds(self):
        f,p,c=self.fixture()
        self.assertEqual(audit([f],[p],[c])["issue_count"],0)
        f["price"]=11.1
        self.assertIn("FILL_OUTSIDE_PRICE_BOUNDS",audit([f],[p],[c])["entries"][0]["issues"])

    def test_preconfirmation_and_cumulative_quantity(self):
        f,p,c=self.fixture()
        f["filled_at"]=c["observed_at"]
        self.assertIn("INVALID_PLAN_CONFIRM_FILL_SEQUENCE",audit([f],[p],[c])["entries"][0]["issues"])
        f,p,c=self.fixture()
        second=dict(f,evidence_id="f2")
        self.assertIn("CUMULATIVE_PLAN_QUANTITY_EXCEEDED",audit([f,second],[p],[c])["entries"][1]["issues"])

    def test_missing_evidence_preserved(self):
        f,_,_=self.fixture()
        r=audit([f],[],[])
        self.assertEqual(len(r["entries"]),1)
        self.assertEqual(r["issue_count"],1)

    def test_boolean_claim_cannot_override_failed_measurement(self):
        f,p,c=self.fixture()
        c["measurements"]["absorption"]["value"]=.1
        result=audit([f],[p],[c])["entries"][0]
        self.assertEqual(result["condition_audit"]["absorption"]["status"],"FAIL")
        self.assertIn("NUMERIC_CONDITION_EVIDENCE_NOT_PASSED",result["issues"])

    def test_unit_window_future_and_missing_not_pass(self):
        _,p,c=self.fixture()
        rule=p["condition_rules"]["auction"]
        evidence=c["measurements"]["auction"]
        for change in ({"unit":"pct"},{"window_seconds":60},{"value":None},{"value":True},
                       {"available_at":"2026-09-07T09:36:00+08:00"}):
            self.assertEqual(numeric_condition(rule,dict(evidence,**change),c["available_at"])["status"],"UNAVAILABLE")


if __name__=="__main__": unittest.main()
