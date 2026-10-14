import unittest

from execution_funnel_audit import audit


class ExecutionFunnelTests(unittest.TestCase):
    sessions=["2026-09-07"]

    def manifest(self,complete=True,count=2):
        return dict(trade_date="2026-09-07",signal_date="2026-09-04",candidate_count=count,
                    universe_complete=complete,data_status="verified",source="synthetic",kind="paper",
                    observed_at="2026-09-04T15:00:00+08:00",available_at="2026-09-04T15:01:00+08:00")

    def decisions(self):
        base=dict(trade_date="2026-09-07",signal_date="2026-09-04",frozen_at="2026-09-04T15:02:00+08:00",
                  available_at="2026-09-04T15:02:01+08:00",source="synthetic",data_status="verified",kind="paper")
        return [dict(base,decision_id="d1",code="600001",board_group="FIRST_BOARD",decision_state="FOCUS",plan_id="p1"),
                dict(base,decision_id="d2",code="000001",board_group="CONSECUTIVE_BOARD",decision_state="WATCH",plan_id=None)]

    def confirmation(self):
        return dict(confirmation_id="c1",plan_id="p1",code="600001",status="TRIGGERED",
                    observed_at="2026-09-07T09:35:00+08:00",available_at="2026-09-07T09:35:01+08:00",
                    expires_at="2026-09-07T10:00:00+08:00",source="synthetic",data_status="verified",evidence_id="c-e",kind="paper")

    def order(self):
        return dict(order_id="o1",plan_id="p1",confirmation_id="c1",code="600001",shares=100,
                    submitted_at="2026-09-07T09:36:00+08:00",available_at="2026-09-07T09:36:01+08:00",
                    expires_at="2026-09-07T10:00:00+08:00",status="SUBMITTED",source="synthetic",
                    data_status="verified",evidence_id="o-e",kind="paper")

    def fill(self,eid,clock,shares):
        return dict(order_id="o1",plan_id="p1",code="600001",side="BUY",shares=shares,price=10,
                    filled_at=f"2026-09-07T{clock}+08:00",available_at=f"2026-09-07T{clock}+08:00",
                    source="synthetic",evidence_id=eid,data_status="verified",kind="paper")

    def call(self,confirmations=None,orders=None,fills=None,asof="2026-09-07T10:01:00+08:00",manifest=None):
        return audit([manifest or self.manifest()],self.decisions(),confirmations or [],orders or [],fills or [],self.sessions,asof)

    def test_two_fill_events_are_one_full_order_and_one_plan(self):
        result=self.call([self.confirmation()],[self.order()],[self.fill("f1","09:37:00",40),self.fill("f2","09:38:00",60)])
        self.assertEqual(result["status"],"PASS")
        self.assertEqual(result["candidate_count"],2)
        self.assertEqual(result["decision_count"],2)
        self.assertEqual(result["full_order_count"],1)
        self.assertEqual(result["buy_fill_event_count"],2)
        self.assertEqual(result["executed_plan_count"],1)
        self.assertEqual(result["known_cash_day_count"],0)
        self.assertIsNone(result["open_position_count"])
        self.assertEqual(result["exit_evidence_status"],"UNAVAILABLE_ENTRY_FUNNEL_ONLY")

    def test_partial_and_unfilled_are_not_discarded(self):
        result=self.call([self.confirmation()],[self.order()],[self.fill("f1","09:37:00",40)])
        self.assertEqual(result["partial_order_count"],1)
        self.assertEqual(result["order_fill_rate"],1)
        self.assertEqual(result["share_fill_rate"],.4)
        result=self.call([self.confirmation()],[self.order()],[])
        self.assertEqual(result["unfilled_order_count"],1)
        self.assertEqual(result["order_fill_rate"],0)

    def test_pending_order_excluded_from_completed_denominator(self):
        result=self.call([self.confirmation()],[self.order()],[],"2026-09-07T09:40:00+08:00")
        self.assertEqual(result["pending_order_count"],1)
        self.assertIsNone(result["order_fill_rate"])

    def test_trigger_without_order_is_not_fill(self):
        result=self.call([self.confirmation()],[],[])
        self.assertEqual(result["triggered_count"],1)
        self.assertEqual(result["no_order_after_trigger_count"],1)
        self.assertEqual(result["executed_plan_count"],0)
        self.assertFalse(result["touch_is_fill"])

    def test_fill_before_order_is_invalid_and_not_counted(self):
        result=self.call([self.confirmation()],[self.order()],[self.fill("f1","09:35:30",100)])
        self.assertEqual(result["status"],"ISSUES_FOUND")
        self.assertEqual(result["order_results"][0]["fill_state"],"INVALID")
        self.assertEqual(result["executed_plan_count"],0)

    def test_incomplete_manifest_makes_cash_day_unknown(self):
        result=self.call(manifest=self.manifest(False))
        self.assertEqual(result["status"],"ISSUES_FOUND")
        self.assertEqual(result["complete_session_count"],0)
        self.assertEqual(result["unknown_session_count"],1)
        self.assertIsNone(result["cash_day_rate"])

    def test_paper_and_user_confirmed_records_cannot_mix(self):
        fill=self.fill("f1","09:37:00",100)
        fill["kind"]="user_confirmed"
        result=self.call([self.confirmation()],[self.order()],[fill])
        self.assertEqual(result["status"],"ISSUES_FOUND")
        self.assertEqual(result["executed_plan_count"],0)

    def test_duplicate_fill_evidence_is_rejected(self):
        fills=[self.fill("same","09:37:00",50),self.fill("same","09:38:00",50)]
        with self.assertRaisesRegex(ValueError,"duplicate evidence_id"):
            self.call([self.confirmation()],[self.order()],fills)

    def test_invalid_trigger_not_counted_or_allowed_to_support_order(self):
        confirmation=self.confirmation()
        confirmation["data_status"]="inferred"
        result=self.call([confirmation],[self.order()],[self.fill("f1","09:37:00",100)])
        self.assertEqual(result["triggered_count"],0)
        self.assertEqual(result["executed_plan_count"],0)
        self.assertEqual(result["order_results"][0]["fill_state"],"INVALID")


if __name__=="__main__": unittest.main()
