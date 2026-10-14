import unittest
from exit_plan_audit import audit


class ExitAuditTests(unittest.TestCase):
    sessions = ["2026-09-04", "2026-09-07", "2026-09-08", "2026-09-09"]

    def plan(self):
        numeric = dict(metric="close_vs_vwap", unit="ratio", window_seconds=300,
                       definition_id="exit-v1", operator="<=", threshold=0, max_age_seconds=60)
        return dict(plan_id="p", code="600001", kind="paper",
                    frozen_at="2026-09-04T09:30:00+08:00", available_at="2026-09-04T09:30:01+08:00",
                    maximum_holding_session_offset=2, allow_beyond_b3=False,
                    exit_rules=[
                        dict(rule_id="hold-strong", effect="HOLD", trigger_mode="NUMERIC",
                             earliest_session_offset=1, latest_session_offset=2,
                             condition_rule=dict(numeric, operator=">", threshold=.01)),
                        dict(rule_id="hold-divergence", effect="HOLD", trigger_mode="NUMERIC",
                             earliest_session_offset=1, latest_session_offset=2, condition_rule=numeric),
                        dict(rule_id="invalid", effect="SELL_ALL", trigger_mode="NUMERIC",
                             earliest_session_offset=1, latest_session_offset=2, condition_rule=numeric),
                        dict(rule_id="time", effect="SELL_ALL", trigger_mode="TIME_STOP",
                             earliest_session_offset=2, latest_session_offset=2),
                    ],
                    holding_scenarios={
                        "strong_continuation": dict(action="HOLD_TO_NEXT_REVIEW", rule_ids=["hold-strong"]),
                        "normal_divergence": dict(action="HOLD_IF_RULES_PASS", rule_ids=["hold-divergence"]),
                        "trend_invalidation": dict(action="SELL_ALL", rule_ids=["invalid"]),
                    })

    def fill(self, side, day, evidence, confirmation=None):
        row = dict(plan_id="p", trade_id="t", code="600001", kind="paper", side=side, shares=100,
                   filled_at=day+"T09:35:00+08:00", evidence_id=evidence)
        if confirmation:
            row["exit_confirmation_id"] = confirmation
        return row

    def confirmation(self, cid="x", rule="invalid"):
        return dict(exit_confirmation_id=cid, plan_id="p", code="600001", kind="paper", rule_id=rule,
                    observed_at="2026-09-07T09:33:00+08:00", available_at="2026-09-07T09:33:01+08:00",
                    expires_at="2026-09-07T09:40:00+08:00", data_status="verified", source="synthetic",
                    evidence_id="trigger-x", measurement=dict(value=-.01, metric="close_vs_vwap", unit="ratio",
                    window_seconds=300, definition_id="exit-v1", observed_at="2026-09-07T09:33:00+08:00",
                    available_at="2026-09-07T09:33:01+08:00", data_status="verified",
                    source="synthetic", evidence_id="m-x"))

    def test_numeric_exit_fill_matches_frozen_rule(self):
        fills=[self.fill("BUY",self.sessions[0],"b"),self.fill("SELL",self.sessions[1],"s","x")]
        result=audit(fills,[self.plan()],[self.confirmation()],self.sessions,"2026-09-08T15:05:00+08:00")
        self.assertEqual(result["status"],"DECLARED_EXIT_RULES_MATCH")
        self.assertEqual(result["trades"][0]["terminal_state"],"CLOSED")

    def test_sell_without_confirmation_is_never_silently_accepted(self):
        fills=[self.fill("BUY",self.sessions[0],"b"),self.fill("SELL",self.sessions[1],"s")]
        result=audit(fills,[self.plan()],[],self.sessions,"2026-09-08T15:05:00+08:00")
        self.assertIn("MISSING_EXIT_CONFIRMATION",result["trades"][0]["issues"])

    def test_open_past_time_stop_is_flagged(self):
        result=audit([self.fill("BUY",self.sessions[0],"b")],[self.plan()],[],self.sessions,
                     "2026-09-08T15:05:00+08:00")
        self.assertIn("MANDATORY_EXIT_NOT_COMPLETED",result["trades"][0]["issues"])

    def test_verified_sell_blockage_remains_unresolved_not_fake_fill(self):
        confirmation=self.confirmation("blocked","time")
        confirmation.update(observed_at="2026-09-08T14:55:00+08:00",available_at="2026-09-08T14:55:01+08:00",
                            expires_at="2026-09-08T15:01:00+08:00",sell_blocked=True,
                            sellability_evidence=dict(state="locked_limit_down",data_status="verified",source="synthetic",
                            evidence_id="locked",observed_at="2026-09-08T14:55:00+08:00",
                            available_at="2026-09-08T14:55:01+08:00"))
        result=audit([self.fill("BUY",self.sessions[0],"b")],[self.plan()],[confirmation],self.sessions,
                     "2026-09-08T15:05:00+08:00")
        self.assertEqual(result["trades"][0]["terminal_state"],"EXIT_BLOCKED_UNRESOLVED")
        self.assertNotIn("MANDATORY_EXIT_NOT_COMPLETED",result["trades"][0]["issues"])

    def test_beyond_b3_requires_prior_hold_rules(self):
        plan=self.plan()
        plan.update(maximum_holding_session_offset=4,allow_beyond_b3=False)
        result=audit([self.fill("BUY",self.sessions[0],"b")],[plan],[],self.sessions,"2026-09-07T15:05:00+08:00")
        self.assertIn("HOLD_BEYOND_B3_NOT_PREAUTHORIZED",result["trades"][0]["issues"])

    def test_unbound_blockage_cannot_excuse_overdue_position(self):
        confirmation=self.confirmation("blocked","hold-strong")
        confirmation.update(observed_at="2026-09-08T14:55:00+08:00",available_at="2026-09-08T14:55:01+08:00",
                            sell_blocked=True,sellability_evidence=dict(state="locked_limit_down",data_status="verified",
                            source="synthetic",evidence_id="locked",observed_at="2026-09-08T14:55:00+08:00",
                            available_at="2026-09-08T14:55:01+08:00"))
        result=audit([self.fill("BUY",self.sessions[0],"b")],[self.plan()],[confirmation],self.sessions,
                     "2026-09-08T15:05:00+08:00")
        self.assertEqual(result["trades"][0]["terminal_state"],"MANDATORY_EXIT_NOT_COMPLETED")
        self.assertIn("UNKNOWN_OR_NONSELL_BLOCKED_RULE",result["trades"][0]["issues"])


if __name__ == "__main__": unittest.main()
