import copy
import json
from pathlib import Path
import tempfile
import unittest

import forum_execution_review as execution


class ForumExecutionTests(unittest.TestCase):
    sessions = ["2026-09-04", "2026-09-07", "2026-09-08", "2026-09-09", "2026-09-10"]
    asof = "2026-09-08T15:10:00+08:00"

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.facts = dict(signal_date="2026-09-04", target_date="2026-09-07", rows=[dict(code="600001", name="样本", boards=2), dict(code="000001", name="反例", boards=2)])
        self.review = dict(formed_at="2026-09-04T18:00:00+08:00", rows=[dict(code="600001", name="样本", boards=2, promotion_judgement="PROMOTE"), dict(code="000001", name="反例", boards=2, promotion_judgement="FAIL")])
        self.predictions = copy.deepcopy(self.review["rows"])

    def payload(self, sell=True, marks=True):
        base = dict(code="600001", kind="user_confirmed", source="user supplied saved broker record", data_status="verified", evidence_origin="USER_CONFIRMED_FILL_RECORD")
        numeric = dict(metric="close_vs_vwap", unit="ratio", window_seconds=300, definition_id="test-1", operator=">", threshold=0, max_age_seconds=60)
        plan = dict(base, plan_id="p1", trade_date="2026-09-07", frozen_at="2026-09-07T09:34:00+08:00", available_at="2026-09-07T09:34:01+08:00",
            entry_start="2026-09-07T09:35:00+08:00", entry_end="2026-09-07T10:00:00+08:00", min_price=9, max_price=11, max_shares=100,
            required_condition_ids=["positive"], condition_rules={"positive": numeric}, maximum_holding_session_offset=3, allow_beyond_b3=False,
            exit_rules=[dict(rule_id="hold", effect="HOLD", trigger_mode="NUMERIC", earliest_session_offset=1, latest_session_offset=3, condition_rule=numeric),
                dict(rule_id="invalid", effect="SELL_ALL", trigger_mode="NUMERIC", earliest_session_offset=1, latest_session_offset=3, condition_rule=dict(numeric, operator="<=", threshold=0)),
                dict(rule_id="time", effect="SELL_ALL", trigger_mode="TIME_STOP", earliest_session_offset=3, latest_session_offset=3)],
            holding_scenarios={"strong_continuation":dict(action="HOLD_TO_NEXT_REVIEW",rule_ids=["hold"]), "normal_divergence":dict(action="HOLD_IF_RULES_PASS",rule_ids=["hold"]), "trend_invalidation":dict(action="SELL_ALL",rule_ids=["invalid"])})
        measurement = dict(value=.01,metric="close_vs_vwap",unit="ratio",window_seconds=300,definition_id="test-1", observed_at="2026-09-07T09:35:00+08:00",available_at="2026-09-07T09:35:01+08:00",data_status="verified",source="saved quote",evidence_id="q1")
        confirm = dict(base,confirmation_id="c1",plan_id="p1", status="TRIGGERED",observed_at="2026-09-07T09:35:00+08:00",available_at="2026-09-07T09:35:01+08:00",expires_at="2026-09-07T10:00:00+08:00",evidence_id="c1",conditions={"positive":True},measurements={"positive":measurement}, cancelled=False)
        order = dict(base,order_id="o1",plan_id="p1",confirmation_id="c1",shares=100,submitted_at="2026-09-07T09:36:00+08:00",available_at="2026-09-07T09:36:01+08:00",expires_at="2026-09-07T10:00:00+08:00",status="SUBMITTED",evidence_id="o1")
        buy = dict(base,evidence_id="b1",fee_evidence_id="b1fee",trade_id="t1",plan_id="p1",confirmation_id="c1",order_id="o1",side="BUY",shares=100,price=10,fee_cash=5.1,filled_at="2026-09-07T09:36:02+08:00",available_at="2026-09-07T09:36:03+08:00")
        sale = dict(base,evidence_id="s1",fee_evidence_id="s1fee",trade_id="t1",plan_id="p1",exit_confirmation_id="x1",side="SELL",shares=100,price=10.5,fee_cash=6.1,filled_at="2026-09-08T09:36:02+08:00",available_at="2026-09-08T09:36:03+08:00")
        xmeasurement=dict(measurement,value=-.01,observed_at="2026-09-08T09:35:00+08:00",available_at="2026-09-08T09:35:01+08:00",evidence_id="xq1")
        exit_confirmation = dict(base,exit_confirmation_id="x1",plan_id="p1",rule_id="invalid",observed_at="2026-09-08T09:35:00+08:00",available_at="2026-09-08T09:35:01+08:00",expires_at="2026-09-08T10:00:00+08:00",evidence_id="x1",measurement=xmeasurement)
        quote = dict(base, date="2026-09-07", close=9.8,observed_at="2026-09-07T15:00:00+08:00",available_at="2026-09-07T15:01:00+08:00",price_basis="unadjusted_no_corporate_action")
        quote2=dict(quote,date="2026-09-08",close=10.5,observed_at="2026-09-08T15:00:00+08:00",available_at="2026-09-08T15:01:00+08:00")
        cash=dict(kind="user_confirmed",evidence_origin="USER_CONFIRMED_FILL_RECORD",initial_cash=10000,opening_positions_empty=True,session="2026-09-07")
        benchmark=dict(base,fill_id="b1",price=9.9,observed_at="2026-09-07T09:36:00+08:00",available_at="2026-09-07T09:36:01+08:00")
        return dict(kind="user_confirmed",plans=[plan],confirmations=[confirm],orders=[order],fills=[buy,sale] if sell else [buy], exit_confirmations=[exit_confirmation] if sell else [],marks=[quote,quote2] if marks else [],benchmarks=[benchmark],initial_cash=10000,initial_cash_evidence=cash)

    def bind(self, payload):
        payload = copy.deepcopy(payload)
        saved = {k:v for k,v in payload.items() if k != "source_bindings"}
        for key in execution.LISTS:
            for row in saved.get(key,[]):
                row.pop("evidence_ref",None)
        saved.get("initial_cash_evidence",{}).pop("evidence_ref",None)
        source = self.root / ("source_" + str(len(list(self.root.glob("source*")))) + ".json")
        source.write_text(json.dumps(saved,ensure_ascii=False),encoding="utf-8")
        payload["source_bindings"]={str(source):execution.sha(source)}
        for key in execution.LISTS:
            for i,row in enumerate(payload.get(key,[])):
                row["evidence_ref"] = dict(path=str(source),sha256=execution.sha(source),json_pointer=f"/{key}/{i}")
        if payload.get("initial_cash_evidence"):
            payload["initial_cash_evidence"]["evidence_ref"]=dict(path=str(source),sha256=execution.sha(source),json_pointer="/initial_cash_evidence")
        return payload

    def call(self, payload=None, asof=None):
        return execution.project(self.facts,self.review,self.predictions,self.sessions,asof or self.asof,self.bind(payload) if payload else None)

    def test_pending_keeps_complete_pool_and_no_actual_returns_or_cash_day(self):
        result=self.call(asof="2026-09-07T08:00:00+08:00")
        self.assertEqual(result["pool_size"],2)
        self.assertTrue(all(r["actual_complete_trade_net_return"] is None and r["actual_buy_quantity"] is None for r in result["rows"]))
        self.assertEqual(result["funnel"]["known_cash_day_count"],0)
        self.assertIsNone(result["funnel"]["cash_day_rate"])

    def test_real_fills_costs_and_t1_close_reuse_account_ledger(self):
        result=self.call(self.payload())
        trade=result["rows"][0]
        self.assertAlmostEqual(trade["actual_complete_trade_net_return"],1043.9/1005.1-1)
        self.assertTrue(trade["strategy_complete_trade_eligible"])
        self.assertAlmostEqual(result["supplied_account_projection"]["equity_curve"][-1]["equity"],10038.8)
        self.assertAlmostEqual(result["observed_slippage"][0]["cost_cash"],10)
        self.assertFalse(result["observed_slippage"][0]["separately_subtracted_from_actual_pnl"])

    def test_open_position_marks_are_not_realized_profit(self):
        result=self.call(self.payload(sell=False))
        self.assertEqual(result["rows"][0]["remaining_quantity"],100)
        self.assertIsNone(result["rows"][0]["actual_complete_trade_net_return"])
        self.assertEqual(result["supplied_account_projection"]["open_trade_count"],1)
        self.assertAlmostEqual(result["supplied_account_projection"]["equity_curve"][0]["equity"],9974.9)

    def test_missing_open_marks_keeps_drawdown_unknown(self):
        result=self.call(self.payload(sell=False,marks=False))
        self.assertIsNone(result["supplied_account_projection"]["account_max_drawdown"])
        self.assertIsNone(result["supplied_account_projection"]["equity_curve"][-1]["equity"])

    def test_observed_open_cannot_be_same_timestamp_order_and_fill(self):
        payload=self.payload()
        payload["orders"][0]["submitted_at"]=payload["confirmations"][0]["observed_at"]
        payload["fills"][0]["filled_at"]=payload["confirmations"][0]["observed_at"]
        with self.assertRaisesRegex(ValueError,"observed open"):
            self.call(payload)

    def test_fill_outside_window_rejected(self):
        payload=self.payload()
        payload["fills"][0].update(filled_at="2026-09-07T10:01:00+08:00",available_at="2026-09-07T10:01:01+08:00")
        with self.assertRaisesRegex(ValueError,"frozen window"):
            self.call(payload)

    def test_fill_before_order_rejected(self):
        payload=self.payload()
        payload["fills"][0].update(filled_at="2026-09-07T09:35:59+08:00")
        with self.assertRaisesRegex(ValueError,"follow its order"):
            self.call(payload)

    def test_t1_same_session_sell_rejected(self):
        payload=self.payload()
        payload["fills"][1].update(filled_at="2026-09-07T10:30:00+08:00",available_at="2026-09-07T10:30:01+08:00")
        with self.assertRaisesRegex(ValueError,r"T\+1"):
            self.call(payload)

    def test_non_session_or_lunch_fill_rejected(self):
        for value in ("2026-09-06T09:36:02+08:00","2026-09-07T12:00:00+08:00"):
            payload=self.payload()
            payload["fills"][0].update(filled_at=value)
            with self.assertRaisesRegex(ValueError,"outside"):
                self.call(payload)

    def test_cross_pool_records_and_dropped_candidates_rejected(self):
        payload=self.payload()
        payload["fills"][0]["code"]="600002"
        with self.assertRaisesRegex(ValueError,"outside the original full pool"):
            self.call(payload)
        self.predictions.pop()
        with self.assertRaisesRegex(ValueError,"exact full pool"):
            self.call()

    def test_source_hash_or_source_pointer_tampering_rejected(self):
        payload=self.bind(self.payload())
        path=next(iter(payload["source_bindings"]))
        Path(path).write_text("{}",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"source hash"):
            execution.project(self.facts,self.review,self.predictions,self.sessions,self.asof,payload)
        payload=self.bind(self.payload())
        payload["fills"][0]["price"]=10.1
        with self.assertRaisesRegex(ValueError,"saved declaration"):
            execution.project(self.facts,self.review,self.predictions,self.sessions,self.asof,payload)

    def test_missing_fees_does_not_create_net_return(self):
        payload=self.payload()
        payload["fills"][0]["fee_cash"]=None
        result=self.call(payload)
        self.assertIsNone(result["rows"][0]["actual_complete_trade_net_return"])
        self.assertIsNone(result["supplied_account_projection"])

    def test_trade_costs_known_without_inventing_initial_account_cash(self):
        payload=self.payload()
        payload.pop("initial_cash")
        payload.pop("initial_cash_evidence")
        result=self.call(payload)
        self.assertAlmostEqual(result["rows"][0]["actual_complete_trade_net_return"],1043.9/1005.1-1)
        self.assertIsNone(result["supplied_account_projection"])

    def test_negative_fee_rejected_even_when_starting_cash_unknown(self):
        payload=self.payload()
        payload.pop("initial_cash")
        payload["fills"][0]["fee_cash"]=-5
        with self.assertRaisesRegex(ValueError,"nonnegative"):
            self.call(payload)

    def test_daily_bar_cannot_be_actual_fill_or_return(self):
        payload=self.payload()
        payload["fills"][0]["evidence_origin"]="DAILY_BAR_INFERRED_OPEN_FILL"
        with self.assertRaisesRegex(ValueError,"not actual execution"):
            self.call(payload)
        payload=self.payload()
        payload["daily_proxy"]=[dict(code="600001",net_return=.1)]
        with self.assertRaisesRegex(ValueError,"separately bound diagnostic"):
            self.call(payload)

    def test_paper_cannot_mix_actual_execution(self):
        payload=self.payload()
        payload["fills"][0]["kind"]="paper"
        with self.assertRaisesRegex(ValueError,"cannot mix"):
            self.call(payload)

    def test_suspended_position_retained_as_blocked_with_no_realized_return(self):
        payload=self.payload(sell=False)
        payload["benchmarks"]=[]
        base=payload["confirmations"][0]
        payload["exit_confirmations"]=[dict(code="600001",kind="user_confirmed",source="saved suspension evidence",data_status="verified",evidence_origin="USER_CONFIRMED_FILL_RECORD",exit_confirmation_id="blocked1",plan_id="p1",rule_id="time",sell_blocked=True,
            observed_at="2026-09-10T15:00:00+08:00",available_at="2026-09-10T15:01:00+08:00",evidence_id="blk1",sellability_evidence=dict(state="suspended",observed_at="2026-09-10T15:00:00+08:00",available_at="2026-09-10T15:01:00+08:00",data_status="verified",source="user saved report",evidence_id="susp1"))]
        result=self.call(payload,asof="2026-09-10T15:10:00+08:00")
        self.assertEqual(result["exit_audit"]["trades"][0]["terminal_state"],"EXIT_BLOCKED_UNRESOLVED")
        self.assertEqual(result["rows"][0]["remaining_quantity"],100)
        self.assertIsNone(result["rows"][0]["actual_complete_trade_net_return"])


if __name__ == "__main__":
    unittest.main()
