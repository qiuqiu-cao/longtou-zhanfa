import copy
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import broker_execution_import_v16 as entry
import broker_exit_import_v16 as exit_import


class BrokerExitImportV16Tests(unittest.TestCase):
    sessions = ["2026-09-07", "2026-09-08", "2026-09-09", "2026-09-10"]

    @staticmethod
    def plan():
        numeric = dict(metric="close_vs_vwap", unit="ratio", window_seconds=300,
                       definition_id="exit-v1", operator="<=", threshold=0, max_age_seconds=60)
        return dict(plan_id="p1", code="600001", kind="user_confirmed",
                    frozen_at="2026-09-07T09:35:10+08:00", available_at="2026-09-07T09:35:11+08:00",
                    maximum_holding_session_offset=3, allow_beyond_b3=False,
                    exit_rules=[
                        dict(rule_id="hold-strong", effect="HOLD", trigger_mode="NUMERIC",
                             earliest_session_offset=1, latest_session_offset=3,
                             condition_rule=dict(numeric, operator=">", threshold=.01)),
                        dict(rule_id="hold-divergence", effect="HOLD", trigger_mode="NUMERIC",
                             earliest_session_offset=1, latest_session_offset=3, condition_rule=numeric),
                        dict(rule_id="invalid", effect="SELL_ALL", trigger_mode="NUMERIC",
                             earliest_session_offset=1, latest_session_offset=3, condition_rule=numeric),
                        dict(rule_id="time", effect="SELL_ALL", trigger_mode="TIME_STOP",
                             earliest_session_offset=3, latest_session_offset=3),
                    ],
                    holding_scenarios={
                        "strong_continuation": dict(action="HOLD_TO_NEXT_REVIEW", rule_ids=["hold-strong"]),
                        "normal_divergence": dict(action="HOLD_IF_RULES_PASS", rule_ids=["hold-divergence"]),
                        "trend_invalidation": dict(action="SELL_ALL", rule_ids=["invalid"]),
                    })

    @staticmethod
    def confirmation(fill_id="XF1"):
        return dict(fill_id=fill_id, exit_confirmation_id="xc1", plan_id="p1", code="600001",
                    kind="user_confirmed", rule_id="invalid",
                    observed_at="2026-09-08T09:34:00+08:00", available_at="2026-09-08T09:34:01+08:00",
                    expires_at="2026-09-08T09:40:00+08:00", data_status="verified",
                    source="test", evidence_id="exit-trigger-1",
                    measurement=dict(value=-.01, metric="close_vs_vwap", unit="ratio", window_seconds=300,
                                     definition_id="exit-v1", observed_at="2026-09-08T09:34:00+08:00",
                                     available_at="2026-09-08T09:34:01+08:00", data_status="verified",
                                     source="test", evidence_id="metric-1"))

    @staticmethod
    def decision(status="ENTER", authorized=True):
        return {"trade_date":"2026-09-07","asof":"09:35","status":status,
                "direct_entry_authorized":authorized,"order_submitted":False,
                "primary_choice":{"code":"600001","maximum_acceptable_price":11.0}}

    def make_entry(self, root, status="ENTER", authorized=True):
        decision, orders, fills = root/"decision.json", root/"entry_orders.csv", root/"entry_fills.csv"
        decision.write_text(json.dumps(self.decision(status, authorized)), encoding="utf-8")
        pd.DataFrame([{"委托编号":"O1","证券代码":"600001","证券名称":"样本","买卖方向":"买入",
                       "委托时间":"09:36:00","委托价格":"11.00","委托数量":"100","委托状态":"已成"}]).to_csv(orders,index=False)
        pd.DataFrame([{"成交编号":"F1","委托编号":"O1","证券代码":"600001","证券名称":"样本",
                       "成交方向":"买入","成交时间":"09:36:01","成交价格":"10.00","成交数量":"100",
                       "佣金":"5","印花税":"0","过户费":"0.1","其他费用":"0"}]).to_csv(fills,index=False)
        return entry.run(decision,orders,fills,"2026-09-07T10:00:00+08:00","test",root/"entry_out")

    @staticmethod
    def exit_frames(day="2026-09-08"):
        orders = entry.normalize_orders(pd.DataFrame([{
            "委托编号":"XO1","证券代码":"600001","证券名称":"样本","买卖方向":"卖出",
            "委托时间":"09:35:00","委托价格":"10.50","委托数量":"100","委托状态":"已成",
        }]), day)
        fills = entry.normalize_fills(pd.DataFrame([{
            "成交编号":"XF1","委托编号":"XO1","证券代码":"600001","证券名称":"样本",
            "成交方向":"卖出","成交时间":"09:35:01","成交价格":"10.50","成交数量":"100",
            "佣金":"5","印花税":"1","过户费":"0.1","其他费用":"0",
        }]), day)
        return orders, fills

    def test_complete_rule_compliant_trade_gets_strategy_net_return(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            report=self.make_entry(root)
            entry_report, buys=exit_import._verify_entry_report(Path(report["report_path"]))
            orders,fills=self.exit_frames()
            result=exit_import.reconcile_exit(entry_report,buys,orders,fills,self.plan(),[self.confirmation()],
                                               self.sessions,"2026-09-08T15:10:00+08:00")
            self.assertTrue(result["strategy_complete_trade_eligible"])
            self.assertGreater(result["strategy_net_return"],0)
            self.assertEqual(result["remaining_quantity"],0)

    def test_off_plan_entry_never_becomes_strategy_complete_trade(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            report=self.make_entry(root,"WATCH_PUBLIC_EVIDENCE_ONLY",False)
            entry_report,buys=exit_import._verify_entry_report(Path(report["report_path"]))
            orders,fills=self.exit_frames()
            result=exit_import.reconcile_exit(entry_report,buys,orders,fills,self.plan(),[self.confirmation()],
                                               self.sessions,"2026-09-08T15:10:00+08:00")
            self.assertTrue(result["fully_closed"])
            self.assertIsNotNone(result["actual_complete_trade_net_return"])
            self.assertFalse(result["strategy_complete_trade_eligible"])
            self.assertIsNone(result["strategy_net_return"])

    def test_same_day_sell_is_rejected_by_t1(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            report=self.make_entry(root)
            entry_report,buys=exit_import._verify_entry_report(Path(report["report_path"]))
            orders,fills=self.exit_frames("2026-09-07")
            confirmation=copy.deepcopy(self.confirmation())
            for field in ("observed_at","available_at","expires_at"):
                confirmation[field]=confirmation[field].replace("2026-09-08","2026-09-07")
            with self.assertRaisesRegex(ValueError, r"T\+1"):
                exit_import.reconcile_exit(entry_report,buys,orders,fills,self.plan(),[confirmation],
                                           self.sessions,"2026-09-07T15:10:00+08:00")

    def test_no_fill_remains_open_and_has_no_complete_return(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            report=self.make_entry(root)
            entry_report,buys=exit_import._verify_entry_report(Path(report["report_path"]))
            orders,fills=self.exit_frames()
            fills=fills.iloc[0:0].copy()
            result=exit_import.reconcile_exit(entry_report,buys,orders,fills,self.plan(),[],self.sessions,
                                               "2026-09-08T15:10:00+08:00")
            self.assertTrue(result["unresolved_open_position"])
            self.assertIsNone(result["actual_complete_trade_net_return"])
            self.assertFalse(result["strategy_complete_trade_eligible"])


if __name__ == "__main__":
    unittest.main()
