import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import broker_execution_import_v16 as entry
import broker_exit_import_v16 as lifecycle
import trade_path_metrics_v16 as path_metrics


class TradePathMetricsV16Tests(unittest.TestCase):
    sessions = ["2026-09-07", "2026-09-08", "2026-09-09", "2026-09-10"]

    def make_entry(self, root, status="ENTER", authorized=True):
        decision = root / "decision.json"
        orders = root / "orders.csv"
        fills = root / "fills.csv"
        decision.write_text(json.dumps({
            "trade_date":"2026-09-07","asof":"09:35","status":status,
            "direct_entry_authorized":authorized,"order_submitted":False,
            "primary_choice":{"code":"600001","maximum_acceptable_price":11.0},
        }), encoding="utf-8")
        pd.DataFrame([{"委托编号":"O1","证券代码":"600001","证券名称":"样本","买卖方向":"买入",
                       "委托时间":"09:36:00","委托价格":"11","委托数量":"100","委托状态":"已成"}]).to_csv(orders,index=False)
        pd.DataFrame([{"成交编号":"F1","委托编号":"O1","证券代码":"600001","证券名称":"样本",
                       "成交方向":"买入","成交时间":"09:36:01","成交价格":"10","成交数量":"100",
                       "佣金":"5","印花税":"0","过户费":"0.1","其他费用":"0"}]).to_csv(fills,index=False)
        report = entry.run(decision,orders,fills,"2026-09-07T10:00:00+08:00","test",root/"entry")
        return lifecycle._verify_entry_report(Path(report["report_path"]))

    @staticmethod
    def plan():
        return {"code":"600001","frozen_at":"2026-09-07T09:35:10+08:00",
                "cost_policy":{"sell_commission_rate":.0003,"sell_min_commission_cash":5,
                               "sell_stamp_tax_rate":.0005,"sell_transfer_fee_rate":.00001,
                               "sell_other_fee_rate":0,"currency":"CNY","policy_id":"test-cost-v1"}}

    @staticmethod
    def marks():
        rows=[]
        for day,prices,state in [
            ("2026-09-08",(10.2,10.7,9.8,10.5),"NORMAL"),
            ("2026-09-09",(10.5,10.8,10.1,10.3),"NORMAL"),
            ("2026-09-10",(10.3,11.0,10.2,10.8),"NORMAL"),
        ]:
            rows.append(dict(date=day,code="600001",open=prices[0],high=prices[1],low=prices[2],close=prices[3],
                             source="test terminal",observed_at=day+"T15:00:00+08:00",
                             available_at=day+"T15:00:01+08:00",data_status="verified_terminal_export",
                             price_basis="unadjusted_no_corporate_action",sellability_state=state))
        return pd.DataFrame(rows)

    def test_complete_window_calculates_net_marks_mfe_and_mae(self):
        with tempfile.TemporaryDirectory() as tmp:
            report,fills=self.make_entry(Path(tmp))
            marks=path_metrics.validate_marks(self.marks(),"600001","2026-09-10T15:05:00+08:00")
            result=path_metrics.calculate(report,fills,marks,self.sessions,self.plan(),"2026-09-10T15:05:00+08:00")
            self.assertTrue(result["window_B1_B3_complete"])
            self.assertTrue(result["B_plus_2_and_B_plus_3_both_positive"])
            self.assertGreater(result["maximum_favourable_excursion_net"],result["B_plus_3_net_close_return"])
            self.assertLess(result["maximum_adverse_excursion_net"],0)
            self.assertTrue(result["strategy_path_metric_eligible"])

    def test_missing_day_keeps_window_metrics_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            report,fills=self.make_entry(Path(tmp))
            raw=self.marks()[self.marks()["date"]!="2026-09-09"]
            marks=path_metrics.validate_marks(raw,"600001","2026-09-10T15:05:00+08:00")
            result=path_metrics.calculate(report,fills,marks,self.sessions,self.plan(),"2026-09-10T15:05:00+08:00")
            self.assertFalse(result["window_B1_B3_complete"])
            self.assertIsNone(result["B_plus_2_net_close_return"])
            self.assertIsNone(result["maximum_favourable_excursion_net"])

    def test_off_plan_entry_has_actual_path_but_not_strategy_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            report,fills=self.make_entry(Path(tmp),"WATCH_PUBLIC_EVIDENCE_ONLY",False)
            marks=path_metrics.validate_marks(self.marks(),"600001","2026-09-10T15:05:00+08:00")
            result=path_metrics.calculate(report,fills,marks,self.sessions,self.plan(),"2026-09-10T15:05:00+08:00")
            self.assertIsNotNone(result["B_plus_1_net_close_return"])
            self.assertFalse(result["strategy_path_metric_eligible"])

    def test_locked_limit_down_is_valued_but_not_called_realisable(self):
        with tempfile.TemporaryDirectory() as tmp:
            report,fills=self.make_entry(Path(tmp))
            raw=self.marks()
            raw.loc[raw["date"].eq("2026-09-08"),"sellability_state"]="LOCKED_LIMIT_DOWN"
            marks=path_metrics.validate_marks(raw,"600001","2026-09-10T15:05:00+08:00")
            result=path_metrics.calculate(report,fills,marks,self.sessions,self.plan(),"2026-09-10T15:05:00+08:00")
            self.assertIsNotNone(result["B_plus_1_net_close_return"])
            self.assertFalse(result["marks"][0]["realisable_at_close"])
            self.assertEqual(result["sellability_blocked_offsets"],[1])

    def test_future_mark_availability_is_rejected(self):
        raw=self.marks()
        raw.loc[0,"available_at"]="2026-09-11T15:00:00+08:00"
        with self.assertRaisesRegex(ValueError,"available"):
            path_metrics.validate_marks(raw,"600001","2026-09-10T15:05:00+08:00")


if __name__ == "__main__":
    unittest.main()
