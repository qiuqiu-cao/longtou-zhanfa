import unittest
from trade_account_ledger import account, completed_statistics, run
import json
import tempfile
from pathlib import Path


class AccountTests(unittest.TestCase):
    dates=["2026-09-04","2026-09-07","2026-09-08"]

    def fill(self,day,side,shares,price,id):
        return dict(kind="paper",source="synthetic",evidence_id=id,fee_evidence_id=id+"-fee",
                    filled_at=day+"T09:35:00+08:00",available_at=day+"T09:35:01+08:00",
                    side=side,shares=shares,price=price,fee_cash=5,trade_id="t1",plan_id="p1",code="600001")

    def marks(self):
        return [dict(date=d,code="600001",close=p,source="synthetic",data_status="verified",
                     observed_at=d+"T15:00:00+08:00",available_at=d+"T15:01:00+08:00",
                     price_basis="unadjusted_no_corporate_action") for d,p in zip(self.dates,[9,8,11])]

    def test_partial_exit_and_open_loss_in_drawdown(self):
        fills=[self.fill(self.dates[0],"BUY",200,10,"b"),self.fill(self.dates[1],"SELL",100,8,"s1"),
               self.fill(self.dates[2],"SELL",100,11,"s2")]
        r=account(10000,fills,self.marks(),self.dates,"paper",self.dates[-1]+"T15:05:00+08:00")
        self.assertEqual(len(r["closed_trades"]),1)
        self.assertEqual(r["closed_trades"][0]["net_profit"],-115)
        self.assertEqual(r["completed_trade_win_rate"],0)
        self.assertAlmostEqual(r["account_max_drawdown"],-.041)

    def test_missing_mark_keeps_open_trade_and_unknown_drawdown(self):
        r=account(10000,[self.fill(self.dates[0],"BUY",100,10,"b")],[],self.dates,"paper",
                  self.dates[-1]+"T15:05:00+08:00")
        self.assertEqual(len(r["open_trades"]),1)
        self.assertIsNone(r["completed_trade_win_rate"])
        self.assertIsNone(r["account_max_drawdown"])

    def test_same_day_sale_and_insufficient_cash_rejected(self):
        buy=self.fill(self.dates[0],"BUY",100,10,"b")
        sale=self.fill(self.dates[0],"SELL",100,11,"s")
        sale.update(filled_at=self.dates[0]+"T10:00:00+08:00",available_at=self.dates[0]+"T10:00:01+08:00")
        with self.assertRaises(ValueError):
            account(10000,[buy,sale],self.marks(),self.dates,"paper",self.dates[-1]+"T15:05:00+08:00")
        with self.assertRaises(ValueError):
            account(500,[buy],self.marks(),self.dates,"paper",self.dates[-1]+"T15:05:00+08:00")

    def test_completed_order_and_profit_factor(self):
        closed=[dict(trade_id=str(i),closed_at=d,net_profit=p,net_return=p/1000)
                for i,d,p in [(0,"2026-09-09",200),(1,"2026-09-07",-100),(2,"2026-09-08",-50)]]
        result=completed_statistics(closed)
        self.assertEqual(result["maximum_consecutive_losses"],2)
        self.assertAlmostEqual(result["profit_factor"],200/150)
        self.assertIsNone(completed_statistics([])["profit_factor"])

    def test_report_roundtrip_open_position(self):
        payload=dict(initial_cash=10000,fills=[self.fill(self.dates[0],"BUY",100,10,"b")],
                     marks=self.marks(),sessions=self.dates,kind="paper",asof=self.dates[-1]+"T15:05:00+08:00")
        with tempfile.TemporaryDirectory() as folder:
            src=Path(folder)/"input.json"
            out=Path(folder)/"report.json"
            src.write_text(json.dumps(payload),encoding="utf-8")
            report=run(src,out)
            self.assertEqual(report["open_trade_count"],1)
            self.assertIsNone(report["completed_trade_win_rate"])
            self.assertEqual(len(report["source_hashes"]),6)
            self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["equity_curve"][-1]["equity"],10095)
            with self.assertRaises(FileExistsError):
                run(src,out)


if __name__=="__main__":
    unittest.main()
