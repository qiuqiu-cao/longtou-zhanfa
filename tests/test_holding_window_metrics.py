import unittest
from holding_window_metrics import evaluate, batch, run
import json
import tempfile
from pathlib import Path
from copy import deepcopy


class WindowTests(unittest.TestCase):
    def fixture(self):
        dates=["2026-09-04","2026-09-07","2026-09-08","2026-09-09"]
        fill=dict(code="600001",kind="paper_confirmed_fill",source="synthetic",evidence_id="fixture",
                  filled_at=dates[0]+"T09:35:00+08:00",available_at=dates[0]+"T09:35:01+08:00",
                  price=10,shares=100,price_basis="unadjusted_no_corporate_action")
        costs=dict(source="synthetic",assumption_id="test",effective_from=dates[0],effective_to=dates[-1],
                   commission=.0003,minimum=5,sell_tax=.0005,slippage=.001)
        bars=[dict(code="600001",date=d,close=p,source="synthetic",data_status="verified",
                   observed_at=d+"T15:00:00+08:00",available_at=d+"T15:01:00+08:00",
                   price_basis=fill["price_basis"]) for d,p in zip(dates[1:],[11,10.5,10.2])]
        return fill,bars,dates,costs

    def test_buy_session_weekend_and_positive_marks(self):
        fill,bars,dates,costs=self.fixture()
        result=evaluate(fill,bars,dates,dates[-1]+"T15:05:00+08:00",costs)
        self.assertEqual(result["horizons"][0]["session"],"2026-09-07")
        self.assertTrue(result["both_B2_B3_positive"])
        self.assertIsNone(result["full_trade_profit"])

    def test_future_close_cannot_be_used(self):
        fill,bars,dates,costs=self.fixture()
        result=evaluate(fill,bars,dates,dates[2]+"T10:00:00+08:00",costs)
        self.assertIsNone(result["both_B2_B3_positive"])
        self.assertEqual(result["horizons"][1]["state"],"NOT_YET_OBSERVABLE")

    def test_missing_close_and_fee_change(self):
        fill,bars,dates,costs=self.fixture()
        costs["effective_to"]=dates[2]
        result=evaluate(fill,bars[:1],dates,dates[-1]+"T15:05:00+08:00",costs)
        self.assertEqual(result["horizons"][1]["state"],"MISSING_CLOSE")
        self.assertEqual(result["horizons"][2]["state"],"COST_SCHEDULE_UNAVAILABLE")

    def test_unconfirmed_touch_is_not_fill(self):
        fill,bars,dates,costs=self.fixture()
        fill["kind"]="touched_limit"
        with self.assertRaises(ValueError):
            evaluate(fill,bars,dates,dates[-1]+"T15:05:00+08:00",costs)

    def test_batch_separates_paper_and_actual_and_preserves_unknown(self):
        fill,bars,dates,costs=self.fixture()
        record=dict(fill=fill,bars=bars,costs=costs)
        actual=deepcopy(record)
        actual["fill"].update(kind="user_confirmed_fill",evidence_id="actual-fixture")
        actual["bars"]=[]
        calendar=dict(sessions=dates,source="synthetic",complete=True,data_status="verified",
                      available_at=dates[0]+"T00:00:00+08:00")
        result=batch([record,actual],calendar,dates[-1]+"T15:05:00+08:00")
        self.assertEqual(len(result["records"]),2)
        self.assertEqual(result["summaries"]["user_confirmed_fill"]["horizons"]["2"]["unresolved"],1)
        self.assertIsNone(result["full_trade_win_rate"])
        with self.assertRaises(ValueError):
            batch([record,record],calendar,dates[-1]+"T15:05:00+08:00")
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            (p/"in.json").write_text(json.dumps([record,actual]),encoding="utf-8")
            (p/"cal.json").write_text(json.dumps(calendar),encoding="utf-8")
            report=run(p/"in.json",p/"cal.json",dates[-1]+"T15:05:00+08:00",p/"out.json")
            self.assertEqual(len(report["source_hashes"]),3)
            with self.assertRaises(FileExistsError):
                run(p/"in.json",p/"cal.json",dates[-1]+"T15:05:00+08:00",p/"out.json")


if __name__=="__main__":
    unittest.main()
