import unittest
from t1_scenario_v44 import net_return,evaluate


class T1ScenarioTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(code="000001",entry_date="2024-01-03",data_status="DAILY_OHLC_MISMATCH",limit_price_proxy=10,
                        observation=dict(state="CLOSE_AT_LIMIT_PROXY"),execution_LABEL_ONLY=dict(state="BELOW_LIMIT_TRADES_OPPORTUNITY_ONLY"))
        self.daily = dict(open=10,high=11,low=9,close=10,pre_close=10,volume=1000)
        self.costs = dict(shares=100,commission=0.0003,minimum=5,sell_tax=0.0005,slippage=0.001)

    def test_costs_reduce_flat_price_return(self):
        self.assertLess(net_return(10,10,**self.costs),0)

    def test_same_day_exit_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(self.row,"2024-01-03",self.daily,self.costs)

    def test_queue_unresolved_no_pnl(self):
        self.row["execution_LABEL_ONLY"]["state"] = "LIMIT_QUEUE_UNRESOLVED"
        self.assertIsNone(evaluate(self.row,"2024-01-04",self.daily,self.costs)["net_t1_mark_return"])

    def test_limit_down_not_realized(self):
        self.daily.update(close=9)
        result = evaluate(self.row,"2024-01-04",self.daily,self.costs)
        self.assertEqual(result["state"],"CONDITIONAL_ENTRY_EXIT_UNRESOLVED")
        self.assertTrue(result["large_loss_proxy"])
        self.assertIsNone(result["realized_return"])

    def test_missing_signal_not_rejection(self):
        del self.row["observation"]
        self.assertEqual(evaluate(self.row,None,None,self.costs)["state"],"SIGNAL_DATA_UNAVAILABLE")

    def test_missing_exit_not_zero(self):
        self.assertIsNone(evaluate(self.row,"2024-01-04",None,self.costs)["net_t1_mark_return"])

    def test_invalid_daily(self):
        self.daily["close"] = float("nan")
        self.assertEqual(evaluate(self.row,"2024-01-04",self.daily,self.costs)["state"],"EXIT_DATA_INVALID")
