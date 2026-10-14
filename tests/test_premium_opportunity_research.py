import unittest
import pandas as pd
from premium_opportunity_research import label_case, labels


def bar(o=10,h=10,l=10,c=10,p=10):
    return dict(open=o,high=h,low=l,close=c,pre_close=p,volume=100)


class PremiumOpportunityTests(unittest.TestCase):
    def test_one_price_positive_remains_unverified(self):
        row=label_case(bar(),bar(11,11,11,11),bar(12,12,12,12,11))
        self.assertEqual(row["label_ONLY"],1)
        self.assertTrue(row["entry_one_price_ONLY"])
        self.assertFalse(row["verified_trade"])

    def test_touch_then_loss_is_failure(self):
        row=label_case(bar(),bar(10,11,9,10),bar(9,9,9,9,10))
        self.assertEqual(row["label_ONLY"],0)
        self.assertTrue(row["exit_down_limit_ONLY"])

    def test_no_touch_is_not_trade_even_if_next_day_rises(self):
        row=label_case(bar(),bar(),bar(12,12,12,12))
        self.assertEqual(row["label_ONLY"],0)
        self.assertIsNone(row["mark_return_ONLY"])

    def test_missing_and_rule_discontinuity_not_zero(self):
        self.assertIsNone(label_case(bar(),None,bar())["label_ONLY"])
        self.assertIsNone(label_case(bar(),bar(p=8),bar())["label_ONLY"])
        self.assertIsNone(label_case(bar(),bar(h=12),bar())["label_ONLY"])

    def test_calendar_gap_not_bridged(self):
        pool=pd.DataFrame([dict(code="000001",signal_date="2025-01-02")])
        daily=pd.DataFrame([dict(code="000001",date=d,**bar()) for d in ("2025-01-02","2025-01-06")]).set_index(["code","date"])
        result=labels(pool,daily,["2025-01-02","2025-01-03","2025-01-06"],"2025-02-01")
        self.assertTrue(pd.isna(result.iloc[0].label_ONLY))
