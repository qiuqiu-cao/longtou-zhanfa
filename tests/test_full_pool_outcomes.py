import unittest
import pandas as pd
from full_pool_outcomes import case, build, path_summary


class OutcomeTests(unittest.TestCase):
    def bar(self, price, prior):
        return dict(open=price, high=price, low=price, close=price, pre_close=prior,
                    volume=100, status_verified=True, st=False, new_listing=False)

    def test_promotion_does_not_need_exit(self):
        result = case(self.bar(10, 9.09), self.bar(11, 10), None)
        self.assertEqual(result["promotion_LABEL_ONLY"], 1)
        self.assertIsNone(result["price_path_LABEL_ONLY"])

    def test_promotion_can_have_negative_mark(self):
        result = case(self.bar(10, 9.09), self.bar(11, 10), self.bar(9.9, 11))
        self.assertEqual(result["promotion_LABEL_ONLY"], 1)
        self.assertEqual(result["big_loss_mark_LABEL_ONLY"], 1)
        self.assertTrue(result["entry_one_price_LABEL_ONLY"])
        self.assertTrue(result["exit_down_limit_LABEL_ONLY"])
        self.assertIsNone(result["actual_T1_net_premium_LABEL_ONLY"])

    def test_missing_and_new_st_not_failure(self):
        bar = self.bar(11, 10)
        bar["st"] = True
        self.assertIsNone(case(self.bar(10, 9.09), bar, None)["promotion_LABEL_ONLY"])

    def test_no_touch_is_not_trade_loss(self):
        result = case(self.bar(10, 9.09), self.bar(10.1, 10), self.bar(10.2, 10.1))
        self.assertEqual(result["price_path_LABEL_ONLY"], 0)
        self.assertIsNone(result["big_loss_mark_LABEL_ONLY"])
        self.assertFalse(result["next_session_touched_limit_LABEL_ONLY"])
        self.assertFalse(result["next_session_one_price_limit_LABEL_ONLY"])

    def test_path_case_records_three_forward_close_marks(self):
        from full_pool_outcomes import path_case
        signal = self.bar(10, 9.09)
        entry = self.bar(11, 10)
        future = [self.bar(11.5, 11), self.bar(12, 11.5), self.bar(11.2, 12)]
        result = path_case(signal, entry, future)
        self.assertIsNotNone(result["b1_net_close_mark_LABEL_ONLY"])
        self.assertIsNotNone(result["b2_net_close_mark_LABEL_ONLY"])
        self.assertIsNotNone(result["b3_net_close_mark_LABEL_ONLY"])

    def test_path_summary_reports_only_known_mark_denominators(self):
        labels = pd.DataFrame({
            "b1_net_close_mark_LABEL_ONLY": [0.01, None],
            "b2_net_close_mark_LABEL_ONLY": [0.02, -0.08],
            "b3_net_close_mark_LABEL_ONLY": [0.03, -0.09],
        })
        result = path_summary(labels)
        self.assertEqual(result["b1"]["known"], 1)
        self.assertEqual(result["b2_b3_both_positive"]["known"], 2)
        self.assertEqual(result["b2_b3_both_positive"]["rate"], .5)

    def test_reserved_exit_not_used(self):
        dates = ["2025-12-24", "2025-12-25", "2025-12-26"]
        daily = pd.DataFrame([dict(code="600001", date=d, **self.bar(p,q))
                              for d,p,q in zip(dates,[10,11,12.1],[9.09,10,11])]).set_index(["code","date"])
        pool = pd.DataFrame([dict(code="600001",signal_date=dates[0])])
        result = build(pool,daily,dates,"2025-12-26").iloc[0]
        self.assertEqual(result.promotion_LABEL_ONLY,1)
        self.assertIsNone(result.price_path_LABEL_ONLY)


if __name__ == "__main__":
    unittest.main()
