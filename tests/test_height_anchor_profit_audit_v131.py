import unittest

import pandas as pd

from height_anchor_profit_audit_v131 import _paired_path_delta, audit, prepare


class HeightAnchorProfitAuditV131Test(unittest.TestCase):
    def _selections(self):
        return pd.DataFrame([
            {"code": "1", "signal_date": "2026-01-05", "selection_method": "MODEL_ALL_TOP1",
             "top_within_3_sessions": 1, "new_challenger_top_within_3": None},
            {"code": "2", "signal_date": "2026-01-06", "selection_method": "MODEL_ALL_TOP1",
             "top_within_3_sessions": 0, "new_challenger_top_within_3": None},
            {"code": "3", "signal_date": "2026-01-07", "selection_method": "MODEL_ALL_TOP1",
             "top_within_3_sessions": 1, "new_challenger_top_within_3": None},
        ])

    def _outcomes(self):
        return pd.DataFrame([
            {"code": "000001", "signal_date": "2026-01-05",
             "price_path_state": "PRICE_OPPORTUNITY_PROXY_ONLY",
             "entry_one_price_LABEL_ONLY": False, "b1_price_path_LABEL_ONLY": 1,
             "b1_net_close_mark_LABEL_ONLY": .02, "b2_net_close_mark_LABEL_ONLY": .03,
             "b3_net_close_mark_LABEL_ONLY": .01},
            {"code": "000002", "signal_date": "2026-01-06",
             "price_path_state": "PRICE_OPPORTUNITY_PROXY_ONLY",
             "entry_one_price_LABEL_ONLY": True, "b1_price_path_LABEL_ONLY": 1,
             "b1_net_close_mark_LABEL_ONLY": .10, "b2_net_close_mark_LABEL_ONLY": .20,
             "b3_net_close_mark_LABEL_ONLY": .30},
            {"code": "000003", "signal_date": "2026-01-07",
             "price_path_state": "PRICE_OPPORTUNITY_PROXY_ONLY",
             "entry_one_price_LABEL_ONLY": False, "b1_price_path_LABEL_ONLY": 0,
             "b1_net_close_mark_LABEL_ONLY": None, "b2_net_close_mark_LABEL_ONLY": None,
             "b3_net_close_mark_LABEL_ONLY": None},
        ])

    def test_queue_and_no_touch_are_not_filled_or_replaced(self):
        rows = prepare(self._selections(), self._outcomes())
        self.assertEqual(rows.execution_class.tolist(), [
            "LIMIT_PRICE_PATH_PROXY", "QUEUE_UNVERIFIABLE", "NO_NEXT_DAY_BOARD_TOUCH"])
        metrics = audit(rows)["methods"]["MODEL_ALL_TOP1"]
        self.assertEqual(metrics["price_path_fill_proxies"], 1)
        self.assertEqual(metrics["b1"]["known"], 1)
        self.assertAlmostEqual(metrics["b1"]["mean_net_close_mark"], .02)

    def test_duplicate_selection_for_method_date_is_rejected(self):
        selections = pd.concat([self._selections(), self._selections().iloc[[0]]])
        with self.assertRaises(ValueError):
            prepare(selections, self._outcomes())

    def test_paired_delta_requires_both_methods_to_have_fill_proxy(self):
        rows = pd.DataFrame([
            {"signal_date": "a", "selection_method": "left",
             "execution_class": "LIMIT_PRICE_PATH_PROXY", "b1_net_close_mark_LABEL_ONLY": .03},
            {"signal_date": "a", "selection_method": "right",
             "execution_class": "LIMIT_PRICE_PATH_PROXY", "b1_net_close_mark_LABEL_ONLY": .01},
            {"signal_date": "b", "selection_method": "left",
             "execution_class": "LIMIT_PRICE_PATH_PROXY", "b1_net_close_mark_LABEL_ONLY": .50},
            {"signal_date": "b", "selection_method": "right",
             "execution_class": "QUEUE_UNVERIFIABLE", "b1_net_close_mark_LABEL_ONLY": -.50},
        ])
        result = _paired_path_delta(rows, "left", "right", 1, bootstrap_samples=100)
        self.assertEqual(result["paired_common_fill_dates"], 1)
        self.assertAlmostEqual(result["mean_return_delta"], .02)


if __name__ == "__main__":
    unittest.main()
