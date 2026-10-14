import unittest

import pandas as pd

from reconcile_target_cases_v18 import reconcile


class ReconcileTests(unittest.TestCase):
    def fixture(self):
        case = dict(code="000017", start="2024-01-01", end="2024-01-02", sessions=2,
                    threshold_pct=20, comparison="greater_than")
        stock = pd.DataFrame(dict(code=["000017"]*2, date=[case["start"], case["end"]],
                                  close=[11, 12.1], pre_close=[10, 11]))
        index = pd.DataFrame(dict(date=stock.date, benchmark_return_pct=[0, 0]))
        return case, stock, index

    def test_match_is_not_training_readiness(self):
        result = reconcile(*self.fixture())
        self.assertEqual(result["result"], "THRESHOLD_DIRECTION_MATCH")
        self.assertFalse(result["training_label_ready"])

    def test_missing_day_rejected(self):
        case, stock, index = self.fixture()
        self.assertEqual(reconcile(case, stock.iloc[:1], index)["result"], "DATA_LIMITED")

    def test_mismatch_preserved(self):
        case, stock, index = self.fixture()
        case["threshold_pct"] = 100
        self.assertEqual(reconcile(case, stock, index)["result"], "MISMATCH")


if __name__ == "__main__":
    unittest.main()
