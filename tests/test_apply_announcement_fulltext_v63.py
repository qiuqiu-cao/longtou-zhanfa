import unittest

import pandas as pd

from apply_announcement_fulltext_v63 import REVIEWS, apply


class FulltextOverlayTests(unittest.TestCase):
    def test_reviewed_closed_rows_clear_but_new_plan_remains_hard(self):
        rows = [{"code": item["code"], "title": item["contains"], "risk_hard": 1,
                 "risk_type": "减持", "risk_window_status": "active"} for item in REVIEWS]
        rows.append({"code": "603077", "title": "和邦生物股东减持股份计划公告",
                     "risk_hard": 1, "risk_type": "减持", "risk_window_status": "active"})
        result, ledger = apply(pd.DataFrame(rows))
        self.assertEqual(len(ledger), 4)
        self.assertEqual(int(result.iloc[:4].risk_hard.sum()), 0)
        self.assertEqual(int(result.iloc[4].risk_hard), 1)


if __name__ == "__main__":
    unittest.main()
