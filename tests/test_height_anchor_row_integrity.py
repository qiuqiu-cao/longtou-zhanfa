import unittest
import pandas as pd
from audit_height_anchor_row_integrity import audit


class RowIntegrityTest(unittest.TestCase):
    def test_missing_label_preserved(self):
        rows = pd.DataFrame({"signal_date": ["a"], "code": ["000001"], "label": [float("nan")]})
        self.assertTrue(audit(rows, rows)["passed"])
        corrupted = rows.copy()
        corrupted["label"] = 1.0
        self.assertFalse(audit(corrupted, rows)["passed"])

    def test_unmatched_and_duplicate_rejected(self):
        rows = pd.DataFrame({"signal_date": ["a"], "code": ["000001"], "label": [1.]})
        self.assertFalse(audit(rows.assign(code="000002"), rows)["passed"])
        with self.assertRaises(ValueError):
            audit(rows, pd.concat([rows, rows]))
