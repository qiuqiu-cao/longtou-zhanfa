import unittest
import pandas as pd
from opening_context_v39 import attach, FIELDS


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(code="000017",signal_date="2024-01-12",entry_date="2024-01-15",promotion_LABEL_ONLY=True)]
        self.features = pd.DataFrame([dict(code="000017",signal_date="2024-01-12",
                feature_available_not_before="2024-01-12T15:05:00+08:00", **{k:1.0 for k in FIELDS})])

    def test_future_rejected(self):
        self.features["feature_available_not_before"] = "2024-01-15T15:05:00+08:00"
        with self.assertRaises(ValueError):
            attach(self.rows,self.features)

    def test_outcome_does_not_change_context(self):
        first = attach(self.rows,self.features)[0]["prior_context"]
        self.rows[0]["promotion_LABEL_ONLY"] = False
        self.assertEqual(first,attach(self.rows,self.features)[0]["prior_context"])

    def test_missing_not_zero(self):
        self.rows[0]["code"] = "000002"
        self.assertIsNone(attach(self.rows,self.features)[0]["prior_context"])

    def test_no_future_column_selected(self):
        self.features["future_profit"] = 99
        self.assertNotIn("future_profit",attach(self.rows,self.features)[0]["prior_context"])
