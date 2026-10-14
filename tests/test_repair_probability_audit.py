import unittest
import pandas as pd
from repair_probability_audit import evaluate, CLASSES


class RepairAuditTest(unittest.TestCase):
    def test_perfect_model_beats_prior(self):
        row = dict(signal_date="2026-09-15", target_date="2026-09-16", target_class=CLASSES[0])
        for i, c in enumerate(CLASSES):
            row["challenger_p_"+c] = float(i == 0)
            row["prior_p_"+c] = 1/3
        result = evaluate(pd.DataFrame([row]), repeats=10)
        self.assertLess(result["challenger_minus_prior_brier"], 0)
        self.assertFalse(result["deployment_authorized"])
        row["prior_p_"+CLASSES[0]] = -1
        with self.assertRaises(ValueError):
            evaluate(pd.DataFrame([row]))
