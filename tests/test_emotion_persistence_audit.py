import unittest
import pandas as pd
from emotion_persistence_audit import evaluate, STATES


class PersistenceAuditTest(unittest.TestCase):
    def rows(self):
        row = dict(signal_date="2026-09-15", target_date="2026-09-16",
                   current_state=STATES[0], target_state=STATES[1])
        for i, state in enumerate(STATES):
            row["p_"+state] = float(i == 1)
            row["marginal_p_"+state] = float(i == 0)
        return pd.DataFrame([row])

    def test_paired_metrics(self):
        result = evaluate(self.rows(), repetitions=10)
        self.assertEqual(result["accuracy_difference"], 1.)
        self.assertEqual(result["model_minus_marginal_accuracy"], 1.)
        self.assertFalse(result["deployment_authorized"])

    def test_invalid_time_and_probabilities(self):
        rows = self.rows()
        rows["target_date"] = rows["signal_date"]
        with self.assertRaises(ValueError):
            evaluate(rows)
        rows = self.rows()
        rows["p_"+STATES[0]] = -1
        with self.assertRaises(ValueError):
            evaluate(rows)
