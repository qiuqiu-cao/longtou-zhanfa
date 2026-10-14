import unittest

import pandas as pd

from market_continue_challenger import CLASSES, gate, make_challenger
from model_upgrade_history import training_rows
from model_upgrade_metrics import classification_metrics
from rolling_forecast_challenger_v274 import make


class MarketContinueChallengerTests(unittest.TestCase):
    def test_only_estimator_class_weight_changes(self):
        baseline = make("forest").get_params()
        challenger = make_challenger().get_params()
        changes = {key for key in baseline
                   if key.startswith("randomforestclassifier__")
                   and baseline[key] != challenger[key]}
        self.assertEqual(changes, {"randomforestclassifier__class_weight"})
        for key in ("strategy", "add_indicator", "keep_empty_features"):
            self.assertEqual(baseline["simpleimputer__" + key], challenger["simpleimputer__" + key])

    def test_missing_continue_labels_cannot_pass(self):
        truth = ["IMPROVE", "WORSEN", None]
        base = classification_metrics(truth, ["WORSEN", "WORSEN", "CONTINUE"], CLASSES)
        new = classification_metrics(truth, ["IMPROVE", "WORSEN", "CONTINUE"], CLASSES)
        result = gate(base, new)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["CONTINUE_recall_strictly_improves"])
        self.assertFalse(result["checks"]["macro_f1_not_lower"])

    def test_continue_improvement_cannot_hide_other_recall_drop(self):
        truth = ["IMPROVE"] * 10 + ["CONTINUE"] * 10 + ["WORSEN"] * 10
        base = classification_metrics(truth, ["IMPROVE"] * 10 + ["IMPROVE"] * 10 + ["WORSEN"] * 10, CLASSES)
        new = classification_metrics(truth, ["CONTINUE"] * 5 + ["IMPROVE"] * 5 + ["CONTINUE"] * 10 + ["WORSEN"] * 10, CLASSES)
        result = gate(base, new)
        self.assertFalse(result["passed"])
        self.assertTrue(result["checks"]["CONTINUE_recall_strictly_improves"])
        self.assertFalse(result["checks"]["IMPROVE_complete_recall_not_lower"])

    def test_labels_on_or_after_fold_start_are_excluded(self):
        frame = pd.DataFrame({"date": ["2026-08-27", "2026-08-28", "2026-08-31"],
                              "target_date": ["2026-08-28", "2026-09-01", "2026-09-01"],
                              "label_valid": [True, True, False],
                              "target": ["CONTINUE", "WORSEN", "IMPROVE"]})
        selected = training_rows(frame, "2026-09-01", "market")
        self.assertEqual(selected.date.tolist(), ["2026-08-27"])


if __name__ == "__main__":
    unittest.main()
