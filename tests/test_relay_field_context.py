import copy
from pathlib import Path
import tempfile
import unittest

import pandas as pd

import relay_field_context as context
import stock_judgement_review as original


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"results/daily_review_20261013T074936Z/model/base_predictions.json"
CHOICE = ROOT/"research/model_upgrade_validation/choice_reasoning_20261013T113350Z/actual_final_DELIVERY/report.json"
RECEIPT = ROOT/"research/model_upgrade_validation/relay_context_20261011T151204Z/source_reconstruction_BEFORE_FITS.json"


class FieldScopeTests(unittest.TestCase):
    def test_missing_scope_never_manufactures_zero_distribution(self):
        d = context.describe(pd.Series([None, None]))
        self.assertEqual((d["rows"], d["known"], d["missing"]), (2, 0, 2))
        self.assertIsNone(d["median"])
        self.assertIn("UNKNOWN_INPUT", context.relative(None, d))
        self.assertEqual(context.relative(.1, d), "TRAINING_FIELD_ALL_UNKNOWN")

    def test_invalid_numeric_values_cannot_be_training_evidence(self):
        with self.assertRaises(ValueError):
            context.describe(pd.Series([1., float("inf")]))
        with self.assertRaises(ValueError):
            context.relative(float("nan"), context.describe(pd.Series([.1, .2])))


class ActualSourceTests(unittest.TestCase):
    def build(self, changed=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            choice = CHOICE
            if changed is not None:
                choice = path/"changed_choice.json"
                choice.write_text(context.original_json(changed), encoding="utf-8")
            return context.build(BASE, choice, RECEIPT, path/"context")

    def test_complete10_exact_training_and_current_fields(self):
        result = self.build()
        base = original.read(BASE)
        self.assertEqual(result["candidate_count"], 10)
        self.assertEqual(result["training_rows"], 3059)
        self.assertEqual(result["training_true_one_word_known"], 0)
        self.assertEqual(result["ranking"], base["relay"]["ranking"])
        self.assertFalse(result["proxy_source_equivalence_verified"])
        calls = {row["code"]: row for row in base["relay"]["candidates"]}
        for row in result["rows"]:
            self.assertEqual(len(row["fields"]), 29)
            self.assertEqual(row["raw_score"], calls[row["code"]]["raw_score"])
            self.assertEqual(row["model_positive"], calls[row["code"]]["predicted_positive"])
            loss = next(f for f in row["fields"] if f["field"] == "previous_limit_tail_loss")
            self.assertIsNone(loss["original_value"])
            self.assertEqual(loss["training_distribution"]["missing"], 0)
        self.assertEqual(result["numeric_fits"], 0)

    def test_self_consistent_fictional_terms_are_rejected(self):
        changed = copy.deepcopy(original.read(CHOICE))
        terms = changed["rows"][0]["positive_model_terms"]
        terms[0]["logit_contribution"] += .01
        terms[1]["logit_contribution"] -= .01
        with self.assertRaisesRegex(ValueError, "fictional"):
            self.build(changed)

    def test_fake_observed_value_with_same_score_is_rejected(self):
        changed = copy.deepcopy(original.read(CHOICE))
        terms = changed["rows"][0]["positive_model_terms"] + changed["rows"][0]["negative_model_terms"]
        next(t for t in terms if t["field"] == "turnover")["signal_value"] += .1
        with self.assertRaisesRegex(ValueError, "fictional"):
            self.build(changed)

    def test_negative_candidate_cannot_be_removed_from_context(self):
        changed = copy.deepcopy(original.read(CHOICE))
        changed["rows"].pop()
        with self.assertRaisesRegex(ValueError, "complete original pool"):
            self.build(changed)

    def test_partial38_feedback_cannot_claim_observed_complete39_tail(self):
        base = original.read(BASE)
        folder = Path(base["feature_archive"])
        manifest, features = original.read(folder/"manifest.json"), original.read(folder/"features.json")
        scope = context.feedback_scope(manifest, features)
        self.assertEqual(scope["previous_pool_count"], 39)
        self.assertEqual(scope["matched_previous_count"], 38)
        self.assertEqual(scope["missing_previous_codes"], ["600825"])
        self.assertFalse(scope["complete_previous_feedback"])
        self.assertFalse(scope["suspension_cause_confirmed"])
        features["market"]["features"]["previous_limit_tail_loss"] = 0.
        with self.assertRaisesRegex(ValueError, "cannot become a known"):
            context.feedback_scope(manifest, features)


if __name__ == "__main__":
    unittest.main()
