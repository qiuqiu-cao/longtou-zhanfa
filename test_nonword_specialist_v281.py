from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import nonword_specialist_challenger_v281 as specialist


class DummyModel:
    classes_ = np.array([0, 1])
    def predict_proba(self, rows):
        p = np.asarray(rows["signal_feature"], dtype=float)
        return np.column_stack((1-p, p))


def history():
    dates = pd.bdate_range("2024-01-02", periods=400).strftime("%Y-%m-%d").tolist()
    rows = []
    for i, day in enumerate(dates[:-1]):
        for code, word in (("000001", 0), ("000002", 1)):
            rows.append(dict(date=day, target_date=dates[i+1], code=code, boards=2,
                daily_true_one_word=word, target=i % 2, label_valid=True,
                signal_feature=.4 if i % 3 else .6))
    return pd.DataFrame(rows), dates[310]


class SpecialistTests(unittest.TestCase):
    def test_training_cohort_and_realization_cutoffs(self):
        frame, cutoff = history()
        train, val, refit, start = specialist.splits(frame, cutoff)
        for part in (train, val, refit):
            self.assertTrue(part.daily_true_one_word.eq(0).all())
            self.assertTrue(part.target_date.lt(cutoff).all())
        self.assertTrue(train.target_date.lt(start).all())
        self.assertEqual(val.date.nunique(), 63)

    def test_current_future_labels_cannot_change_model_or_gate(self):
        frame, cutoff = history()
        changed = frame.copy()
        changed.loc[changed.target_date.ge(cutoff), "target"] = 1-changed.loc[changed.target_date.ge(cutoff), "target"]
        seen = []
        def fake_fit(rows, fields, kind, channel):
            seen.append(rows.copy())
            return DummyModel()
        with patch.object(specialist, "fit", side_effect=fake_fit):
            _, first = specialist.frozen(frame, ["signal_feature"], cutoff)
            _, second = specialist.frozen(changed, ["signal_feature"], cutoff)
        self.assertEqual(first, second)
        self.assertTrue(all(x.daily_true_one_word.eq(0).all() for x in seen))
        self.assertTrue(all(x.target_date.lt(cutoff).all() for x in seen))

    def test_all_rows_preserved_and_outside_domain_abstains(self):
        rows, cutoff = history()
        rows = rows[rows.date == cutoff].copy()
        rows.loc[rows.code == "000002", "signal_feature"] = .99
        log = dict(cutoff=cutoff, gate=dict(validation_qualified=False, threshold=.6))
        result = specialist.forecast(DummyModel(), rows, ["signal_feature"], log)
        self.assertEqual(len(result), len(rows))
        outside = result[result.daily_true_one_word.eq(1)]
        self.assertTrue(outside.probability.isna().all())
        self.assertTrue(outside.binary_call.eq("UNCERTAIN").all())
        self.assertEqual(result.ranking_first.sum(), 1)
        self.assertFalse(result.gated_ranking_call.any())

    def test_gate_uses_prediction_score_and_prior_qualification(self):
        rows, cutoff = history()
        rows = rows[rows.date == cutoff].copy()
        rows.loc[rows.code == "000001", "signal_feature"] = .7
        log = dict(cutoff=cutoff, gate=dict(validation_qualified=True, threshold=.65))
        result = specialist.forecast(DummyModel(), rows, ["signal_feature"], log)
        self.assertEqual(result.gated_ranking_call.sum(), 1)
        log = deepcopy(log); log["gate"]["threshold"] = .75
        result = specialist.forecast(DummyModel(), rows, ["signal_feature"], log)
        self.assertFalse(result.gated_ranking_call.any())

    def test_missing_outcome_keeps_entire_date_unresolved(self):
        rows, cutoff = history()
        rows = rows[rows.date == cutoff].copy()
        rows.loc[rows.code == "000002", "label_valid"] = False
        self.assertTrue(specialist.complete_rows(rows).empty)

    def test_high_negative_accuracy_does_not_become_promotion_precision(self):
        rows, cutoff = history()
        rows = rows[rows.date == cutoff].copy()
        rows["target"] = 0
        rows["signal_feature"] = .1
        p = specialist.forecast(DummyModel(), rows, ["signal_feature"],
            dict(cutoff=cutoff, gate=dict(validation_qualified=False, threshold=.6)))
        summary = specialist.summarize(p)
        self.assertEqual(summary["nonword_binary_accuracy"], 1)
        self.assertEqual(summary["full_nonword_top1"]["accuracy"], 0)
        self.assertEqual(summary["positive_binary_precision"]["n"], 0)
        self.assertEqual(summary["gated_coverage"], 0)


if __name__ == "__main__":
    unittest.main()
