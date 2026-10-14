"""Causality, weights, missingness and independent fixed-comparison checks."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

import relay_class_balance_research as research


class FixedRelayBalanceTests(unittest.TestCase):
    def training(self):
        return pd.DataFrame({"date": ["2026-08-27"]*4+["2026-08-28"]*2,
            "target_date": ["2026-08-28"]*4+["2026-08-31"]*2,
            "target": [0., 0., 0., 1., 0., 1.], "label_valid": [True]*6,
            "boards": [2., 3., 4., 2., 4., 3.], "price": [4., 5., 6., 7., 8., 9.]})

    def test_class_mass_equal_and_total_daily_mass_preserved(self):
        train = self.training(); w, factors, masses = research.training_weights(train)
        self.assertAlmostEqual(w.sum(), 2.)
        self.assertAlmostEqual(masses[0], 1.25); self.assertAlmostEqual(masses[1], .75)
        adjusted = w * train.target.map(factors)
        self.assertAlmostEqual(adjusted[train.target.eq(0)].sum(), 1.)
        self.assertAlmostEqual(adjusted[train.target.eq(1)].sum(), 1.)
        self.assertAlmostEqual(adjusted.sum(), w.sum())

    def test_future_targets_features_do_not_change_train_factors_or_scores(self):
        train = self.training()
        future = pd.DataFrame({"date": ["2026-09-01"], "target_date": ["2026-09-02"], "target": [0.],
            "label_valid": [True], "boards": [20.], "price": [100.]})
        combined = pd.concat([train, future], ignore_index=True)
        before = research.history.training_rows(combined, "2026-09-01", "relay")
        altered = combined.copy(); altered.loc[altered.date.ge("2026-09-01"), ["target", "boards", "price"]] = [1., 999., -999.]
        after = research.history.training_rows(altered, "2026-09-01", "relay")
        pd.testing.assert_frame_equal(before, after)
        a, _ = research.fit_once(before, ["boards", "price"])
        b, _ = research.fit_once(after, ["boards", "price"])
        np.testing.assert_array_equal(a.predict_proba(train[["boards", "price"]]), b.predict_proba(train[["boards", "price"]]))

    def test_label_cutoff_and_single_class_fail_without_fallback(self):
        train = self.training(); train.loc[0, "target_date"] = "2026-09-01"
        kept = research.history.training_rows(train, "2026-09-01", "relay")
        self.assertEqual(len(kept), 5)
        train.target = 0.
        with self.assertRaises(ValueError): research.fit_once(train, ["boards", "price"])

    def test_forbidden_feature_and_all_unknown_training_rejected(self):
        train = self.training(); train["trade_close"] = 100.
        with self.assertRaises(ValueError): research.fit_once(train, ["trade_close"])
        train.price = np.nan
        with self.assertRaises(ValueError): research.fit_once(train, ["price"])

    def test_nonconvergence_is_failure_not_retry(self):
        model = unittest.mock.MagicMock(); model.steps = [("logisticregression", object())]
        model.fit.side_effect = ConvergenceWarning("synthetic convergence failure")
        with patch("rolling_forecast_challenger_v274.make", return_value=model):
            with self.assertRaises(ConvergenceWarning): research.fit_once(self.training(), ["boards", "price"])
        self.assertEqual(model.fit.call_count, 1)

    def test_unknown_labels_predictions_and_coverage_independent_counts(self):
        frame = pd.DataFrame({"date": ["2026-08-27"]*4, "target_date": ["2026-08-28"]*4,
            "code": ["000001", "000002", "000003", "000004"], "boards": [2]*4,
            "label_valid": [True, True, True, False], "target": [1., 0., 1., np.nan],
            "predicted": [1., 1., np.nan, 0.], "status": ["PREDICTED_UNCALIBRATED"]*2+["UNKNOWN_INPUT", "PREDICTED_UNCALIBRATED"],
            "raw_score": [.8, .6, np.nan, .2], "actual_one_word_daily": [0., 0., 0., 0.]})
        rows = research.project(frame, "R1", ["2026-08-27", "2026-08-28"])
        m = research.statistics(rows, ["2026-08-27", "2026-08-28"], {"start": "2026-08-27", "end": "2026-08-27"})
        self.assertEqual(m["original_registered_count"], 4); self.assertEqual(m["scoreable_count"], 2)
        self.assertEqual(m["confusion"], {"TP": 1, "TN": 0, "FP": 1, "FN": 0})
        self.assertEqual(m["complete_positive_Recall"], .5)
        self.assertEqual(m["true_nonword_rank_first"]["scoreable_days"], 0)
        self.assertIsNone(m["calibrated_Brier"])

    def test_exact_official_target_required(self):
        frame = pd.DataFrame({"date": ["2026-09-30"], "target_date": ["2026-10-01"], "code": ["000001"],
            "boards": [2], "label_valid": [True], "target": [1.], "predicted": [1.], "raw_score": [.8], "status": ["PREDICTED_UNCALIBRATED"]})
        self.assertIsNone(research.project(frame, "R1", ["2026-09-30", "2026-10-08"])[0]["truth"])

    def test_rank_ties_and_real_word_unknown_not_proxy_filled(self):
        rows = [{"code": "000002", "signal_date": "2026-08-27", "boards": 2, "prediction": False, "truth": False, "raw_score": .6, "actual_one_word_daily": 0},
                {"code": "000001", "signal_date": "2026-08-27", "boards": 2, "prediction": False, "truth": True, "raw_score": .6, "actual_one_word_daily": 0}]
        m = research.statistics(rows, ["2026-08-27"], {"start": "2026-08-27", "end": "2026-08-27"})
        self.assertEqual(m["true_nonword_rank_first"]["daily_rows"][0]["selected_code"], "000001")
        self.assertFalse(m["true_nonword_rank_first"]["daily_rows"][0]["original_prediction"])
        rows[0]["actual_one_word_daily"] = None
        m = research.statistics(rows, ["2026-08-27"], {"start": "2026-08-27", "end": "2026-08-27"})
        self.assertEqual(m["true_nonword_rank_first"]["daily_rows"][0]["status"], "UNKNOWN_OHLC_IDENTITY")


if __name__ == "__main__": unittest.main()
