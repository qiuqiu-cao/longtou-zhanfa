"""Synthetic contracts only: never open/fill/fits the historical archives."""
import copy
import json
from io import StringIO
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd

import model_upgrade_history as history


class SyntheticModel:
    classes_ = np.asarray(["CONTINUE", "IMPROVE", "WORSEN"])

    def predict_proba(self, frame):
        return np.tile([.5, .25, .25], (len(frame), 1))


class HistoryTimingAndCohorts(unittest.TestCase):
    def test_cached_market_uses_original_csv_empty_string_roundtrip(self):
        combined = pd.DataFrame({"date": ["2026-08-31", "2026-09-01"],
            "emotion_warnings": ["", "synthetic warning"], "x": [.123456789123, np.nan]})
        saved = pd.read_csv(StringIO(combined.to_csv(index=False)))
        self.assertTrue(pd.isna(saved.loc[0, "emotion_warnings"]))
        history._assert_cached_market_equivalent(combined, saved)
        changed = saved.copy()
        changed.loc[0, "x"] += 1e-6
        with self.assertRaises(AssertionError):
            history._assert_cached_market_equivalent(combined, changed)

    def test_nonconvergence_rejected_without_alternative_fit(self):
        from sklearn.exceptions import ConvergenceWarning
        frame = pd.DataFrame({"x": [0., 1.], "target": [0, 1]})
        def nonconvergent(*args, **kwargs):
            warnings.warn("synthetic convergence failure", ConvergenceWarning)
        with patch("rolling_forecast_challenger_v274.fit", side_effect=nonconvergent) as fit:
            with self.assertRaisesRegex(ValueError, "FIT_NONCONVERGENCE"):
                history._fit(frame, ["x"], "relay")
            self.assertEqual(fit.call_count, 1)

    def test_market_predecessor_missing_and_future_mutation(self):
        sessions = ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-07"]
        # Sept 2 is absent: Sept 3 must not use Sept 1 as its predecessor.
        frame = pd.DataFrame({"date": [sessions[i] for i in [0, 2, 3, 4]],
            "yesterday_premium": [-.01, -.02, -.01, .01],
            "promotion_mid": [.7, .6, .5, .2], "failed_rate": [.1, .2, .3, .4]})
        observed = history.interactions(frame, "market", sessions)
        self.assertTrue(observed.loc[1, list(history.MARKET_INTERACTIONS)].isna().all())
        self.assertEqual(observed.loc[2, history.MARKET_INTERACTIONS[0]], 1.)
        self.assertEqual(observed.loc[3, history.MARKET_INTERACTIONS[0]], 0.)
        changed = frame.copy()
        changed.loc[3, ["promotion_mid", "failed_rate", "yesterday_premium"]] = [1e6, -1e6, -1e6]
        revised = history.interactions(changed, "market", sessions)
        pd.testing.assert_frame_equal(observed.iloc[:3], revised.iloc[:3])
        frame.loc[2, "yesterday_premium"] = np.nan
        missing = history.interactions(frame, "market", sessions)
        self.assertTrue(missing.loc[2, list(history.MARKET_INTERACTIONS)].isna().all())

    def test_complete_same_height_group_includes_unknown_failure_and_one_word(self):
        frame = pd.DataFrame({"date": ["2026-08-31"] * 3, "code": ["000001", "000002", "000003"],
            "boards": [2] * 3, "same_height_rivals": [2] * 3,
            "signal_amount": [10., 20., np.nan], "turnover": [1., 2., 3.],
            "target": [0., 1., np.nan], "label_valid": [True, True, False],
            "actual_one_word_daily": [0., 1., np.nan]})
        augmented = history.interactions(frame, "relay")
        self.assertEqual(augmented.code.tolist(), frame.code.tolist())
        self.assertTrue(pd.isna(augmented.loc[2, history.RELAY_INTERACTIONS[0]]))
        self.assertTrue(np.isfinite(augmented.loc[2, history.RELAY_INTERACTIONS[1]]))
        with self.assertRaisesRegex(ValueError, "incomplete original"):
            history.interactions(frame.iloc[:2], "relay")
        # Labels and one-word flags cannot affect pre-signal interactions.
        changed = frame.copy()
        changed["target"], changed["actual_one_word_daily"] = [1., 0., 1.], [1., 0., 0.]
        pd.testing.assert_frame_equal(augmented[list(history.RELAY_INTERACTIONS)],
            history.interactions(changed, "relay")[list(history.RELAY_INTERACTIONS)])

    def test_training_last252_eligible_signal_dates_and_strict_label_boundary(self):
        dates = pd.bdate_range("2024-01-01", periods=300).strftime("%Y-%m-%d").tolist()
        frame = pd.DataFrame({"date": dates[:-1], "target_date": dates[1:],
            "target": np.arange(299) % 2, "label_valid": True, "code": "000001"})
        cutoff = dates[-1]
        frame.loc[40, "label_valid"] = False
        frame.loc[41, "target"] = np.nan
        # A late-arriving older signal label cannot enter until it is known.
        frame.loc[0, "target_date"] = cutoff
        chosen = history.training_rows(frame, cutoff, "relay")
        self.assertEqual(chosen.date.nunique(), 252)
        self.assertTrue(chosen.target_date.lt(cutoff).all())
        self.assertTrue(chosen.date.lt(cutoff).all())
        modified = frame.copy()
        modified.loc[modified.target_date.ge(cutoff), "target"] = 100000
        pd.testing.assert_frame_equal(chosen, history.training_rows(modified, cutoff, "relay"))

    def test_exact_target_rejects_skip_without_dropping_rows(self):
        sessions = ["2026-09-24", "2026-09-28", "2026-09-29", "2026-09-30"]
        frame = pd.DataFrame({"date": sessions[:2], "target_date": [sessions[1], sessions[3]],
            "target": ["CONTINUE", "WORSEN"], "label_valid": True})
        result, changed = history.enforce_exact_targets(frame, sessions)
        self.assertEqual(len(result), 2)
        self.assertEqual(changed, 1)
        self.assertEqual(result.label_valid.tolist(), [True, False])

    def test_reproduction_rejects_universe_classes_and_scores(self):
        baseline = pd.DataFrame({"date": ["2026-09-01", "2026-09-02"],
            "predicted": ["CONTINUE", "WORSEN"], "confidence": [.7, .8]})
        self.assertTrue(history.compare_reproduction(baseline.copy(), baseline, "market")["passed"])
        for field, value in [("date", "2026-09-03"), ("predicted", "IMPROVE"), ("confidence", .800001)]:
            changed = baseline.copy()
            changed.loc[1, field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                history.compare_reproduction(changed, baseline, "market")

    def test_protocol_recipe_bindings_and_source_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "synthetic.csv"
            source.write_text("date,x\n2026-01-01,1\n", encoding="utf-8")
            prepared = {"provenance": {"source_hashes": {str(source): history.sha(source)}, "dependency_hashes": {}}}
            protocol = {"recipe": history.recipe(), "input_hashes": prepared["provenance"]["source_hashes"]}
            history._validate_protocol(protocol, prepared)
            changed = copy.deepcopy(protocol)
            changed["recipe"]["relay_window"] = 120
            with self.assertRaisesRegex(ValueError, "recipe"):
                history._validate_protocol(changed, prepared)
            source.write_text("changed after registration", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                history._validate_protocol(protocol, prepared)

    def test_history_must_complete_before_freezing_models(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "reproduction").mkdir()
            prepared = {"provenance": {"source_hashes": {}, "dependency_hashes": {}}}
            protocol = {"recipe": history.recipe()}
            bound = history.digest(protocol)
            (output / "reproduction/report.json").write_text(json.dumps({
                "status": "BOTH_BASES_REPRODUCED", "protocol_sha256": bound}), encoding="utf-8")
            with patch.object(history, "_fit") as fit:
                with self.assertRaises(FileNotFoundError):
                    history.freeze_module("market", output, protocol, prepared)
                fit.assert_not_called()

    def test_september_base_and_freeze_reuse_bound_cache_without_refitting(self):
        dates = pd.bdate_range("2024-01-01", "2026-10-01").strftime("%Y-%m-%d").tolist()
        frame = pd.DataFrame({"date": dates[:-1], "target_date": dates[1:], "label_valid": True,
            "target": np.resize(["CONTINUE", "IMPROVE", "WORSEN"], len(dates) - 1),
            "limit_count": 20., "yesterday_premium": -.01, "promotion_mid": .4, "failed_rate": .2})
        fields = ["limit_count", "yesterday_premium", "promotion_mid", "failed_rate"]
        prepared = {"market": frame, "market_fields": fields,
            "provenance": {"source_hashes": {}, "dependency_hashes": {}, "sessions": dates}}
        protocol = {"recipe": history.recipe()}
        bound = history.digest(protocol)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "reproduction").mkdir()
            (output / "reproduction/report.json").write_text(json.dumps({
                "status": "REPRODUCED", "protocol_sha256": bound}), encoding="utf-8")
            train = history.training_rows(frame, history.FREEZE_CUTOFF, "market")
            history._save_model(output / "reproduction/BASE_artifacts/market", SyntheticModel(), fields,
                train, "market", "BASE", prepared, bound, history.FREEZE_CUTOFF)
            with patch.object(history, "_fit", side_effect=lambda *a: SyntheticModel()) as fit:
                result = history.evaluate_module("market", output, protocol, prepared)
                # 2 earlier folds *3 arms plus Sept C1/C2; BASE came from reproduction.
                self.assertEqual(fit.call_count, 8)
                for entry in result["fit_logs"]:
                    self.assertLess(entry["training_max_label_date"], entry["fold"]["start"])
            with patch.object(history, "_fit") as fit:
                frozen = history.freeze_module("market", output / "models", protocol, prepared)
                fit.assert_not_called()
                self.assertEqual(set(frozen["manifests"]), set(history.ARMS))
                for arm, manifest in frozen["manifests"].items():
                    cached = output / "market_history/september_fit_cache" / arm / "model.joblib"
                    self.assertEqual(history.sha(cached), manifest["sha256"])
                with self.assertRaises(FileExistsError):
                    history.freeze_module("market", output / "models", protocol, prepared)


if __name__ == "__main__":
    unittest.main()
