import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from relay_expert_model import RelayExpertClassifier, PARAMETERS, RELAY_FIELDS


class ExpertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(20261007)
        cls.frame = pd.DataFrame(rng.normal(size=(80, len(RELAY_FIELDS))), columns=RELAY_FIELDS)
        cls.frame["signal_one_word"] = np.repeat([0., 1.], 40)
        cls.frame.loc[0, "turnover"] = np.nan
        cls.frame["previous_limit_tail_loss"] = np.nan
        cls.labels = np.tile([0, 1, 0, 1], 20)
        cls.weights = np.repeat(.1, 80)
        cls.h1 = RelayExpertClassifier("H1").fit(cls.frame, cls.labels, cls.weights)
        cls.h2 = RelayExpertClassifier("H2").fit(cls.frame, cls.labels, cls.weights)

    def test_fixed_parameters_and_no_early_stopping(self):
        self.assertEqual(self.h1.global_.n_iter_, 160)
        self.assertFalse(self.h2.global_.early_stopping)
        self.assertEqual(self.h1.global_.get_params()["random_state"], PARAMETERS["random_state"])

    def test_branches_partition_full_training_members(self):
        self.assertEqual(self.h2.members_by_branch_, {"0": 40, "1": 40})
        self.assertEqual(self.h2.members_, 80)
        self.assertEqual(self.h2.unknown_proxy_training_members_, 0)

    def test_unknown_proxy_global_fallback_not_false_identity(self):
        unknown = self.frame.iloc[:3].copy(); unknown["signal_one_word"] = np.nan
        np.testing.assert_allclose(self.h2.predict_proba(unknown), self.h1.predict_proba(unknown), atol=1e-12)

    def test_routing_uses_only_original_signal_proxy(self):
        x = self.frame.iloc[:5].copy()
        actual = self.h2.predict_proba(x)
        expected = self.h2.models_[0.].predict_proba(self.h2.preprocessor_.transform(x))
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_saved_model_load_exact_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "h2.joblib"; joblib.dump(self.h2, path)
            np.testing.assert_allclose(joblib.load(path).predict_proba(self.frame), self.h2.predict_proba(self.frame), atol=1e-12)

    def test_future_label_column_rejected_and_cannot_enter_features(self):
        frame = self.frame.copy(); frame["target_day_result"] = 999
        with self.assertRaises(ValueError): self.h2.predict_proba(frame)

    def test_missing_fields_and_invalid_proxy_rejected(self):
        with self.assertRaises(ValueError): self.h2.predict_proba(self.frame.drop(columns="price"))
        invalid = self.frame.copy(); invalid.loc[0, "signal_one_word"] = .5
        with self.assertRaises(ValueError): self.h2.predict_proba(invalid)

    def test_single_class_branch_cannot_be_synthetically_completed(self):
        labels = self.labels.copy(); labels[:40] = 0
        with self.assertRaises(ValueError): RelayExpertClassifier("H2").fit(self.frame, labels, self.weights)

    def test_original_per_day_weights_not_rebalanced_by_branch(self):
        seen = []
        class Estimator:
            def __init__(self, **kwargs): self.n_iter_ = kwargs["max_iter"]
            def fit(self, x, y, sample_weight):
                seen.append(np.asarray(sample_weight).copy()); return self
        weights = np.linspace(.1, 1., 80)
        with patch("relay_expert_model.HistGradientBoostingClassifier", Estimator):
            RelayExpertClassifier("H2").fit(self.frame, self.labels, weights)
        np.testing.assert_array_equal(seen[0], weights)
        np.testing.assert_array_equal(seen[1], weights[:40])
        np.testing.assert_array_equal(seen[2], weights[40:])


class ExpertProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol = Path("research/model_upgrade_validation/relay_expert_20261011T162713Z/protocol_BEFORE_FITS.json")

    def test_prior_experiment_namespace_not_mutated(self):
        import relay_context_shadow as old
        import relay_expert_shadow as new
        before = dict(old.r.ARMS)
        runtime = new.runtime(self.protocol)
        self.assertEqual(list(runtime.r.ARMS), ["H1", "H2"])
        self.assertEqual(old.r.ARMS, before)

    def test_failed_historical_arms_cannot_register_shadow(self):
        from relay_expert_shadow import runtime
        with self.assertRaisesRegex(ValueError, "no fixed historical"):
            runtime(self.protocol).approved_research(self.protocol)

    def test_repeat_fit_rejected_before_training(self):
        import relay_expert_research as research
        with self.assertRaisesRegex(ValueError, "already attempted"):
            research.compare(self.protocol)

    def test_architecture_forward_gate_uses_h_names_and_h1_tie(self):
        from relay_expert_shadow import runtime
        from test_relay_context_research import ContextTests
        days, rows, reasons = ContextTests().synthetic_window()
        renamed = {"BASE": rows["BASE"], "H1": rows["E1"], "H2": rows["E2"]}
        result = runtime(self.protocol).assess(renamed, reasons, days, days)
        self.assertEqual(result["winner"], "H1")
        self.assertTrue(result["challenges"]["H2"]["passed"])


if __name__ == "__main__": unittest.main()
