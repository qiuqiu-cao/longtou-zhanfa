import unittest

import pandas as pd

from relay_phase_response_research import features, baseline_rows, TERMS, INTERACTIONS


class PhaseResponseTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({"date": ["2026-09-01"] * 2, "code": ["600001", "600002"],
            "signal_one_word": [0., 1.], "target": [0., 1.], "boards": [2, 3], "emotion_score": [.7, .7]})

    def context(self, phase="加速"):
        return {"rows": [{"date": "2026-09-01", "code": c, "phase_feature": phase} for c in ["600001", "600002"]]}

    def test_original_columns_are_exactly_preserved(self):
        f = self.frame()
        result = features(f, self.context())
        pd.testing.assert_frame_equal(f, result[f.columns])
        self.assertEqual(len(result.columns) - len(f.columns), 14)

    def test_fixed_phase_dummy_and_proxy_products(self):
        f = features(self.frame(), self.context())
        self.assertEqual(f[TERMS].sum(axis=1).tolist(), [1., 1.])
        self.assertEqual(f[INTERACTIONS].sum(axis=1).tolist(), [0., 1.])

    def test_unknown_phase_not_zero_or_inferred_from_proxy(self):
        result = features(self.frame(), self.context(None))
        self.assertTrue(result[TERMS + INTERACTIONS].isna().all().all())

    def test_missing_candidate_or_other_day_cannot_replace_scope(self):
        context = self.context()
        context["rows"][0]["date"] = "2026-09-02"
        with self.assertRaisesRegex(ValueError, "complete"):
            features(self.frame(), context)

    def test_future_label_changes_do_not_change_new_inputs(self):
        f = self.frame()
        first = features(f, self.context())
        f["target"] = [1., 0.]
        later = features(f, self.context())
        pd.testing.assert_frame_equal(first[TERMS + INTERACTIONS], later[TERMS + INTERACTIONS])

    def test_missing_proxy_keeps_interactions_unknown(self):
        f = self.frame()
        f.loc[0, "signal_one_word"] = float("nan")
        result = features(f, self.context())
        self.assertTrue(result.loc[0, INTERACTIONS].isna().all())

    def test_saved_csv_unknown_truth_and_false_call_preserved(self):
        f = pd.DataFrame({"truth": [float("nan"), False], "prediction": [False, True],
            "actual_one_word_daily": [float("nan"), 0.], "legacy_proxy": [0., 1.], "raw_score": [.2, .6]})
        rows = baseline_rows(f)
        self.assertIsNone(rows[0]["truth"])
        self.assertIs(rows[0]["prediction"], False)
        self.assertIs(rows[1]["truth"], False)

    def test_unregistered_phase_rejected(self):
        with self.assertRaisesRegex(ValueError, "unregistered"):
            features(self.frame(), self.context("后来判定牛市"))


if __name__ == "__main__":
    unittest.main()
