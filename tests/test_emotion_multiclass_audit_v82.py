import unittest

import pandas as pd

from emotion_multiclass_audit_v82 import STATES, _group_metrics, latest_prediction, walk_forward


class EmotionMulticlassAuditTests(unittest.TestCase):
    def test_prediction_does_not_use_target_transition(self):
        states = ["启动", "一致", "退潮", "启动", "一致"]
        frame = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=len(states)),
                              "emotion_state": states})
        result = walk_forward(frame, initial_transitions=1, alpha=1)
        first = result.iloc[0]
        # At signal index 1 only 启动->一致 is known.  The future 一致->退潮
        # transition must not have influenced this probability.
        self.assertAlmostEqual(first["p_退潮"], 1 / 7)
        self.assertAlmostEqual(first["p_启动"], 1 / 7)

    def test_probabilities_sum_to_one(self):
        states = (STATES * 4)[:24]
        frame = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=len(states)),
                              "emotion_state": states})
        result = walk_forward(frame, initial_transitions=3)
        self.assertTrue(((result[[f"p_{s}" for s in STATES]].sum(axis=1) - 1).abs() < 1e-12).all())
        latest = latest_prediction(frame)
        self.assertAlmostEqual(sum(latest["probabilities"].values()), 1.0)
        self.assertAlmostEqual(sum(latest["group_probabilities"].values()), 1.0)
        self.assertEqual(latest["status"], "DIAGNOSTIC_ONLY")

    def test_group_metrics_keep_repair_separate(self):
        rows = pd.DataFrame([{
            "target_state": "修复", **{f"p_{state}": (1.0 if state == "修复" else 0.0) for state in STATES}
        }])
        metrics = _group_metrics(rows)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["per_group"]["REPAIR"]["recall"], 1.0)

    def test_macro_f1_penalizes_unpredicted_supported_group(self):
        rows = pd.DataFrame([
            {"target_state": target, **{f"p_{state}": (1.0 if state == "大分歧" else 0.0)
                                        for state in STATES}}
            for target in ("大分歧", "修复")
        ])
        metrics = _group_metrics(rows)
        self.assertEqual(metrics["per_group"]["REPAIR"]["f1"], 0.0)
        self.assertLess(metrics["macro_f1"], metrics["per_group"]["RISK_OFF"]["f1"])

    def test_group_predictions_are_reproducible(self):
        states = (STATES * 30)[:180]
        frame = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=len(states)),
                              "emotion_state": states})
        rows = walk_forward(frame, initial_transitions=20)
        self.assertEqual(_group_metrics(rows), _group_metrics(rows.copy()))

    def test_unknown_state_rejected(self):
        frame = pd.DataFrame({"date": ["2026-01-01", "2026-01-02"],
                              "emotion_state": ["启动", "未知"]})
        with self.assertRaisesRegex(ValueError, "unknown emotion"):
            walk_forward(frame, initial_transitions=0)


if __name__ == "__main__":
    unittest.main()
