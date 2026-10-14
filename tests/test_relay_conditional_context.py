import unittest

import pandas as pd

from relay_conditional_context import cells, decomposition, training_parts, outcomes, phase_context


class ConditionalContextTests(unittest.TestCase):
    def frame(self, positives, board=2, state="修复", proxy=0):
        f = pd.DataFrame({"date": ["2025-01-02"] * len(positives), "boards": [board] * len(positives),
            "emotion_state": [state] * len(positives), "signal_one_word": [proxy] * len(positives),
            "target": positives, "full_day_weight": [1 / len(positives)] * len(positives)})
        return cells(f)

    def test_single_common_cell_is_within_change(self):
        r = decomposition(self.frame([1, 0]), self.frame([0, 0]))
        for mode in ["raw_rows", "equal_full_day_weight"]:
            self.assertEqual(r[mode]["delta"], .5)
            self.assertEqual(r[mode]["within_cell_rate_change"], .5)
            self.assertEqual(r[mode]["candidate_composition_change"], 0.)

    def test_mixture_change_does_not_become_within_cell_evidence(self):
        recent = pd.concat([self.frame([1] * 3, board=2), self.frame([0], board=3)], ignore_index=True)
        earlier = pd.concat([self.frame([1], board=2), self.frame([0] * 3, board=3)], ignore_index=True)
        r = decomposition(recent, earlier)["raw_rows"]
        self.assertEqual(r["delta"], .5)
        self.assertEqual(r["within_cell_rate_change"], 0.)
        self.assertEqual(r["candidate_composition_change"], .5)

    def test_unmatched_cells_keep_full_mass(self):
        r = decomposition(self.frame([1, 1], board=2), self.frame([0, 0], board=3))
        self.assertEqual(r["raw_rows"]["delta"], 1.)
        self.assertEqual(r["raw_rows"]["unmatched_cell_contribution"], 1.)
        self.assertEqual(r["raw_rows"]["common_recent_mass"], 0.)

    def test_small_cell_and_missing_state_remain_explicit(self):
        r = decomposition(self.frame([1], state=None), self.frame([0], state=None))
        self.assertEqual(r["cell_rows"][0]["original_emotion"], "UNKNOWN")
        self.assertFalse(r["cell_rows"][0]["adequate_descriptive_support_both"])

    def test_labels_after_prediction_month_never_enter_training(self):
        days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-02", periods=253)]
        frame = pd.DataFrame({"date": days[:-1], "target_date": days[1:], "target": [0, 1] * 126,
            "label_valid": True, "boards": 2, "emotion_state": "加速", "signal_one_word": 0.})
        future = frame.iloc[-1:].copy()
        future["date"] = "2025-12-31"
        future["target_date"] = "2026-01-02"
        f = pd.concat([frame, future], ignore_index=True)
        recent, earlier = training_parts(f, "2025-01-01")
        self.assertEqual(recent.date.nunique(), 63)
        self.assertEqual(earlier.date.nunique(), 189)
        self.assertLess(recent.target_date.max(), "2025-01-01")
        self.assertNotIn("2025-12-31", recent.date.tolist() + earlier.date.tolist())

    def test_unknown_outcome_stays_in_metrics_denominator(self):
        rows = [{"signal_date": "2025-01-02", "truth": True, "prediction": False},
            {"signal_date": "2025-01-02", "truth": None, "prediction": True}]
        r = outcomes(rows)
        self.assertEqual((r["n"], r["scoreable"], r["unknown"], r["FN"]), (2, 1, 1, 1))
        self.assertEqual(r["coverage"], .5)

    def test_phase_supplement_uses_only_exact_signal_day(self):
        relay = pd.DataFrame({"date": ["2026-09-01"], "code": ["600001"], "emotion_state": [None], "target": [1.]})
        market = pd.DataFrame({"date": ["2026-09-01", "2026-09-02"], "emotion_state": ["加速", "退潮"]})
        row = phase_context(relay, market)[0]
        self.assertEqual(row["phase_feature"], "加速")
        self.assertIsNone(row["original_relay_phase"])
        relay["target"] = 0.
        self.assertEqual(phase_context(relay, market)[0], row)

    def test_conflicting_archived_phases_stay_unknown(self):
        relay = pd.DataFrame({"date": ["2026-09-01"], "code": ["600001"], "emotion_state": ["加速"]})
        market = pd.DataFrame({"date": ["2026-09-01"], "emotion_state": ["退潮"]})
        row = phase_context(relay, market)[0]
        self.assertIsNone(row["phase_feature"])
        self.assertEqual(row["status"], "ARCHIVED_PHASE_CONFLICT")


if __name__ == "__main__":
    unittest.main()
