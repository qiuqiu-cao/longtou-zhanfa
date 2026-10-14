import unittest

import pandas as pd

from leader_target_contract_v16 import build_labels, evaluate_locked_signals


class TargetContractTests(unittest.TestCase):
    def setUp(self):
        self.calendar = ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04"]
        self.obs = pd.DataFrame([{"code": "003005", "signal_date": self.calendar[0], "episode_id": "a"}])
        self.path = pd.DataFrame([{"code": "003005", "date": d, "is_limit_up": 1,
                                   "indicator_status": "verified", "deviation_10d_pct": v,
                                   "deviation_30d_pct": w}
                                  for d, v, w in zip(self.calendar, [70, 90, 105, 120], [130, 160, 190, 205])])

    def test_separate_targets_and_first_hit(self):
        row = build_labels(self.obs, self.path, self.calendar).iloc[0]
        self.assertEqual(row.target_100_sessions_to_hit, 2)
        self.assertEqual(row.target_200_sessions_to_hit, 3)
        self.assertEqual(row.target_either_label, 1)

    def test_broken_board_ends_relay_even_if_later_hits(self):
        self.path.loc[1, "is_limit_up"] = 0
        row = build_labels(self.obs, self.path, self.calendar).iloc[0]
        self.assertEqual(row.target_either_label, 0)

    def test_censored_future_not_failure(self):
        row = build_labels(self.obs, self.path.iloc[:2], self.calendar).iloc[0]
        self.assertIsNone(row.target_either_label)

    def test_hit_100_does_not_imply_200(self):
        row = build_labels(self.obs, self.path.iloc[:3], self.calendar).iloc[0]
        self.assertEqual(row.target_100_label, 1)
        self.assertIsNone(row.target_200_label)

    def test_already_hit_is_excluded(self):
        self.path.loc[0, "deviation_10d_pct"] = 100
        row = build_labels(self.obs, self.path, self.calendar).iloc[0]
        self.assertEqual(row.outcome_status, "EXCLUDED_ALREADY_HIT")
        self.assertIsNone(row.target_either_label)

    def test_unverified_indicator_censors(self):
        self.path.loc[1, "indicator_status"] = "inferred"
        self.assertIsNone(build_labels(self.obs, self.path, self.calendar).iloc[0].target_either_label)

    def test_duplicate_dates_rejected(self):
        with self.assertRaises(ValueError):
            build_labels(self.obs, pd.concat([self.path, self.path.iloc[:1]]), self.calendar)

    def test_cluster_dedup_uses_first_signal_not_best_outcome(self):
        signals = pd.DataFrame([{"code": "003005", "signal_date": d, "episode_id": "a",
                                 "selected": 1, "split": "blind", "target_either_label": y}
                                for d, y in zip(self.calendar, [0, 1, 1, 1])])
        report = evaluate_locked_signals(signals)
        self.assertEqual(report["independent_episodes"], 1)
        self.assertEqual(report["precision"], 0)
        self.assertFalse(report["empirical_80pct_gate"])

    def test_eighty_percent_is_not_readiness(self):
        signals = pd.DataFrame([{"code": str(i).zfill(6), "signal_date": self.calendar[0],
                                 "episode_id": str(i), "selected": 1, "split": "blind",
                                 "target_either_label": int(i < 80)} for i in range(100)])
        report = evaluate_locked_signals(signals)
        self.assertTrue(report["empirical_80pct_gate"])
        self.assertLess(report["wilson_95_lower"], 0.8)
        self.assertEqual(report["status"], "DIAGNOSTIC_ONLY")


if __name__ == "__main__":
    unittest.main()
