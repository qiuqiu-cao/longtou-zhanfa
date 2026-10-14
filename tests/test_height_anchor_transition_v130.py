import unittest

import pandas as pd

from height_anchor_transition_v130 import _paired_delta, build_labels


class HeightAnchorTransitionV130Test(unittest.TestCase):
    def test_candidate_can_become_top_after_old_anchor_disappears(self):
        rows = pd.DataFrame([
            {"signal_date": "2026-01-05", "code": "000001", "boards_for_comparison": 3,
             "height_gap_to_known_top": 0},
            {"signal_date": "2026-01-05", "code": "000002", "boards_for_comparison": 2,
             "height_gap_to_known_top": 1},
            {"signal_date": "2026-01-06", "code": "000002", "boards_for_comparison": 3,
             "height_gap_to_known_top": 0},
            {"signal_date": "2026-01-07", "code": "000002", "boards_for_comparison": 4,
             "height_gap_to_known_top": 0},
            {"signal_date": "2026-01-08", "code": "000003", "boards_for_comparison": 1,
             "height_gap_to_known_top": 0},
        ])
        result = build_labels(rows)
        challenger = result[(result.signal_date == "2026-01-05") & (result.code == "000002")].iloc[0]
        self.assertEqual(challenger.top_next_session, 1)
        self.assertEqual(challenger.top_within_3_sessions, 1)
        self.assertEqual(challenger.new_challenger_top_within_3, 1)

    def test_last_dates_have_no_complete_three_session_label(self):
        rows = pd.DataFrame([
            {"signal_date": date, "code": "000001", "boards_for_comparison": 1,
             "height_gap_to_known_top": 0}
            for date in ["2026-01-05", "2026-01-06", "2026-01-07"]
        ])
        result = build_labels(rows)
        self.assertTrue(pd.isna(result.iloc[-1].top_within_3_sessions))

    def test_paired_delta_is_date_matched(self):
        model = pd.DataFrame({"signal_date": ["a", "b"], "y": [1, 1]})
        baseline = pd.DataFrame({"signal_date": ["a", "b"], "y": [0, 1]})
        result = _paired_delta(model, baseline, "y", bootstrap_samples=100)
        self.assertEqual(result["sessions"], 2)
        self.assertEqual(result["precision_delta"], .5)


if __name__ == "__main__":
    unittest.main()
