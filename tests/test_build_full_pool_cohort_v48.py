import unittest

import pandas as pd

from build_full_pool_cohort_v48 import build


class BuildFullPoolCohortV48Tests(unittest.TestCase):
    def test_keeps_first_and_relay_and_excludes_left_censored(self):
        frame = pd.DataFrame([
            {"code": "000001", "signal_date": "2024-01-02", "boards": 1, "start_known": True,
             "board_group": "FIRST_BOARD", "board_status": "verified", "candidate_scope": "scope"},
            {"code": "000002", "signal_date": "2024-01-02", "boards": 3, "start_known": True,
             "board_group": "CONSECUTIVE_BOARD", "board_status": "verified", "candidate_scope": "scope"},
            {"code": "000003", "signal_date": "2024-01-02", "boards": 4, "start_known": False,
             "board_group": "UNKNOWN_LEFT_CENSORED", "board_status": "verified", "candidate_scope": "scope"},
        ])
        cohort, report = build(frame, pd.Series(["2024-01-02", "2024-01-03", "2024-01-04"]),
                               "2024-01-02", "2024-01-02")
        self.assertEqual(set(cohort["board_group"]), {"FIRST_BOARD", "CONSECUTIVE_BOARD"})
        self.assertEqual(report["eligible_rows"], 2)
        self.assertEqual(report["excluded_left_censored_or_unknown"], 1)


if __name__ == "__main__":
    unittest.main()
