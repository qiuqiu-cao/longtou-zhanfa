import unittest

import pandas as pd

from survivorship_sensitivity_v4 import (
    as_bool, attach_sse_point_in_time_eligibility, attach_streaks, build_sensitivity,
)


class SurvivorshipSensitivityV4Tests(unittest.TestCase):
    def test_boolean_parser_does_not_treat_false_string_as_true(self):
        parsed = as_bool(pd.Series(["True", "False", "1", "0", None]))
        self.assertEqual(parsed.tolist(), [True, False, True, False, False])

    def test_streak_resets_on_ineligible_or_non_limit_day(self):
        bars = pd.DataFrame({
            "code": ["000001"] * 5,
            "is_limit_10": [True, True, True, True, True],
        })
        eligible = pd.Series([True, True, False, True, True])
        result = attach_streaks(bars, eligible, "boards")
        self.assertEqual(result["boards"].tolist(), [1, 2, 0, 1, 2])

    def test_verified_height_change_is_separate_from_upper_bound(self):
        bars = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "code": ["000001", "600001"],
            "is_limit_10": [True, True], "touched_limit_10": [True, True], "is_down_limit_10": [False, False],
            "verified_eligible": [True, False], "upper_bound_eligible": [True, True],
            "verified_boards": [3, 0], "upper_boards": [3, 5],
        })
        market = pd.DataFrame([{
            "date": "2024-01-01", "emotion_state": "升温", "limit_count": 10, "consecutive_count": 2,
            "max_board": 2, "failed_count": 1, "down_limit_count": 0, "failed_rate": 1 / 11,
        }])
        result = build_sensitivity(bars, market).iloc[0]
        self.assertEqual(result["verified_revised_max_board"], 3)
        self.assertEqual(result["upper_revised_max_board"], 5)
        self.assertEqual(result["sensitivity_class"], "VERIFIED_HEIGHT_CHANGE")

    def test_sse_inferred_interval_cannot_become_verified_eligible(self):
        bars = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02", "2024-02-02"]),
            "code": ["600001", "600001"],
        })
        master = pd.DataFrame([
            {"code": "600001", "exchange": "SSE", "valid_from": "2024-01-01", "valid_to": "2024-01-31",
             "source_verified": True, "board": "MAINBOARD", "listing_status": "LISTED", "st_status": "NORMAL",
             "price_limit_regime": "MAINBOARD_10"},
            {"code": "600001", "exchange": "SSE", "valid_from": "2024-02-01", "valid_to": "2024-02-28",
             "source_verified": False, "board": "MAINBOARD", "listing_status": "LISTED", "st_status": "NORMAL",
             "price_limit_regime": "MAINBOARD_10"},
        ])
        result = attach_sse_point_in_time_eligibility(bars, master)
        self.assertEqual(result["sse_status_verified"].tolist(), [True, False])
        self.assertEqual(result["sse_verified_eligible"].tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
