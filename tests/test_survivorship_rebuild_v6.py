import unittest

import pandas as pd

from security_master_v4 import validate_master
from survivorship_rebuild_v6 import correct_current_raw, _apply_end_cutoff


class SurvivorshipRebuildV6Tests(unittest.TestCase):
    def test_known_ineligible_is_removed_and_unknown_breadth_is_retained(self):
        bars = pd.DataFrame([
            {"date": "2025-01-02", "code": "600001", "name": "甲", "st": False,
             "new_listing": False, "status_verified": False, "universe_complete": False},
            {"date": "2025-01-02", "code": "600002", "name": "乙", "st": False,
             "new_listing": False, "status_verified": False, "universe_complete": False},
            {"date": "2025-01-02", "code": "600003", "name": "未知", "st": False,
             "new_listing": False, "status_verified": False, "universe_complete": False},
        ])
        base = {
            "valid_from": "2025-01-01", "valid_to": "2025-12-31", "exchange": "SSE",
            "board": "MAINBOARD", "listing_status": "LISTED", "listing_date": "2020-01-01",
            "delisting_date": "", "source_type": "sse_official", "source_ref": "official",
            "source_timestamp": "2026-09-01T00:00:00+00:00", "data_status": "verified_import",
        }
        master = validate_master(pd.DataFrame([
            {**base, "code": "600001", "name": "甲历史", "st_status": "NORMAL",
             "price_limit_regime": "MAINBOARD_10"},
            {**base, "code": "600002", "name": "ST乙", "st_status": "ST",
             "price_limit_regime": "ST_5"},
        ]))
        corrected, meta = correct_current_raw(bars, master)
        self.assertEqual(set(corrected["code"]), {"600001", "600003"})
        self.assertEqual(corrected.loc[corrected["code"].eq("600001"), "name"].iloc[0], "甲历史")
        self.assertTrue(corrected.loc[corrected["code"].eq("600001"), "status_verified"].iloc[0])
        self.assertFalse(corrected.loc[corrected["code"].eq("600003"), "status_verified"].iloc[0])
        self.assertEqual(meta["officially_ineligible_rows_removed"], 1)
        self.assertEqual(meta["unverified_breadth_rows_retained"], 1)

    def test_historical_cutoff_is_applied_before_rebuild(self):
        bars = pd.DataFrame([{"date": "2026-08-31", "code": "600001"},
                             {"date": "2026-09-01", "code": "600002"}])
        result = _apply_end_cutoff(bars, "2026-08-31", "test bars")
        self.assertEqual(result["code"].tolist(), ["600001"])


if __name__ == "__main__":
    unittest.main()
