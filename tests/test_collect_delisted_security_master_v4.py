import unittest

import pandas as pd

from collect_delisted_security_master_v4 import build_delisted_intervals, regime_for_status


class DelistedSecurityMasterV4Tests(unittest.TestCase):
    def test_regime_mapping_is_explicit(self):
        self.assertEqual(regime_for_status("NORMAL"), "MAINBOARD_10")
        self.assertEqual(regime_for_status("ST"), "ST_5")
        self.assertEqual(regime_for_status("DELISTING_RISK"), "OTHER")

    def test_listed_and_delisted_intervals_do_not_overlap(self):
        delists = pd.DataFrame([{
            "code": "000004", "final_name": "国华退", "listing_date": pd.Timestamp("1990-12-01"),
            "delisting_date": pd.Timestamp("2024-01-10"), "exchange": "SZSE",
        }])
        names = pd.DataFrame([
            {"code": "000004", "change_date": pd.Timestamp("2024-01-05"),
             "before_name": "国华网安", "after_name": "ST国华"},
            {"code": "000004", "change_date": pd.Timestamp("2024-01-09"),
             "before_name": "ST国华", "after_name": "国华退"},
        ])
        frame = build_delisted_intervals(
            delists, names, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-31"),
            "2026-09-01T00:00:00+00:00",
        )
        self.assertEqual(frame["listing_status"].tolist(), ["LISTED", "LISTED", "LISTED", "DELISTED"])
        self.assertEqual(frame["valid_to"].tolist(), ["2024-01-04", "2024-01-08", "2024-01-09", "2024-01-31"])
        self.assertEqual(frame.iloc[-1]["price_limit_regime"], "NONE")


if __name__ == "__main__":
    unittest.main()
