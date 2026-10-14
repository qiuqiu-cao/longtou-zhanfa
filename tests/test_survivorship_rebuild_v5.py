import unittest

import pandas as pd

from survivorship_rebuild_v5 import verified_recovered_raw


class SurvivorshipRebuildV5Tests(unittest.TestCase):
    def test_only_verified_normal_mainboard_rows_are_rebuilt(self):
        bars = pd.DataFrame({
            "date": ["2024-01-02", "2024-02-02", "2024-03-02"],
            "code": ["600001"] * 3, "open": [10, 10, 10], "high": [11, 11, 11],
            "low": [10, 10, 10], "close": [11, 11, 11], "pre_close": [10, 10, 10],
            "volume": [1, 1, 1], "amount": [1, 1, 1], "turnover": [1, 1, 1],
        })
        master = pd.DataFrame([
            {"code": "600001", "valid_from": "2024-01-01", "valid_to": "2024-01-31",
             "source_verified": True, "board": "MAINBOARD", "listing_status": "LISTED",
             "st_status": "NORMAL", "price_limit_regime": "MAINBOARD_10", "name": "甲公司"},
            {"code": "600001", "valid_from": "2024-02-01", "valid_to": "2024-02-29",
             "source_verified": True, "board": "MAINBOARD", "listing_status": "LISTED",
             "st_status": "ST", "price_limit_regime": "ST_5", "name": "ST甲"},
            {"code": "600001", "valid_from": "2024-03-01", "valid_to": "2024-03-31",
             "source_verified": False, "board": "MAINBOARD", "listing_status": "LISTED",
             "st_status": "NORMAL", "price_limit_regime": "MAINBOARD_10", "name": "甲公司"},
        ])
        result = verified_recovered_raw(bars, master)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["name"], "甲公司")
        self.assertTrue(bool(result.iloc[0]["status_verified"]))


if __name__ == "__main__":
    unittest.main()
