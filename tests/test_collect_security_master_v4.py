import tempfile
import unittest
from pathlib import Path

import pandas as pd

from collect_security_master_v4 import (
    build_intervals,
    first_limited_date,
    name_status,
    verify_name_history,
)


class SecurityCollectorTests(unittest.TestCase):
    def test_name_status_is_conservative(self):
        self.assertEqual(name_status("S*ST 示例"), "*ST")
        self.assertEqual(name_status("SST示例"), "ST")
        self.assertEqual(name_status("退市示例"), "DELISTING_RISK")
        self.assertEqual(name_status("普通股份"), "NORMAL")

    def test_registration_ipo_uses_sixth_trading_day(self):
        days = list(pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-12"]))
        self.assertEqual(first_limited_date(pd.Timestamp("2026-01-05"), days), pd.Timestamp("2026-01-12"))

    def test_history_must_end_at_current_name(self):
        current = pd.DataFrame([{"code": "000001", "current_name": "现在", "listing_date": "1991-01-01"}])
        changes = pd.DataFrame([{"code": "000001", "change_date": pd.Timestamp("2020-01-01"), "before_name": "过去", "after_name": "错误"}])
        with self.assertRaises(ValueError):
            verify_name_history(current, changes)

    def test_builds_non_overlapping_st_intervals(self):
        current = pd.DataFrame([{"code": "000001", "current_name": "普通", "listing_date": pd.Timestamp("2020-01-01")}])
        changes = pd.DataFrame([
            {"code": "000001", "change_date": pd.Timestamp("2024-01-03"), "before_name": "普通", "after_name": "ST普通"},
            {"code": "000001", "change_date": pd.Timestamp("2024-01-05"), "before_name": "ST普通", "after_name": "普通"},
        ])
        frame = build_intervals(
            current, changes, list(pd.date_range("2024-01-01", "2024-01-10", freq="B")),
            pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-10"),
            "2026-08-31T00:00:00+00:00", "a" * 64,
        )
        self.assertEqual(frame["st_status"].tolist(), ["NORMAL", "ST", "NORMAL"])
        self.assertEqual(frame["price_limit_regime"].tolist(), ["MAINBOARD_10", "ST_5", "MAINBOARD_10"])
        self.assertEqual(frame["valid_to"].tolist(), ["2024-01-02", "2024-01-04", "2024-01-10"])


if __name__ == "__main__":
    unittest.main()
