import unittest

import pandas as pd

from historical_universe_v4 import (
    effective_szse_name,
    longest_consecutive,
    missing_window_universe,
    normalize_history,
    round_limit_price,
    zfill_code,
)


class HistoricalUniverseV4Tests(unittest.TestCase):
    def test_limit_rounding_matches_exchange_half_up(self):
        self.assertEqual(round_limit_price(10.05), 11.06)
        self.assertEqual(round_limit_price(10.05, -0.10), 9.05)

    def test_excel_numeric_codes_are_normalized(self):
        self.assertEqual(zfill_code("4.0"), "000004")
        self.assertEqual(zfill_code(600290), "600290")

    def test_mixed_exchange_date_types_parse_without_losing_rows(self):
        values = pd.Series(["2024-01-01", pd.Timestamp("2024-01-02")])
        parsed = pd.to_datetime(values, errors="coerce", format="mixed")
        self.assertEqual(parsed.notna().sum(), 2)

    def test_effective_name_uses_dated_chain(self):
        events = pd.DataFrame([
            {"change_date": pd.Timestamp("2024-01-03"), "before_name": "普通", "after_name": "ST普通"},
            {"change_date": pd.Timestamp("2024-01-05"), "before_name": "ST普通", "after_name": "普通"},
        ])
        self.assertEqual(effective_szse_name(events, "普通", pd.Timestamp("2024-01-02")), "普通")
        self.assertEqual(effective_szse_name(events, "普通", pd.Timestamp("2024-01-04")), "ST普通")

    def test_history_normalization_keeps_status_and_10pct_limit_separate(self):
        raw = pd.DataFrame([
            {"日期": "2024-01-02", "开盘": 10.0, "收盘": 11.0, "最高": 11.0, "最低": 10.0,
             "成交量": 100, "成交额": 1000, "换手率": 1, "涨跌额": 1.0, "涨跌幅": 10.0, "振幅": 10.0},
            {"日期": "2024-01-03", "开盘": 11.0, "收盘": 12.1, "最高": 12.1, "最低": 11.0,
             "成交量": 100, "成交额": 1100, "换手率": 1, "涨跌额": 1.1, "涨跌幅": 10.0, "振幅": 10.0},
        ])
        security = pd.Series({"code": "000001", "exchange": "SZSE", "final_name": "普通", "delisting_date": "2024-02-01"})
        names = pd.DataFrame(columns=["code", "change_date", "before_name", "after_name"])
        frame = normalize_history(raw, security, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-31"), names)
        self.assertTrue(frame["is_limit_10"].all())
        self.assertTrue(frame["st_status"].eq("NORMAL").all())
        self.assertEqual(longest_consecutive(frame["is_limit_10"]), 2)

    def test_current_universe_omissions_are_explicit(self):
        delists = pd.DataFrame([
            {"code": "600001", "exchange": "SSE", "listing_date": pd.Timestamp("2000-01-01"),
             "delisting_date": pd.Timestamp("2024-01-01")},
        ])
        import historical_universe_v4 as module
        original = module.current_codes
        module.current_codes = lambda: set()
        try:
            result = missing_window_universe(delists, pd.Timestamp("2023-01-01"), pd.Timestamp("2025-01-01"))
        finally:
            module.current_codes = original
        self.assertTrue(result.iloc[0]["missing_from_current_universe"])


if __name__ == "__main__":
    unittest.main()
