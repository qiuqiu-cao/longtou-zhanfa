import unittest

import pandas as pd

from official_window_universe_v50 import build


class OfficialWindowUniverseV50Tests(unittest.TestCase):
    def test_current_and_window_delisted_form_disjoint_official_universe(self):
        sz = pd.DataFrame([{
            "code": "1", "name": "甲", "listing_date": "2020-01-01", "valid_to": "2026-09-04",
            "exchange": "SZSE", "board": "MAINBOARD", "source_type": "szse_official",
            "data_status": "verified_import",
        }])
        sh = pd.DataFrame([{
            "A_STOCK_CODE": "600001", "SEC_NAME_CN": "乙", "LIST_DATE": "20200102",
            "LIST_BOARD": "1", "STOCK_TYPE": "1",
        }])
        old = pd.DataFrame([{
            "code": "600002", "final_name": "丙", "listing_date": "2000-01-01",
            "delisting_date": "2024-05-01", "exchange": "SSE",
            "source_ref": "https://www.sse.com.cn/assortment/stock/list/delisting/",
        }, {
            "code": "600003", "final_name": "窗口外", "listing_date": "2000-01-01",
            "delisting_date": "2020-01-01", "exchange": "SSE",
            "source_ref": "https://www.sse.com.cn/assortment/stock/list/delisting/",
        }])
        universe, missing, report = build(
            sz, sh, old, {"000001", "600001"}, "2023-11-28", "2026-08-31")
        self.assertEqual(set(universe["code"]), {"000001", "600001", "600002"})
        self.assertEqual(missing["code"].tolist(), ["600002"])
        self.assertEqual(report["daily_missing_codes"], 1)
        self.assertFalse(report["survivorship_bias_resolved"])

    def test_non_exchange_delisted_source_is_rejected(self):
        sz = pd.DataFrame(columns=["code", "name", "listing_date", "valid_to", "exchange",
                                   "board", "source_type", "data_status"])
        sh = pd.DataFrame(columns=["A_STOCK_CODE", "SEC_NAME_CN", "LIST_DATE", "LIST_BOARD", "STOCK_TYPE"])
        old = pd.DataFrame([{
            "code": "600002", "final_name": "丙", "listing_date": "2000-01-01",
            "delisting_date": "2024-05-01", "exchange": "SSE", "source_ref": "community",
        }])
        with self.assertRaisesRegex(ValueError, "non-exchange"):
            build(sz, sh, old, set(), "2023-11-28", "2026-08-31")


if __name__ == "__main__":
    unittest.main()
