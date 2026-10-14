import unittest

import pandas as pd

from collect_sse_security_master_v4 import build_intervals


class SSESecurityMasterV4Tests(unittest.TestCase):
    def base_delist(self):
        return pd.DataFrame([{
            "code": "600001", "final_name": "退市甲", "listing_date": "2000-01-01",
            "delisting_date": "2024-12-31", "exchange": "SSE", "source_ref": "official",
        }])

    def test_verified_pdf_event_sets_exact_st_boundary(self):
        index = pd.DataFrame([
            {"SECURITY_CODE": "600001", "SECURITY_NAME": "甲公司", "SSEDATE": "2023-01-01", "ORG_BULLETIN_ID": "1"},
            {"SECURITY_CODE": "600001", "SECURITY_NAME": "*ST甲", "SSEDATE": "2024-05-06", "ORG_BULLETIN_ID": "2"},
        ])
        index["SSEDATE"] = pd.to_datetime(index["SSEDATE"])
        events = pd.DataFrame([{
            "code": "600001", "parse_status": "VERIFIED_EVENT", "before_name": "甲公司",
            "after_name": "*ST甲", "effective_date": "2024-05-06", "url": "official-pdf",
        }])
        intervals, _ = build_intervals(
            self.base_delist(), index, events, pd.Timestamp("2023-01-01"),
            pd.Timestamp("2024-12-31"), "2026-01-01T00:00:00+00:00",
        )
        listed = intervals[intervals.listing_status.eq("LISTED")]
        self.assertEqual(listed.iloc[0].valid_to, "2024-05-05")
        self.assertEqual(listed.iloc[1].valid_from, "2024-05-06")
        self.assertEqual(listed.iloc[1].st_status, "*ST")
        self.assertEqual(listed.iloc[1].data_status, "verified_import")

    def test_metadata_only_change_is_inferred(self):
        index = pd.DataFrame([
            {"SECURITY_CODE": "600001", "SECURITY_NAME": "甲公司", "SSEDATE": "2023-01-01", "ORG_BULLETIN_ID": "1"},
            {"SECURITY_CODE": "600001", "SECURITY_NAME": "ST甲", "SSEDATE": "2024-05-06", "ORG_BULLETIN_ID": "2"},
        ])
        index["SSEDATE"] = pd.to_datetime(index["SSEDATE"])
        events = pd.DataFrame(columns=["code", "parse_status", "before_name", "after_name", "effective_date", "url"])
        intervals, _ = build_intervals(
            self.base_delist(), index, events, pd.Timestamp("2023-01-01"),
            pd.Timestamp("2024-12-31"), "2026-01-01T00:00:00+00:00",
        )
        listed = intervals[intervals.listing_status.eq("LISTED")]
        self.assertEqual(listed.iloc[1].data_status, "inferred")
        self.assertEqual(listed.iloc[1].st_status, "ST")

    def test_stable_official_metadata_remains_verified(self):
        index = pd.DataFrame([
            {"SECURITY_CODE": "600001", "SECURITY_NAME": "甲公司", "SSEDATE": "2023-01-01", "ORG_BULLETIN_ID": "1"},
            {"SECURITY_CODE": "600001", "SECURITY_NAME": "甲公司", "SSEDATE": "2024-06-01", "ORG_BULLETIN_ID": "2"},
        ])
        index["SSEDATE"] = pd.to_datetime(index["SSEDATE"])
        events = pd.DataFrame(columns=["code", "parse_status", "before_name", "after_name", "effective_date", "url"])
        intervals, _ = build_intervals(
            self.base_delist(), index, events, pd.Timestamp("2023-01-01"),
            pd.Timestamp("2024-12-31"), "2026-01-01T00:00:00+00:00",
        )
        listed = intervals[intervals.listing_status.eq("LISTED")]
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed.iloc[0].data_status, "verified_import")
        self.assertEqual(listed.iloc[0].st_status, "NORMAL")


if __name__ == "__main__":
    unittest.main()
