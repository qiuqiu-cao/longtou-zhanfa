import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import sse_current_master_v5 as module
from sse_current_master_v5 import (
    chain_is_trusted,
    downgrade_untrusted_intervals,
    event_chain,
    filter_action_rows,
    month_segments,
    verified_manual_resolution_ids,
)


class SSECurrentMasterV5Tests(unittest.TestCase):
    def test_month_segments_are_complete_and_non_overlapping(self):
        result = month_segments("2024-01-15", "2024-03-02")
        self.assertEqual(result, [
            ("2024-01-15", "2024-01-31"),
            ("2024-02-01", "2024-02-29"),
            ("2024-03-01", "2024-03-02"),
        ])

    def test_verified_chain_must_reach_current_official_name(self):
        events = pd.DataFrame([
            {"code": "600001", "parse_status": "VERIFIED_EVENT", "before_name": "甲公司",
             "after_name": "ST甲", "effective_date": "2024-05-06", "announcement_date": "2024-04-30",
             "bulletin_id": "1", "url": "official"},
        ])
        chain, warnings = event_chain("600001", "*ST甲", events, pd.Timestamp("2023-11-28"))
        self.assertEqual(len(chain), 1)
        self.assertTrue(any("current-tail-mismatch" in warning for warning in warnings))

    def test_duplicate_identical_event_is_deduplicated(self):
        events = pd.DataFrame([
            {"code": "600001", "parse_status": "VERIFIED_EVENT", "before_name": "甲公司",
             "after_name": "ST甲", "effective_date": "2024-05-06", "announcement_date": "2024-04-30",
             "bulletin_id": "1", "url": "official"},
            {"code": "600001", "parse_status": "VERIFIED_EVENT", "before_name": "甲公司",
             "after_name": "ST甲", "effective_date": "2024-05-06", "announcement_date": "2024-04-30",
             "bulletin_id": "2", "url": "official-correction"},
        ])
        chain, warnings = event_chain("600001", "ST甲", events, pd.Timestamp("2023-11-28"))
        self.assertEqual(len(chain), 1)
        self.assertEqual(warnings, [])

    def test_successful_zero_action_search_writes_typed_empty_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current = pd.DataFrame([{
                "A_STOCK_CODE": "600001", "SEC_NAME_CN": "甲公司", "LIST_DATE": "20000101",
            }])
            old = (module.RAW, module.INDEX_CSV, module.EVENTS_CSV)
            module.RAW, module.INDEX_CSV, module.EVENTS_CSV = (
                root / "raw", root / "index.csv", root / "events.csv")
            try:
                with patch.object(module, "gap_codes", return_value=["600001"]), \
                     patch.object(module, "collect_current_list", return_value=current), \
                     patch.object(module, "query_action_month", return_value=([], [])):
                    collected = module.collect_actions("2026-09-04", "2026-09-04", 1)
                self.assertEqual(collected["event_candidates"], 0)
                self.assertEqual(pd.read_csv(module.INDEX_CSV).shape[0], 0)
                extracted = module.extract_events()
                self.assertEqual(extracted["verified_events"], 0)
                self.assertIn("parse_status", pd.read_csv(module.EVENTS_CSV).columns)
            finally:
                module.RAW, module.INDEX_CSV, module.EVENTS_CSV = old

    def test_archived_action_rows_can_be_rescoped_without_network(self):
        rows = [{"SECURITY_CODE": "600001", "SECURITY_NAME": "甲", "SSEDATE": "2026-09-01",
                 "TITLE": "关于变更证券简称的公告", "URL": "official", "ORG_BULLETIN_ID": "1"},
                {"SECURITY_CODE": "600002", "SECURITY_NAME": "乙", "SSEDATE": "2026-09-01",
                 "TITLE": "关于变更证券简称的公告", "URL": "official", "ORG_BULLETIN_ID": "2"}]
        result = filter_action_rows(rows, {"600002"})
        self.assertEqual(result["SECURITY_CODE"].tolist(), ["600002"])
        self.assertTrue(bool(result.iloc[0]["event_candidate"]))

    def test_unresolved_action_prevents_verified_chain(self):
        self.assertFalse(chain_is_trusted([], 1))
        self.assertTrue(chain_is_trusted([], 0))

    def test_untrusted_chain_downgrades_every_interval_for_code(self):
        rows = [
            {"code": "600000", "data_status": "verified_import"},
            {"code": "600001", "data_status": "inferred"},
            {"code": "600001", "data_status": "verified_import"},
        ]
        downgrade_untrusted_intervals(rows, 1, trusted=False)
        self.assertEqual(rows[0]["data_status"], "verified_import")
        self.assertEqual(rows[1]["data_status"], "inferred")
        self.assertEqual(rows[2]["data_status"], "inferred")

    def test_manual_resolution_is_bound_to_review_required_text_hash(self):
        events = pd.DataFrame([{
            "code": "600001", "bulletin_id": "abc", "text_sha256": "deadbeef",
            "parse_status": "REVIEW_REQUIRED",
        }])
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "review.csv"
            pd.DataFrame([{
                "code": "600001", "bulletin_id": "abc", "text_sha256": "deadbeef",
                "resolution": "VERIFIED_NO_STATE_TRANSITION", "reviewed_at": "2026-09-07T12:00:00+08:00",
                "reviewer": "document_review", "evidence": "official text says status remains unchanged",
            }]).to_csv(ledger, index=False)
            accepted, errors = verified_manual_resolution_ids(events, ledger)
            self.assertEqual(accepted, {("600001", "abc")})
            self.assertEqual(errors, [])

            frame = pd.read_csv(ledger)
            frame.loc[0, "text_sha256"] = "wrong"
            frame.to_csv(ledger, index=False)
            with self.assertRaisesRegex(ValueError, "text-hash-mismatch"):
                verified_manual_resolution_ids(events, ledger)


if __name__ == "__main__":
    unittest.main()
