import unittest

import pandas as pd

from current_full_pool_snapshot import COL, build


def pool_row(code, boards=1):
    return {COL["code"]: code, COL["name"]: "SYNTHETIC", COL["change"]: 10,
            COL["last"]: 11, COL["amount"]: 100, COL["float_cap"]: 1000,
            COL["turnover"]: 5, COL["first_seal"]: 93000, COL["failed_count"]: 0,
            COL["limit_stat"]: "1/1", COL["industry"]: "TEST", COL["boards"]: boards}


def failed_row(code):
    row = pool_row(code)
    row.pop(COL["boards"])
    row[COL["limit_price"]] = 11
    return row


def master(code="000001"):
    return pd.DataFrame([{
        "code": code, "valid_from": "2025-01-01", "valid_to": "2026-09-04",
        "exchange": "SZSE", "board": "MAINBOARD", "listing_status": "LISTED",
        "st_status": "NORMAL", "price_limit_regime": "MAINBOARD_10",
        "source_type": "szse_official", "source_ref": "official", "data_status": "verified_import",
    }])


def manifest(limit=2, failed=1):
    return {"date": "20260904", "collected_at": "2026-09-05T10:00:00",
            "datasets": {"limit_up": {"status": "verified", "rows": limit, "source": "provider"},
                         "failed_board": {"status": "verified", "rows": failed, "source": "provider"}}}


class CurrentFullPoolSnapshotTests(unittest.TestCase):
    def test_first_relay_and_failed_are_separate(self):
        closing = pd.DataFrame([pool_row("000001", 1), pool_row("600001", 2)])
        frame, report = build(closing, pd.DataFrame([failed_row("000002")]), manifest(), master(), "2026-09-04")
        self.assertEqual(report["mainboard_first_board_rows"], 1)
        self.assertEqual(report["mainboard_consecutive_board_rows"], 1)
        self.assertEqual(report["mainboard_failed_board_rows"], 1)
        self.assertTrue(report["main_and_failed_pools_disjoint"])
        self.assertEqual(set(frame.pool_assignment), {"CLOSING_LIMIT_UP", "FAILED_BOARD_OBSERVATION"})

    def test_missing_sse_master_is_unavailable_not_inferred(self):
        closing = pd.DataFrame([pool_row("000001"), pool_row("600001")])
        frame, report = build(closing, pd.DataFrame([failed_row("000002")]), manifest(), master(), "2026-09-04")
        sse = frame[frame.code.eq("600001")].iloc[0]
        self.assertEqual(sse.security_status, "unavailable")
        self.assertIsNone(sse.is_st)
        self.assertEqual(report["verified_eligible_closing_rows"], 1)

    def test_non_mainboard_is_reported_out_of_scope(self):
        closing = pd.DataFrame([pool_row("000001"), pool_row("920001")])
        frame, report = build(closing, pd.DataFrame([failed_row("000002")]), manifest(), master(), "2026-09-04")
        self.assertEqual(report["non_mainboard_closing_rows"], 1)
        self.assertEqual(frame[frame.code.eq("920001")].iloc[0].eligibility_status, "OUT_OF_SCOPE_NON_MAINBOARD")

    def test_manifest_count_mismatch_fails(self):
        closing = pd.DataFrame([pool_row("000001")])
        with self.assertRaisesRegex(ValueError, "row count"):
            build(closing, pd.DataFrame([failed_row("000002")]), manifest(), master(), "2026-09-04")

    def test_output_never_authorizes_entry(self):
        closing = pd.DataFrame([pool_row("000001"), pool_row("600001")])
        _, report = build(closing, pd.DataFrame([failed_row("000002")]), manifest(), master(), "2026-09-04")
        self.assertFalse(report["entry_authorized"])
        self.assertEqual(report["full_market_completeness"], "not_independently_verified")

    def test_security_complete_status_requires_closing_and_failed_pool(self):
        closing = pd.DataFrame([pool_row("000001")])
        complete_master = pd.concat([master("000001"), master("000002")], ignore_index=True)
        frame, report = build(closing, pd.DataFrame([failed_row("000002")]),
                              manifest(limit=1, failed=1), complete_master, "2026-09-04")
        self.assertEqual(report["verified_eligible_closing_rows"], 1)
        self.assertEqual(report["verified_eligible_failed_board_rows"], 1)
        self.assertTrue(report["point_in_time_security_scope_complete"])
        self.assertNotIn("SSE official security reconstruction was unavailable when this snapshot was built",
                         report["limitations"])


if __name__ == "__main__":
    unittest.main()
