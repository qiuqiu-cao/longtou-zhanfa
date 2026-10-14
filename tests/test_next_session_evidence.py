import unittest
import pandas as pd

from leader_workflow_v30 import next_session_watchlist


class NextSessionTests(unittest.TestCase):
    def run_case(self, snapshot_changes=None, row_changes=None, sessions=None, asof=None):
        row = dict(code="600001", name="SYNTHETIC", boards=1, is_mainboard=1, is_st=0,
                   is_delisting_period=0, no_price_limit=0, security_status="verified", snapshot_kind="closing",
                   is_limit_up=1, limit_up_status="verified", identity_status="verified",
                   observed_at="2026-09-04T15:00:00+08:00",
                   available_at="2026-09-04T15:01:00+08:00", source="synthetic")
        row.update(row_changes or {})
        snap = dict(code="600001", price=10, is_st=0, tradability="buyable", source="synthetic",
                    data_status="verified", observed_at="2026-09-07T09:35:00+08:00",
                    available_at="2026-09-07T09:35:00+08:00")
        snap.update(snapshot_changes or {})
        calendar = dict(source="synthetic", data_status="verified", complete=True,
                        available_at="2026-09-01T00:00:00+08:00",
                        sessions=sessions or ["2026-09-04", "2026-09-07", "2026-09-08"])
        return next_session_watchlist(pd.DataFrame([row]), "2026-09-04T15:05:00+08:00",
                                     asof or "2026-09-07T09:35:00+08:00", [snap], calendar)

    def test_weekend_identity_and_timestamp_preserved(self):
        result = self.run_case()
        row = result["candidates"][0]
        self.assertEqual(row["boards"], 1)
        self.assertEqual(row["observed_at"], "2026-09-04T15:00:00+08:00")
        self.assertEqual(row["opening_review"]["status"], "CURRENT_EVIDENCE_ONLY")
        self.assertEqual(row["execution_observation"], "REQUIRES_LIVE_CONFIRMATION")
        self.assertIsNone(result["comprehensive_first_choice"])
        self.assertFalse(result["position_created"])

    def test_future_delayed_and_wrong_day_rejected(self):
        for changes in ({"available_at": "2026-09-07T09:36:00+08:00"},
                        {"observed_at": "2026-09-07T09:25:00+08:00"},
                        {"observed_at": "2026-09-04T09:35:00+08:00"},
                        {"data_status": "delayed"}):
            row = self.run_case(changes)["candidates"][0]
            self.assertEqual(row["opening_review"]["status"], "DATA_LIMITED")
            self.assertEqual(row["execution_observation"], "UNKNOWN")

    def test_missing_quote_no_replacement(self):
        result = self.run_case({"code": "600002"})
        self.assertEqual(result["session_binding"]["unmatched_opening_codes"], ["600002"])
        self.assertIn("missing_current_snapshot", result["candidates"][0]["opening_review"]["issues"])
        self.assertEqual(len(result["candidates"]), 1)

    def test_overnight_st_and_suspension(self):
        self.assertEqual(self.run_case({"is_st": 1})["candidates"][0]["opening_review"]["status"],
                         "CURRENT_OUT_OF_SCOPE")
        self.assertEqual(self.run_case({"tradability": "suspended"})["candidates"][0]["execution_observation"],
                         "UNBUYABLE")

    def test_intraday_pool_not_mistaken_for_close(self):
        row = self.run_case(row_changes={"observed_at": "2026-09-04T14:00:00+08:00"})["candidates"][0]
        self.assertIn("signal_is_not_closing_snapshot", row["opening_review"]["issues"])

    def test_skipped_session_and_nontrading_date(self):
        for asof in ("2026-09-08T09:35:00+08:00", "2026-09-05T09:35:00+08:00"):
            with self.assertRaises(ValueError):
                self.run_case(asof=asof)

    def test_calendar_order(self):
        with self.assertRaises(ValueError):
            self.run_case(sessions=["2026-09-07", "2026-09-04"])


if __name__ == "__main__":
    unittest.main()
