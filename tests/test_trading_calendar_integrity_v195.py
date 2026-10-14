import unittest

import pandas as pd

from trading_calendar_integrity_v195 import audit


class TradingCalendarIntegrityV195Tests(unittest.TestCase):
    def inputs(self):
        official = pd.DataFrame([
            {"date": "2026-09-01", "status": "SUCCESS"},
            {"date": "2026-09-02", "status": "SUCCESS"},
            {"date": "2026-09-03", "status": "SUCCESS"},
            {"date": "2026-09-04", "status": "SUCCESS"},
        ])
        provider = pd.DataFrame([
            {"date": date, "code": code}
            for date in official.date for code in ("000001", "600000")
        ])
        observations = pd.DataFrame([
            {"signal_date": "2026-09-01", "target_date": "2026-09-04"},
            {"signal_date": "2026-09-04", "target_date": None},
        ])
        return official, provider, observations

    def test_sparse_review_dates_cannot_skip_real_sessions(self):
        calendar, rows, report = audit(
            *self.inputs(), "2026-09-01", "2026-09-04")
        self.assertEqual(calendar.date.tolist(), [
            "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04"])
        first = rows.loc[rows.signal_date.eq("2026-09-01")].iloc[0]
        self.assertEqual(first.correct_next_session, "2026-09-02")
        self.assertFalse(first.old_target_matches_calendar)
        self.assertIn("SPARSE_OBSERVATION_CALENDAR", first.issue)
        self.assertEqual(report["status"], "CALENDAR_LABEL_REBUILD_REQUIRED")
        self.assertFalse(report["governance"][
            "daily_review_dates_may_define_trading_calendar"])

    def test_incomplete_provider_cross_section_is_not_verified(self):
        official, provider, observations = self.inputs()
        provider = provider[~(
            provider.date.eq("2026-09-03") & provider.code.eq("600000"))]
        calendar, _, report = audit(
            official, provider, observations, "2026-09-01", "2026-09-04")
        row = calendar.loc[calendar.date.eq("2026-09-03")].iloc[0]
        self.assertFalse(row.provider_cross_section_complete)
        self.assertEqual(report["provider_complete_sessions"], 3)

    def test_duplicate_observation_rows_do_not_inflate_session_count(self):
        official, provider, observations = self.inputs()
        observations = pd.concat([observations, observations.iloc[[0]]], ignore_index=True)
        _, rows, report = audit(
            official, provider, observations, "2026-09-01", "2026-09-04")
        self.assertEqual(len(rows), 2)
        self.assertEqual(report["observation_signal_sessions"], 2)

    def test_separate_pool_date_inventory_prevents_false_missing_flag(self):
        official, provider, observations = self.inputs()
        pools = pd.DataFrame({"date": official.date})
        _, rows, report = audit(
            official, provider, observations, "2026-09-01", "2026-09-04", pools)
        last = rows.loc[rows.signal_date.eq("2026-09-04")].iloc[0]
        self.assertFalse(last.correct_next_pool_available)
        first = rows.loc[rows.signal_date.eq("2026-09-01")].iloc[0]
        self.assertTrue(first.correct_next_pool_available)
        self.assertEqual(report["available_pool_sessions"], 4)


if __name__ == "__main__":
    unittest.main()
