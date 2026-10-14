import unittest

import public_forward_audit_v16 as audit
import public_live_monitor_v1 as monitor


class PublicForwardAuditV16Tests(unittest.TestCase):
    @staticmethod
    def record(date, run_id, *, complete=True, verification="PASS", statuses=None):
        return {
            "version": monitor.VERSION, "trade_date": date, "run_id": run_id,
            "weekday": True, "verification": verification,
            "run_status": "CAPTURE_AND_DECISION_SEQUENCE_COMPLETE_PUBLIC_ONLY" if complete else "CAPTURE_SEQUENCE_INCOMPLETE",
            "completed_captures": 6 if complete else 5,
            "completed_decisions": 3 if complete else 2,
            "failed_decisions": 0, "decision_statuses": statuses or ["CASH", "CASH", "CASH"],
            "direct_entry_authorized": False,
        }

    def test_complete_unique_date_counts_all_public_decisions_but_no_trades(self):
        result = audit.summarize([
            self.record("2026-09-07", "one", statuses=["CASH", "WATCH_PUBLIC_PRICE_ONLY", "CANCEL"]),
        ], {})
        self.assertEqual(result["eligible_complete_sessions"], 1)
        self.assertEqual(result["public_decision_observations"], 3)
        self.assertEqual(result["cash_decisions"], 1)
        self.assertEqual(result["watch_decisions"], 1)
        self.assertEqual(result["cancel_decisions"], 1)
        self.assertEqual(result["completed_trades"], 0)
        self.assertIsNone(result["complete_trade_win_rate"])

    def test_duplicate_date_is_ineligible_instead_of_best_run_selection(self):
        records = [self.record("2026-09-07", "one"), self.record("2026-09-07", "two")]
        result = audit.summarize(records, {})
        self.assertEqual(result["duplicate_trade_dates"], ["2026-09-07"])
        self.assertEqual(result["eligible_complete_sessions"], 0)
        self.assertEqual(result["public_decision_observations"], 0)

    def test_failed_or_incomplete_run_never_enters_eligible_denominator(self):
        records = [
            self.record("2026-09-07", "bad", verification="FAIL"),
            self.record("2026-09-08", "partial", complete=False),
        ]
        result = audit.summarize(records, {})
        self.assertEqual(result["eligible_complete_sessions"], 0)
        self.assertEqual(result["verification_failed_runs"], 1)


if __name__ == "__main__":
    unittest.main()
