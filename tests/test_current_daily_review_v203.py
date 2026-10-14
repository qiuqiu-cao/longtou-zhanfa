import unittest

from current_daily_review import _index_emotion_divergence_summary


class CurrentDailyReviewV203Test(unittest.TestCase):
    def evidence(self):
        return {
            "version": "v203-index-emotion-divergence-audit-1",
            "status": "DIAGNOSTIC_ONLY",
            "data_quality": {
                "index_status": "DELAYED_PROVIDER_OBSERVED_NOT_OFFICIAL_RECONCILED",
                "emotion_rows_missing_on_index_calendar": 2,
            },
            "latest_context": {
                "date": "2026-09-16", "sh_a_return_pct": .7,
                "sz_a_return_pct": 1.3,
                "previous_limit_median_return": .009,
                "divergence_category": "BOTH_UP_SHORT_POSITIVE",
                "emotion_state": "修复",
            },
            "weekly_clustered_comparisons": {"full": {
                "next_adverse_state": {
                    "observed_mean_difference_a_minus_b": .07,
                    "weekly_cluster_bootstrap_ci95": [-.06, .22],
                }}},
            "replication_gate": {"next_adverse_state": {
                "replication_gate_passed": False}},
            "governance": {
                "broad_index_direction_equals_short_term_emotion": False,
                "index_up_may_override_short_loss_effect": False,
                "divergence_may_be_displayed_as_context": True,
                "divergence_direct_emotion_gate_allowed": False,
                "divergence_direct_stock_reranking_allowed": False,
                "divergence_entry_authorized": False,
            },
            "interpretation": "separate layers",
        }

    def test_binds_context_without_promoting_entry(self):
        result = _index_emotion_divergence_summary(
            self.evidence(), "2026-09-16")
        self.assertEqual(result["status"], "BOUND_DIAGNOSTIC_ONLY")
        self.assertFalse(result["replication_gate_passed"])
        self.assertFalse(result["entry_authorized"])

    def test_rejects_date_mismatch(self):
        with self.assertRaisesRegex(ValueError, "does not match signal date"):
            _index_emotion_divergence_summary(
                self.evidence(), "2026-09-15")

    def test_rejects_override_permission(self):
        evidence = self.evidence()
        evidence["governance"]["index_up_may_override_short_loss_effect"] = True
        with self.assertRaisesRegex(ValueError, "cannot authorize"):
            _index_emotion_divergence_summary(evidence, "2026-09-16")


if __name__ == "__main__":
    unittest.main()
