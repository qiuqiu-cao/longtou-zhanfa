import unittest

import pandas as pd

from apply_current_reset_evidence_v193 import apply_reset_evidence


class ApplyCurrentResetEvidenceV193Tests(unittest.TestCase):
    def _distances(self):
        return pd.DataFrame([
            {"signal_date": "2026-09-16", "code": "605058",
             "benchmark_status": "provider_observed_not_official_reconciled",
             "reset_status": "unavailable", "corporate_action_status": "unavailable",
             "security_eligibility_status": "unavailable",
             "regulatory_distance_status": "unavailable",
             "research_indicator_status": "inferred_unreconciled",
             "provider_aligned_30_return_coverage": True,
             "ranking_allowed": False, "buy_signal": False,
             "hundred_percent_path_potential": None},
            {"signal_date": "2026-09-16", "code": "000993",
             "benchmark_status": "provider_observed_not_official_reconciled",
             "reset_status": "unavailable", "corporate_action_status": "unavailable",
             "security_eligibility_status": "unavailable",
             "regulatory_distance_status": "unavailable",
             "research_indicator_status": "inferred_unreconciled",
             "provider_aligned_30_return_coverage": True,
             "ranking_allowed": False, "buy_signal": False,
             "hundred_percent_path_potential": None},
        ])

    def _reset(self):
        common = {
            "severe_event_count_since_rule_effective": 0,
            "latest_exchange_publication_date": None,
            "latest_reset_effective_session": None,
        }
        return pd.DataFrame([
            {"code": "605058", **common,
             "reset_chronology_status": "verified_no_exchange_event_since_rule_effective",
             "official_exchange_chronology_complete": True,
             "source_level": "SSE_OFFICIAL_EXCHANGE_DAILY_FEED",
             "query_coverage": "2026-07-06~2026-09-16"},
            {"code": "000993", **common,
             "reset_chronology_status": "company_index_no_event_found_not_exchange_exhaustive",
             "official_exchange_chronology_complete": False,
             "source_level": "CNINFO_OFFICIAL_COMPANY_DISCLOSURE_INDEX",
             "query_coverage": "2026-07-06~2026-09-17"},
        ])

    def test_only_complete_exchange_chronology_passes_reset_gate(self):
        result = apply_reset_evidence(self._distances(), self._reset()).set_index("code")
        self.assertEqual(result.loc["605058", "reset_status"], "verified")
        self.assertEqual(result.loc["000993", "reset_status"], "unavailable")
        self.assertTrue(result.loc["605058", "official_reset_evidence"])
        self.assertFalse(result.loc["000993", "official_reset_evidence"])

    def test_overlay_never_promotes_distance_or_trade(self):
        result = apply_reset_evidence(self._distances(), self._reset())
        self.assertTrue(result["regulatory_distance_status"].eq("unavailable").all())
        self.assertFalse(result["ranking_allowed"].any())
        self.assertFalse(result["buy_signal"].any())

    def test_candidate_coverage_mismatch_fails(self):
        with self.assertRaises(ValueError):
            apply_reset_evidence(self._distances(), self._reset().iloc[:1])


if __name__ == "__main__":
    unittest.main()
