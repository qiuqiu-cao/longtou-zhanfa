import unittest

import pandas as pd

from theme_membership_evidence_v169 import build, validate


class ThemeMembershipEvidenceV169Tests(unittest.TestCase):
    def pool(self):
        return pd.DataFrame([
            {"signal_date": "2026-09-16", "code": "605058", "name": "澳弘电子",
             "boards": 4, "amount": 4.6e8, "first_seal_time": 93436,
             "pool_assignment": "CLOSING_LIMIT_UP"},
            {"signal_date": "2026-09-16", "code": "002913", "name": "奥士康",
             "boards": 1, "amount": 8.3e8, "first_seal_time": 112557,
             "pool_assignment": "CLOSING_LIMIT_UP"},
        ])

    def evidence(self, driver_status="secondary_market_narrative"):
        return {
            "schema_version": "v169-member-theme-evidence-1",
            "signal_date": "2026-09-16",
            "drivers": [{
                "theme_group_id": "pcb_direct", "theme": "PCB直接制造",
                "business_branch": "direct_pcb_manufacturing",
                "driver_summary": "盘后复盘将两股归为PCB，但未验证主动催化",
                "driver_source_url": "https://example.test/review",
                "driver_first_available_at": "2026-09-16T16:06:53+08:00",
                "driver_claim_status": driver_status,
                "expectation_status": "unknown",
                "expectation_assessed_at": "2026-09-16T19:00:00+08:00",
            }],
            "members": [
                {"code": "605058", "theme_group_id": "pcb_direct",
                 "business_branch": "direct_pcb_manufacturing",
                 "membership_claim": "公司直接生产印制电路板",
                 "membership_source_url": "https://example.test/605058.pdf",
                 "membership_source_tier": "A_PRIMARY_EXCHANGE_FILING",
                 "membership_published_at": "2025-04-17",
                 "membership_available_at": "2025-04-17",
                 "membership_claim_status": "verified_primary"},
                {"code": "002913", "theme_group_id": "pcb_direct",
                 "business_branch": "direct_pcb_manufacturing",
                 "membership_claim": "公司直接生产高精密印制电路板",
                 "membership_source_url": "https://example.test/002913.pdf",
                 "membership_source_tier": "A_PRIMARY_EXCHANGE_FILING",
                 "membership_published_at": "2024-03-15",
                 "membership_available_at": "2024-03-15",
                 "membership_claim_status": "verified_primary"},
            ],
        }

    def test_verified_membership_with_secondary_driver_is_not_positive(self):
        result = build(self.pool(), self.evidence(), "2026-09-16T20:00:00+08:00")
        self.assertTrue(result.membership_verified.all())
        self.assertFalse(result.driver_verified.any())
        self.assertTrue(result.theme_status.eq(
            "verified_membership_driver_unverified").all())
        self.assertFalse(result.positive_catalyst_factor_eligible.any())
        self.assertFalse(result.formal_high_low_mapping_eligible.any())

    def test_branch_mismatch_is_rejected_instead_of_forcing_same_theme(self):
        evidence = self.evidence()
        evidence["members"][1]["business_branch"] = "pcb_materials"
        with self.assertRaisesRegex(ValueError, "business branch does not match"):
            validate(evidence, "2026-09-16T20:00:00+08:00")

    def test_same_day_date_only_driver_time_fails_closed(self):
        evidence = self.evidence()
        evidence["drivers"][0]["driver_first_available_at"] = "2026-09-16"
        with self.assertRaisesRegex(ValueError, "date-only evidence not admissible"):
            validate(evidence, "2026-09-16T20:00:00+08:00")

    def test_verified_driver_still_needs_non_saturated_expectation(self):
        evidence = self.evidence("verified_official_media")
        evidence["drivers"][0]["expectation_status"] = "fully_traded"
        result = build(self.pool(), evidence, "2026-09-16T20:00:00+08:00")
        self.assertTrue(result.driver_verified.all())
        self.assertTrue(result.theme_status.eq("verified_membership_and_driver").all())
        self.assertFalse(result.positive_catalyst_factor_eligible.any())
        self.assertFalse(result.formal_high_low_mapping_eligible.any())


if __name__ == "__main__":
    unittest.main()
