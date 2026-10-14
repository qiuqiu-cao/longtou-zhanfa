import unittest

import pandas as pd

from build_current_competition_snapshot_v114 import build


class CompetitionSnapshotTests(unittest.TestCase):
    def inputs(self):
        pool = pd.DataFrame([
            {"signal_date": "2026-09-11", "code": "000001", "name": "甲", "boards": 3,
             "pool_assignment": "CLOSING_LIMIT_UP", "is_mainboard": True, "amount": 100,
             "first_seal_time": 92500, "observed_at": "2026-09-11T15:00:00+08:00",
             "available_at": "2026-09-11T18:00:00+08:00"},
            {"signal_date": "2026-09-11", "code": "000002", "name": "乙", "boards": 2,
             "pool_assignment": "CLOSING_LIMIT_UP", "is_mainboard": True, "amount": 300,
             "first_seal_time": 93000, "observed_at": "2026-09-11T15:00:00+08:00",
             "available_at": "2026-09-11T18:00:00+08:00"},
            {"signal_date": "2026-09-11", "code": "000003", "name": "丙", "boards": 2,
             "pool_assignment": "CLOSING_LIMIT_UP", "is_mainboard": True, "amount": 200,
             "first_seal_time": 93100, "observed_at": "2026-09-11T15:00:00+08:00",
             "available_at": "2026-09-11T18:00:00+08:00"},
            {"signal_date": "2026-09-11", "code": "000004", "name": "未知", "boards": 1,
             "pool_assignment": "CLOSING_LIMIT_UP", "is_mainboard": True, "amount": 50,
             "first_seal_time": 100000, "observed_at": "2026-09-11T15:00:00+08:00",
             "available_at": "2026-09-11T18:00:00+08:00"},
        ])
        themes = pd.DataFrame([
            {"signal_date": "2026-09-11", "code": "000001",
             "theme_status": "verified_explicit_source_roster", "theme_role": "THEME_HEIGHT_LEADER",
             "theme_group_id": "a", "theme": "A", "primary_evidence_id": "src", "source_tier": "C",
             "theme_support_ex_self": 2, "same_height_rivals": 0, "theme_ladder_verified": True,
             "evidence_published_at": "2026-09-11T17:00:00+08:00",
             "identity_tags": "THEME_HEIGHT_LEADER|FIRST_SEAL_PIONEER", "capacity_rank_in_theme": 3},
            {"signal_date": "2026-09-11", "code": "000002",
             "theme_status": "verified_explicit_source_roster", "theme_role": "SAME_HEIGHT_CORE",
             "theme_group_id": "a", "theme": "A", "primary_evidence_id": "src", "source_tier": "C",
             "theme_support_ex_self": 2, "same_height_rivals": 1, "theme_ladder_verified": True,
             "evidence_published_at": "2026-09-11T17:00:00+08:00",
             "identity_tags": "SAME_HEIGHT_CORE|CAPACITY_CORE_BY_CLOSE_AMOUNT", "capacity_rank_in_theme": 1},
            {"signal_date": "2026-09-11", "code": "000003",
             "theme_status": "verified_explicit_source_roster", "theme_role": "SAME_HEIGHT_CORE",
             "theme_group_id": "a", "theme": "A", "primary_evidence_id": "src", "source_tier": "C",
             "theme_support_ex_self": 2, "same_height_rivals": 1, "theme_ladder_verified": True,
             "evidence_published_at": "2026-09-11T17:00:00+08:00",
             "identity_tags": "SAME_HEIGHT_CORE", "capacity_rank_in_theme": 2},
        ])
        pool_manifest = {"signal_date": "2026-09-11", "mainboard_closing_limit_rows": 4,
                         "complete_within_archived_roster": True,
                         "full_market_completeness": "not_proven_archived_roster_only"}
        theme_manifest = {"signal_date": "2026-09-11"}
        return pool, pool_manifest, themes, theme_manifest

    def test_global_height_and_same_theme_competition_are_distinct(self):
        result, metadata = build(*self.inputs(), "2026-09-11T20:00:00+08:00")
        rows = result.set_index("code")
        self.assertEqual(rows.loc["000001", "competition_identity"], "height_anchor")
        self.assertEqual(rows.loc["000001", "boards"], 3)
        self.assertEqual(rows.loc["000002", "competition_identity"], "capacity_core")
        events = rows.loc["000002", "events"]
        self.assertIn("THEME_SAME_HEIGHT_RIVALS", events)
        self.assertIn("GLOBAL_HIGHER_BOARD_PRESSURE", events)
        self.assertEqual(metadata["verified_theme_candidate_count"], 3)
        self.assertEqual(metadata["unknown_candidate_codes"], ["000004"])
        self.assertFalse(metadata["entry_authorized"])

    def test_unknown_theme_candidate_is_not_filled_as_negative_evidence(self):
        result, _ = build(*self.inputs(), "2026-09-11T20:00:00+08:00")
        self.assertNotIn("000004", set(result.code))

    def test_asof_before_input_availability_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not available by asof"):
            build(*self.inputs(), "2026-09-11T17:30:00+08:00")

    def test_manifest_date_mismatch_is_rejected(self):
        pool, pool_manifest, themes, theme_manifest = self.inputs()
        theme_manifest["signal_date"] = "2026-09-10"
        with self.assertRaisesRegex(ValueError, "different signal dates"):
            build(pool, pool_manifest, themes, theme_manifest, "2026-09-11T20:00:00+08:00")

    def test_v150_fields_propagate_without_upgrading_legacy_rows(self):
        pool, pool_manifest, themes, theme_manifest = self.inputs()
        themes["theme_status"] = "verified_specific_point_in_time_schema"
        themes["theme_granularity"] = "specific_subtheme"
        themes["driver_type"] = "industry"
        themes["driver_summary"] = "已核验催化"
        themes["driver_first_available_at"] = "2026-09-11T14:00:00+08:00"
        themes["driver_claim_status"] = "verified_fact"
        themes["expectation_status"] = "partially_traded"
        themes["expectation_basis"] = "尚未一致加速"
        themes["positive_catalyst_factor_eligible"] = True
        result, metadata = build(
            pool, pool_manifest, themes, theme_manifest, "2026-09-11T20:00:00+08:00"
        )
        self.assertTrue(result.theme_factor_schema_status.eq("V150_COMPLETE").all())
        self.assertTrue(result.positive_catalyst_factor_eligible.all())
        self.assertEqual(metadata["v150_factor_schema_candidate_count"], 3)


if __name__ == "__main__":
    unittest.main()
