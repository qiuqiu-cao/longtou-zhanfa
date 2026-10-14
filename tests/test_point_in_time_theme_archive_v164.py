import unittest
from pathlib import Path

import pandas as pd

from point_in_time_theme_archive_v164 import assemble


def pool(date="2026-09-01", full=False):
    frame = pd.DataFrame([
        {"signal_date": date, "code": "600001", "name": "甲", "boards": 1,
         "board_group": "FIRST_BOARD", "pool_assignment": "CLOSING_LIMIT_UP"},
        {"signal_date": date, "code": "600002", "name": "乙", "boards": 2,
         "board_group": "CONSECUTIVE_BOARD", "pool_assignment": "CLOSING_LIMIT_UP"},
    ])
    return {date: {"path": Path("pool.csv"), "frame": frame, "manifest_path": None,
                   "manifest": {"full_market_completeness": "verified_full_market" if full else
                                "not_proven_archived_roster_only",
                                "complete_within_archived_roster": True}, "revision": 1}}


def role(date="2026-09-01", published="2026-09-01T18:00:00+08:00"):
    frame = pd.DataFrame([{
        "signal_date": date, "code": "600001", "theme_status": "verified_explicit_source_roster",
        "theme_group_id": "specific.a", "theme": "细分A", "theme_role": "THEME_HEIGHT_LEADER",
        "primary_evidence_id": "e1", "source_tier": "B", "theme_support_ex_self": 1,
        "same_height_rivals": 0, "theme_ladder_verified": True,
        "evidence_published_at": published, "evidence_status": "verified",
    }])
    return {date: [{"path": Path("roles.csv"), "frame": frame, "manifest_path": None,
                    "manifest": {}, "verified": 1, "revision": 1}]}


class PointInTimeThemeArchiveV164Tests(unittest.TestCase):
    def test_uncovered_candidate_is_preserved_as_unknown_not_zero(self):
        archive, daily, report, _ = assemble(
            pool(), role(), pd.DataFrame(columns=["signal_date", "code"]),
            ["2026-09-01", "2026-09-02"],
        )
        unknown = archive.loc[archive.code.eq("600002")].iloc[0]
        self.assertEqual(unknown.theme_status, "unavailable")
        self.assertTrue(pd.isna(unknown.theme_support_ex_self))
        self.assertFalse(unknown.missing_is_zero)
        self.assertEqual(daily.iloc[0].theme_coverage, .5)
        self.assertEqual(report["candidate_rows"], 2)

    def test_late_evidence_is_not_point_in_time(self):
        archive, _, _, _ = assemble(
            pool(), role(published="2026-09-02T10:00:00+08:00"),
            pd.DataFrame(columns=["signal_date", "code"]),
            ["2026-09-01", "2026-09-02"],
        )
        self.assertFalse(archive.loc[archive.code.eq("600001")].iloc[0]
                         .point_in_time_available_for_next_session)

    def test_membership_driver_text_does_not_complete_factor_schema(self):
        registry = pd.DataFrame([{
            "signal_date": "2026-09-01", "code": "600001", "evidence_id": "e1",
            "usable_at_signal": True, "theme_granularity": "specific",
            "driver": "政策催化", "effective_published_at": "2026-09-01T12:00:00+08:00",
        }])
        archive, _, report, _ = assemble(
            pool(full=True), role(), registry, ["2026-09-01", "2026-09-02"],
        )
        row = archive.loc[archive.code.eq("600001")].iloc[0]
        self.assertEqual(row.driver_summary, "政策催化")
        self.assertTrue(pd.isna(row.driver_type))
        self.assertTrue(pd.isna(row.expectation_status))
        self.assertFalse(row.theme_factor_model_eligible)
        self.assertEqual(report["model_eligible_rows"], 0)

    def test_best_role_revision_is_selected(self):
        roles = role()
        second = roles["2026-09-01"][0].copy()
        second["revision"] = 2
        second["path"] = Path("roles_v2.csv")
        second["frame"] = pd.concat([second["frame"], pd.DataFrame([{
            **second["frame"].iloc[0].to_dict(), "code": "600002", "theme_group_id": "specific.b",
        }])], ignore_index=True)
        second["verified"] = 2
        roles["2026-09-01"].append(second)
        archive, daily, _, _ = assemble(
            pool(), roles, pd.DataFrame(columns=["signal_date", "code"]),
            ["2026-09-01", "2026-09-02"],
        )
        self.assertEqual(daily.iloc[0].verified_theme_candidate_count, 2)
        self.assertTrue(archive.theme_status.astype(str).str.startswith("verified").all())


if __name__ == "__main__":
    unittest.main()
