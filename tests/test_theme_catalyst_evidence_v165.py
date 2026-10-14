import unittest

import pandas as pd

from theme_catalyst_evidence_v165 import build_backlog, validate


def evidence(**updates):
    row = {
        "signal_date": "2026-09-01", "theme_group_id": "specific.a", "theme": "细分A",
        "theme_granularity": "specific", "driver_type": "POLICY",
        "driver_summary": "正式政策发布", "source_url": "https://example.com/policy",
        "source_published_at": "2026-09-01T10:00:00+08:00",
        "driver_first_available_at": "2026-09-01T10:00:00+08:00",
        "driver_first_source_url": "https://example.com/policy",
        "first_time_basis": "official_timestamp", "driver_claim_status": "verified_official",
        "novelty_status": "new_information", "expectation_status": "underrecognized",
        "expectation_evidence": "盘前尚无同题材涨停，收盘形成首轮扩散",
        "expectation_observed_at": "2026-09-01T15:05:00+08:00",
        "expectation_source_url": "https://example.com/market-close",
        "available_at": "2026-09-01T18:00:00+08:00", "evidence_status": "verified",
    }
    row.update(updates)
    return pd.DataFrame([row])


class ThemeCatalystEvidenceV165Tests(unittest.TestCase):
    def test_complete_verified_evidence_passes_schema_not_model_authorization(self):
        rows, report = validate(evidence(), ["2026-09-01", "2026-09-02"])
        self.assertTrue(rows.iloc[0].factor_schema_ready)
        self.assertTrue(rows.iloc[0].hard_fact_use_allowed)
        self.assertFalse(rows.iloc[0].positive_buy_reason_allowed)
        self.assertFalse(rows.iloc[0].model_use_allowed)
        self.assertEqual(report["status"], "EVIDENCE_SCHEMA_READY_FOR_FACTOR_TESTING")

    def test_rumor_is_never_hard_fact(self):
        rows, _ = validate(evidence(driver_claim_status="rumor", evidence_status="inferred"),
                           ["2026-09-01", "2026-09-02"])
        self.assertFalse(rows.iloc[0].factor_schema_ready)
        self.assertFalse(rows.iloc[0].hard_fact_use_allowed)

    def test_late_evidence_fails_point_in_time(self):
        rows, _ = validate(evidence(available_at="2026-09-02T10:00:00+08:00"),
                           ["2026-09-01", "2026-09-02"])
        self.assertFalse(rows.iloc[0].point_in_time_available)
        self.assertIn("LATE_FOR_NEXT_SESSION_PREOPEN", rows.iloc[0].validation_issues)

    def test_roster_archive_builds_one_task_per_theme(self):
        archive = pd.DataFrame([
            {"signal_date": "2026-09-01", "code": "600001", "theme_group_id": "a",
             "theme": "A", "boards": 1, "point_in_time_available_for_next_session": True},
            {"signal_date": "2026-09-01", "code": "600002", "theme_group_id": "a",
             "theme": "A", "boards": 3, "point_in_time_available_for_next_session": True},
            {"signal_date": "2026-09-01", "code": "600003", "theme_group_id": None,
             "theme": None, "boards": 1, "point_in_time_available_for_next_session": False},
        ])
        backlog = build_backlog(archive)
        self.assertEqual(len(backlog), 1)
        self.assertEqual(backlog.iloc[0].candidate_count, 2)
        self.assertEqual(backlog.iloc[0].collection_priority, "HIGH")
        self.assertEqual(backlog.iloc[0].first_time_basis, "unavailable")


if __name__ == "__main__":
    unittest.main()
