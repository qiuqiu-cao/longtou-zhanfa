import tempfile
import unittest
from pathlib import Path

import pandas as pd

from theme_factor_readiness_v149 import REQUIRED_FACTOR_FIELDS, audit, discover


def record(root, date="2026-09-01", complete=True, include_schema=True, coverage=1.0):
    folder = root / date
    folder.mkdir(parents=True)
    competition = folder / "candidate_competition_snapshot.csv"
    pd.DataFrame([{
        "signal_date": date, "code": "600001",
        "observed_at": f"{date}T15:00:00+08:00",
        "available_at": f"{date}T15:10:00+08:00", "data_status": "verified",
    }]).to_csv(competition, index=False)
    theme = folder / "candidate_theme_roles.csv"
    data = {"code": ["600001"]}
    if include_schema:
        data.update({field: ["x"] for field in REQUIRED_FACTOR_FIELDS})
    pd.DataFrame(data).to_csv(theme, index=False)
    theme_manifest = folder / "theme_manifest.json"
    theme_manifest.write_text("{}", encoding="utf-8")
    manifest = {
        "signal_date": date,
        "closing_candidate_count": 1,
        "verified_theme_candidate_count": int(coverage),
        "coverage": coverage,
        "full_market_completeness": "verified_full_market" if complete else "not_proven_archived_roster_only",
        "complete_within_declared_roster": True,
        "source_hashes": {
            str(theme.resolve()): "x",
            str(theme_manifest.resolve()): "y",
        },
    }
    competition_manifest = folder / "manifest.json"
    competition_manifest.write_text(__import__("json").dumps(manifest), encoding="utf-8")
    return {
        "competition_manifest_path": competition_manifest,
        "competition_path": competition,
        "theme_path": theme,
        "theme_manifest_path": theme_manifest,
        "manifest": manifest,
    }


class ThemeFactorReadinessTests(unittest.TestCase):
    def test_complete_session_can_be_model_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            item = record(Path(tmp))
            _, daily, report = audit([item], minimum_sessions=1, minimum_coverage=.8)
            self.assertTrue(daily.iloc[0].leader_factor_model_ready)
            self.assertEqual(report["status"], "READY_FOR_TIME_ISOLATED_THEME_FACTOR_MODEL")

    def test_archived_roster_or_legacy_schema_is_not_training_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = record(root / "a", complete=False)
            second = record(root / "b", date="2026-09-02", include_schema=False)
            _, daily, report = audit([first, second], minimum_sessions=1)
            self.assertFalse(daily.leader_factor_model_ready.any())
            self.assertEqual(report["allowed_use"], "coverage_audit_and_daily_descriptive_review_only")
            self.assertFalse(report["missing_values_treated_as_zero"])

    def test_low_coverage_can_still_be_point_in_time_but_not_descriptive_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            item = record(root, coverage=0.0)
            revisions, _, _ = audit([item], minimum_sessions=1, minimum_coverage=.8)
            self.assertTrue(revisions.iloc[0].point_in_time_competition)
            self.assertFalse(revisions.iloc[0].descriptive_review_ready)

    def test_discovery_ignores_unrelated_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record(root / "good")
            other = root / "other"
            other.mkdir()
            (other / "manifest.json").write_text('{"status":"OTHER"}', encoding="utf-8")
            self.assertEqual(len(discover(root)), 1)


if __name__ == "__main__":
    unittest.main()
