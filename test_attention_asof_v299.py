"""Isolated clock fixtures; the current view never changes the old forecast."""
from copy import deepcopy
import unittest

from attention_asof_v299 import project
from attention_observations_v299 import ENDPOINTS


class AttentionAsOfTests(unittest.TestCase):
    def fixture(self):
        common = {"available_at": "2026-10-02T23:40:00+08:00", "raw_path": "ISOLATED_UNIT_FIXTURE", "raw_sha256": "UNIT_FIXTURE_NOT_SOURCE"}
        rank = {**common, "calculated_at": "2026-10-02T23:30:00+08:00", "rank": 16}
        row = {"code": "600825", "errors": {}, "latest": rank,
            "profiles": {**common, "latest": {"calculated_at": "2026-10-02T23:00:00+08:00", "new_user_share": 0.3, "established_user_share": 0.7, "verified_user_count": None}},
            "keywords": {**common, "rows": [{"calculated_at": "2026-10-02T23:00:00+08:00", "hit_count_provider": 48}]},
            "related": {**common, "rows": [{"calculated_at": "2026-10-02T23:00:00+08:00", "provider_rate_literal": "7.72%"}]},
            "intraday": {**common, "rows": [{"calculated_at": "2026-10-02T23:30:00+08:00", "rank": 16}],
                "first_rank": 14, "last_rank": 16, "rank_improvement_first_minus_last": -2, "window_seconds": 84600,
                "missing_grid_times": [], "identity_limitation": "request only", "same_clock_latest_checked": False}}
        return {"recorded_at": "2026-10-02T23:40:01+08:00", "scope": {"codes": ["600825"]}, "observations": [row]}

    def test_complete_archive_availability_prevents_backfill(self):
        result = project(self.fixture(), "2026-10-02T23:40:00+08:00")
        self.assertTrue(all(result["observations"][0][k]["status"] == "UNKNOWN" for k in ENDPOINTS))

    def test_current_view_has_diagnostic_observations_but_consensus_stays_unknown(self):
        source = self.fixture(); before = deepcopy(source)
        result = project(source, "2026-10-02T23:40:02+08:00"); row = result["observations"][0]
        self.assertTrue(all(row[k]["status"] != "UNKNOWN" for k in ENDPOINTS))
        for key in ["forum_heat", "discussion_change", "market_agreement", "bull_bear_extremes"]:
            self.assertEqual(row[key]["status"], "UNKNOWN")
        self.assertEqual(source, before)
        self.assertEqual(row["intraday"]["status"], "inferred")

    def test_freshness_is_rechecked_at_view_time_not_only_capture(self):
        result = project(self.fixture(), "2026-10-03T02:00:00+08:00")
        self.assertTrue(all(result["observations"][0][k]["status"] == "UNKNOWN" for k in ENDPOINTS))

    def test_previous_day_intraday_not_presented_as_current_day_change(self):
        result = project(self.fixture(), "2026-10-03T00:00:00+08:00")
        self.assertEqual(result["observations"][0]["intraday"]["status"], "UNKNOWN")
        self.assertIn("另一自然日", result["observations"][0]["intraday"]["reason"])

    def test_missing_field_retained_and_future_calculation_not_admitted(self):
        source = self.fixture(); source["observations"][0]["keywords"] = None
        source["observations"][0]["errors"]["keywords"] = "provider timeout"
        source["observations"][0]["latest"]["calculated_at"] = "2026-10-03T00:00:00+08:00"
        result = project(source, "2026-10-02T23:40:02+08:00")
        self.assertEqual(result["observations"][0]["keywords"]["reason"], "provider timeout")
        self.assertEqual(result["observations"][0]["latest"]["status"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
