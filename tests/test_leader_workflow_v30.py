import unittest
import pandas as pd
from leader_workflow_v30 import watchlist, holding, clean


class WorkflowTests(unittest.TestCase):
    def candidate(self):
        return dict(code="003005", name="SYNTHETIC_TEST_NOT_MARKET", boards=7,
                    is_mainboard=1, is_st=0, valid_identity=1, is_limit_up=1, limit_up_status="verified",
                    deviation_10d_pct=95, deviation_30d_pct=180,
                    indicator_status="verified", identity_status="verified",
                    observed_at="2026-09-04T09:25:00+08:00", available_at="2026-09-04T09:25:10+08:00",
                    source="synthetic_fixture", tradability="buyable", emotion_state="退潮", notice_titles="普通异动")

    def fixture(self):
        position = dict(position_id="test-paper", position_kind="paper", code="003005",
                        entry_at="2026-09-03T09:35:00+08:00", entry_price=10, shares=100)
        quote = dict(code="003005", observed_at="2026-09-04T09:35:00+08:00",
                     available_at="2026-09-04T09:35:10+08:00", data_status="verified",
                     source="synthetic_fixture", price=11, leader_identity="intact",
                     indicator_status="verified", deviation_10d_pct=90, deviation_30d_pct=180,
                     sellability="sellable", emotion_state="升温", theme_feedback="supportive")
        return position, quote

    def test_no_automatic_position_or_probability_winner(self):
        result = watchlist(pd.DataFrame([self.candidate()]), "2026-09-04T09:26:00+08:00")
        self.assertIsNone(result["probability_first_choice"])
        self.assertFalse(result["position_created"])
        self.assertEqual(result["candidates"][0]["candidate_status"], "DATA_LIMITED")
        self.assertIn("closing_snapshot_required", result["candidates"][0]["data_issues"])

    def test_old_intraday_requires_refresh(self):
        result = watchlist(pd.DataFrame([self.candidate()]), "2026-09-04T10:00:00+08:00")
        self.assertEqual(result["candidates"][0]["execution_observation"], "REFRESH_REQUIRED")

    def test_invalid_timestamp_preserves_data_limited(self):
        row = self.candidate()
        row["observed_at"] = None
        result = watchlist(pd.DataFrame([row]), "2026-09-04T09:26:00+08:00")
        self.assertEqual(result["candidates"][0]["candidate_status"], "DATA_LIMITED")

    def test_position_kind_required(self):
        position, quote = self.fixture()
        position.pop("position_kind")
        with self.assertRaises(ValueError):
            holding(position, quote, "2026-09-04T09:36:00+08:00")

    def test_review_change_and_blocked_execution(self):
        position, quote = self.fixture()
        previous = holding(position, quote, "2026-09-04T09:36:00+08:00")
        quote.update(leader_identity="lost", sellability="locked_limit_down", emotion_state="退潮")
        result = holding(position, quote, "2026-09-04T09:37:00+08:00", previous=previous)
        self.assertEqual(result["decision"], "EXIT_PLAN")
        self.assertEqual(result["execution_state"], "SELL_BLOCKED")
        self.assertEqual(result["changes_since_previous_review"]["decision"]["previous"], "HOLD_REVIEW")
        self.assertFalse(result["scheduled_monitoring_enabled"])

    def test_no_mixing_paper_actual(self):
        position, quote = self.fixture()
        previous = holding(position, quote, "2026-09-04T09:36:00+08:00")
        position["position_kind"] = "user_confirmed"
        with self.assertRaises(ValueError):
            holding(position, quote, "2026-09-04T09:37:00+08:00", previous=previous)

    def test_future_previous_review_rejected(self):
        position, quote = self.fixture()
        previous = holding(position, quote, "2026-09-04T09:37:00+08:00")
        with self.assertRaises(ValueError):
            holding(position, quote, "2026-09-04T09:36:00+08:00", previous=previous)

    def test_json_nonfinite_is_missing(self):
        self.assertEqual(clean({"x": [float("nan"), float("inf")]}), {"x": [None, None]})


if __name__ == "__main__":
    unittest.main()
