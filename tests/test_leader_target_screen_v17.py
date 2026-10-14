import unittest

import pandas as pd

from leader_target_screen_v17 import screen


class TargetScreenTests(unittest.TestCase):
    def row(self, **updates):
        row = dict(code="003005", name="测试", boards=7, is_mainboard=1, is_st=0,
                   valid_identity=1, deviation_10d_pct=95, deviation_30d_pct=180,
                   indicator_status="verified", identity_status="verified",
                   observed_at="2026-09-04T09:25:00+08:00",
                   available_at="2026-09-04T09:25:10+08:00", source="test_fixture",
                   tradability="buyable", emotion_state="退潮", notice_titles="普通异动公告")
        row.update(updates)
        return row

    def run_row(self, **updates):
        return screen(pd.DataFrame([self.row(**updates)]), "2026-09-04T09:26:00+08:00").iloc[0]

    def test_high_board_retreat_and_notice_are_not_vetoes(self):
        row = self.run_row()
        self.assertEqual(row.candidate_status, "TARGET_RESEARCH_CANDIDATE")
        self.assertEqual(row.execution_permission, "NO_ENTER")
        self.assertIsNone(row.target_probability)

    def test_other_notice_disclosed_not_silently_removed(self):
        row = self.run_row(notice_titles="业绩预亏、减持计划")
        self.assertEqual(row.notice_titles, "业绩预亏、减持计划")
        self.assertEqual(row.candidate_status, "TARGET_RESEARCH_CANDIDATE")

    def test_exact_thresholds_stop_new_target_relay(self):
        for field, value in [("deviation_10d_pct", 100), ("deviation_30d_pct", 200)]:
            self.assertEqual(self.run_row(**{field: value}).candidate_status, "TARGET_ALREADY_REACHED")

    def test_locked_board_is_candidate_but_not_fill(self):
        row = self.run_row(tradability="locked_limit_up")
        self.assertEqual(row.candidate_status, "TARGET_RESEARCH_CANDIDATE")
        self.assertEqual(row.execution_observation, "UNBUYABLE")

    def test_unverified_or_future_information_is_not_a_signal(self):
        for update in [dict(indicator_status="inferred"), dict(deviation_10d_pct=None),
                       dict(available_at="2026-09-04T09:35:00+08:00"),
                       dict(observed_at="2026-09-03T15:00:00+08:00")]:
            self.assertEqual(self.run_row(**update).candidate_status, "DATA_LIMITED")

    def test_mainboard_nonst_scope_remains(self):
        self.assertEqual(self.run_row(is_st=1).candidate_status, "OUT_OF_SCOPE")
        self.assertEqual(self.run_row(is_mainboard=0).candidate_status, "OUT_OF_SCOPE")

    def test_duplicate_stock_rejected(self):
        with self.assertRaises(ValueError):
            screen(pd.DataFrame([self.row(), self.row()]), "2026-09-04T09:26:00+08:00")


if __name__ == "__main__":
    unittest.main()
