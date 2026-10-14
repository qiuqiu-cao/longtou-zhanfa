import unittest

import pandas as pd

from limit_up_universe import screen
from leader_workflow_v30 import watchlist


class LimitUpUniverseTests(unittest.TestCase):
    asof = "2026-09-04T15:05:00+08:00"

    def row(self, **changes):
        result = dict(code="600001", name="SYNTHETIC_NOT_MARKET", boards=1,
                      is_mainboard=1, is_st=0, is_limit_up=1,
                      is_delisting_period=0, no_price_limit=0, security_status="verified", snapshot_kind="closing",
                      limit_up_status="verified", identity_status="verified",
                      observed_at="2026-09-04T15:00:00+08:00",
                      available_at="2026-09-04T15:01:00+08:00", source="synthetic")
        result.update(changes)
        return result

    def status(self, **changes):
        return screen(pd.DataFrame([self.row(**changes)]), self.asof).iloc[0].candidate_status

    def test_first_and_high_boards_without_abnormality_fields(self):
        for boards in (1, 2, 6, 15):
            self.assertEqual(self.status(boards=boards), "LIMIT_UP_OBSERVATION_CANDIDATE")

    def test_target_and_notice_not_scope_veto(self):
        self.assertEqual(self.status(deviation_10d_pct=110, deviation_30d_pct=220,
                                    notice_titles="普通异动", emotion_state="退潮"),
                         "LIMIT_UP_OBSERVATION_CANDIDATE")

    def test_st_and_nonsealed_excluded(self):
        self.assertEqual(self.status(is_st=1), "OUT_OF_SCOPE")
        self.assertEqual(self.status(is_limit_up=0, boards=0), "OUT_OF_SCOPE")
        self.assertEqual(self.status(code="300001", is_mainboard=0), "OUT_OF_SCOPE")

    def test_missing_not_zero(self):
        for changes in ({"boards": None}, {"is_limit_up": None}, {"is_st": None},
                        {"limit_up_status": "inferred"}, {"boards": 1.5},
                        {"boards": 0}, {"code": "300001"}):
            self.assertEqual(self.status(**changes), "DATA_LIMITED")

    def test_time_boundaries(self):
        for changes in ({"available_at": "2026-09-04T15:06:00+08:00"},
                        {"observed_at": "2026-09-03T15:00:00+08:00"},
                        {"observed_at": None}):
            self.assertEqual(self.status(**changes), "DATA_LIMITED")

    def test_groups_and_no_automatic_winner(self):
        result = watchlist(pd.DataFrame([self.row(), self.row(code="000001", boards=8),
                                        self.row(code="600002", is_st=1)]), self.asof)
        self.assertEqual(result["candidate_groups"],
                         {"FIRST_BOARD": ["600001"], "CONSECUTIVE_BOARD": ["000001"]})
        self.assertIsNone(result["comprehensive_first_choice"])
        self.assertFalse(result["position_created"])

    def test_locked_not_entry_or_deleted(self):
        row = screen(pd.DataFrame([self.row(tradability="locked_limit_up")]), self.asof).iloc[0]
        self.assertEqual(row.candidate_status, "LIMIT_UP_OBSERVATION_CANDIDATE")
        self.assertEqual(row.execution_observation, "UNBUYABLE")
        self.assertEqual(row.execution_permission, "NO_ENTER")

    def test_duplicate_code_rejected(self):
        with self.assertRaises(ValueError):
            screen(pd.DataFrame([self.row(), self.row()]), self.asof)

    def test_delisting_unlimited_and_intraday(self):
        self.assertEqual(self.status(is_delisting_period=1), "OUT_OF_SCOPE")
        self.assertEqual(self.status(no_price_limit=1), "OUT_OF_SCOPE")
        self.assertEqual(self.status(no_price_limit=None), "DATA_LIMITED")
        self.assertEqual(self.status(snapshot_kind="intraday"), "DATA_LIMITED")
        self.assertEqual(self.status(observed_at="2026-09-04T14:59:00+08:00"), "DATA_LIMITED")

    def test_close_failed_and_missing_pools_do_not_mix(self):
        frame=pd.DataFrame([self.row(),self.row(code="000001",boards=8),
                            self.row(code="600002",boards=0,is_limit_up=0,touched_limit=1,touch_status="verified"),
                            self.row(code="600003",boards=0,is_limit_up=0)])
        result=watchlist(frame,self.asof)
        self.assertEqual(result["failed_board_observation_codes"],["600002"])
        self.assertEqual(result["pool_summary"]["closing_limit_count"],2)
        self.assertEqual(result["pool_summary"]["board_level_counts"],{"1":1,"8":1})
        self.assertEqual(result["pool_summary"]["touch_evidence_missing_count"],1)
        self.assertIsNone(result["pool_summary"]["missing_market_codes"])
        self.assertNotIn("600002",result["candidate_groups"]["FIRST_BOARD"])

    def test_delayed_archive_can_be_used_after_signal_date_when_declared(self):
        row = self.row(signal_date="2026-09-04", available_at="2026-09-05T10:00:00+08:00")
        result = screen(pd.DataFrame([row]), "2026-09-05T21:00:00+08:00").iloc[0]
        self.assertEqual(result.candidate_status, "LIMIT_UP_OBSERVATION_CANDIDATE")
        self.assertEqual(result.archive_availability, "DELAYED_BEFORE_DECISION")

    def test_failed_board_does_not_require_relay_identity(self):
        row = self.row(boards=0, is_limit_up=0, touched_limit=1, touch_status="verified",
                       identity_status="not_applicable")
        result = screen(pd.DataFrame([row]), self.asof).iloc[0]
        self.assertEqual(result.pool_assignment, "FAILED_BOARD_OBSERVATION")


if __name__ == "__main__":
    unittest.main()
