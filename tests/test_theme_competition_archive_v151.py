import copy
import unittest

import pandas as pd

from theme_competition_archive_v151 import build


ASOF = "2026-09-17T20:00:00+08:00"


def inputs(full_market=False):
    pool = pd.DataFrame([
        {"candidate_id": "a", "signal_date": "2026-09-17", "code": "600001", "name": "甲",
         "boards": 2, "board_group": "CONSECUTIVE_BOARD", "pool_assignment": "CLOSING_LIMIT_UP",
         "is_mainboard": True, "is_st": False, "is_delisting_period": False,
         "security_status": "verified", "limit_up_status": "verified", "identity_status": "verified",
         "observed_at": "2026-09-17T15:00:00+08:00", "available_at": "2026-09-17T15:10:00+08:00",
         "amount": 100, "turnover": 10, "first_seal_time": 93000, "failed_seal_count": 0},
        {"candidate_id": "b", "signal_date": "2026-09-17", "code": "000001", "name": "乙",
         "boards": 1, "board_group": "FIRST_BOARD", "pool_assignment": "CLOSING_LIMIT_UP",
         "is_mainboard": True, "is_st": False, "is_delisting_period": False,
         "security_status": "verified", "limit_up_status": "verified", "identity_status": "verified",
         "observed_at": "2026-09-17T15:00:00+08:00", "available_at": "2026-09-17T15:10:00+08:00",
         "amount": 50, "turnover": 5, "first_seal_time": 94000, "failed_seal_count": 1},
    ])
    pool_manifest = {
        "signal_date": "2026-09-17", "mainboard_closing_limit_rows": 2,
        "complete_within_archived_roster": True,
        "full_market_completeness": "verified_full_market" if full_market else "not_proven_archived_roster_only",
    }
    competition = pd.DataFrame([{
        "signal_date": "2026-09-17", "code": "600001",
        "observed_at": "2026-09-17T15:00:00+08:00",
        "available_at": "2026-09-17T15:15:00+08:00", "data_status": "verified",
        "theme_group_id": "specific", "theme": "细分", "theme_granularity": "specific_subtheme",
        "theme_role": "THEME_HEIGHT_LEADER", "competition_identity": "height_anchor",
        "theme_factor_schema_status": "V150_COMPLETE", "driver_type": "industry",
        "driver_summary": "催化", "driver_first_available_at": "2026-09-17T14:00:00+08:00",
        "driver_claim_status": "verified_fact", "expectation_status": "partially_traded",
        "expectation_basis": "尚未一致", "positive_catalyst_factor_eligible": True,
        "same_board_peer_count": 0, "higher_board_peer_count": 0, "same_theme_peer_count": 1,
        "same_theme_board_peer_count": 0, "higher_theme_peer_count": 0, "events": "[]",
    }])
    comp_manifest = {"signal_date": "2026-09-17", "closing_candidate_count": 2,
                     "verified_theme_candidate_count": 1}
    return pool, pool_manifest, competition, comp_manifest


class ThemeCompetitionArchiveTests(unittest.TestCase):
    def test_unknown_candidate_remains_explicit_missing_not_zero(self):
        result, metadata = build(*inputs(), ASOF)
        unknown = result[result.code.eq("000001")].iloc[0]
        self.assertEqual(unknown.theme_factor_status, "UNAVAILABLE")
        self.assertTrue(pd.isna(unknown.theme_group_id))
        self.assertTrue(pd.isna(unknown.theme_membership_known))
        self.assertFalse(unknown.missing_theme_is_negative)
        self.assertEqual(metadata["unavailable_codes"], ["000001"])

    def test_full_market_and_v150_are_both_required_for_training(self):
        result, _ = build(*inputs(full_market=False), ASOF)
        self.assertFalse(result.eligible_for_leader_factor_training.any())
        result, metadata = build(*inputs(full_market=True), ASOF)
        self.assertTrue(result.loc[result.code.eq("600001"), "eligible_for_leader_factor_training"].iloc[0])
        self.assertFalse(result.loc[result.code.eq("000001"), "eligible_for_leader_factor_training"].iloc[0])
        self.assertEqual(metadata["model_eligible_count"], 1)

    def test_extra_competition_code_and_future_evidence_fail_closed(self):
        values = inputs()
        extra = values[2].iloc[0].copy()
        extra["code"] = "600999"
        bad_comp = pd.concat([values[2], extra.to_frame().T], ignore_index=True)
        bad_manifest = dict(values[3], verified_theme_candidate_count=2)
        with self.assertRaises(ValueError):
            build(values[0], values[1], bad_comp, bad_manifest, ASOF)
        values = list(inputs())
        values[2].loc[0, "available_at"] = "2026-09-18T09:00:00+08:00"
        with self.assertRaises(ValueError):
            build(*values, ASOF)

    def test_declared_pool_count_must_match(self):
        values = list(inputs())
        values[1] = copy.deepcopy(values[1])
        values[1]["mainboard_closing_limit_rows"] = 3
        with self.assertRaises(ValueError):
            build(*values, ASOF)


if __name__ == "__main__":
    unittest.main()
