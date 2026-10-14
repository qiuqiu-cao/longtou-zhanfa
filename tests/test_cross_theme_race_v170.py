import unittest

import pandas as pd

from cross_theme_race_v170 import build


class CrossThemeRaceV170Tests(unittest.TestCase):
    def pool(self, include_unknown=False):
        rows = [
            ("600001", 3, 100), ("600002", 1, 40),
            ("000001", 2, 80), ("000002", 1, 30), ("000003", 1, 20),
        ]
        if include_unknown:
            rows.append(("600009", 1, 10))
        return pd.DataFrame([{
            "signal_date": "2026-09-16", "code": code, "name": code,
            "boards": boards, "amount": amount,
            "pool_assignment": "CLOSING_LIMIT_UP", "is_mainboard": True,
        } for code, boards, amount in rows])

    def manifest(self, count=5, verified=False):
        return {
            "signal_date": "2026-09-16", "mainboard_closing_limit_rows": count,
            "full_market_completeness": "verified" if verified else "not_independently_verified",
        }

    def roles(self, active=False):
        groups = {
            "600001": ("A", "题材A"), "600002": ("A", "题材A"),
            "000001": ("B", "题材B"), "000002": ("B", "题材B"),
            "000003": ("B", "题材B"),
        }
        return pd.DataFrame([{
            "signal_date": "2026-09-16", "code": code,
            "theme_status": "verified_membership_and_driver" if active
            else "observed_explicit_secondary_roster",
            "theme_group_id": group, "theme": theme,
            "positive_catalyst_factor_eligible": active,
            "primary_evidence_id": "https://example.test/source",
            "evidence_available_at": "2026-09-17T04:10:00+08:00",
        } for code, (group, theme) in groups.items()])

    def test_first_board_and_relay_use_distinct_race_fields(self):
        result, meta = build(
            self.pool(), self.manifest(), [self.roles()],
            "2026-09-17T04:35:00+08:00")
        rows = result.set_index("code")
        first = rows.loc["600002"]
        relay = rows.loc["000001"]
        self.assertEqual(first.race_lane, "FIRST_BOARD_NEW_CYCLE_RACE")
        self.assertEqual(first.first_cross_theme_competing_candidate_count, 2)
        self.assertTrue(first.first_existing_relay_backed)
        self.assertTrue(pd.isna(first.relay_cross_theme_higher_candidate_count))
        self.assertEqual(relay.race_lane, "RELAY_HEIGHT_RACE")
        self.assertEqual(relay.relay_cross_theme_higher_candidate_count, 1)
        self.assertEqual(relay.relay_own_theme_first_support_count, 2)
        self.assertTrue(pd.isna(relay.first_cross_theme_competing_candidate_count))
        self.assertTrue(meta["lane_contract"]["cross_lane_common_winner_score_forbidden"])
        self.assertFalse(meta["winner_claimed"])

    def test_stronger_membership_evidence_overrides_secondary_without_becoming_active(self):
        overlay = self.roles().query("code in ['600001', '600002']").copy()
        overlay["theme_status"] = "verified_membership_driver_unverified"
        overlay["theme_group_id"] = "A_DIRECT"
        overlay["theme"] = "题材A直接制造"
        overlay["business_branch"] = "direct_manufacturing"
        overlay["evidence_available_at"] = "2026-09-17T04:30:00+08:00"
        result, meta = build(
            self.pool(), self.manifest(), [self.roles(), overlay],
            "2026-09-17T04:35:00+08:00")
        rows = result.set_index("code")
        self.assertEqual(rows.loc["600001", "theme_group_id"], "A_DIRECT")
        self.assertEqual(
            rows.loc["600001", "theme_evidence_confidence"],
            "VERIFIED_MEMBERSHIP_DRIVER_UNVERIFIED")
        self.assertEqual(
            rows.loc["600001", "cross_theme_race_status"],
            "DESCRIPTIVE_MEMBERSHIP_ONLY_DRIVER_UNVERIFIED")
        self.assertEqual(
            rows.loc["600002", "first_cross_theme_competing_theme_count"], 0)
        self.assertEqual(
            rows.loc["600002", "cross_confidence_known_candidate_count"], 3)
        self.assertEqual(rows.loc["600002", "race_peer_evidence_scope"],
                         "SAME_CONFIDENCE_ONLY")
        self.assertEqual(meta["model_eligible_count"], 0)

    def test_unknown_candidate_blocks_training_and_stays_unknown(self):
        result, meta = build(
            self.pool(include_unknown=True), self.manifest(6, verified=True),
            [self.roles(active=True)], "2026-09-17T04:35:00+08:00")
        unknown = result[result.code.eq("600009")].iloc[0]
        self.assertEqual(unknown.cross_theme_race_status,
                         "UNKNOWN_THEME_NO_RACE_INFERENCE")
        self.assertTrue(pd.isna(unknown.first_cross_theme_competing_theme_count))
        self.assertFalse(unknown.missing_theme_is_weak_theme)
        self.assertFalse(meta["complete_theme_scope"])
        self.assertEqual(meta["model_eligible_count"], 0)

    def test_complete_active_theme_scope_can_only_enable_training_not_entry(self):
        result, meta = build(
            self.pool(), self.manifest(verified=True), [self.roles(active=True)],
            "2026-09-17T04:35:00+08:00")
        self.assertTrue(result.eligible_for_cross_theme_race_training.all())
        self.assertTrue(result.cross_theme_race_status.eq(
            "MODEL_ELIGIBLE_COMPLETE_SCOPE").all())
        self.assertFalse(result.entry_authorized.any())
        self.assertEqual(meta["model_eligible_count"], 5)

    def test_future_evidence_fails_closed(self):
        roles = self.roles()
        roles.loc[0, "evidence_available_at"] = "2026-09-17T04:36:00+08:00"
        with self.assertRaisesRegex(ValueError, "after asof"):
            build(self.pool(), self.manifest(), [roles],
                  "2026-09-17T04:35:00+08:00")


if __name__ == "__main__":
    unittest.main()
