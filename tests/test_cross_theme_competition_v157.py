import unittest

import pandas as pd

from cross_theme_competition_v157 import build


class CrossThemeCompetitionV157Tests(unittest.TestCase):
    def inputs(self, unknown=True, full_market=True):
        rows = [
            ("a3", "600001", 3, "A", "题材A", 100),
            ("a1", "600002", 1, "A", "题材A", 40),
            ("b2", "000001", 2, "B", "题材B", 80),
            ("b1", "000002", 1, "B", "题材B", 30),
            ("b1x", "000003", 1, "B", "题材B", 20),
        ]
        frame = pd.DataFrame([{
            "candidate_id": candidate, "signal_date": "2026-09-17", "code": code,
            "name": code, "boards": boards,
            "board_group": "FIRST_BOARD" if boards == 1 else "CONSECUTIVE_BOARD",
            "amount": amount, "theme_group_id": theme_id, "theme": theme,
            "theme_factor_status": "V150_POINT_IN_TIME",
            "eligible_for_leader_factor_training": True,
            "full_market_complete": full_market,
        } for candidate, code, boards, theme_id, theme, amount in rows])
        if unknown:
            frame = pd.concat([frame, pd.DataFrame([{
                "candidate_id": "u", "signal_date": "2026-09-17", "code": "600009",
                "name": "未知", "boards": 1, "board_group": "FIRST_BOARD", "amount": 10,
                "theme_group_id": pd.NA, "theme": pd.NA,
                "theme_factor_status": "UNAVAILABLE",
                "eligible_for_leader_factor_training": False,
                "full_market_complete": full_market,
            }])], ignore_index=True)
        manifest = {"signal_date": "2026-09-17", "closing_candidate_count": len(frame)}
        return frame, manifest

    def test_first_board_and_relay_receive_different_cross_theme_factors(self):
        result, metadata = build(*self.inputs(unknown=False))
        rows = result.set_index("code")
        self.assertEqual(rows.loc["600002", "first_board_other_theme_first_board_count"], 2)
        self.assertTrue(rows.loc["600002", "first_board_theme_has_existing_relay"])
        self.assertEqual(rows.loc["000001", "relay_cross_theme_higher_count"], 1)
        self.assertEqual(rows.loc["000001", "relay_same_theme_lower_support_count"], 2)
        self.assertIn("CROSS_THEME_HIGHER_PRESSURE", rows.loc["000001", "cross_theme_events"])
        self.assertTrue(metadata["cross_theme_scope_complete"])
        self.assertEqual(metadata["model_eligible_count"], 5)

    def test_unknown_theme_blocks_training_for_every_candidate(self):
        result, metadata = build(*self.inputs(unknown=True))
        self.assertFalse(metadata["full_theme_scope"])
        self.assertFalse(result.eligible_for_cross_theme_factor_training.any())
        unknown = result[result.code.eq("600009")].iloc[0]
        self.assertTrue(pd.isna(unknown.cross_theme_same_height_candidate_count))
        self.assertFalse(unknown.missing_theme_is_weak_theme)

    def test_incomplete_full_market_scope_blocks_training(self):
        result, metadata = build(*self.inputs(unknown=False, full_market=False))
        self.assertFalse(metadata["cross_theme_scope_complete"])
        self.assertFalse(result.eligible_for_cross_theme_factor_training.any())

    def test_board_group_conflict_is_rejected(self):
        frame, manifest = self.inputs(unknown=False)
        frame.loc[0, "board_group"] = "FIRST_BOARD"
        with self.assertRaisesRegex(ValueError, "board_group"):
            build(frame, manifest)


if __name__ == "__main__":
    unittest.main()
