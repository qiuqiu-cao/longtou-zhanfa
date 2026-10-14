import unittest

import pandas as pd

from first_board_supply_base_rate_audit_v185 import build


class FirstBoardSupplyBaseRateAuditV185Tests(unittest.TestCase):
    def data(self):
        candidates, outcomes = [], []
        for day, date in enumerate(pd.bdate_range("2024-01-02", periods=160)):
            supply = ([5, 10, 20, 30] if day < 112
                      else [6, 12, 24, 36])[day % 4]
            for within in range(supply):
                code = f"{day * 100 + within:06d}"
                promoted = within == 0 or (supply <= 6 and within == 1)
                candidates.append({
                    "code": code, "signal_date": date.strftime("%Y-%m-%d"),
                    "board_group": "FIRST_BOARD",
                })
                outcomes.append({
                    "code": code, "signal_date": date.strftime("%Y-%m-%d"),
                    "promotion_LABEL_ONLY": int(promoted),
                })
        return pd.DataFrame(candidates), pd.DataFrame(outcomes)

    def manifest(self):
        return {"target_contract": {"no_touch_rows_retained_as_negative": True}}

    def test_any_success_is_separate_from_candidate_rate(self):
        candidates, outcomes = self.data()
        report, _ = build(candidates, outcomes, self.manifest(), bootstrap_draws=200)
        self.assertFalse(report["daily_any_promotion_is_discriminative"])
        validation = report["summaries"]["validation"]
        self.assertEqual(validation["HIGH_SUPPLY"]["daily_any_promotion_rate"], 1.0)
        self.assertEqual(validation["LOW_SUPPLY"]["daily_any_promotion_rate"], 1.0)
        self.assertLess(validation["HIGH_SUPPLY"]["candidate_promotion_rate"],
                        validation["LOW_SUPPLY"]["candidate_promotion_rate"])

    def test_supply_cannot_rank_or_authorize_entry(self):
        candidates, outcomes = self.data()
        report, _ = build(candidates, outcomes, self.manifest(), bootstrap_draws=200)
        governance = report["governance"]
        self.assertFalse(governance["any_first_board_promoted_may_rank_a_candidate"])
        self.assertFalse(governance["first_board_supply_positive_ranking_factor_allowed"])
        self.assertFalse(governance["entry_authorized"])
        self.assertTrue(
            governance["requires_candidate_theme_identity_and_execution_confirmation"])

    def test_missing_labels_are_not_filled_as_zero(self):
        candidates, outcomes = self.data()
        outcomes.loc[0, "promotion_LABEL_ONLY"] = None
        report, daily = build(candidates, outcomes, self.manifest(), bootstrap_draws=200)
        self.assertFalse(report["target_contract"][
            "missing_promotion_labels_filled_as_zero"])
        self.assertLess(daily.iloc[0]["known"], daily.iloc[0]["supply"])

    def test_uncorrected_source_is_rejected(self):
        candidates, outcomes = self.data()
        with self.assertRaisesRegex(ValueError, "no-touch-negative"):
            build(candidates, outcomes, {}, bootstrap_draws=200)


if __name__ == "__main__":
    unittest.main()
