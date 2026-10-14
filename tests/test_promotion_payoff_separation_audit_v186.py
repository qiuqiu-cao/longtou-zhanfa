import unittest

import pandas as pd

from promotion_payoff_separation_audit_v186 import build


class PromotionPayoffSeparationAuditV186Tests(unittest.TestCase):
    def data(self):
        candidates, outcomes = [], []
        for day, date in enumerate(pd.bdate_range("2024-01-02", periods=210)):
            for lane, boards in enumerate((1, 2)):
                code = f"{day * 2 + lane:06d}"
                promotion = day % 2 == 0 or boards == 2
                touched = promotion or day % 5 == 0
                one_price = promotion and day % 4 == 0
                b2 = .02 if touched and not one_price and day % 3 == 0 else -.03
                b3 = .01 if touched and not one_price and day % 6 == 0 else -.04
                candidates.append({
                    "code": code, "signal_date": date.strftime("%Y-%m-%d"),
                    "boards": boards,
                    "board_group": "FIRST_BOARD" if boards == 1
                    else "CONSECUTIVE_BOARD",
                })
                outcomes.append({
                    "code": code, "signal_date": date.strftime("%Y-%m-%d"),
                    "promotion_LABEL_ONLY": int(promotion),
                    "next_session_touched_limit_LABEL_ONLY": touched,
                    "next_session_one_price_limit_LABEL_ONLY": one_price,
                    "b1_big_loss_mark_LABEL_ONLY": bool(touched and not promotion),
                    "b2_net_close_mark_LABEL_ONLY": b2 if touched and not one_price else None,
                    "b3_net_close_mark_LABEL_ONLY": b3 if touched and not one_price else None,
                    "actual_entry_verified": False,
                    "complete_trade_profit_LABEL_ONLY": None,
                })
        return pd.DataFrame(candidates), pd.DataFrame(outcomes)

    def contract(self):
        return {"target_coverage": {"no_touch_rows_retained_as_negative": True}}

    def test_promotion_and_payoff_are_separate_targets(self):
        candidates, outcomes = self.data()
        report, _ = build(candidates, outcomes, self.contract(), bootstrap_draws=200)
        validation = report["summaries"]["ALL"]["validation"]
        self.assertGreater(validation["promotion_rate"],
                           validation["full_denominator_buyable_b2_positive_rate"])
        self.assertGreater(validation["promoted_without_buyable_b2_positive_rate"], .5)
        self.assertFalse(report["promotion_is_buyable_b2_payoff_equivalent"])
        self.assertFalse(report[
            "promotion_is_buyable_b2_b3_sustained_equivalent"])

    def test_one_price_and_no_touch_remain_negative_in_full_denominator(self):
        candidates, outcomes = self.data()
        report, observations = build(
            candidates, outcomes, self.contract(), bootstrap_draws=200)
        self.assertTrue(report["target_contract"][
            "one_price_rows_retained_as_negative"])
        self.assertFalse(observations.loc[observations.one_price, "b2_positive"].any())
        self.assertFalse(observations.loc[~observations.touchable, "b2_positive"].any())

    def test_promotion_cannot_authorize_trade_ranking(self):
        candidates, outcomes = self.data()
        report, _ = build(candidates, outcomes, self.contract(), bootstrap_draws=200)
        governance = report["governance"]
        self.assertFalse(governance[
            "promotion_probability_may_be_presented_as_trade_win_probability"])
        self.assertFalse(governance[
            "promotion_probability_may_replace_b2_b3_payoff_target"])
        self.assertFalse(governance["entry_authorized"])
        self.assertTrue(governance[
            "requires_separate_buyability_payoff_and_big_loss_models"])

    def test_uncorrected_contract_is_rejected(self):
        candidates, outcomes = self.data()
        with self.assertRaisesRegex(ValueError, "no-touch-negative"):
            build(candidates, outcomes, {}, bootstrap_draws=200)


if __name__ == "__main__":
    unittest.main()
