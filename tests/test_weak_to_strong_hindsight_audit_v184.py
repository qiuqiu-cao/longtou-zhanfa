import unittest

import pandas as pd

from weak_to_strong_hindsight_audit_v184 import build


class WeakToStrongHindsightAuditV184Tests(unittest.TestCase):
    def report(self):
        return {
            "version": "v182-opening-gap-payoff-tradeoff-1",
            "status": "DIAGNOSTIC_ONLY",
            "target_contract": {
                "no_touch_rows_retained_as_negative": True,
                "next_session_one_price_rows_retained_as_negative": True,
                "actual_fills": 0, "complete_trades": 0,
            },
        }

    def observations(self):
        rows = []
        dates = pd.bdate_range("2024-01-02", periods=210)
        for index, date in enumerate(dates):
            for lane, band in enumerate(("LE_0", "GT_0_LE_5")):
                touched = lane == 1 or index % 10 == 0
                positive = touched and index % 3 == 0
                rows.append({
                    "code": f"{index * 2 + lane + 1:06d}",
                    "signal_date": date.strftime("%Y-%m-%d"),
                    "boards": 1 if index % 2 == 0 else 2,
                    "board_group": "FIRST_BOARD" if index % 2 == 0
                    else "CONSECUTIVE_BOARD",
                    "opening_gap_band": band, "touchable": touched,
                    "one_price": False, "target": positive,
                    "b1_big_loss": touched and not positive,
                    "b2_mark": .02 if positive else (-.04 if touched else None),
                })
        return pd.DataFrame(rows)

    def test_full_and_future_conditioned_denominators_are_separate(self):
        _, report = build(self.observations(), self.report(), bootstrap_draws=20)
        block = report["hindsight_inflation"]["ALL"]["validation_low_open"]
        self.assertLess(block["full_opening_denominator_success_rate"],
                        block["later_touch_conditioned_b2_positive_rate"])
        self.assertTrue(block["later_touch_is_future_information_at_09_30"])
        self.assertFalse(report["denominator_contract"]["later_touch_available_at_09_30"])

    def test_low_open_is_neither_auto_buy_nor_auto_reject(self):
        _, report = build(self.observations(), self.report(), bootstrap_draws=20)
        governance = report["governance"]
        self.assertFalse(governance["low_open_automatic_weak_to_strong_buy_allowed"])
        self.assertFalse(governance["low_open_automatic_reject_allowed"])
        self.assertTrue(governance[
            "requires_live_reclaim_vwap_volume_and_theme_confirmation"])
        self.assertFalse(governance["entry_authorized"])

    def test_invalid_source_contract_is_rejected(self):
        report = self.report()
        report["target_contract"]["no_touch_rows_retained_as_negative"] = False
        with self.assertRaisesRegex(ValueError, "source contract"):
            build(self.observations(), report, bootstrap_draws=10)

    def test_duplicate_candidate_is_rejected(self):
        observations = self.observations()
        observations = pd.concat([observations, observations.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "keys must be unique"):
            build(observations, self.report(), bootstrap_draws=10)


if __name__ == "__main__":
    unittest.main()
