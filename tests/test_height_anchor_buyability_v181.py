import unittest

import numpy as np
import pandas as pd

from height_anchor_buyability_v181 import CHALLENGER_FEATURES, build


class HeightAnchorBuyabilityV181Tests(unittest.TestCase):
    def frames(self, sessions=240):
        feature_rows, outcome_rows = [], []
        dates = pd.bdate_range("2024-01-02", periods=sessions)
        for day_index, date in enumerate(dates):
            signal_date = date.strftime("%Y-%m-%d")
            for candidate_index in (0, 1):
                code = f"{day_index * 2 + candidate_index + 1:06d}"
                target = (day_index + candidate_index) % 5 == 0
                row = {
                    "code": code, "signal_date": signal_date,
                    "board_group": "CONSECUTIVE_BOARD",
                    "height_gap_to_known_top": 0,
                    "feature_available_not_before": signal_date + "T15:05:00+08:00",
                }
                for feature_index, column in enumerate(CHALLENGER_FEATURES):
                    row[column] = float((day_index + candidate_index + feature_index) % 11)
                row["boards"] = float(3 + day_index % 4)
                row["sole_known_top_proxy"] = 0.0
                feature_rows.append(row)
                outcome_rows.append({
                    "code": code, "signal_date": signal_date,
                    "next_session_touched_limit_LABEL_ONLY": bool(target),
                    "next_session_one_price_limit_LABEL_ONLY": False,
                    "b2_net_close_mark_LABEL_ONLY": .02 if target else np.nan,
                    "actual_entry_verified": False,
                    "complete_trade_profit_LABEL_ONLY": np.nan,
                })
        return pd.DataFrame(feature_rows), pd.DataFrame(outcome_rows)

    def test_no_touch_and_one_price_are_explicit_negatives(self):
        features, outcomes = self.frames()
        no_touch_index, one_price_index = outcomes.index[-2], outcomes.index[-1]
        no_touch_code = outcomes.loc[no_touch_index, "code"]
        one_price_code = outcomes.loc[one_price_index, "code"]
        outcomes.loc[no_touch_index, "next_session_touched_limit_LABEL_ONLY"] = False
        outcomes.loc[no_touch_index, "b2_net_close_mark_LABEL_ONLY"] = np.nan
        outcomes.loc[one_price_index, "next_session_touched_limit_LABEL_ONLY"] = True
        outcomes.loc[one_price_index, "next_session_one_price_limit_LABEL_ONLY"] = True
        outcomes.loc[one_price_index, "b2_net_close_mark_LABEL_ONLY"] = .10
        predictions, report = build(features, outcomes, "2026-01-01",
                                    bootstrap_draws=40)
        self.assertFalse(bool(predictions.loc[
            predictions.code.eq(no_touch_code), "target"].iloc[0]))
        self.assertFalse(bool(predictions.loc[
            predictions.code.eq(one_price_code), "target"].iloc[0]))
        self.assertTrue(report["target_contract"]["no_touch_rows_retained_as_negative"])

    def test_touchable_unknown_b2_is_excluded(self):
        features, outcomes = self.frames()
        outcomes.loc[0, "next_session_touched_limit_LABEL_ONLY"] = True
        outcomes.loc[0, "next_session_one_price_limit_LABEL_ONLY"] = False
        outcomes.loc[0, "b2_net_close_mark_LABEL_ONLY"] = np.nan
        _, report = build(features, outcomes, "2026-01-01", bootstrap_draws=40)
        self.assertEqual(report["target_contract"][
            "touchable_rows_with_unknown_b2_excluded"], 1)

    def test_rejects_features_available_after_close(self):
        features, outcomes = self.frames()
        features.loc[0, "feature_available_not_before"] = (
            features.loc[0, "signal_date"] + "T15:06:00+08:00")
        with self.assertRaisesRegex(ValueError, "signal close"):
            build(features, outcomes, "2026-01-01", bootstrap_draws=20)

    def test_never_authorizes_height_anchor_as_buy(self):
        features, outcomes = self.frames()
        _, report = build(features, outcomes, "2026-01-01", bootstrap_draws=40)
        governance = report["governance"]
        self.assertFalse(governance["height_anchor_can_auto_fill_relay_primary"])
        self.assertFalse(governance["height_anchor_can_auto_fill_current_buy"])
        self.assertFalse(governance["candidate_probability_claim_allowed"])
        self.assertFalse(governance["entry_authorized"])


if __name__ == "__main__":
    unittest.main()
