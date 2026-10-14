import unittest

import pandas as pd

from opening_gap_emotion_interaction_v183 import build


class OpeningGapEmotionInteractionV183Tests(unittest.TestCase):
    def fixture(self):
        observations, features = [], []
        dates = pd.bdate_range("2024-01-02", periods=210)
        states = ["market_state_repair", "market_state_acceleration",
                  "market_state_retreat"]
        for index, date in enumerate(dates):
            state = states[index % 3]
            for lane, band in enumerate(("GT_5_LE_7", "GT_7")):
                code = f"{index * 2 + lane + 1:06d}"
                observations.append({
                    "code": code, "signal_date": date.strftime("%Y-%m-%d"),
                    "opening_gap_band": band, "touchable": True,
                    "one_price": lane == 1 and index % 2 == 0,
                    "target": lane == 0, "b1_big_loss": lane == 1,
                    "b2_mark": .02 if lane == 0 else -.04,
                })
                row = {name: 0 for name in (
                    "market_state_start", "market_state_warming",
                    "market_state_acceleration", "market_state_consensus",
                    "market_state_divergence", "market_state_retreat",
                    "market_state_repair")}
                row.update({
                    "code": code, "signal_date": date.strftime("%Y-%m-%d"),
                    state: 1,
                })
                features.append(row)
        return pd.DataFrame(observations), pd.DataFrame(features)

    def test_regime_groups_cover_every_valid_state(self):
        rows, report = build(*self.fixture(), bootstrap_draws=20)
        self.assertEqual(set(rows.regime_group), {"SUPPORTIVE", "HOT", "ADVERSE"})
        self.assertEqual(report["scope"]["invalid_or_nonunique_state_rows_excluded"], 0)
        self.assertEqual(report["scope"]["sessions"], 210)

    def test_interaction_never_authorizes_entry_or_waives_risk(self):
        _, report = build(*self.fixture(), bootstrap_draws=20)
        governance = report["governance"]
        self.assertFalse(governance["emotion_can_override_above_seven_risk_flag"])
        self.assertFalse(governance["opening_gap_emotion_interaction_factor_allowed"])
        self.assertFalse(governance["direct_stock_reranking_allowed"])
        self.assertFalse(governance["entry_authorized"])

    def test_nonunique_state_is_excluded_not_coerced(self):
        observations, features = self.fixture()
        features.loc[0, "market_state_start"] = 1
        _, report = build(observations, features, bootstrap_draws=20)
        self.assertEqual(report["scope"]["invalid_or_nonunique_state_rows_excluded"], 1)

    def test_duplicate_candidate_is_rejected(self):
        observations, features = self.fixture()
        observations = pd.concat([observations, observations.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "keys must be unique"):
            build(observations, features, bootstrap_draws=10)


if __name__ == "__main__":
    unittest.main()
