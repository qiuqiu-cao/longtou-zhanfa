import unittest

import numpy as np
import pandas as pd

from emotion_actionable_risk_v144 import FEATURES, _targets, run


def fixture(rows=520):
    rng = np.random.default_rng(144)
    signal = rng.normal(size=rows)
    states = ["修复"] * rows
    for index in range(rows - 1):
        states[index + 1] = "大分歧" if signal[index] > 0 else "升温"
    data = pd.DataFrame({"date": pd.bdate_range("2024-01-02", periods=rows),
                         "emotion_state": states})
    for index, feature in enumerate(FEATURES):
        data[feature] = signal + rng.normal(scale=.2 + index / 100, size=rows)
    risk = np.where(signal > 0, .55, .35)
    data["emotion_next_divergence_probability"] = risk * .7
    data["emotion_next_retreat_probability"] = risk * .3
    remaining = 1 - risk
    data["emotion_next_start_probability"] = remaining * .1
    data["emotion_next_warming_probability"] = remaining * .25
    data["emotion_next_acceleration_probability"] = remaining * .35
    data["emotion_next_consensus_probability"] = remaining * .1
    data["emotion_next_repair_probability"] = remaining * .2
    risk_state = pd.Series(states).isin({"大分歧", "退潮"}).astype(float).to_numpy()
    data["failed_rate"] = .2 + .2 * risk_state
    data["previous_limit_tail_loss"] = -.03 - .04 * risk_state
    data["down_limit_count"] = 5 + 15 * risk_state
    data["yesterday_premium"] = .02 - .03 * risk_state
    data["promotion_mid"] = .5 - .2 * risk_state
    return data


class EmotionActionableRiskV144Tests(unittest.TestCase):
    def test_targets_use_future_states_but_features_are_current_only(self):
        data = _targets(fixture())
        self.assertEqual(bool(data.iloc[0].risk_off_next),
                         data.iloc[1].emotion_state in {"大分歧", "退潮"})
        self.assertFalse(any(name.startswith("next_") for name in FEATURES))

    def test_nested_selection_never_uses_outer_labels(self):
        report, predictions = run(fixture())
        self.assertFalse(report["inner_selection"]["outer_labels_used_for_selection"])
        self.assertGreater(len(predictions), 150)
        self.assertTrue(set(predictions.risk_band) <= {"LOW", "MIXED", "HIGH"})
        self.assertFalse(report["policy"]["positive_buy_bonus_allowed"])
        self.assertFalse(report["policy"]["individual_stock_ranking_use_allowed"])

    def test_three_session_failure_is_not_hidden(self):
        report, _ = run(fixture())
        block = report["three_session_falsification"]
        self.assertIn("promotion_gate_passed", block)
        self.assertEqual(block["allowed_to_claim_reliable_two_to_three_day_regime_forecast"],
                         block["promotion_gate_passed"])

    def test_missing_economic_outcome_rejected(self):
        with self.assertRaisesRegex(ValueError, "promotion_mid"):
            _targets(fixture().drop(columns=["promotion_mid"]))


if __name__ == "__main__":
    unittest.main()
