import unittest

import numpy as np
import pandas as pd

from calibrate_emotion_exposure_v85 import apply_latest, calibrate


class EmotionExposureCalibrationTests(unittest.TestCase):
    def frame(self):
        n = 120
        actual = np.array([i % 3 == 0 for i in range(n)])
        p = np.where(actual, .62, .38)
        return pd.DataFrame({
            "signal_date": pd.date_range("2025-01-01", periods=n).strftime("%Y-%m-%d"),
            "target_date": pd.date_range("2025-01-02", periods=n).strftime("%Y-%m-%d"),
            "target_state": np.where(actual, "大分歧", "修复"),
            "p_大分歧": p, "p_退潮": 0.0,
        })

    def test_threshold_selected_on_development_and_tested_later(self):
        result = calibrate(self.frame())
        self.assertEqual(result["split"]["development_n"], 84)
        self.assertEqual(result["split"]["validation_n"], 36)
        self.assertEqual(result["validation"]["fn"], 0)
        self.assertEqual(result["validation"]["fp"], 0)
        self.assertTrue(result["gate_accepted"])

    def test_latest_gate_never_authorizes_direct_buy(self):
        calibration = {"selected_threshold": .5, "gate_accepted": True,
                       "gate_status": "ACCEPTED_VALIDATION_ADVANTAGE"}
        forecast = {"latest_prediction": {"signal_date": "2026-09-09",
                    "group_probabilities": {"RISK_OFF": .6}}}
        decision = apply_latest(calibration, forecast)
        self.assertEqual(decision["exposure_regime"], "DEFENSIVE")
        self.assertFalse(decision["direct_buy_authorized"])
        self.assertEqual(decision["default_position"], 0)

    def test_rejected_gate_is_only_a_lean(self):
        calibration = {"selected_threshold": .5, "gate_accepted": False,
                       "gate_status": "REJECTED_NO_VALIDATION_ADVANTAGE"}
        forecast = {"latest_prediction": {"signal_date": "2026-09-09",
                    "group_probabilities": {"RISK_OFF": .6}}}
        decision = apply_latest(calibration, forecast)
        self.assertEqual(decision["exposure_regime"], "DIAGNOSTIC_LEAN_ONLY")
        self.assertEqual(decision["high_board_relay"], "UNDECIDED_BY_MODEL")

    def test_short_sample_rejected(self):
        with self.assertRaisesRegex(ValueError, "insufficient"):
            calibrate(self.frame().head(20))


if __name__ == "__main__":
    unittest.main()
