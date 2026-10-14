import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from settle_emotion_forecast_v84 import STATES, settle


class SettleEmotionForecastTests(unittest.TestCase):
    def test_settles_without_mutating_forecast(self):
        with tempfile.TemporaryDirectory() as tmp:
            forecast_path = Path(tmp) / "forecast.json"
            market_path = Path(tmp) / "market.csv"
            output = Path(tmp) / "settlement.json"
            probabilities = {state: 0.0 for state in STATES}
            probabilities.update({"大分歧": .6, "修复": .4})
            forecast = {
                "signal_date": "2026-09-09", "target_date": "2026-09-10",
                "frozen_at": "2026-09-10T03:00:00+08:00",
                "probabilities": probabilities,
                "outcome_status": "UNAVAILABLE_BEFORE_TARGET_CLOSE",
            }
            forecast_path.write_text(json.dumps(forecast, ensure_ascii=False), encoding="utf-8")
            original = forecast_path.read_bytes()
            pd.DataFrame([{"date": "2026-09-10", "emotion_state": "大分歧"}]).to_csv(market_path, index=False)
            result = settle(forecast_path, market_path, output)
            self.assertTrue(result["seven_state_correct"])
            self.assertTrue(result["three_regime_correct"])
            self.assertEqual(result["trade_result_status"], "NOT_A_TRADE_SIGNAL")
            self.assertEqual(forecast_path.read_bytes(), original)

    def test_missing_target_outcome_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            forecast_path = Path(tmp) / "forecast.json"
            market_path = Path(tmp) / "market.csv"
            probabilities = {state: 1 / len(STATES) for state in STATES}
            forecast_path.write_text(json.dumps({
                "signal_date": "2026-09-09", "target_date": "2026-09-10",
                "frozen_at": "2026-09-10T03:00:00+08:00", "probabilities": probabilities,
                "outcome_status": "UNAVAILABLE_BEFORE_TARGET_CLOSE"}, ensure_ascii=False), encoding="utf-8")
            pd.DataFrame([{"date": "2026-09-09", "emotion_state": "大分歧"}]).to_csv(market_path, index=False)
            with self.assertRaisesRegex(ValueError, "missing"):
                settle(forecast_path, market_path, Path(tmp) / "out.json")


if __name__ == "__main__":
    unittest.main()
