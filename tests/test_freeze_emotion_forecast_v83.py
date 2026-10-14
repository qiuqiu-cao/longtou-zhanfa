import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from freeze_emotion_forecast_v83 import freeze


class FreezeEmotionForecastTests(unittest.TestCase):
    def test_freezes_probabilities_and_zero_exposure_when_adverse(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "report.json"
            output = Path(tmp) / "frozen.json"
            source.write_text(json.dumps({
                "status": "DIAGNOSTIC_ONLY", "method": "markov", "evaluation": {},
                "latest_prediction": {"signal_date": "2026-09-09", "current_state": "大分歧",
                    "probabilities": {"大分歧": .4, "退潮": .2},
                    "adverse_probability": .6, "supportive_probability": .4,
                    "direction_call": "偏分歧或退潮"}}, ensure_ascii=False), encoding="utf-8")
            class FrozenNow:
                @classmethod
                def now(cls, tz):
                    from datetime import datetime
                    return datetime(2026, 9, 10, 8, 0, tzinfo=tz)
            with patch("freeze_emotion_forecast_v83.datetime", FrozenNow):
                result = freeze(source, output, "2026-09-10")
            self.assertEqual(result["decision_use"]["high_board_relay"], "DISABLED")
            self.assertEqual(result["decision_use"]["default_position"], 0)
            self.assertIsNone(result["outcome"])

    def test_refuses_after_preopen(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "report.json"
            source.write_text(json.dumps({"latest_prediction": {"signal_date": "2026-09-09"}}), encoding="utf-8")
            class FrozenNow:
                @classmethod
                def now(cls, tz):
                    from datetime import datetime
                    return datetime(2026, 9, 10, 9, 16, tzinfo=tz)
            with patch("freeze_emotion_forecast_v83.datetime", FrozenNow):
                with self.assertRaisesRegex(ValueError, "pre-open"):
                    freeze(source, Path(tmp) / "out.json", "2026-09-10")

    def test_refuses_on_any_later_date_even_before_0915(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "report.json"
            source.write_text(json.dumps({"latest_prediction": {"signal_date": "2026-09-09"}}), encoding="utf-8")
            class FrozenNow:
                @classmethod
                def now(cls, tz):
                    from datetime import datetime
                    return datetime(2026, 9, 11, 8, 0, tzinfo=tz)
            with patch("freeze_emotion_forecast_v83.datetime", FrozenNow):
                with self.assertRaisesRegex(ValueError, "pre-open"):
                    freeze(source, Path(tmp) / "out.json", "2026-09-10")


if __name__ == "__main__":
    unittest.main()
