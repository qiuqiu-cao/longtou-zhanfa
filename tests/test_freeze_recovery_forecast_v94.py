import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from freeze_recovery_forecast_v94 import freeze


class FreezeRecoveryForecastTests(unittest.TestCase):
    def test_freeze_is_non_actionable(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            report.write_text(json.dumps({"version": "v", "promotion_gate_passed": True,
                "latest_research_prediction": {"signal_date": "2026-09-09",
                    "current_state": "大分歧", "recovery_probability": .53,
                    "continued_risk_off_probability": .47, "top_class": "RECOVERY"}},
                ensure_ascii=False), encoding="utf-8")
            class FrozenNow:
                @classmethod
                def now(cls, tz):
                    from datetime import datetime
                    return datetime(2026, 9, 10, 8, 0, tzinfo=tz)
            with patch("freeze_recovery_forecast_v94.datetime", FrozenNow):
                result = freeze(report, "2026-09-10", Path(tmp) / "frozen.json")
            self.assertFalse(result["decision_authorized"])
            self.assertEqual(result["position"], 0)
            self.assertIsNone(result["outcome"])


if __name__ == "__main__":
    unittest.main()
