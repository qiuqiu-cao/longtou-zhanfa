import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from freeze_regime_research_forecast_v88 import freeze


class FreezeRegimeResearchForecastTests(unittest.TestCase):
    def test_failed_model_gate_remains_non_actionable(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            report.write_text(json.dumps({
                "version": "x", "promotion": "x", "promotion_gate_passed": False,
                "latest_research_prediction": {"signal_date": "2026-09-09",
                    "probabilities": {"RISK_ON": .2, "RISK_OFF": .5, "REPAIR": .3},
                    "top_class": "RISK_OFF"}}, ensure_ascii=False), encoding="utf-8")
            class FrozenNow:
                @classmethod
                def now(cls, tz):
                    from datetime import datetime
                    return datetime(2026, 9, 10, 8, 0, tzinfo=tz)
            output = Path(tmp) / "frozen.json"
            with patch("freeze_regime_research_forecast_v88.datetime", FrozenNow):
                result = freeze(report, "2026-09-10", output)
            self.assertFalse(result["decision_authorized"])
            self.assertEqual(result["position"], 0)
            self.assertFalse(result["model_promotion_gate_passed"])


if __name__ == "__main__":
    unittest.main()
