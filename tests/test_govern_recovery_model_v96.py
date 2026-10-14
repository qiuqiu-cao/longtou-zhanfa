import json
import tempfile
import unittest
from pathlib import Path

from govern_recovery_model_v96 import govern


class GovernRecoveryModelTests(unittest.TestCase):
    def test_uncertainty_failure_overrides_point_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.json"
            uncertainty = Path(tmp) / "uncertainty.json"
            model.write_text(json.dumps({"promotion_gate_passed": True,
                "latest_research_prediction": {"recovery_probability": .53}}), encoding="utf-8")
            uncertainty.write_text(json.dumps({"stable_gate_passed": False}), encoding="utf-8")
            result = govern(model, uncertainty, Path(tmp) / "verdict.json")
            self.assertEqual(result["status"], "REJECTED_UNCERTAINTY_GATE")
            self.assertFalse(result["effective_historical_gate_passed"])
            self.assertFalse(result["deployment_allowed"])


if __name__ == "__main__":
    unittest.main()
