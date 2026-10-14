import json
import tempfile
import unittest
from pathlib import Path

from lock_emotion_risk_veto_v91 import lock


class LockEmotionRiskVetoTests(unittest.TestCase):
    def test_lock_never_authorizes_deployment_or_buy(self):
        with tempfile.TemporaryDirectory() as tmp:
            economic = Path(tmp) / "economic.json"
            ensemble = Path(tmp) / "ensemble.json"
            economic.write_text(json.dumps({
                "economic_gate_passed": True, "fixed_high_risk_cutoff": .5,
                "selected_challenger_weight": .3,
                "nested_time_integrity": {"outer_blind_n": 202},
                "outer_blind_risk_off_detection": {"precision": .7},
            }), encoding="utf-8")
            ensemble.write_text(json.dumps({
                "version": "v", "latest_research_prediction": {
                    "signal_date": "2026-09-09", "probabilities": {"RISK_OFF": .6}}
            }), encoding="utf-8")
            result = lock(economic, ensemble, Path(tmp) / "lock.json")
            self.assertTrue(result["current_shadow_application"]["triggered"])
            self.assertTrue(result["paper_rule"]["never_creates_buy_signal"])
            self.assertFalse(result["deployment_allowed"])

    def test_failed_economic_gate_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            economic = Path(tmp) / "economic.json"
            ensemble = Path(tmp) / "ensemble.json"
            economic.write_text(json.dumps({"economic_gate_passed": False}), encoding="utf-8")
            ensemble.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "did not pass"):
                lock(economic, ensemble, Path(tmp) / "lock.json")


if __name__ == "__main__":
    unittest.main()
