import json
import tempfile
import unittest
from pathlib import Path

import risk_gate_lock_v12 as lock


class RiskGateLockV12Tests(unittest.TestCase):
    def _artifacts(self, root: Path):
        script = root / "model.py"
        script.write_text("print('frozen')\n", encoding="utf-8")
        source = root / "input.csv"
        source.write_text("x\n1\n", encoding="utf-8")
        result = root / "result"
        result.mkdir()
        card = {
            "status": "REJECTED_DIAGNOSTIC_ONLY", "blind_opened": True,
            "blind_reuse_prohibited": True, "deployment_gate_passed": False,
            "veto_only_gate_passed": False, "script_sha256": lock.sha256_file(script),
            "input": {"path": str(source), "sha256": lock.sha256_file(source)},
        }
        (result / "model_card.json").write_text(json.dumps(card), encoding="utf-8")
        (result / "model_card.md").write_text("rejected\n", encoding="utf-8")
        (result / "threshold_audit.csv").write_text("x\n1\n", encoding="utf-8")
        return result, script

    def test_lock_verifies_and_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, script = self._artifacts(root)
            output = root / "lock.json"
            lock.build_lock(result, script, output)
            self.assertEqual(lock.verify_lock(output)["status"], "PASS")
            with self.assertRaisesRegex(RuntimeError, "immutable"):
                lock.build_lock(result, script, output)

    def test_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, script = self._artifacts(root)
            output = root / "lock.json"
            lock.build_lock(result, script, output)
            script.write_text("print('changed')\n", encoding="utf-8")
            checked = lock.verify_lock(output)
            self.assertEqual(checked["status"], "FAIL")
            self.assertIn("model_script hash mismatch", checked["errors"])


if __name__ == "__main__":
    unittest.main()
