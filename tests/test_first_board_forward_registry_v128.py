import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import first_board_forward_registry_v128 as registry


class FirstBoardForwardRegistryV128Test(unittest.TestCase):
    def test_freeze_snapshot_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = root / "registry"
            code = root / "code.py"
            artifact = root / "report.json"
            scores = root / "scores.csv"
            code.write_text("frozen\n", encoding="utf-8")
            artifact.write_text("{}", encoding="utf-8")
            signal_date = date.today().isoformat()
            data_end = (date.today() - timedelta(days=1)).isoformat()
            pd.DataFrame([
                {"code": "1", "name": "A", "signal_date": signal_date,
                 "thin_control_risk": 1, "research_score_not_probability": .9},
                {"code": "2", "name": "B", "signal_date": signal_date,
                 "thin_control_risk": 0, "research_score_not_probability": .8},
            ]).to_csv(scores, index=False)
            with patch.object(registry, "CODE_FILES", [code]):
                registry.freeze(store, data_end, [artifact])
                result = registry.snapshot(store, scores, signal_date, "15:10")
                self.assertEqual(result["shadow_top1"][0]["code"], "000002")
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    registry.snapshot(store, scores, signal_date, "15:10")
                event = json.loads((store / "ledger.jsonl").read_text(encoding="utf-8").splitlines()[-1])
                snapshot = store / event["snapshot_file"]
                snapshot.write_text(snapshot.read_text(encoding="utf-8") + " ", encoding="utf-8")
                self.assertEqual(registry.verify_ledger(store / "ledger.jsonl", store)["status"], "FAIL")

    def test_snapshot_rejects_frozen_or_earlier_date(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store, code, artifact, scores = root / "r", root / "c.py", root / "a", root / "s.csv"
            code.write_text("x", encoding="utf-8")
            artifact.write_text("x", encoding="utf-8")
            pd.DataFrame([{"code": "1", "name": "A", "signal_date": "2026-09-16",
                           "thin_control_risk": 0,
                           "research_score_not_probability": .5}]).to_csv(scores, index=False)
            with patch.object(registry, "CODE_FILES", [code]):
                registry.freeze(store, "2026-09-16", [artifact])
                with self.assertRaisesRegex(ValueError, "later"):
                    registry.snapshot(store, scores, "2026-09-16", "15:10")


if __name__ == "__main__":
    unittest.main()
