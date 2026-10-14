import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch
from absorption_forward_registry_v140 import initialize, snapshot_decision, verify
from absorption_forward_protocol_v139 import protocol


class RegistryTest(unittest.TestCase):
    signal_date = (date.today() + timedelta(days=1)).isoformat()
    trade_date = (date.today() + timedelta(days=2)).isoformat()

    def registry(self, directory):
        root = Path(directory); (root/"protocol.json").write_text(json.dumps(protocol()), encoding="utf-8")
        return root

    def decision(self, path):
        row = {"code": "600001", "name": "甲", "boards": 1, "board_group": "FIRST_BOARD",
               "minute_data_status": "VERIFIED_PUBLIC_MINUTE_BAR_FRESH", "close_to_vwap_proxy": -.01,
               "absorption_shadow_gate_v139": "SHADOW_VETO_BELOW_VWAP"}
        path.write_text(json.dumps({"signal_date": self.signal_date, "trade_date": self.trade_date, "asof": "10:00",
            "direct_entry_authorized": False, "absorption_shadow_gate_live_effect": "NONE_RESEARCH_ONLY",
            "first_board_candidates": [row], "relay_candidates": []}), encoding="utf-8")

    def test_snapshot_is_complete_immutable_and_unfilled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.registry(directory)
            with patch("absorption_forward_registry_v140.DEPENDENCIES", []):
                initialize(root)
                source = root/"decision.json"; self.decision(source)
                result = snapshot_decision(root, source, f"{self.trade_date}T10:01:00+08:00")
                saved = json.loads(Path(result["snapshot"]).read_text(encoding="utf-8"))
                self.assertFalse(saved["candidates"][0]["actual_fill_verified"])
                self.assertEqual(verify(root)["status"], "PASS")
                with self.assertRaises(ValueError):
                    snapshot_decision(root, source, f"{self.trade_date}T10:02:00+08:00")

    def test_non_1000_or_nonshadow_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.registry(directory)
            with patch("absorption_forward_registry_v140.DEPENDENCIES", []):
                initialize(root)
                source = root/"decision.json"; self.decision(source)
                value = json.loads(source.read_text()); value["asof"] = "09:35"
                source.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(ValueError):
                    snapshot_decision(root, source)

    def test_backfill_and_pre_freeze_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.registry(directory)
            with patch("absorption_forward_registry_v140.DEPENDENCIES", []):
                initialize(root)
                source = root/"decision.json"; self.decision(source)
                with self.assertRaisesRegex(ValueError, "same-day"):
                    wrong_day = (date.fromisoformat(self.trade_date) + timedelta(days=1)).isoformat()
                    snapshot_decision(root, source, f"{wrong_day}T10:01:00+08:00")
