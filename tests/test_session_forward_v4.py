import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import math
import pandas as pd

import open_confirmation_v4 as opening
import session_forward_v4 as session


SHANGHAI = ZoneInfo("Asia/Shanghai")


class SessionForwardV4Tests(unittest.TestCase):
    def _inputs(self, root: Path, two_required: bool = False, no_required: bool = False) -> tuple[Path, Path]:
        candidates = pd.DataFrame([
            {"date": "2026-09-01", "code": "600001", "name": "甲", "boards": 3,
             "valid_identity": 0 if no_required else 1, "one_word_proxy": 0,
             "emotion_state": "升温", "announcement_risk": 0, "buyable_eod": 1},
            {"date": "2026-09-01", "code": "600002", "name": "乙", "boards": 2,
             "valid_identity": 1 if two_required else 0, "one_word_proxy": 0,
             "emotion_state": "升温", "announcement_risk": 0, "buyable_eod": 1},
        ])
        decisions = ["REJECT" if no_required else "WATCH", "WATCH" if two_required else "REJECT"]
        signals = pd.DataFrame([
            {"signal_date": "2026-09-01", "code": code, "decision": decision,
             "freeze_id": "freeze-1", "snapshot_sha256": "snapshot-hash",
             "signal_event_hash": "signal-event"}
            for code, decision in zip(["600001", "600002"], decisions)
        ])
        candidate_path = root / "candidates.csv"
        signal_path = root / "signals.csv"
        candidates.to_csv(candidate_path, index=False)
        signals.to_csv(signal_path, index=False)
        return candidate_path, signal_path

    def _snapshot(self, root: Path, asof: str = "09:25", codes: tuple[str, ...] = ("600001",)) -> Path:
        rows = []
        for code in codes:
            row = {
                "signal_date": "2026-09-01", "trade_date": "2026-09-02", "asof": asof,
                "code": code, "name": "甲" if code == "600001" else "乙",
                "source": "broker", "source_timestamp": f"2026-09-02 {asof}:00",
                "data_status": "verified_broker_export", "theme_sync_basis": "",
                "theme_evidence_id": "", "theme_group_id": "",
                "theme_member_coverage": math.nan, "theme_observed_peers": math.nan,
            }
            for field in opening.OPEN_FIELDS:
                row[field] = math.nan
            row["auction_gap"] = 0.03
            row["auction_amount_ratio"] = 0.08
            row["auction_cancel_ratio"] = 0.10
            rows.append(row)
        path = root / f"snapshot_{asof.replace(':', '')}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _register(self, root: Path, **kwargs):
        candidates, signals = self._inputs(root, **kwargs)
        output = root / "output"
        registered = session.register_plan(
            "2026-09-01", "2026-09-02", output, candidates, signals,
            now=datetime(2026, 9, 1, 16, 0, tzinfo=SHANGHAI),
        )
        return output, candidates, registered

    def test_plan_freezes_required_and_shadow_codes_before_trade(self):
        with tempfile.TemporaryDirectory() as directory:
            output, _, registered = self._register(Path(directory))
            plan = registered["plan"]
            self.assertEqual(plan["required_codes"], ["600001"])
            self.assertEqual(plan["shadow_codes"], ["600001", "600002"])
            self.assertEqual(registered["verification"]["status"], "PASS")
            self.assertTrue((output / plan["session_id"] / "session_plan.json").exists())

    def test_plan_after_trade_day_capture_start_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates, signals = self._inputs(root)
            with self.assertRaisesRegex(RuntimeError, "before trade-day 09:15"):
                session.build_plan(
                    "2026-09-01", "2026-09-02", candidates, signals,
                    now=datetime(2026, 9, 2, 9, 15, tzinfo=SHANGHAI),
                )

    def test_capture_is_bound_unique_and_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, _, registered = self._register(root)
            session_id = registered["plan"]["session_id"]
            source = self._snapshot(root)
            recorded = session.record_capture(
                session_id, "09:25", source, output, theme_sync_registry=pd.DataFrame(),
                now=datetime(2026, 9, 2, 9, 25, 30, tzinfo=SHANGHAI),
            )
            self.assertTrue(recorded["event"]["capture_complete"])
            self.assertFalse(recorded["event"]["late_capture"])
            with self.assertRaisesRegex(RuntimeError, "already registered"):
                session.record_capture(
                    session_id, "09:25", source, output, theme_sync_registry=pd.DataFrame(),
                    now=datetime(2026, 9, 2, 9, 26, tzinfo=SHANGHAI),
                )

    def test_partial_capture_is_preserved_not_called_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, _, registered = self._register(root, two_required=True)
            session_id = registered["plan"]["session_id"]
            recorded = session.record_capture(
                session_id, "09:25", self._snapshot(root), output,
                theme_sync_registry=pd.DataFrame(),
                now=datetime(2026, 9, 2, 9, 25, 20, tzinfo=SHANGHAI),
            )
            self.assertFalse(recorded["event"]["capture_complete"])
            self.assertEqual(recorded["event"]["missing_required_codes"], ["600002"])

    def test_assessment_with_missing_0935_capture_is_not_training_eligible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, candidates, registered = self._register(root)
            session_id = registered["plan"]["session_id"]
            session.record_capture(
                session_id, "09:25", self._snapshot(root), output,
                theme_sync_registry=pd.DataFrame(),
                now=datetime(2026, 9, 2, 9, 25, 20, tzinfo=SHANGHAI),
            )
            assessed = session.assess_session(
                session_id, output, candidates,
                now=datetime(2026, 9, 2, 10, 5, tzinfo=SHANGHAI),
            )
            self.assertFalse(assessed["event"]["training_eligible"])
            self.assertEqual(assessed["verification"]["status"], "PASS")

    def test_tampering_with_plan_file_breaks_ledger_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, _, registered = self._register(root)
            plan_path = output / registered["plan"]["session_id"] / "session_plan.json"
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            payload["required_codes"] = []
            plan_path.write_text(json.dumps(payload), encoding="utf-8")
            verification = session.verify_ledger(output)
            self.assertEqual(verification["status"], "FAIL")
            self.assertTrue(any("plan file hash mismatch" in item for item in verification["errors"]))

    def test_monitor_distinguishes_no_execution_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, _, registered = self._register(root, no_required=True)
            status = session.monitor(
                output, now=datetime(2026, 9, 2, 10, 5, tzinfo=SHANGHAI)
            )
            self.assertEqual(status["sessions"][0]["status"], "NO_EXECUTION_CANDIDATES")
            self.assertEqual(registered["verification"]["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
