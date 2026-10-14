import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from build_minute_request_pack_v49 import build


class MinuteRequestPackTests(unittest.TestCase):
    def test_build_excludes_hash_verified_complete_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            requests = root / "requests.json"
            requests.write_text(json.dumps([["sz.000017", "2024-01-15"],
                                             ["sh.600001", "2024-01-15"]]), encoding="utf-8")
            cohort = root / "cohort.csv"
            pd.DataFrame([
                {"code": "000017", "provider_code": "sz.000017", "signal_date": "2024-01-12",
                 "entry_date": "2024-01-15", "boards": 1, "board_group": "FIRST_BOARD"},
                {"code": "600001", "provider_code": "sh.600001", "signal_date": "2024-01-12",
                 "entry_date": "2024-01-15", "boards": 1, "board_group": "FIRST_BOARD"},
            ]).to_csv(cohort, index=False)
            archive = root / "archive" / "batch_0001"
            archive.mkdir(parents=True)
            raw = archive / "sz.000017_2024-01-15.json"
            raw.write_text(json.dumps({"code": "sz.000017", "requested_date": "2024-01-15",
                                       "rows": []}), encoding="utf-8")
            digest = hashlib.sha256(raw.read_bytes()).hexdigest()
            (archive / "manifest.json").write_text(json.dumps({"cases": [{
                "code": "sz.000017", "date": "2024-01-15", "rows": 48,
                "error_code": "0", "raw_sha256": digest}]}), encoding="utf-8")
            output = root / "out"
            manifest = build(requests, cohort, [archive.parent], output, 50)
            self.assertEqual(manifest["already_complete"], 1)
            self.assertEqual(manifest["remaining"], 1)
            remaining = json.loads((output / "remaining_requests.json").read_text(encoding="utf-8"))
            self.assertEqual(remaining, [["sh.600001", "2024-01-15"]])

    def test_replay_missing_pack_includes_reconciliation_and_relay_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            requests = root / "requests.json"
            requests.write_text(json.dumps([["sz.000017", "2024-01-15"]]), encoding="utf-8")
            cohort = root / "cohort.csv"
            pd.DataFrame([
                {"code": "000017", "provider_code": "sz.000017", "signal_date": "2024-01-12",
                 "entry_date": "2024-01-15", "boards": 1, "board_group": "FIRST_BOARD"},
                {"code": "600001", "provider_code": "sh.600001", "signal_date": "2024-01-12",
                 "entry_date": "2024-01-15", "boards": 2, "board_group": "CONSECUTIVE_BOARD"},
            ]).to_csv(cohort, index=False)
            replay = root / "signals.json"
            replay.write_text(json.dumps({"missing": [
                {"code": "000017", "entry_date": "2024-01-15", "state": "MISSING_ARCHIVE"},
                {"code": "600001", "entry_date": "2024-01-15", "state": "DAILY_OHLC_CONTRADICTION"},
            ]}), encoding="utf-8")
            output = root / "out"
            manifest = build(requests, cohort, [], output, 50, replay)
            self.assertEqual(manifest["requested"], 2)
            self.assertEqual(manifest["remaining"], 2)
            self.assertEqual(manifest["remediation_reason_counts"], {
                "MISSING_ARCHIVE": 1, "DAILY_OHLC_CONTRADICTION": 1,
            })
            frame = pd.read_csv(output / "remaining_requests.csv")
            self.assertEqual(set(frame["board_group"]), {"FIRST_BOARD", "CONSECUTIVE_BOARD"})


if __name__ == "__main__":
    unittest.main()
