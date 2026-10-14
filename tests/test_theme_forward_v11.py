import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import theme_forward_v11 as forward


class ThemeForwardV11Tests(unittest.TestCase):
    def _fixtures(self, root: Path, peer_count: int = 1, broad: bool = False):
        plan = {
            "session_id": "session-1", "model_lock_id": "lock-1",
            "signal_date": "2026-09-01", "trade_date": "2026-09-02",
            "candidate_code": "003005", "candidate_name": "竞业达",
        }
        plan_path = root / "plan.json"
        plan_path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
        rows = [{"signal_date": "2026-09-01", "theme_group_id": "ai", "code": "003005", "name": "竞业达", "boards": 3}]
        rows.extend(
            {"signal_date": "2026-09-01", "theme_group_id": "ai", "code": f"60000{i}", "name": f"同伴{i}", "boards": 1}
            for i in range(1, peer_count + 1)
        )
        roster_path = root / "roster.csv"
        pd.DataFrame(rows).to_csv(roster_path, index=False)
        roles_path = root / "roles.csv"
        pd.DataFrame([{
            "date": "2026-09-01", "code": "003005", "theme_group_id": "ai", "theme": "AI应用",
            "theme_role": "BROAD_THEME_REFERENCE" if broad else "PIONEER",
            "theme_status": "verified_same_day_broad_group" if broad else "verified_same_day_group",
            "theme_specific_enough": not broad, "theme_ladder_verified": False,
        }]).to_csv(roles_path, index=False)
        return plan_path, roster_path, roles_path

    def test_one_peer_is_frozen_as_insufficient(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._fixtures(root, peer_count=1)
            result = forward.bind_theme(*paths, root / "out", datetime(2026, 9, 1, 20, tzinfo=ZoneInfo("Asia/Shanghai")))
            self.assertEqual(result["binding"]["theme_decision"], "INSUFFICIENT_PEER_ROSTER")
            self.assertEqual(result["binding"]["expected_peer_count"], 1)
            self.assertEqual(forward.verify_binding(Path(result["binding_path"]))["status"], "PASS")

    def test_broad_group_cannot_be_upgraded_by_many_peers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._fixtures(root, peer_count=3, broad=True)
            result = forward.bind_theme(*paths, root / "out", datetime(2026, 9, 1, 20, tzinfo=ZoneInfo("Asia/Shanghai")))
            self.assertEqual(result["binding"]["theme_decision"], "BROAD_THEME_REJECT")

    def test_tampered_role_file_breaks_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, roster, roles = self._fixtures(root, peer_count=2)
            result = forward.bind_theme(plan, roster, roles, root / "out", datetime(2026, 9, 1, 20, tzinfo=ZoneInfo("Asia/Shanghai")))
            roles.write_text(roles.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            checked = forward.verify_binding(Path(result["binding_path"]))
            self.assertEqual(checked["status"], "FAIL")
            self.assertIn("roles artifact hash mismatch", checked["errors"])


if __name__ == "__main__":
    unittest.main()
