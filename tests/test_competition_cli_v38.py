import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd
from short_term_environment import METRICS


class CompetitionCLITests(unittest.TestCase):
    def test_score_with_competition_and_no_position(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            candidate = dict(code="003005", name="SYNTHETIC_NOT_MARKET", boards=7,
                             is_mainboard=1, is_st=0, valid_identity=1,
                             deviation_10d_pct=95, deviation_30d_pct=180,
                             indicator_status="verified", identity_status="verified",
                             observed_at="2026-09-04T09:35:00+08:00", available_at="2026-09-04T09:35:00+08:00",
                             source="synthetic_fixture", tradability="buyable",
                             emotion_state="升温", notice_titles="synthetic_none")
            pd.DataFrame([candidate]).to_csv(root/"candidate.csv", index=False)
            current = dict(observed_at=candidate["observed_at"], available_at=candidate["available_at"],
                           source="synthetic_fixture", data_status="verified", universe_complete=True,
                           scope_id="synthetic-theme", candidates=[dict(code="003005", boards=7, seal_state="open")])
            previous = dict(current, observed_at="2026-09-04T09:30:00+08:00",
                            available_at="2026-09-04T09:30:00+08:00",
                            candidates=[dict(code="003005", boards=7, seal_state="sealed")])
            for name, value in (("current.json", current), ("previous.json", previous)):
                (root/name).write_text(json.dumps(value), encoding="utf-8")
            environment = dict(scope_id="synthetic-mainboard", metrics={
                key: dict(value=.2, unit=unit, sample_count=20, definition_id=key,
                          cohort_id="fixed", source="synthetic_fixture", data_status="verified",
                          observed_at=candidate["observed_at"], available_at=candidate["available_at"])
                for key, (_, _, unit) in METRICS.items()})
            (root/"environment.json").write_text(json.dumps(environment), encoding="utf-8")
            command = [sys.executable, str(Path(__file__).resolve().parents[1]/"leader_workflow_v30.py"), "score",
                       "--input", str(root/"candidate.csv"), "--asof", candidate["available_at"],
                       "--competition", str(root/"current.json"), "--previous-competition", str(root/"previous.json"),
                       "--environment", str(root/"environment.json"),
                       "--output", str(root/"output.json")]
            result = subprocess.run(command, capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = json.loads((root/"output.json").read_text(encoding="utf-8"))
            self.assertFalse(output["position_created"])
            self.assertIsNone(output["probability_first_choice"])
            review = output["candidates"][0]["competition_review"]
            self.assertEqual(review["events"][0]["type"], "SUBJECT_OPENED")
            self.assertEqual(len(output["source_hashes"]), 4)
            self.assertEqual(output["short_term_environment"]["status"], "DIAGNOSTIC_ONLY")
            self.assertEqual(len(output["short_term_environment"]["evidence"]), 6)
            self.assertEqual(output["premium_model_status"], "NOT_VALIDATED")
            current["available_at"] = "2026-09-04T09:36:00+08:00"
            (root/"current.json").write_text(json.dumps(current), encoding="utf-8")
            command[-1] = str(root/"invalid.json")
            failed = subprocess.run(command, capture_output=True, timeout=20)
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse((root/"invalid.json").exists())
