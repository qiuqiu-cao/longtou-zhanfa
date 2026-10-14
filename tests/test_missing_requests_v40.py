import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class MissingRequestsTests(unittest.TestCase):
    def test_resume_preserves_missing_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"cohort.csv").write_text("code,provider_code,entry_date\n000001,sz.000001,2024-01-03\n000002,sz.000002,2024-01-03\n",encoding="utf-8")
            archive = root/"minute_batch"; archive.mkdir()
            rawpath = archive/"sz.000001_2024-01-03.json"
            rawpath.write_text(json.dumps(dict(frequency="5",code="sz.000001",requested_date="2024-01-03",error_code="0",rows=[[] for _ in range(48)])),encoding="utf-8")
            case = dict(code="sz.000001",date="2024-01-03",rows=48,error_code="0",raw_sha256=hashlib.sha256(rawpath.read_bytes()).hexdigest())
            (archive/"manifest.json").write_text(json.dumps(dict(cases=[case])),encoding="utf-8")
            command = [sys.executable,str(Path(__file__).resolve().parents[1]/"prepare_missing_requests_v40.py"),
                       "--cohort",str(root/"cohort.csv"),"--archive-root",str(root),"--output",str(root/"missing.json")]
            result = subprocess.run(command,capture_output=True,timeout=15)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads((root/"missing.json").read_text()),[["sz.000002","2024-01-03"]])
            rawpath.write_text("{}",encoding="utf-8")
            command[-1] = str(root/"invalid.json")
            result = subprocess.run(command,capture_output=True,timeout=15)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse((root/"invalid.json").exists())
