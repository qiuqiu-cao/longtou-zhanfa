import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from target_event_library_v19 import load_reconciliations


class BatchTests(unittest.TestCase):
    def fixture(self, root):
        source = root / "source.json"
        source.write_text("{}", encoding="utf-8")
        report = root / "report.json"
        payload = dict(sha256={str(source): hashlib.sha256(source.read_bytes()).hexdigest()}, cases=[{"code": "000759"}])
        report.write_text(json.dumps(payload), encoding="utf-8")
        return source, report

    def test_duplicate_batch_does_not_duplicate_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            _, report = self.fixture(Path(directory))
            self.assertEqual(len(load_reconciliations([report, report])), 1)

    def test_changed_input_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source, report = self.fixture(Path(directory))
            source.write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_reconciliations([report])


if __name__ == "__main__":
    unittest.main()
