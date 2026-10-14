import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from collect_missing_batches_v48 import requested_successful_keys, successful_keys


class CollectMissingBatchesV48Tests(unittest.TestCase):
    def test_successful_keys_requires_48_rows_and_matching_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch = root / "batch_0001"
            batch.mkdir()
            raw = batch / "sz.000001_2024-01-02.json"
            raw.write_text("{}", encoding="utf-8")
            digest = hashlib.sha256(raw.read_bytes()).hexdigest()
            manifest = {"cases": [
                {"code": "sz.000001", "date": "2024-01-02", "rows": 48,
                 "error_code": "0", "raw_sha256": digest},
                {"code": "sz.000002", "date": "2024-01-02", "rows": 0,
                 "error_code": "1", "raw_sha256": "x"},
            ]}
            (batch / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(successful_keys(root), {("sz.000001", "2024-01-02")})
            self.assertEqual(requested_successful_keys(
                root, {("sh.600001", "2024-01-02")}), set())


if __name__ == "__main__":
    unittest.main()
