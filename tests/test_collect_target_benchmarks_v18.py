import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from collect_target_benchmarks_v18 import collect


class Response:
    url = "https://example.test/fixture"
    content = json.dumps({"data": {"klines": ["2026-09-01,100,100,101,99,1,100",
                                              "2026-09-02,100,102,103,99,2,200"]}}).encode()

    def raise_for_status(self):
        pass

    def json(self):
        return json.loads(self.content)


class BenchmarkCollectorTests(unittest.TestCase):
    def test_archive_keeps_first_return_missing_and_provenance(self):
        with tempfile.TemporaryDirectory() as temp, patch("requests.get", return_value=Response()) as get:
            target = Path(temp) / "archive"
            result = collect(target, "2026-09-01", "2026-09-02")
            frame = pd.read_csv(target / "sh000002.csv")
            self.assertTrue(pd.isna(frame.iloc[0].benchmark_return_pct))
            self.assertAlmostEqual(frame.iloc[1].benchmark_return_pct, 2)
            self.assertFalse(result["official_reconciled"])
            self.assertEqual(len(result["indexes"]["sh000002"]["raw_sha256"]), 64)
            self.assertIn("timeout", get.call_args.kwargs)
            with self.assertRaises(FileExistsError):
                collect(target, "2026-09-01", "2026-09-02")

    def test_failed_providers_are_archived_as_unavailable(self):
        with tempfile.TemporaryDirectory() as temp, patch("requests.get", side_effect=TimeoutError("fixture")):
            result = collect(Path(temp) / "archive", "2026-09-01", "2026-09-02")
            self.assertTrue(all(x["status"] == "unavailable" for x in result["indexes"].values()))
            self.assertTrue(all(len(x["errors"]) == 2 for x in result["indexes"].values()))


if __name__ == "__main__":
    unittest.main()
