import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from build_coverage_prefix_v48 import build


class BuildCoveragePrefixV48Tests(unittest.TestCase):
    def test_uses_only_contiguous_complete_prefix_and_ignores_later_complete_day(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cohort_path = root / "cohort.csv"
            pd.DataFrame([
                {"code": "000001", "entry_date": "2024-01-02", "board_group": "FIRST_BOARD"},
                {"code": "000002", "entry_date": "2024-01-03", "board_group": "CONSECUTIVE_BOARD"},
                {"code": "000003", "entry_date": "2024-01-04", "board_group": "FIRST_BOARD"},
            ]).to_csv(cohort_path, index=False)
            coverage_path = root / "coverage.json"
            coverage_path.write_text(json.dumps({
                "cohort_rows": 3,
                "coverage_by_entry_date": [
                    {"entry_date": "2024-01-02", "cohort_rows": 1,
                     "missing_rows": 0, "complete": True},
                    {"entry_date": "2024-01-03", "cohort_rows": 1,
                     "missing_rows": 1, "complete": False},
                    {"entry_date": "2024-01-04", "cohort_rows": 1,
                     "missing_rows": 0, "complete": True},
                ],
            }), encoding="utf-8")
            result = build(cohort_path, coverage_path, root / "output")
            self.assertEqual(result["complete_sessions"], 1)
            self.assertEqual(result["cohort_rows"], 1)
            self.assertEqual(result["first_incomplete_entry_date"], "2024-01-03")
            self.assertEqual(result["later_complete_dates_ignored"], ["2024-01-04"])
            saved = pd.read_csv(root / "output" / "cohort.csv", dtype={"code": str})
            self.assertEqual(saved["code"].tolist(), ["000001"])

    def test_rejects_report_for_different_cohort(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cohort_path = root / "cohort.csv"
            pd.DataFrame([{"code": "000001", "entry_date": "2024-01-02",
                           "board_group": "FIRST_BOARD"}]).to_csv(cohort_path, index=False)
            coverage_path = root / "coverage.json"
            coverage_path.write_text(json.dumps({"cohort_rows": 2,
                                                  "coverage_by_entry_date": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "row count"):
                build(cohort_path, coverage_path, root / "output")


if __name__ == "__main__":
    unittest.main()
