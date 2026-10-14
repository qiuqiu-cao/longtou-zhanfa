import tempfile
import unittest
from pathlib import Path

import pandas as pd

import sse_current_master_v5 as core
from sse_breadth_master_v7 import derive_scope, freeze_scope


class SSEBreadthMasterV7Tests(unittest.TestCase):
    def test_scope_is_daily_mainboard_minus_master(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bars = root / "bars.csv"
            master = root / "master.csv"
            pd.DataFrame({"code": ["600001", "600002", "000001", "300001"]}).to_csv(bars, index=False)
            pd.DataFrame({"code": ["600001", "000001"]}).to_csv(master, index=False)
            self.assertEqual(derive_scope(bars, master), ["600002"])

    def test_frozen_scope_does_not_change_after_master_expands(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bars = root / "bars.csv"
            master = root / "master.csv"
            scope = root / "scope.csv"
            pd.DataFrame({"code": ["600001", "600002"]}).to_csv(bars, index=False)
            pd.DataFrame({"code": ["600001"]}).to_csv(master, index=False)
            first = freeze_scope(bars, master, scope)
            pd.DataFrame({"code": ["600001", "600002"]}).to_csv(master, index=False)
            second = freeze_scope(bars, master, scope)
            self.assertEqual(first["scope_codes"], 1)
            self.assertEqual(second["scope_codes"], 1)
            self.assertEqual(second["scope_source"], "IMMUTABLE_ARCHIVED_SCOPE")

    def test_core_gap_scope_is_resolved_at_call_time(self):
        with tempfile.TemporaryDirectory() as temp:
            scope = Path(temp) / "scope.csv"
            pd.DataFrame({
                "code": ["600002"],
                "security_status": ["UNVERIFIED_POINT_IN_TIME_STATUS"],
            }).to_csv(scope, index=False)
            original = core.GAP_AUDIT
            try:
                core.GAP_AUDIT = scope
                self.assertEqual(core.gap_codes(), ["600002"])
            finally:
                core.GAP_AUDIT = original


if __name__ == "__main__":
    unittest.main()
