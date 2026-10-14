import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from collect_limit_pools_timeout_v204 import collect


class TimeoutCollectorV204Test(unittest.TestCase):
    def test_partial_failure_is_unavailable_not_verified_empty(self):
        good = pd.DataFrame({"代码": ["000001"]})
        functions = {
            "stock_zt_pool_em": lambda date: good,
            "stock_zt_pool_zbgc_em": lambda date: good,
            "stock_zt_pool_dtgc_em": lambda date: good,
            "stock_zt_pool_previous_em": lambda date: good,
            "stock_zt_pool_strong_em": lambda date: (_ for _ in ()).throw(
                RuntimeError("provider down")),
        }
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "archive"
            patches = [patch(f"akshare.{name}", value)
                       for name, value in functions.items()]
            for item in patches:
                item.start()
                self.addCleanup(item.stop)
            report = collect("20260917", str(target), retries=1)
            self.assertEqual(report["datasets"]["strong"]["status"], "unavailable")
            self.assertFalse((target / "strong.csv").exists())
            self.assertEqual(report["datasets"]["limit_up"]["rows"], 1)

    def test_existing_archive_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "archive"
            target.mkdir()
            with self.assertRaises(FileExistsError):
                collect("20260917", str(target), retries=1)


if __name__ == "__main__":
    unittest.main()
