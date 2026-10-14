import unittest
import pandas as pd
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from absorption_forward_report_v142 import trade_metrics, build
import absorption_forward_registry_v141 as registry
from absorption_forward_protocol_v139 import protocol


class ForwardReportTest(unittest.TestCase):
    def test_metrics_do_not_call_proxy_account_drawdown(self):
        frame=pd.DataFrame({"net_return_after_costs":[.1,-.05,-.08,.02]})
        result=trade_metrics(frame)
        self.assertEqual(result["completed_trades"],4)
        self.assertEqual(result["maximum_consecutive_losses"],2)
        self.assertIsNone(result["account_max_drawdown"])
        self.assertLess(result["trade_sequence_max_drawdown_proxy"],0)

    def test_empty_is_explicit(self):
        result=trade_metrics(pd.DataFrame({"net_return_after_costs":[]}))
        self.assertEqual(result["completed_trades"],0); self.assertIsNone(result["win_rate"])

    def test_empty_registry_is_zero_sample_not_zero_win_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/"protocol.json").write_text(json.dumps(protocol()),encoding="utf-8")
            with patch("absorption_forward_registry_v141.DEPENDENCIES",[]): registry.initialize(root)
            result=build(root)
            self.assertEqual(result["snapshot_sessions"],0)
            self.assertIsNone(result["actual_fills"]["win_rate"])
            self.assertFalse(result["challenge_70pct_testable"])
