import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from collect_historical_5min_v33 import worker


class CollectorFrequencyTests(unittest.TestCase):
    def test_graceful_budget_logs_out_without_new_query(self):
        with tempfile.TemporaryDirectory() as directory:
            sdk = SimpleNamespace(login=Mock(return_value=SimpleNamespace(error_code="0",error_msg="")),
                                  query_history_k_data_plus=Mock(),logout=Mock())
            with patch.dict("sys.modules", {"baostock":sdk}), patch("socket.setdefaulttimeout"), patch("collect_historical_5min_v33.time.monotonic",side_effect=[0,36,36]):
                worker(Path(directory),[["sz.000017","2024-01-15"]])
            sdk.query_history_k_data_plus.assert_not_called()
            sdk.logout.assert_called_once()
            self.assertEqual(json.loads((Path(directory)/"worker_stop.json").read_text())["reason"],"GRACEFUL_TIME_BUDGET")

    def test_frequency_and_field_contract(self):
        for frequency in ("d", "5"):
            with self.subTest(frequency=frequency), tempfile.TemporaryDirectory() as directory:
                result = SimpleNamespace(error_code="0", error_msg="", fields=["date"],
                                         next=Mock(side_effect=[True, False]),
                                         get_row_data=Mock(return_value=["2024-01-15"]))
                sdk = SimpleNamespace(login=Mock(return_value=SimpleNamespace(error_code="0", error_msg="")),
                                      query_history_k_data_plus=Mock(return_value=result), logout=Mock())
                with patch.dict("sys.modules", {"baostock": sdk}), patch("socket.setdefaulttimeout"):
                    worker(Path(directory), [["sz.000017", "2024-01-15"]], frequency)
                args, kwargs = sdk.query_history_k_data_plus.call_args
                self.assertEqual(kwargs["frequency"], frequency)
                self.assertEqual(kwargs["adjustflag"], "3")
                self.assertEqual("time" in args[1].split(","), frequency == "5")
                saved = json.loads((Path(directory)/"sz.000017_2024-01-15.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["frequency"], frequency)
                sdk.logout.assert_called_once()

    def test_login_error_does_not_query_or_invent_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            sdk = SimpleNamespace(login=Mock(return_value=SimpleNamespace(error_code="10002007", error_msg="network")),
                                  query_history_k_data_plus=Mock(), logout=Mock())
            with patch.dict("sys.modules", {"baostock": sdk}), patch("socket.setdefaulttimeout"):
                worker(Path(directory), [["sz.000017", "2024-01-15"]], "d")
            sdk.query_history_k_data_plus.assert_not_called()
            self.assertFalse((Path(directory)/"sz.000017_2024-01-15.json").exists())
