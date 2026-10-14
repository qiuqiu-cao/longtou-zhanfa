import tempfile
import unittest
from pathlib import Path
from absorption_forward_protocol_v139 import classify, freeze


class ForwardProtocolTest(unittest.TestCase):
    def test_fails_closed_and_never_runs_early(self):
        self.assertEqual(classify("09:35", "VERIFIED_PUBLIC_MINUTE_BAR_FRESH", -.1),
                         "NOT_DUE_BEFORE_1000")
        self.assertEqual(classify("10:00", "STALE", .1), "DATA_LIMITED")
        self.assertEqual(classify("10:00", "VERIFIED_PUBLIC_MINUTE_BAR_FRESH", -.001),
                         "SHADOW_VETO_BELOW_VWAP")

    def test_freeze_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"protocol.json"
            result = freeze(path)
            self.assertEqual(len(result["sha256"]), 64)
            with self.assertRaises(FileExistsError):
                freeze(path)
