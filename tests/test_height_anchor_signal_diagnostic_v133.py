import unittest

import pandas as pd

from height_anchor_signal_diagnostic_v133 import _auc, _quintiles


class HeightAnchorSignalDiagnosticV133Test(unittest.TestCase):
    def test_auc_keeps_unknown_targets_out(self):
        rows = pd.DataFrame({
            "signal_date": ["a", "b", "c"], "score": [.1, .9, .8],
            "target": [0, 1, None],
        })
        result = _auc(rows, "score", "target")
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["auc"], 1.0)

    def test_quintiles_run_low_to_high(self):
        rows = pd.DataFrame({"score": range(10), "target": [0] * 5 + [1] * 5})
        result = _quintiles(rows, "score", "target")
        self.assertLess(result[0]["target_rate"], result[-1]["target_rate"])


if __name__ == "__main__":
    unittest.main()
