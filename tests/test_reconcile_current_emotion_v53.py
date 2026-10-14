import unittest

import pandas as pd

from reconcile_current_emotion_v53 import _previous_pool


class ReconcileCurrentEmotionV53Tests(unittest.TestCase):
    def test_previous_promotion_denominator_is_mainboard_nonst_only(self):
        raw = pd.DataFrame([
            {"代码": "600001", "名称": "沪市", "连板数": 1},
            {"代码": "002001", "名称": "深市", "连板数": 2},
            {"代码": "300001", "名称": "创业板", "连板数": 1},
            {"代码": "688001", "名称": "科创板", "连板数": 1},
            {"代码": "600002", "名称": "ST风险", "连板数": 1},
        ])
        result = _previous_pool(raw)
        self.assertEqual(result.code.tolist(), ["600001", "002001"])
        self.assertEqual(result.boards.tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()
