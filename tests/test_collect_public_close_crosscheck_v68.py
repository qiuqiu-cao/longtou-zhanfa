import unittest

import pandas as pd

from collect_public_close_crosscheck_v68 import classify, limit_price


class PublicCloseCrosscheckTests(unittest.TestCase):
    def test_rounding_and_close_failed_split(self):
        self.assertEqual(limit_price(10.48), 11.53)
        frame = pd.DataFrame([
            {"code": "600001", "provider_name": "A", "open": 10.5,
             "previous_close": 10.48, "last": 11.53, "high": 11.53, "low": 10.5,
             "amount": 1e8, "quote_timestamp": "2026-09-09T15:01:00+08:00"},
            {"code": "600002", "provider_name": "B", "open": 10.5,
             "previous_close": 10.48, "last": 10.98, "high": 11.53, "low": 10.4,
             "amount": 2e8, "quote_timestamp": "2026-09-09T15:01:00+08:00"},
        ])
        _, closing, failed, fresh = classify(frame, "2026-09-09")
        self.assertEqual(fresh, 2)
        self.assertEqual(closing.code.tolist(), ["600001"])
        self.assertEqual(failed.code.tolist(), ["600002"])


if __name__ == "__main__":
    unittest.main()
