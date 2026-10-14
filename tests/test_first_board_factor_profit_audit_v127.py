import unittest

from first_board_factor_profit_audit_v127 import wilson


class FirstBoardFactorProfitAuditV127Test(unittest.TestCase):
    def test_wilson_interval_contains_rate(self):
        low, high = wilson(4, 10)
        self.assertLess(low, .4)
        self.assertGreater(high, .4)

    def test_wilson_empty_is_missing(self):
        self.assertIsNone(wilson(0, 0))


if __name__ == "__main__":
    unittest.main()
