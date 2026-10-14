"""Exact last relay appearance and original proxy counterexamples."""
import unittest
import pandas as pd

from relay_turnover_reference_context import legacy_clock_proxy, turnover_references


class ContextTests(unittest.TestCase):
    def history(self):
        return pd.DataFrame([{"date": "2026-09-29", "code": "000001", "turnover": .05},
                             {"date": "2026-10-20", "code": "000001", "turnover": .8}])

    def test_gap_uses_last_pool_instead_of_previous_official_session(self):
        pools = {"2026-10-12": [{"code": "000002", "boards": 2, "turnover_pct": 7}],
                 "2026-10-13": [{"code": "000001", "boards": 2, "turnover_pct": 10}]}
        r = turnover_references(self.history(), pools, "2026-10-13", "2026-09-30", "2026-10-12")[0]
        self.assertEqual(r["turnover_reference_date"], "2026-09-29")
        self.assertEqual(r["reproduced_turnover_vs_prev"], 2)
        self.assertFalse(r["reference_is_previous_official_session"])

    def test_missing_turnover_does_not_erase_last_observed_reference(self):
        pools = {"2026-10-12": [{"code": "000001", "boards": 2, "turnover_pct": None}],
                 "2026-10-13": [{"code": "000001", "boards": 2, "turnover_pct": 10}]}
        r = turnover_references(self.history(), pools, "2026-10-13", "2026-09-30", "2026-10-12")[0]
        self.assertEqual(r["turnover_reference_date"], "2026-09-29")

    def test_first_board_does_not_update_relay_reference_and_future_ignored(self):
        pools = {"2026-10-12": [{"code": "000001", "boards": 1, "turnover_pct": 30}],
                 "2026-10-13": [{"code": "000001", "boards": 2, "turnover_pct": 10}],
                 "2026-10-14": [{"code": "000001", "boards": 3, "turnover_pct": 20}]}
        r = turnover_references(self.history(), pools, "2026-10-13", "2026-09-30", "2026-10-12")[0]
        self.assertEqual(r["reproduced_turnover_vs_prev"], 2)

    def test_current_missing_or_unknown_reference_keeps_ratio_unknown(self):
        pools = {"2026-10-13": [{"code": "000001", "boards": 2, "turnover_pct": None},
                                {"code": "000002", "boards": 2, "turnover_pct": 10}]}
        rows = turnover_references(self.history(), pools, "2026-10-13", "2026-09-30", "2026-10-12")
        self.assertTrue(all(r["reproduced_turnover_vs_prev"] is None for r in rows))
        self.assertIsNone(rows[1]["turnover_reference_date"])

    def test_duplicate_and_missing_current_pool_rejected(self):
        row = {"code": "000001", "boards": 2, "turnover_pct": 10}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            turnover_references(self.history(), {"2026-10-13": [row, row]}, "2026-10-13", "2026-09-30", "2026-10-12")
        with self.assertRaisesRegex(ValueError, "unavailable"):
            turnover_references(self.history(), {}, "2026-10-13", "2026-09-30", "2026-10-12")

    def test_original_clock_rule_and_ohlc_are_separate(self):
        self.assertEqual(legacy_clock_proxy(92500, 92500, 0, 2), 1)
        self.assertEqual(legacy_clock_proxy(92500, 93100, 0, 2), 0)
        self.assertEqual(legacy_clock_proxy(92500, 92500, 1, 2), 0)
        self.assertEqual(legacy_clock_proxy(92500, 92500, 0, 3), 0)
        # The old rule accepts these unknowns; preserve and disclose that fact.
        self.assertEqual(legacy_clock_proxy(92500, 92500, None, None), 1)
        self.assertEqual(legacy_clock_proxy(None, 92500, 0, 2), 0)


if __name__ == "__main__":
    unittest.main()
