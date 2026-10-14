import unittest
import pandas as pd
from build_opening_cohort_v35 import select


class CohortTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([
            dict(code="000017", signal_date="2024-01-16", boards=6, start_known=True, target_label=1),
            dict(code="600593", signal_date="2024-01-16", boards=2, start_known=False, target_label=0),
            dict(code="000017", signal_date="2024-01-17", boards=7, start_known=True, target_label=1)])

    def test_outcome_independence(self):
        result = select(self.frame)
        changed = self.frame.copy()
        changed["target_label"] = 99
        pd.testing.assert_frame_equal(result, select(changed))
        self.assertNotIn("target_label", result)

    def test_preserves_high_board_and_unknown_start(self):
        result = select(self.frame)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.boards.max(), 6)
        self.assertFalse(result.start_known.all())
        self.assertEqual(set(result.provider_code), {"sz.000017", "sh.600593"})
        self.assertEqual(set(result.entry_date), {"2024-01-17"})

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            select(pd.concat([self.frame, self.frame]))
