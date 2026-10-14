import unittest
import pandas as pd
from sole_height_benchmark_v45 import select


class SoleHeightTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([dict(code="000001",signal_date="2024-01-02",entry_date="2024-01-03",boards=6,start_known=True),
                             dict(code="000002",signal_date="2024-01-02",entry_date="2024-01-03",boards=3,start_known=True)])

    def test_high_board_retained(self):
        self.assertEqual(select(self.frame())[0]["selected_code"],"000001")

    def test_outcomes_unused(self):
        frame = self.frame(); frame["tomorrow_profit"] = [0,1]
        before = select(frame); frame["tomorrow_profit"] = [1,0]
        self.assertEqual(before,select(frame))

    def test_tie_no_arbitrary_winner(self):
        frame = self.frame(); frame["boards"] = 6
        self.assertIsNone(select(frame)[0]["selected_code"])

    def test_unknown_competitor_no_false_uniqueness(self):
        frame = self.frame(); frame.loc[1,"start_known"] = False
        self.assertEqual(select(frame)[0]["state"],"HEIGHT_EVIDENCE_INCOMPLETE")
