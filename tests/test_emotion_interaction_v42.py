import unittest
import pandas as pd
from emotion_interaction_v42 import interactions


class InteractionTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(dict(boards=[3],market_yesterday_premium=[0.02],sole_known_top_proxy=[1],
                                 market_failed_rate=[None],height_gap_to_known_top=[0],market_promotion_mid=[0.5]))

    def test_missing_preserved(self):
        output,names = interactions(self.frame())
        self.assertAlmostEqual(output[names[0]].iloc[0],0.06)
        self.assertTrue(pd.isna(output[names[1]].iloc[0]))

    def test_outcomes_unused(self):
        frame = self.frame(); frame["future_profit"] = 999
        output,names = interactions(frame)
        frame["future_profit"] = -999
        pd.testing.assert_frame_equal(output[names],interactions(frame)[0][names])

    def test_existing_column_rejected(self):
        output,_ = interactions(self.frame())
        with self.assertRaises(ValueError):
            interactions(output)
