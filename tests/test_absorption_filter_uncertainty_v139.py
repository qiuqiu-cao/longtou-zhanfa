import unittest
import pandas as pd
from absorption_filter_uncertainty_v139 import audit


class AbsorptionUncertaintyTest(unittest.TestCase):
    def test_direction_and_unknown_handling(self):
        cohort = pd.DataFrame([
            dict(code="1", signal_date="2026-01-01", entry_date="2026-01-02", board_group="FIRST_BOARD"),
            dict(code="2", signal_date="2026-01-01", entry_date="2026-01-02", board_group="FIRST_BOARD")])
        feature = pd.DataFrame([
            dict(code="1", date="2026-01-02", feature_usable=True, close_1000_above_cum_vwap=True),
            dict(code="2", date="2026-01-02", feature_usable=True, close_1000_above_cum_vwap=False)])
        outcome = pd.DataFrame([
            dict(code="1", signal_date="2026-01-01", b2_net_close_mark_LABEL_ONLY=.1),
            dict(code="2", signal_date="2026-01-01", b2_net_close_mark_LABEL_ONLY=None)])
        result = audit(cohort, feature, outcome, repetitions=10)
        group = result["groups"]["FIRST_BOARD"]
        self.assertEqual(group["not_above_vwap"]["b2_known"], 0)
        self.assertEqual(group["above_vwap"]["b2_mean"], .1)
        self.assertFalse(result["research_veto_authorized"])
