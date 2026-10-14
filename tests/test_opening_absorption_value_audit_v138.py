import unittest
import pandas as pd
from opening_absorption_value_audit_v138 import audit


class AbsorptionValueAuditTest(unittest.TestCase):
    def test_unknown_outcome_not_counted_and_no_fill_claim(self):
        cohort = pd.DataFrame([dict(code="1", signal_date="2026-01-01", entry_date="2026-01-02",
                                    board_group="FIRST_BOARD")])
        feature = pd.DataFrame([dict(code="1", date="2026-01-02", feature_usable=True,
                                    first_below_cum_vwap_at=None, first_reclaim_cum_vwap_at=None,
                                    close_1000_above_cum_vwap=True)])
        outcome = pd.DataFrame([dict(code="1", signal_date="2026-01-01",
                                    b2_net_close_mark_LABEL_ONLY=None, actual_entry_verified=False)])
        result = audit(cohort, feature, outcome)
        cell = result["groups"]["FIRST_BOARD"]["above_vwap_1000"]["True"]
        self.assertEqual(cell["b2_proxy_known"], 0)
        self.assertEqual(result["actual_verified_fills"], 0)
        self.assertFalse(result["deployment_authorized"])
