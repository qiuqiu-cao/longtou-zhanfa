import unittest

import pandas as pd

from intraday_readiness_v9 import snapshot_coverage


class IntradayReadinessV9Tests(unittest.TestCase):
    def row(self, asof):
        values = {
            "signal_date": "2026-09-01", "trade_date": "2026-09-02", "asof": asof,
            "code": "600001", "name": "A", "source": "terminal",
            "source_timestamp": f"2026-09-02 {asof}",
            "data_status": "verified_derived_intraday_v9",
            "theme_sync_basis": "unavailable", "theme_evidence_id": "",
            "theme_group_id": "", "theme_member_coverage": None,
            "theme_observed_peers": None,
        }
        fields = {
            "auction_gap": 0.03, "auction_amount_ratio": 0.04,
            "auction_cancel_ratio": 0.10, "expectation_gap": None,
            "vwap_slope_5m": None, "vwap_slope_15m": None,
            "pullback_volume_ratio": None, "vwap_reclaimed": None,
            "reseal_minute": None, "theme_sync_open": None,
            "rivals_eliminated": None, "below_vwap_expanding": None,
            "failed_seal_twice": None,
        }
        values.update(fields)
        if asof == "09:35":
            values.update({
                "vwap_slope_5m": 0.01, "pullback_volume_ratio": 0.5,
                "vwap_reclaimed": 1, "reseal_minute": 4,
                "below_vwap_expanding": 0, "failed_seal_twice": 0,
            })
        return values

    def test_technical_sequence_is_not_full_without_verified_theme_sync(self):
        frame = pd.DataFrame([self.row("09:25"), self.row("09:35")])
        result = snapshot_coverage(frame)
        self.assertEqual(result["technical_complete_sequences"], 1)
        self.assertEqual(result["full_complete_sequences"], 0)
        self.assertEqual(result["capture_coverage"], 0.0)

    def test_missing_0935_is_not_a_complete_sequence(self):
        result = snapshot_coverage(pd.DataFrame([self.row("09:25")]))
        self.assertEqual(result["technical_complete_sequences"], 0)


if __name__ == "__main__":
    unittest.main()
