import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from intraday_evidence_v9 import append_snapshots, build_snapshots, validate_auction


class IntradayEvidenceV9Tests(unittest.TestCase):
    def candidates(self):
        return pd.DataFrame({
            "date": ["2026-09-01"], "code": ["600001"], "name": ["测试股份"],
            "boards": [3],
        })

    def auction(self):
        rows = []
        for point, amount, price in (
            ("09:20", 100.0, 9.40), ("09:24:50", 50.0, 9.50), ("09:25", 50.0, 9.5454545),
        ):
            rows.append({
                "signal_date": "2026-09-01", "trade_date": "2026-09-02",
                "asof": point, "code": "600001", "name": "测试股份", "source": "broker csv",
                "source_timestamp": f"2026-09-02 {point}:00" if point.count(":") == 1 else f"2026-09-02 {point}",
                "data_status": "verified_broker_export", "matched_amount": amount,
                "indicative_price": price, "prior_close": 9.090909,
                "prior_day_amount": 1000.0,
            })
        return pd.DataFrame(rows)

    def minutes(self):
        rows = []
        values = [
            ("09:31", 9.50, 9.60, 9.40, 9.50, 100),
            ("09:32", 9.50, 10.00, 9.45, 9.80, 80),
            ("09:33", 9.80, 9.85, 9.60, 9.70, 60),
            ("09:34", 9.70, 10.00, 9.65, 9.70, 70),
            ("09:35", 9.70, 10.00, 9.70, 10.00, 120),
        ]
        for point, open_, high, low, close, volume in values:
            rows.append({
                "signal_date": "2026-09-01", "trade_date": "2026-09-02",
                "bar_end": f"2026-09-02 {point}:00", "code": "600001", "name": "测试股份",
                "source": "terminal csv", "source_timestamp": "2026-09-02 09:35:00",
                "data_status": "verified_terminal_export", "open": open_, "high": high,
                "low": low, "close": close, "volume_shares": volume,
                "amount_yuan": close * volume, "limit_up_price": 10.0,
            })
        return pd.DataFrame(rows)

    def test_primitive_fields_derive_auction_and_reseal_factors(self):
        result = build_snapshots(
            self.auction(), self.minutes(), self.candidates(),
            "2026-09-01", "2026-09-02", "09:35",
        )
        row = result.iloc[0]
        self.assertAlmostEqual(row["auction_gap"], 0.05, places=5)
        self.assertAlmostEqual(row["auction_amount_ratio"], 0.05)
        self.assertAlmostEqual(row["auction_cancel_ratio"], 0.50)
        self.assertEqual(
            row["auction_cancel_ratio_semantics"],
            "LEVEL1_MATCHED_AMOUNT_NET_DECAY_PROXY_NOT_GROSS_CANCELLATION",
        )
        self.assertEqual(row["failed_seal_twice"], 1.0)
        self.assertEqual(row["reseal_minute"], 5.0)
        self.assertEqual(row["data_status"], "verified_derived_intraday_v9")

    def test_future_bar_mutation_does_not_change_0935_factors(self):
        base = self.minutes()
        future = base.iloc[-1].copy()
        future["bar_end"] = "2026-09-02 09:40:00"
        future["source_timestamp"] = "2026-09-02 09:40:00"
        future["close"] = 1.0
        future["low"] = 1.0
        future["open"] = 1.0
        future["high"] = 10.0
        mutated = pd.concat([base, future.to_frame().T], ignore_index=True)
        first = build_snapshots(self.auction(), base, self.candidates(), "2026-09-01", "2026-09-02", "09:35")
        second = build_snapshots(self.auction(), mutated, self.candidates(), "2026-09-01", "2026-09-02", "09:35")
        columns = [
            "vwap_slope_5m", "pullback_volume_ratio", "vwap_reclaimed",
            "reseal_minute", "below_vwap_expanding", "failed_seal_twice",
        ]
        np.testing.assert_allclose(first[columns].to_numpy(float), second[columns].to_numpy(float), equal_nan=True)

    def test_unverified_auction_export_is_rejected(self):
        auction = self.auction()
        auction.loc[0, "data_status"] = "inferred"
        with self.assertRaisesRegex(ValueError, "verified raw export"):
            validate_auction(auction)

    def test_auction_timestamp_after_asof_is_rejected(self):
        auction = self.auction()
        auction.loc[0, "source_timestamp"] = "2026-09-02 09:20:11"
        with self.assertRaisesRegex(ValueError, "point-in-time capture"):
            validate_auction(auction)

    def test_derived_archive_rejects_duplicate_immutable_key(self):
        snapshots = build_snapshots(
            self.auction(), self.minutes(), self.candidates(),
            "2026-09-01", "2026-09-02", "09:35",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.csv"
            append_snapshots(snapshots, path)
            with self.assertRaisesRegex(ValueError, "immutable V9 snapshot keys"):
                append_snapshots(snapshots, path)


if __name__ == "__main__":
    unittest.main()
