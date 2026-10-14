import unittest
import json
import tempfile
from pathlib import Path
import pandas as pd
from opening_absorption_features_v138 import features, load_union


class OpeningAbsorptionTest(unittest.TestCase):
    def bars(self):
        times = ["20260102093500000", "20260102094000000", "20260102094500000",
                 "20260102095000000", "20260102095500000", "20260102100000000"]
        close = [10, 9.8, 9.9, 10.1, 10.2, 10.2]
        return pd.DataFrame(dict(time=times, open=close, high=[x+.1 for x in close],
                                 low=[x-.1 for x in close], close=close,
                                 volume=[100, 50, 60, 120, 90, 80],
                                 amount=[1000, 500, 600, 1200, 900, 800]))

    def test_completed_bar_reclaim_and_boundaries(self):
        bars = self.bars()
        daily = pd.Series(dict(open=10, high=bars.high.max(), low=bars.low.min(), close=10.2))
        result = features("sh.600001", "2026-01-02", bars, 1, daily)
        self.assertEqual(result["first_below_cum_vwap_at"], "20260102094000000")
        self.assertEqual(result["first_reclaim_cum_vwap_at"], "20260102095000000")
        self.assertFalse(result["auction_verified"])
        self.assertFalse(result["queue_or_fill_verified"])
        self.assertTrue(result["feature_usable"])

    def test_daily_contradiction_blocks_feature_use(self):
        result = features("sh.600001", "2026-01-02", self.bars(), 1,
                          pd.Series(dict(open=99, high=100, low=1, close=99)))
        self.assertEqual(result["daily_reconciliation_status"], "DAILY_OHLC_CONTRADICTION")
        self.assertFalse(result["feature_usable"])

    def test_requires_exact_completed_window(self):
        with self.assertRaises(ValueError):
            features("sh.600001", "2026-01-02", self.bars().iloc[:-1], 1)

    def test_invalid_identity_is_disclosed_not_silently_used(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); batch = root / "batch_1"; batch.mkdir()
            raw = {"code": "sh.600001", "requested_date": "2026-01-02", "error_code": "0",
                   "fields": ["date", "time", "code", "open", "high", "low", "close", "volume", "amount", "adjustflag"],
                   "rows": []}
            raw_path = batch / "sh.600001_2026-01-02.json"
            raw_path.write_text(json.dumps(raw), encoding="utf-8")
            import hashlib
            digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
            (batch / "manifest.json").write_text(json.dumps({"cases": [{"code": "sh.600001",
                "date": "2026-01-02", "error_code": "0", "raw_sha256": digest}]}), encoding="utf-8")
            payloads, _, _, invalid = load_union([root])
            self.assertEqual(payloads, {})
            self.assertEqual(len(invalid), 1)
