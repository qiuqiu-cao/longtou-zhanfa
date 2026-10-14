import tempfile
import unittest
from pathlib import Path

import pandas as pd

import public_minute_capture_v1 as minute


class PublicMinuteCaptureV1Tests(unittest.TestCase):
    def test_jsonp_parser(self):
        text = ('/*guard*/\nvar _data=([{"day":"2026-09-07 09:30:00","open":"10.50",'
                '"high":"10.60","low":"10.40","close":"10.55","volume":"1000",'
                '"amount":"10550"}]);')
        result = minute.parse_minute_response(text, "600001")
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["code"], "600001")
        self.assertEqual(result.iloc[0]["close"], 10.55)

    @staticmethod
    def bars(code="600001", day="2026-09-07"):
        prices = [
            ("09:30:00", 10.50, 10.60, 10.40, 10.50, 1000),
            ("09:31:00", 10.50, 11.00, 10.50, 10.80, 1200),
            ("09:32:00", 10.80, 10.85, 10.55, 10.60, 1500),
            ("09:33:00", 10.60, 11.00, 10.60, 10.90, 1300),
            ("09:34:00", 10.90, 11.00, 10.90, 11.00, 1800),
        ]
        return pd.DataFrame([
            {"code": code, "minute": f"{day} {clock}", "open": open_, "high": high,
             "low": low, "close": close, "volume_shares": volume,
             "amount": close * volume}
            for clock, open_, high, low, close, volume in prices
        ])

    def test_measurements_make_reseal_and_failed_seals_reproducible(self):
        result = minute.measure_candidate(
            self.bars(), 10.0, "2026-09-07T09:35:05+08:00", "2026-09-07", "09:35",
        )
        self.assertEqual(result["minute_data_status"], "VERIFIED_PUBLIC_MINUTE_BAR_FRESH")
        self.assertEqual(result["failed_seal_minute_count_proxy"], 2)
        self.assertTrue(result["failed_seal_twice_proxy"])
        self.assertEqual(result["reseal_minute_proxy"], 4.0)
        self.assertEqual(result["evidence_level"], "INFERRED_FROM_PUBLIC_1MIN_BARS")
        self.assertFalse(result["direct_entry_authorized"])

    def test_previous_session_bars_are_stale(self):
        result = minute.measure_candidate(
            self.bars(day="2026-09-04"), 10.0,
            "2026-09-07T09:35:05+08:00", "2026-09-07", "09:35",
        )
        self.assertEqual(result["minute_data_status"], "UNAVAILABLE_NO_BARS_BEFORE_ASOF")

    def test_capture_archives_bars_and_never_authorises_enter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = root / "candidates.csv"
            pd.DataFrame([{
                "signal_date": "2026-09-04", "code": "600001", "name": "样本", "boards": 2,
                "pool_assignment": "CLOSING_LIMIT_UP", "is_mainboard": True,
                "amount": 100_000_000, "last_price": 10.0,
            }]).to_csv(candidates, index=False)

            def fetcher(codes):
                return minute.MinuteFetchResult(self.bars(), {}, 1, {"600001": "abc"})

            result = minute.capture(
                candidates, "2026-09-07", "09:35", root / "out",
                now="2026-09-07T09:35:05+08:00", fetcher=fetcher,
            )
            self.assertEqual(result["status"], "CONDITIONAL_PUBLIC_MINUTE_EVIDENCE_FRESH")
            self.assertFalse(result["direct_entry_authorized"])
            measures = pd.read_csv(Path(result["capture_dir"]) / "measurements.csv")
            self.assertFalse(measures["direct_entry_authorized"].any())
            with self.assertRaisesRegex(FileExistsError, "immutable"):
                minute.capture(
                    candidates, "2026-09-07", "09:35", root / "out",
                    now="2026-09-07T09:35:05+08:00", fetcher=fetcher,
                )


if __name__ == "__main__":
    unittest.main()
