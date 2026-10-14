import tempfile
import unittest
from pathlib import Path

import pandas as pd

import public_live_capture_v1 as live


class PublicLiveCaptureV1Tests(unittest.TestCase):
    def test_sina_response_parser_keeps_provider_timestamp(self):
        body = (
            'var hq_str_sh600001="样本A,10.00,9.50,10.20,10.30,9.90,10.19,10.20,'
            '100000,1010000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,'
            '2026-09-07,09:25:01,00";\n'
        )
        result = live.parse_sina_response(body)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["code"], "600001")
        self.assertEqual(result.iloc[0]["quote_timestamp"], "2026-09-07T09:25:01+08:00")
        self.assertEqual(result.iloc[0]["amount"], 1_010_000)

    def test_price_limit_uses_half_up_exchange_style_rounding(self):
        self.assertEqual(live.limit_price_10(6.55), 7.21)

    def test_freshness_distinguishes_fresh_delayed_and_prior_session(self):
        now = "2026-09-07T09:25:10+08:00"
        self.assertEqual(live.quote_freshness("2026-09-07T09:25:01+08:00", now)[0],
                         "VERIFIED_PUBLIC_QUOTE_FRESH")
        self.assertEqual(live.quote_freshness("2026-09-07T09:24:00+08:00", now)[0],
                         "DELAYED_PUBLIC_QUOTE")
        self.assertEqual(live.quote_freshness("2026-09-04T15:00:00+08:00", now)[0],
                         "STALE_PREVIOUS_SESSION")

    def test_load_candidates_keeps_first_and_relay_but_not_failed_or_growth_board(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidates.csv"
            pd.DataFrame([
                {"signal_date": "2026-09-04", "code": "600001", "name": "首板", "boards": 1,
                 "pool_assignment": "CLOSING_LIMIT_UP", "is_mainboard": True, "amount": 1e8, "last_price": 10},
                {"signal_date": "2026-09-04", "code": "000002", "name": "连板", "boards": 3,
                 "pool_assignment": "CLOSING_LIMIT_UP", "is_mainboard": True, "amount": 2e8, "last_price": 12},
                {"signal_date": "2026-09-04", "code": "600003", "name": "炸板", "boards": 0,
                 "pool_assignment": "FAILED_BOARD_OBSERVATION", "is_mainboard": True, "amount": 2e8, "last_price": 8},
                {"signal_date": "2026-09-04", "code": "300001", "name": "创业板", "boards": 1,
                 "pool_assignment": "CLOSING_LIMIT_UP", "is_mainboard": False, "amount": 2e8, "last_price": 8},
            ]).to_csv(path, index=False)
            result = live.load_candidates(path)
            self.assertEqual(set(result["code"]), {"600001", "000002"})
            self.assertEqual(set(result["boards"]), {1, 3})

    @staticmethod
    def fake_quotes(codes):
        rows = []
        for code in sorted(codes):
            previous = 10.0
            last = 10.2 if code != "600002" else 11.0
            rows.append({
                "code": code, "provider_name": code, "open": 10.1,
                "previous_close": previous, "last": last, "high": last,
                "low": 10.0, "bid": last - 0.01, "ask": last,
                "volume_shares": 100_000.0, "amount": 1_020_000.0,
                "quote_timestamp": "2026-09-07T09:25:01+08:00", "provider_status": "00",
            })
        return live.FetchResult(pd.DataFrame(rows), [], len(rows), ["abc"])

    def test_capture_is_immutable_and_never_authorises_enter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_path = root / "candidates.csv"
            universe_path = root / "universe.csv"
            output = root / "output"
            pd.DataFrame([
                {"signal_date": "2026-09-04", "code": "600001", "name": "首板", "boards": 1,
                 "pool_assignment": "CLOSING_LIMIT_UP", "is_mainboard": True,
                 "amount": 100_000_000, "last_price": 10.0},
                {"signal_date": "2026-09-04", "code": "000002", "name": "连板", "boards": 2,
                 "pool_assignment": "CLOSING_LIMIT_UP", "is_mainboard": True,
                 "amount": 200_000_000, "last_price": 10.0},
            ]).to_csv(candidate_path, index=False)
            pd.DataFrame([
                {"code": "600001", "name": "首板"}, {"code": "000002", "name": "连板"},
            ]).to_csv(universe_path, index=False)
            result = live.capture(
                candidate_path, "2026-09-07", "09:25", output, universe_path,
                now="2026-09-07T09:25:05+08:00", quote_fetcher=self.fake_quotes,
            )
            self.assertEqual(result["status"], "CONDITIONAL_PUBLIC_PRICE_FRESH")
            self.assertFalse(result["direct_entry_authorized"])
            quotes = pd.read_csv(Path(result["capture_dir"]) / "candidate_quotes.csv")
            self.assertFalse(quotes["direct_entry_authorized"].any())
            self.assertTrue((quotes["public_feed_decision"] == "WATCH_PUBLIC_PRICE_ONLY").all())
            with self.assertRaisesRegex(FileExistsError, "immutable"):
                live.capture(
                    candidate_path, "2026-09-07", "09:25", output, universe_path,
                    now="2026-09-07T09:25:05+08:00", quote_fetcher=self.fake_quotes,
                )

    def test_stale_quotes_force_cash(self):
        candidates = pd.DataFrame([{
            "signal_date": "2026-09-04", "code": "600001", "name": "样本", "boards": 1,
            "signal_amount": 1e8, "signal_close": 10.0,
        }])
        quotes = self.fake_quotes(["600001"]).quotes
        quotes["quote_timestamp"] = "2026-09-04T15:00:00+08:00"
        result = live.build_candidate_snapshot(
            candidates, quotes, "2026-09-07T09:25:05+08:00", "2026-09-07", "09:25",
        )
        self.assertEqual(result.iloc[0]["public_feed_decision"], "CASH_DATA_NOT_FRESH")
        self.assertFalse(result.iloc[0]["direct_entry_authorized"])

    def test_capture_fetches_candidates_before_optional_breadth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_path = root / "candidates.csv"
            universe_path = root / "universe.csv"
            pd.DataFrame([{
                "signal_date": "2026-09-04", "code": "600001", "name": "候选", "boards": 1,
                "pool_assignment": "CLOSING_LIMIT_UP", "is_mainboard": True,
                "amount": 100_000_000, "last_price": 10.0,
            }]).to_csv(candidate_path, index=False)
            pd.DataFrame([
                {"code": "600001", "name": "候选"}, {"code": "000002", "name": "广度样本"},
            ]).to_csv(universe_path, index=False)
            calls = []

            def ordered_fetch(codes):
                normalized = set(codes)
                calls.append(normalized)
                return self.fake_quotes(normalized)

            result = live.capture(
                candidate_path, "2026-09-07", "09:25", root / "out", universe_path,
                now="2026-09-07T09:25:05+08:00", quote_fetcher=ordered_fetch,
            )
            self.assertEqual(calls, [{"600001"}, {"000002"}])
            self.assertEqual(result["candidate_requested_quote_codes"], 1)
            self.assertEqual(result["breadth_requested_quote_codes"], 1)
            self.assertEqual(result["requested_quote_codes"], 2)

    def test_provider_failure_creates_fail_closed_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_path = root / "candidates.csv"
            pd.DataFrame([{
                "signal_date": "2026-09-04", "code": "600001", "name": "候选", "boards": 1,
                "pool_assignment": "CLOSING_LIMIT_UP", "is_mainboard": True,
                "amount": 100_000_000, "last_price": 10.0,
            }]).to_csv(candidate_path, index=False)

            def failed_fetch(codes):
                return live.FetchResult(pd.DataFrame(), ["timeout"], len(set(codes)), [])

            result = live.capture(
                candidate_path, "2026-09-07", "09:25", root / "out", None,
                now="2026-09-07T09:25:05+08:00", quote_fetcher=failed_fetch,
            )
            self.assertEqual(result["status"], "CASH_PUBLIC_DATA_INCOMPLETE_OR_NOT_ON_TIME")
            self.assertEqual(result["candidate_quote_rows"], 0)
            self.assertEqual(result["provider_errors"], ["candidate: timeout"])
            quotes = pd.read_csv(Path(result["capture_dir"]) / "candidate_quotes.csv")
            self.assertEqual(quotes.iloc[0]["public_feed_decision"], "CASH_DATA_NOT_FRESH")


if __name__ == "__main__":
    unittest.main()
