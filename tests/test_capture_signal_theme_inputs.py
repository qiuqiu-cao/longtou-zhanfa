"""Signal-day source boundaries; real receipts are offline availability fixtures.

These checks do not create historical features, labels, or forward predictions.
"""
import copy
import json
import unittest
from datetime import datetime
from pathlib import Path

import capture_signal_theme_inputs as capture
from public_outcome_sources_v293 import parse_daily


SIGNAL = "2026-09-30"
AVAILABLE = SIGNAL + "T16:30:00+08:00"
ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "research" / "closing_source_availability_20261002T210734261773Z"


def calendar():
    return {
        "status": "verified",
        "source": "https://example.com/synthetic-official-calendar",
        "scope": ["2026-09-29", "2026-10-09"],
        "sessions": ["2026-09-29", SIGNAL, "2026-10-08", "2026-10-09"],
    }


def quote_payload():
    return json.loads((PROBE / "raw" / "000678_eastmoney_quote.response").read_bytes())


def changed_quote(**fields):
    payload = quote_payload()
    payload["data"].update(fields)
    return payload


def epoch(timestamp):
    return int(datetime.fromisoformat(timestamp).timestamp())


class SignalDaySourceTests(unittest.TestCase):
    def test_window_accepts_close_and_cutoff_in_shanghai_timezone(self):
        for clock in (
            SIGNAL + "T15:00:00+08:00",
            SIGNAL + "T16:30:00+08:00",
            SIGNAL + "T07:00:00+00:00",
            SIGNAL + "T08:30:00+00:00",
        ):
            with self.subTest(clock=clock):
                capture.validate_window(SIGNAL, calendar(), clock)

    def test_window_rejects_late_backdated_future_closed_and_unknown_calendar(self):
        cases = [
            (SIGNAL, calendar(), SIGNAL + "T14:59:59+08:00"),
            (SIGNAL, calendar(), SIGNAL + "T16:30:01+08:00"),
            (SIGNAL, calendar(), "2026-10-01T15:30:00+08:00"),
            (SIGNAL, calendar(), "2026-09-29T15:30:00+08:00"),
            ("2026-10-01", calendar(), "2026-10-01T15:30:00+08:00"),
            ("2026-10-09", calendar(), "2026-10-09T15:30:00+08:00"),
            (SIGNAL, {**calendar(), "status": "unverified"}, AVAILABLE),
            (SIGNAL, {**calendar(), "source": ""}, AVAILABLE),
            (SIGNAL, {**calendar(), "sessions": [SIGNAL]}, AVAILABLE),
            (SIGNAL, calendar(), SIGNAL + "T15:30:00"),
        ]
        for signal, cal, clock in cases:
            with self.subTest(signal=signal, clock=clock, status=cal["status"]):
                with self.assertRaises(ValueError):
                    capture.validate_window(signal, cal, clock)

    def test_real_two_provider_receipts_parse_and_reconcile_offline(self):
        for code, expected_upper in (("000678", 1289), ("600825", 1035)):
            with self.subTest(code=code):
                payload = json.loads((PROBE / "raw" / (code + "_eastmoney_quote.response")).read_bytes())
                parsed = capture.parse_quote(payload, code, SIGNAL, AVAILABLE)
                self.assertEqual(parsed["code"], code)
                self.assertEqual(parsed["upper_cents"], expected_upper)
                self.assertEqual(parsed["values_cents"]["close"], expected_upper)
                self.assertEqual(parsed["observed_at"][:10], SIGNAL)
                native_daily = json.loads((PROBE / "raw" / (code + "_tencent_daily.response")).read_bytes())
                daily = parse_daily(native_daily, code, SIGNAL, SIGNAL)
                self.assertFalse(daily["adjusted"])
                capture.reconcile_quote_daily(parsed, daily, SIGNAL)

    def test_quote_code_name_and_required_fields_cannot_be_substituted(self):
        bad_payloads = [
            changed_quote(f57="600825"),
            changed_quote(f58=""),
            changed_quote(f58=None),
            {"data": None},
        ]
        missing = quote_payload()
        missing["data"].pop("f51")
        bad_payloads.append(missing)
        for payload in bad_payloads:
            with self.subTest(data=payload.get("data")):
                with self.assertRaises(ValueError):
                    capture.parse_quote(payload, "000678", SIGNAL, AVAILABLE)

    def test_quote_timestamp_must_be_same_day_after_close_and_already_available(self):
        bad_times = (
            SIGNAL + "T14:59:59+08:00",
            "2026-09-29T16:00:00+08:00",
            "2026-10-08T15:01:00+08:00",
            SIGNAL + "T16:30:01+08:00",
        )
        for time in bad_times:
            with self.subTest(time=time):
                with self.assertRaises(ValueError):
                    capture.parse_quote(changed_quote(f86=epoch(time)), "000678", SIGNAL, AVAILABLE)
        with self.assertRaises(ValueError):
            capture.parse_quote(quote_payload(), "000678", SIGNAL, SIGNAL + "T15:01:00+08:00")

    def test_every_price_field_requires_positive_exact_cent_grid(self):
        for field in ("f43", "f44", "f45", "f46", "f51", "f52", "f60"):
            for value in ("12.891", "NaN", "Infinity", 0, -1):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        capture.parse_quote(changed_quote(**{field: value}), "000678", SIGNAL, AVAILABLE)

    def test_ohlc_and_reported_price_limits_must_be_consistent(self):
        inconsistent = (
            {"f46": "13.00"},
            {"f43": "10.50"},
            {"f44": "10.68"},
            {"f45": "12.90"},
            {"f51": "12.88"},
            {"f52": "10.70"},
            {"f52": "12.90"},
        )
        for fields in inconsistent:
            with self.subTest(fields=fields):
                with self.assertRaises(ValueError):
                    capture.parse_quote(changed_quote(**fields), "000678", SIGNAL, AVAILABLE)

    def test_daily_missing_date_or_one_cent_disagreement_is_rejected(self):
        parsed = capture.parse_quote(quote_payload(), "000678", SIGNAL, AVAILABLE)
        daily_payload = json.loads((PROBE / "raw" / "000678_tencent_daily.response").read_bytes())
        daily = parse_daily(daily_payload, "000678", SIGNAL, SIGNAL)
        missing = copy.deepcopy(daily)
        missing["days"].pop(SIGNAL)
        with self.assertRaises(ValueError):
            capture.reconcile_quote_daily(parsed, missing, SIGNAL)
        for price in ("open", "close", "high", "low"):
            modified = copy.deepcopy(daily)
            modified["days"][SIGNAL][price] += 1
            with self.subTest(price=price):
                with self.assertRaises(ValueError):
                    capture.reconcile_quote_daily(parsed, modified, SIGNAL)


if __name__ == "__main__":
    unittest.main()
