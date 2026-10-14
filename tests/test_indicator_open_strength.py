import hashlib
import json
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from research.indicator_observations import open_strength as s


CALENDAR = {"status": "verified", "source": "synthetic_official_calendar_fixture",
    "sessions": ["2026-09-30", "2026-10-08", "2026-10-09"]}


def intake():
    return [dict(code=code, signal_date="2026-09-30", boards=2, name=code,
        provider_same_height_codes=[other for other in ("000001", "000002", "600003") if other != code])
        for code in ("000001", "000002", "600003")]


def frozen():
    return s.freeze_groups(intake(), "2026-09-30", "2026-10-08", CALENDAR,
        frozen_at="2026-10-07T16:00:00+08:00", original_roster_path="synthetic.csv",
        original_roster_sha256="a"*64)


def observations(kind):
    rows = []
    for code, price in zip(("000001", "000002", "600003"), (11.0, 10.5, 9.5)):
        row = dict(code=code, close=10.0 if kind == "daily" else price,
            adjustment="none", price_unit="yuan", source="Tencent_daily" if kind == "daily" else "explicit_minute_fixture",
            raw_path="synthetic_source.json", raw_sha256="b"*64,
            available_at="2026-10-02T12:00:00+08:00" if kind == "daily" else "2026-10-08T09:36:00+08:00",
            binding_status="RAW_FIELDS_REBUILT")
        if kind == "daily":
            row.update(date="2026-09-30", independent=True)
        else:
            row.update(bar_end="2026-10-08T09:35:00+08:00", timestamp_side="end", interval_seconds=60)
        rows.append(row)
    return rows


def derive(closes=None, minutes=None, cutoff="2026-10-08T09:36:01+08:00", derived=None, group=None):
    return s.derive_open_strength(group or frozen(), observations("daily") if closes is None else closes,
        observations("minute") if minutes is None else minutes, CALENDAR, cutoff,
        derived_at=derived or cutoff)


class OpenStrengthTests(unittest.TestCase):
    def test_exact_ex_self_median_and_percentage_points(self):
        row = derive()[0]
        self.assertEqual(row["original_peer_codes"], ["000002", "600003"])
        self.assertAlmostEqual(row["own_return_0935_pct"], 10)
        self.assertAlmostEqual(row["peer_return_0935_median_pct"], 0)
        self.assertAlmostEqual(row["relative_strength_0935_pp"], 10)
        self.assertEqual(row["relative_strength_unit"], "percentage_points")
        self.assertEqual(row["peer_coverage"], 1)
        self.assertEqual(row["status"], "OBSERVED_RESEARCH_ONLY")
        self.assertEqual(row["forward_prediction_samples_added"], 0)
        self.assertFalse(row["production_score_changed"])

    def test_missing_peer_keeps_original_group(self):
        row = derive(minutes=observations("minute")[:-1])[0]
        self.assertEqual(row["original_peer_count"], 2)
        self.assertEqual(row["missing_peer_codes"], ["600003"])
        self.assertIsNone(row["relative_strength_0935_pp"])
        self.assertAlmostEqual(row["partial_peer_median_pct_for_gap_description"], 5)
        self.assertIsNone(row["peer_return_0935_median_pct"])

    def test_missing_own_unknown(self):
        row = derive(minutes=observations("minute")[1:])[0]
        self.assertIsNone(row["own_return_0935_pct"])
        self.assertIsNone(row["relative_strength_0935_pp"])
        self.assertEqual(row["original_peer_count"], 2)

    def test_no_peers_never_substitutes_zero(self):
        rows = intake()[:1]
        rows[0]["provider_same_height_codes"] = []
        group = s.freeze_groups(rows, "2026-09-30", "2026-10-08", CALENDAR,
            frozen_at="2026-10-07T16:00:00+08:00", original_roster_path="synthetic.csv", original_roster_sha256="a"*64)
        row = derive(group=group)[0]
        self.assertIsNone(row["peer_coverage"])
        self.assertIsNone(row["relative_strength_0935_pp"])

    def test_future_target_is_waiting_without_value_or_source_availability(self):
        rows = derive(cutoff="2026-10-07T16:01:00+08:00")
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["status"] == "WAIT_TARGET_OBSERVATION" for row in rows))
        self.assertTrue(all(row["relative_strength_0935_pp"] is None and row["available_at"] is None for row in rows))

    def test_0936_received_cannot_be_used_at_0935(self):
        row = derive(cutoff="2026-10-08T09:35:05+08:00")[0]
        self.assertIsNone(row["relative_strength_0935_pp"])
        self.assertEqual(row["missing_reasons"]["000001"], "SOURCE_NOT_AVAILABLE_BY_CUTOFF")

    def test_source_cannot_arrive_before_observation(self):
        rows = observations("minute")
        rows[0]["available_at"] = "2026-10-08T09:34:59+08:00"
        self.assertIsNone(derive(minutes=rows)[0]["relative_strength_0935_pp"])

    def test_units_adjustment_side_interval_and_exact_day_rejected(self):
        changes = (("price_unit", "fen"), ("adjustment", "qfq"), ("timestamp_side", "start"),
                   ("interval_seconds", 300), ("bar_end", "2026-09-30T09:35:00+08:00"),
                   ("bar_end", "2026-10-08T09:34:00+08:00"), ("close", float("nan")),
                   ("close", 0), ("binding_status", "DECLARED_ONLY"))
        for key, value in changes:
            with self.subTest(key=key, value=value):
                rows = observations("minute")
                rows[0][key] = value
                self.assertIsNone(derive(minutes=rows)[0]["relative_strength_0935_pp"])

    def test_exact_independent_signal_day_close_required(self):
        for key, value in (("date", "2026-09-29"), ("independent", False), ("source", "explicit_minute_fixture")):
            with self.subTest(key=key):
                rows = observations("daily")
                rows[0][key] = value
                self.assertIsNone(derive(closes=rows)[0]["relative_strength_0935_pp"])

    def test_duplicate_prices_unknown_not_last_wins(self):
        rows = observations("minute")
        rows.append(deepcopy(rows[0]))
        self.assertEqual(derive(minutes=rows)[0]["missing_reasons"]["000001"], "DUPLICATE_CODE_OBSERVATIONS")

    def test_extreme_finite_price_pair_cannot_emit_nonfinite_return(self):
        closes, minutes = observations("daily"), observations("minute")
        closes[0]["close"], minutes[0]["close"] = 1e-308, 1e308
        row = derive(closes=closes, minutes=minutes)[0]
        self.assertIsNone(row["relative_strength_0935_pp"])
        self.assertEqual(row["missing_reasons"]["000001"], "NONFINITE_DERIVED_RETURN")

    def test_future_derivation_cutoff_rejected(self):
        with self.assertRaises(ValueError):
            derive(derived="2026-10-08T09:35:00+08:00")

    def test_original_group_hash_change_rejected(self):
        group = frozen()
        group["groups"][0]["original_peer_codes"].pop()
        with self.assertRaises(ValueError):
            derive(group=group)

    def test_target_must_be_exact_next_official_session(self):
        with self.assertRaises(ValueError):
            s.freeze_groups(intake(), "2026-09-30", "2026-10-09", CALENDAR,
                frozen_at="2026-10-07T16:00:00+08:00", original_roster_path="x", original_roster_sha256="a"*64)

    def test_subset_peer_reselection_and_late_freeze_rejected(self):
        rows = intake()
        rows[0]["provider_same_height_codes"].pop()
        with self.assertRaises(ValueError):
            s.freeze_groups(rows, "2026-09-30", "2026-10-08", CALENDAR,
                frozen_at="2026-10-07T16:00:00+08:00", original_roster_path="x", original_roster_sha256="a"*64)
        with self.assertRaises(ValueError):
            s.freeze_groups(intake(), "2026-09-30", "2026-10-08", CALENDAR,
                frozen_at="2026-10-08T09:31:00+08:00", original_roster_path="x", original_roster_sha256="a"*64)

    def test_postclose_replay_never_gets_forward_credit(self):
        row = derive(cutoff="2026-10-08T15:30:00+08:00")[0]
        self.assertEqual(row["mode"], "POSTCLOSE_REPLAY")
        self.assertEqual(row["forward_prediction_samples_added"], 0)

    def test_offline_adapter_rebuilds_and_refuses_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.json"
            raw.write_text(json.dumps([dict(code="000001", close=10, date="2026-09-30", price_unit="yuan", adjustment="none", independent=True, source="synthetic_daily")]), encoding="utf-8")
            past = s.stamp("2026-10-02T11:59:00+08:00").timestamp()
            os.utime(raw, (past, past))
            binding = dict(code="000001", raw_path=str(raw.resolve()), raw_sha256=s.file_sha(raw), raw_record_index=0, available_at="2026-10-02T12:00:00+08:00")
            bundle = root / "bundle.json"
            bundle.write_text(json.dumps({"records": [binding]}), encoding="utf-8")
            self.assertEqual(s.load_bound_observations(bundle)["records"][0]["close"], 10)
            binding["close"] = 99
            bundle.write_text(json.dumps({"records": [binding]}), encoding="utf-8")
            result = s.load_bound_observations(bundle)
            self.assertEqual(result["records"], [])
            self.assertIn("differs from raw", result["failures"][0]["reason"])

    def test_adapter_rejects_changed_raw_and_backdated_availability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.json"
            raw.write_text('[{"code":"000001","close":10}]', encoding="utf-8")
            binding = dict(code="000001", raw_path=str(raw.resolve()), raw_sha256="a"*64, raw_record_index=0, available_at="2026-10-02T12:00:00+08:00")
            bundle = root / "bundle.json"
            bundle.write_text(json.dumps({"records": [binding]}), encoding="utf-8")
            self.assertEqual(len(s.load_bound_observations(bundle)["failures"]), 1)
            binding["raw_sha256"] = s.file_sha(raw)
            bundle.write_text(json.dumps({"records": [binding]}), encoding="utf-8")
            self.assertIn("formed after", s.load_bound_observations(bundle)["failures"][0]["reason"])

    def test_collector_future_date_stops_before_request_or_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            with patch.object(s, "now", return_value="2026-10-07T16:00:00+08:00"):
                with self.assertRaises(ValueError):
                    s.collect_target_minutes(frozen(), CALENDAR, out, get=lambda *a, **k: self.fail("unexpected network"))
            self.assertFalse(out.exists())

    def test_collector_one_request_each_retains_403_and_unconfirmed_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "roster.csv"
            original.write_text("code\n000001\n", encoding="utf-8")
            group = s.freeze_groups(intake(), "2026-09-30", "2026-10-08", CALENDAR,
                frozen_at="2026-10-07T16:00:00+08:00", original_roster_path=str(original), original_roster_sha256=s.file_sha(original))
            calls = []
            class Response:
                def __init__(self, status):
                    self.status_code = status
                    self.content = b'var _data=([{"day":"2026-10-08 09:35:00","close":"10"}]);'
                def raise_for_status(self):
                    if self.status_code == 403:
                        raise RuntimeError("403 Forbidden")
            def get(url, **kwargs):
                calls.append(kwargs)
                return Response(403 if kwargs["params"]["symbol"] == "sz000001" else 200)
            with patch.object(s, "now", return_value="2026-10-08T09:36:00+08:00"):
                result = s.collect_target_minutes(group, CALENDAR, root / "out", get=get)
            self.assertEqual(len(calls), 3)
            self.assertTrue(all(call["timeout"] == (5, 8) for call in calls))
            self.assertEqual(result["requests"][0]["status"], "UNAVAILABLE")
            self.assertEqual(len(list((root/"out").glob("*.response"))), 3)
            self.assertEqual(len(result["observations"]), 2)
            self.assertEqual(result["observations"][0]["timestamp_side"], "unknown")
            self.assertFalse(result["timestamp_and_adjustment_verified"])

    def _collect_payload(self, payload):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "roster.csv"
            original.write_text("code\n000001\n", encoding="utf-8")
            group = s.freeze_groups(intake(), "2026-09-30", "2026-10-08", CALENDAR,
                frozen_at="2026-10-07T16:00:00+08:00", original_roster_path=str(original), original_roster_sha256=s.file_sha(original))
            class Response:
                status_code = 200
                content = ("var _data=(" + json.dumps(payload) + ");").encode()
                def raise_for_status(self):
                    pass
            with patch.object(s, "now", return_value="2026-10-08T09:36:00+08:00"):
                result = s.collect_target_minutes(group, CALENDAR, root / "out", get=lambda *a, **k: Response())
            self.assertEqual(len(list((root/"out").glob("*.response"))), 3)
            return result

    def test_collector_mixed_older_and_target_dates_retained_without_old_substitution(self):
        result = self._collect_payload([
            {"day": "2026-09-30 09:35:00", "close": "99"},
            {"day": "2026-09-30 15:00:00", "close": "98"},
            {"day": "2026-10-08 09:35:00", "close": "10"}])
        self.assertEqual(len(result["observations"]), 3)
        self.assertTrue(all(row["returned_dates"] == ["2026-09-30", "2026-10-08"] for row in result["requests"]))
        self.assertTrue(all(row["close"] == "10" and row["timestamp_side"] == "unknown" for row in result["observations"]))
        self.assertFalse(result["timestamp_and_adjustment_verified"])

    def test_collector_future_response_date_rejected_with_raw_preserved(self):
        result = self._collect_payload([
            {"day": "2026-10-08 09:35:00", "close": "10"},
            {"day": "2026-10-09 09:35:00", "close": "11"}])
        self.assertEqual(result["observations"], [])
        self.assertTrue(all(row["status"] == "UNAVAILABLE" and row["raw_sha256"] for row in result["requests"]))

    def test_collector_missing_or_duplicate_target_0935_rejected(self):
        payloads = ([{"day": "2026-09-30 09:35:00", "close": "10"}],
                    [{"day": "2026-10-08 09:35:00", "close": "10"}] * 2)
        for payload in payloads:
            with self.subTest(payload=payload):
                result = self._collect_payload(payload)
                self.assertEqual(result["observations"], [])
                self.assertTrue(all(row["status"] == "UNAVAILABLE" for row in result["requests"]))


if __name__ == "__main__":
    unittest.main()
