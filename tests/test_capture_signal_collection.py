"""Offline collection contracts, with synthetic clocks and protocol/roster rows.

Two saved public price responses are fixtures only. These tests perform no HTTP,
create no real prediction or forward sample, and do not certify historical PIT.
"""
import copy
import csv
import json
import os
from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import capture_signal_theme_inputs as capture


SIGNAL = "2026-09-30"
CLOCK = SIGNAL + "T16:20:00+08:00"
ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "research" / "closing_source_availability_20261002T210734261773Z" / "raw"


def read_json(path):
    return json.loads(Path(path).read_bytes())


class SignalCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="signal_collection_contract_")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.calendar = self.folder / "calendar.json"
        self.calendar.write_text(json.dumps({
            "status": "verified", "source": "https://example.test/offline-calendar",
            "scope": ["2026-09-29", "2026-10-09"],
            "sessions": ["2026-09-29", SIGNAL, "2026-10-08", "2026-10-09"],
        }), encoding="utf-8")
        self.protocol = self.folder / "protocol.json"
        self.protocol.write_text(json.dumps({
            "prospective_information_mode": "B_FIXED_EVENING",
            "prospective_knowledge_cutoff_local": "16:30:00 Asia/Shanghai",
            "registered_at": "2026-09-29T08:00:00+08:00",
            "fixture_only": True,
        }), encoding="utf-8")
        self.roster = self.folder / "prior.csv"
        self.rows = [{
            "code": code, "name": name, "original_pool_date": "2026-09-29",
            "target_date": SIGNAL, "original_boards": "2",
            "original_available_at": "2026-09-29T16:15:00+08:00",
            "roster_derived_at": "2026-09-29T16:20:00+08:00",
            "fixture_only": "True",
        } for code, name in (("000678", "襄阳轴承"), ("600825", "新华传媒"))]
        self.write_roster()
        self.clock = CLOCK
        self.calls = []

    def write_roster(self):
        with self.roster.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        epoch = datetime.fromisoformat("2026-09-29T16:21:00+08:00").timestamp()
        os.utime(self.roster, (epoch, epoch))

    def arguments(self, output=None, signal=SIGNAL):
        return (signal, self.calendar, self.protocol,
                output or self.folder / "capture", self.roster, capture.sources.sha(self.roster))

    def fake_fetch(self, *, wrong_pool_date=False, penny_conflict=False,
                   late_pool=False, late_quote=False, future_daily_row=False,
                   forbidden_pool=False, failing_code="600825", extra_pool_code=None):
        """Keep exact request parameters and raw bytes, never call the network."""
        endpoint_kinds = {capture.sources.EM_BASE + endpoint: kind
                          for kind, endpoint in capture.sources.ENDPOINTS.items()}

        def fetch(url, parameters, path):
            parameters = copy.deepcopy(parameters)
            self.calls.append((url, parameters))
            requested = self.clock
            kind = endpoint_kinds.get(url)
            if kind is not None:
                pool = []
                if kind == "limit_up":
                    pool.append({"c": "000678", "n": "襄阳轴承", "p": 12890,
                                 "ztp": 12890, "lbc": 2})
                    if extra_pool_code:
                        pool.append({"c": extra_pool_code, "n": "合成当前候选",
                                     "p": 12890, "ztp": 12890, "lbc": 1})
                response = {"data": {"qdate": "20261008" if wrong_pool_date and kind == "limit_up"
                                      else "20260930", "tc": len(pool), "pool": pool}}
                status, http_status = "RAW_JSON_RECEIVED", 200
                if forbidden_pool and kind == "limit_up":
                    response = {"data": None, "fixture_error": "forbidden source"}
                    status, http_status = "UNAVAILABLE", 403
            else:
                code = (parameters["param"].split(",")[0][2:] if url == capture.sources.DAY_URL
                        else parameters["secid"].split(".")[1])
                if code == failing_code or code not in {"000678", "600825"}:
                    response = {"data": None, "fixture_error": "unavailable source"}
                    status, http_status = "UNAVAILABLE", 503
                elif url == capture.sources.DAY_URL:
                    response = read_json(PROBE / (code + "_tencent_daily.response"))
                    if penny_conflict and code == "000678":
                        response["data"]["sz000678"]["day"][0][2] = "12.88"
                    if future_daily_row and code == "000678":
                        future = copy.deepcopy(response["data"]["sz000678"]["day"][-1])
                        future[0] = "2026-10-08"
                        response["data"]["sz000678"]["day"].append(future)
                    status, http_status = "RAW_JSON_RECEIVED", 200
                else:
                    response = read_json(PROBE / (code + "_eastmoney_quote.response"))
                    status, http_status = "RAW_JSON_RECEIVED", 200
            available = self.clock
            if (late_pool and kind == "limit_up") or (
                    late_quote and url == capture.QUOTE_URL and code == "000678"):
                available = SIGNAL + "T16:30:01+08:00"
                self.clock = available
            path.write_bytes(json.dumps(response, ensure_ascii=False).encode("utf-8"))
            receipt = {"url": url, "parameters": parameters, "requested_at": requested,
                       "available_at": available, "raw_path": str(path.resolve()),
                       "raw_sha256": capture.sources.sha(path), "http_status": http_status,
                       "status": status}
            if status == "UNAVAILABLE":
                receipt["error"] = "synthetic 503 availability failure; no retry"
            return receipt
        return fetch

    def run_collection(self, output=None, **scenario):
        with mock.patch.object(capture, "now", side_effect=lambda: self.clock), \
                mock.patch.object(capture.sources, "fetch", side_effect=self.fake_fetch(**scenario)):
            return capture.collect(*self.arguments(output=output))

    def test_closed_future_and_stale_sessions_reject_before_http_or_output_directory(self):
        cases = (("2026-10-03", "2026-10-03T16:10:00+08:00"),
                 ("2026-10-08", "2026-10-03T16:10:00+08:00"),
                 (SIGNAL, "2026-10-03T16:10:00+08:00"),
                 (SIGNAL, SIGNAL + "T14:59:59+08:00"),
                 (SIGNAL, SIGNAL + "T16:30:01+08:00"))
        for index, (signal, clock) in enumerate(cases):
            output = self.folder / ("refused_" + str(index))
            with self.subTest(signal=signal, clock=clock), \
                    mock.patch.object(capture, "now", return_value=clock), \
                    mock.patch.object(capture.sources, "fetch") as fetch:
                with self.assertRaises(ValueError):
                    capture.collect(*self.arguments(output=output, signal=signal))
                fetch.assert_not_called()
                self.assertFalse(output.exists())

    def test_prior_absent_from_current_pool_and_failed_sources_remain_unknown_in_union(self):
        result = self.run_collection(extra_pool_code="002001")
        self.assertEqual(result["requested_codes"], ["000678", "002001", "600825"])
        self.assertEqual(result["prior_roster_rows"], 2)
        self.assertEqual(result["prior_roster_retained_rows"], 2)
        self.assertEqual(result["prior_roster_retention"], 1.0)
        self.assertEqual(result["descriptive_price_matched_rows"], 1)
        with (self.folder / "capture" / "prior_roster_observations.csv").open(encoding="utf-8-sig", newline="") as handle:
            retained = {row["code"]: row for row in csv.DictReader(handle)}
        absent = retained["600825"]
        self.assertEqual(absent["current_provider_closing_pool_presence"], "False")
        self.assertEqual(absent["price_observation_status"], "UNKNOWN")
        self.assertEqual(absent["current_promotion_label"], "")
        self.assertEqual(absent["absence_from_pool_is_failure"], "False")
        self.assertEqual(absent["close_cents"], "")
        self.assertEqual(len(self.calls), 4 + 2 * 3)
        for url, parameters in self.calls:
            if url == capture.sources.DAY_URL:
                self.assertEqual(parameters["param"].split(",")[3], SIGNAL)
                self.assertNotIn("2026-10-08", parameters["param"])
            elif "date" in parameters:
                self.assertEqual(parameters["date"], "20260930")
        self.assertEqual(result["new_settled_forward_samples"], 0)
        self.assertFalse(result["v151_ready"])
        self.assertFalse(result["model_upgraded"])

    def test_response_after_cutoff_is_retained_raw_but_not_normalized_or_retried(self):
        result = self.run_collection(late_pool=True)
        folder = self.folder / "capture"
        self.assertTrue((folder / "raw" / "limit_up.json").is_file())
        normalized = read_json(folder / "normalized.json")
        self.assertNotIn("limit_up", normalized["pools"])
        self.assertTrue(all(row["status"] == "UNKNOWN" for row in normalized["observations"].values()))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(result["status"], "LATE_ARCHIVE_NOT_PIT_ELIGIBLE")
        self.assertFalse(result["prepared_by_cutoff"])
        self.assertFalse((folder / "closing_pool.csv").exists())
        self.assertTrue(any("after fixed cutoff" in failure["reason"] for failure in result["failures"]))

    def test_prior_original_and_target_dates_must_bind_adjacent_official_sessions(self):
        for index, changes in enumerate(({"original_pool_date": "2026-09-28"},
                                        {"target_date": "2026-10-08"},
                                        {"roster_derived_at": ""},
                                        {"original_available_at": "2026-10-08T15:00:00+08:00"})):
            before = copy.deepcopy(self.rows)
            self.rows[0].update(changes)
            self.write_roster()
            output = self.folder / ("invalid_prior_" + str(index))
            with self.subTest(changes=changes), mock.patch.object(capture, "now", return_value=CLOCK), \
                    mock.patch.object(capture.sources, "fetch") as fetch:
                with self.assertRaises(ValueError):
                    capture.collect(*self.arguments(output=output))
                fetch.assert_not_called()
                self.assertFalse(output.exists())
            self.rows = before

    def test_late_quote_raw_survives_but_cannot_make_a_matched_observation(self):
        result = self.run_collection(late_quote=True)
        folder = self.folder / "capture"
        self.assertTrue((folder / "raw" / "000678_quote.json").is_file())
        normalized = read_json(folder / "normalized.json")
        self.assertIn("limit_up", normalized["pools"])
        observation = normalized["observations"]["000678"]
        self.assertEqual(observation["status"], "UNKNOWN")
        self.assertNotIn("quote", observation)
        self.assertEqual(result["descriptive_price_matched_rows"], 0)
        self.assertTrue(any(failure["code"] == "000678" and failure["kind"] == "quote"
                            and "after fixed cutoff" in failure["reason"] for failure in result["failures"]))
        self.assertFalse(result["prepared_by_cutoff"])

    def test_unrequested_future_daily_row_rejects_entire_price_observation(self):
        result = self.run_collection(future_daily_row=True)
        folder = self.folder / "capture"
        raw = read_json(folder / "raw" / "000678_daily.json")
        self.assertEqual(raw["data"]["sz000678"]["day"][-1][0], "2026-10-08")
        observation = read_json(folder / "normalized.json")["observations"]["000678"]
        self.assertEqual(observation["status"], "UNKNOWN")
        self.assertNotIn("quote", observation)
        self.assertTrue(any("future daily row" in failure["reason"] for failure in result["failures"]))
        self.assertEqual(result["new_settled_forward_samples"], 0)

    def test_pool_returned_target_day_is_rejected_not_relabelled_as_signal_day(self):
        result = self.run_collection(wrong_pool_date=True)
        folder = self.folder / "capture"
        self.assertEqual(read_json(folder / "raw" / "limit_up.json")["data"]["qdate"], "20261008")
        self.assertNotIn("limit_up", read_json(folder / "normalized.json")["pools"])
        self.assertIsNone(result["provider_closing_rows"])
        self.assertFalse((folder / "closing_pool.csv").exists())
        self.assertTrue(any("returned date differs" in failure["reason"] for failure in result["failures"]))
        with (folder / "prior_roster_observations.csv").open(encoding="utf-8-sig", newline="") as handle:
            self.assertTrue(all(row["current_provider_closing_pool_presence"] == "" for row in csv.DictReader(handle)))

    def test_one_cent_independent_close_conflict_cannot_emit_matched_price_fact_or_label(self):
        result = self.run_collection(penny_conflict=True)
        normalized = read_json(self.folder / "capture" / "normalized.json")
        observation = normalized["observations"]["000678"]
        self.assertEqual(observation["status"], "UNKNOWN")
        self.assertNotIn("quote", observation)
        self.assertIsNone(observation["promotion_label"])
        self.assertEqual(result["descriptive_price_matched_rows"], 0)
        self.assertTrue(any("price conflict: close" in failure["reason"] for failure in result["failures"]))
        self.assertFalse(result["v151_ready"])

    def test_unmodified_archive_verifies_for_research_but_not_historical_cutoff_use(self):
        self.run_collection()
        folder = self.folder / "capture"
        verified = capture.verify_archive(folder)
        self.assertEqual(verified["status"], "BOUND_RESEARCH_ARCHIVE_VERIFIED")
        self.assertEqual(verified["prior_roster_rows"], 2)
        self.assertEqual(verified["requested_codes"], 2)
        self.assertFalse(verified["prepared_by_cutoff"])
        self.assertFalse(verified["v151_ready"])
        with self.assertRaisesRegex(ValueError, "not actually formed by the fixed cutoff"):
            capture.verify_archive(folder, require_by_cutoff=True)

    def test_first_pool_403_stops_same_host_without_bypassing_independent_source_requests(self):
        result = self.run_collection(forbidden_pool=True)
        folder = self.folder / "capture"
        pool_calls = [(url, parameters) for url, parameters in self.calls
                      if url.startswith(capture.sources.EM_BASE)]
        self.assertEqual(len(pool_calls), 1)
        self.assertEqual(pool_calls[0][0], capture.sources.EM_BASE + capture.sources.ENDPOINTS["limit_up"])
        self.assertTrue((folder / "raw" / "limit_up.json").is_file())
        self.assertEqual(len([url for url, _ in self.calls if url == capture.sources.DAY_URL]), 2)
        self.assertEqual(len([url for url, _ in self.calls if url == capture.QUOTE_URL]), 2)
        self.assertEqual(len(self.calls), 5)
        receipts = {entry["kind"]: entry for entry in result["requests"]
                    if entry["kind"] in capture.sources.ENDPOINTS}
        self.assertEqual(receipts["limit_up"]["http_status"], 403)
        self.assertEqual(receipts["limit_up"]["status"], "UNAVAILABLE")
        for kind in ("failed_board", "limit_down", "previous_limit"):
            self.assertEqual(receipts[kind]["status"], "NOT_REQUESTED_SOURCE_FORBIDDEN")
            self.assertIsNone(receipts[kind]["requested_at"])
            self.assertNotIn("raw_path", receipts[kind])
        normalized = read_json(folder / "normalized.json")
        self.assertEqual(normalized["pools"], {})
        self.assertEqual(set(normalized["observations"]), {"000678", "600825"})
        self.assertEqual(normalized["observations"]["600825"]["status"], "UNKNOWN")
        self.assertIsNone(normalized["observations"]["600825"]["promotion_label"])
        self.assertEqual(result["prior_roster_retained_rows"], 2)
        self.assertEqual(len([failure for failure in result["failures"]
                              if failure["kind"] in capture.sources.ENDPOINTS]), 4)
        self.assertIsNone(result["provider_closing_rows"])
        self.assertEqual(result["new_settled_forward_samples"], 0)

    def test_archive_rejects_changed_raw_missing_request_and_removed_prior_observation(self):
        for mutation in ("raw", "request", "prior_observation"):
            self.clock = CLOCK
            folder = self.folder / ("tamper_" + mutation)
            self.run_collection(output=folder)
            manifest = read_json(folder / "manifest.json")
            formation = read_json(folder / "formation_receipt.json")
            if mutation == "raw":
                with (folder / "raw" / "000678_quote.json").open("ab") as handle:
                    handle.write(b"\n")
                expected_error = "bound artifact changed"
            else:
                if mutation == "request":
                    manifest["requests"].pop()
                    expected_error = "original request scope or failures missing"
                else:
                    observed_path = folder / "prior_roster_observations.csv"
                    with observed_path.open(encoding="utf-8-sig", newline="") as handle:
                        reader = csv.DictReader(handle)
                        headers, rows = reader.fieldnames, list(reader)
                    with observed_path.open("w", encoding="utf-8-sig", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=headers)
                        writer.writeheader()
                        writer.writerows(rows[:1])
                    manifest["artifact_hashes"]["prior_roster_observations.csv"] = capture.sources.sha(observed_path)
                    expected_error = "full original roster was not retained"
                # Synthetic rebindings reach the semantic scope/retention gate;
                # they grant no real provenance, review, or PIT eligibility.
                (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                formation["manifest_sha256"] = capture.sources.sha(folder / "manifest.json")
                (folder / "formation_receipt.json").write_text(json.dumps(formation), encoding="utf-8")
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, expected_error):
                capture.verify_archive(folder)


if __name__ == "__main__":
    unittest.main()
