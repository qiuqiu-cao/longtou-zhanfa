"""Provider observations are not market consensus or executable trading facts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import attention_observations_v299 as a

AVAILABLE = "2026-10-02T23:40:00+08:00"


def response(rows):
    return {"status": 0, "code": 0, "message": "OK", "data": rows}


class AttentionObservationTests(unittest.TestCase):
    def test_actual_probe_field_shapes_are_parseable(self):
        base = a.ROOT / "data/v299/attention_source_probe_20261002_v1"
        report = json.loads((base / "report.json").read_text(encoding="utf-8"))
        index = {r["kind"]: r for r in report["requests"]}
        for kind, parser in [("profiles", a.parse_profiles), ("keywords", a.parse_keywords), ("related", a.parse_related)]:
            entry = index[kind]
            parsed = parser(json.loads(Path(entry["raw_path"]).read_bytes()), "600825", entry["available_at"])
            self.assertGreater(len(parsed["rows"]), 0)

    def test_nonzero_shares_with_provider_zero_count_never_become_zero_users(self):
        row = {"srcSecurityCode": "SH600825", "innerCode": "gsOPAQUE", "calcTime": "2026-10-02 22:00:00",
            "uidCount": 0, "newUidRate": "38.22%", "oldUidRate": "61.78%"}
        result = a.parse_profiles(response([row]), "600825", AVAILABLE)
        self.assertTrue(result["latest"]["zero_count_with_nonzero_shares"])
        self.assertIsNone(result["latest"]["verified_user_count"])
        self.assertFalse(result["historical_rows_are_live_features"])

    def test_security_echo_mismatch_rejected_for_three_explicit_sources(self):
        rows = [
            (a.parse_profiles, {"srcSecurityCode": "SH600826"}),
            (a.parse_keywords, {"srcSecurityCode": "SH600826"}),
            (a.parse_related, {"srcSecurityCode": "SH600826"})]
        for parser, row in rows:
            with self.assertRaisesRegex(ValueError, "security identity mismatch"):
                parser(response([row]), "600825", AVAILABLE)

    def test_percentage_sum_and_invalid_values_rejected(self):
        row = {"srcSecurityCode": "SH600825", "calcTime": "2026-10-02 22:00:00", "newUidRate": "80%", "oldUidRate": "50%"}
        with self.assertRaisesRegex(ValueError, "percentages inconsistent"):
            a.parse_profiles(response([row]), "600825", AVAILABLE)
        for value in ["nan%", "101%", "-1%", "0.5", True]:
            with self.assertRaises(ValueError): a.percent(value)

    def test_actual_full_day_grid_uses_china_date_and_retains_request_identity_gap(self):
        base = a.ROOT / "data/v299/attention_source_probe_20261002_v1"
        probe = json.loads((base / "report.json").read_text(encoding="utf-8"))
        entry = next(r for r in probe["requests"] if r["kind"] == "intraday")
        result = a.parse_intraday(json.loads(Path(entry["raw_path"]).read_bytes()), "600825", entry["available_at"], None)
        self.assertEqual(len(result["rows"]), 142)
        self.assertFalse(result["security_code_returned_in_each_row"])
        self.assertFalse(result["same_clock_latest_checked"])
        self.assertEqual(result["missing_grid_times"], [])

    def test_duplicate_future_unordered_and_cross_date_intraday_rejected(self):
        base = [{"calcTime": "2026-10-02 00:00:00", "rank": 20}, {"calcTime": "2026-10-02 00:10:00", "rank": 18}]
        examples = [base + [deepcopy(base[0])], [{"calcTime": "2026-10-03 00:00:00", "rank": 18}], list(reversed(base)),
            [base[0], {"calcTime": "2026-10-01 23:50:00", "rank": 18}]]
        for rows in examples:
            with self.assertRaises(ValueError): a.parse_intraday(response(rows), "600825", AVAILABLE, None)

    def test_same_clock_rank_disagreement_rejected_and_missing_slots_not_zero_filled(self):
        rows = [{"calcTime": "2026-10-02 00:20:00", "rank": 18}]
        latest = {"calculated_at": "2026-10-02T00:20:00+08:00", "rank": 17}
        with self.assertRaisesRegex(ValueError, "endpoints disagree"):
            a.parse_intraday(response(rows), "600825", AVAILABLE, latest)
        latest["rank"] = 18
        result = a.parse_intraday(response(rows), "600825", AVAILABLE, latest)
        self.assertEqual(len(result["missing_grid_times"]), 2)
        self.assertEqual(len(result["rows"]), 1)

    def test_keyword_hits_do_not_become_discussion_count_or_theme_confirmation(self):
        row = {"srcSecurityCode": "SH600825", "calcTime": "2026-10-02 23:00:00", "conceptId": "BK0672", "conceptName": "沪企改革", "hitCount": 48}
        result = a.parse_keywords(response([row]), "600825", AVAILABLE)
        self.assertEqual(result["rows"][0]["hit_count_provider"], 48)
        self.assertIsNone(result["discussion_count"])
        self.assertFalse(result["active_trading_theme_verified"])
        self.assertFalse(result["hit_count_denominator_and_unique_users_verified"])

    def test_related_rate_keeps_opaque_semantics_and_does_not_expand_candidates(self):
        row = {"srcSecurityCode": "SH600825", "calcTime": "2026-10-02 22:00:00", "followSrcSecurityCode": "SH601811", "rate": "7.72%"}
        result = a.parse_related(response([row]), "600825", AVAILABLE)
        self.assertFalse(result["rate_is_verified_stock_return"])
        self.assertFalse(result["related_rows_expand_relay_denominator"])
        self.assertFalse(result["same_theme_or_controller_verified"])

    def test_all_failed_requests_keep_candidate_and_unknown_consensus(self):
        scope = {"codes": ["600825"]}
        entries = [{"code": "600825", "kind": k, "method": "POST", "url": a.query("600825", k)[0],
            "query": a.query("600825", k)[1], "requested_at": AVAILABLE, "available_at": AVAILABLE,
            "status": "UNAVAILABLE", "error": "ISOLATED TEST TIMEOUT"} for k in a.ENDPOINTS]
        result = a.normalize(scope, entries, AVAILABLE)
        row = result["observations"][0]
        self.assertEqual(len(row["errors"]), 5)
        for field in ["forum_heat", "discussion_change", "market_agreement", "bull_bear_extremes"]:
            self.assertIsNone(row[field])
        self.assertFalse(result["trade_authorized"])
        self.assertFalse(result["probability_adjustment_authorized"])
        with self.assertRaisesRegex(ValueError, "five requests"):
            a.normalize(scope, entries[:-1], AVAILABLE)

    def test_raw_hash_change_and_request_scope_change_rejected_as_explicit_gaps(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "raw.json"; path.write_text(json.dumps(response([])), encoding="utf-8")
            entries = [{"code": "600825", "kind": k, "method": "POST", "url": a.query("600825", k)[0],
                "query": a.query("600825", k)[1], "requested_at": AVAILABLE, "available_at": AVAILABLE,
                "http_status": 200, "status": "RAW_JSON_RECEIVED", "raw_path": str(path), "raw_sha256": "0"*64} for k in a.ENDPOINTS]
            result = a.normalize({"codes": ["600825"]}, entries, AVAILABLE)
            self.assertTrue(all("raw response changed" in e for e in result["observations"][0]["errors"].values()))
            entries[0]["query"]["srcSecurityCode"] = "SH600826"
            result = a.normalize({"codes": ["600825"]}, entries, AVAILABLE)
            self.assertIn("request code or endpoint scope", result["observations"][0]["errors"]["latest"])


if __name__ == "__main__":
    unittest.main()
