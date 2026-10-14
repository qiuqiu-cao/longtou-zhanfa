from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from daily_research_runner_v292 import ROOT, write_new
from next_session_expectation_contract_v271 import load_registered
from public_outcome_sources_v293 import cent, parse_pool, parse_daily, reconstruct
from public_outcome_bridge_v293 import candidate_truth, legacy_signal, verify, source_score
from source_bound_daily_runner_v293 import run

ARCHIVE=ROOT/"data/v293/public_outcome_sources_20261002_v1"
REPLAY=ROOT/"results/v293/source_pipeline_replay_20261002_v1"


def fixture(directory):
    path=Path(directory)/"feedback.json"
    write_new(path,{"data":{"pool":[{"c":"600001","ylbc":2}]}})
    candidate={"code":"600001","boards":2}
    signal={"rows":[{"code":"600001","boards":2,"price_cents":1000}],
        "available_at":"2026-09-29T16:00:00+08:00","raw_path":"SIGNAL_TEST_FIXTURE"}
    target={"rows":[{"code":"600001","boards":3,"price_cents":1100}],
        "available_at":"2026-09-30T16:00:00+08:00","raw_path":"TARGET_TEST_FIXTURE"}
    feedback={"rows":[{"code":"600001","price_cents":1100,"upper_cents_explicit":1100}],
        "available_at":"2026-09-30T16:00:00+08:00","raw_path":str(path)}
    daily={"600001":{"days":{"2026-09-29":{"open":980,"close":1000,"high":1000,"low":980},
        "2026-09-30":{"open":1100,"close":1100,"high":1100,"low":1100}},
        "quote":{"status":"POSITIONAL_FIELDS_INFERRED","observed_at":"2026-09-30T15:10:00+08:00",
            "values":{"open":1100,"close":1100,"high":1100,"low":1100,"previous_close":1000,"upper_candidate":1100}},
        "available_at":"2026-09-30T16:00:00+08:00","raw_path":"DAY_TEST_FIXTURE"}}
    return candidate,signal,target,feedback,daily


def evaluate(values):
    return candidate_truth(*values,"2026-09-29","2026-09-30")


class PublicOutcomeTests(unittest.TestCase):
    def test_source_date_fallback_rejected(self):
        with self.assertRaises(ValueError):
            parse_pool({"data":{"qdate":20260930,"tc":0,"pool":[]}},"2026-09-29","limit_up")

    def test_missing_total_and_duplicate_denominator_rejected(self):
        row={"c":"600001","n":"TEST","p":10000,"lbc":2}
        with self.assertRaises(ValueError):
            parse_pool({"data":{"qdate":20260930,"pool":[row]}},"2026-09-30","limit_up")
        with self.assertRaises(ValueError):
            parse_pool({"data":{"qdate":20260930,"tc":2,"pool":[row,row]}},"2026-09-30","limit_up")

    def test_exact_cent_grid_has_no_one_cent_tolerance(self):
        self.assertEqual(cent("10.990"),1099)
        with self.assertRaises(ValueError): cent("10.991")

    def test_empty_response_requires_explicit_zero_total(self):
        value=parse_pool({"data":{"qdate":20260930,"tc":0,"pool":[]}},"2026-09-30","limit_up")
        self.assertEqual(value["rows"],[])
        with self.assertRaises(ValueError):
            parse_pool({"data":{"qdate":20260930,"tc":1,"pool":[]}},"2026-09-30","limit_up")

    def test_adjusted_days_and_duplicate_day_dates_rejected(self):
        with self.assertRaises(ValueError):
            parse_daily({"data":{"sh600001":{"qfqday":[["2026-09-30","10","10","10","10","1"]]}}},"600001","2026-09-29","2026-09-30")
        row=["2026-09-30","10","10","10","10","1"]
        with self.assertRaises(ValueError):
            parse_daily({"data":{"sh600001":{"day":[row,row]}}},"600001","2026-09-29","2026-09-30")

    def test_positive_requires_all_price_and_continuity_checks(self):
        with tempfile.TemporaryDirectory() as d:
            result=evaluate(fixture(d))
            self.assertEqual(result["status"],"verified")
            self.assertIs(result["promoted"],True)
            self.assertFalse(result["exchange_security_status_verified"])

    def test_one_cent_below_upper_is_negative_when_absent_from_pool(self):
        with tempfile.TemporaryDirectory() as d:
            values=list(fixture(d)); values[2]["rows"]=[]
            values[3]["rows"][0]["price_cents"]=1099
            values[4]["600001"]["days"]["2026-09-30"].update(close=1099,low=1099)
            values[4]["600001"]["quote"]["values"].update(close=1099,low=1099)
            result=evaluate(values)
            self.assertIs(result["promoted"],False)
            self.assertEqual(result["status"],"verified")

    def test_one_cent_below_upper_in_pool_is_a_conflict_not_positive(self):
        with tempfile.TemporaryDirectory() as d:
            values=list(fixture(d)); values[2]["rows"][0]["price_cents"]=1099
            values[3]["rows"][0]["price_cents"]=1099
            values[4]["600001"]["days"]["2026-09-30"].update(close=1099,low=1099)
            values[4]["600001"]["quote"]["values"].update(close=1099,low=1099)
            result=evaluate(values)
            self.assertEqual(result["reason"],"BELOW_LIMIT_STOCK_REPORTED_IN_CLOSING_POOL")
            self.assertIsNone(result["promoted"])

    def test_signal_close_below_daily_high_is_unresolved(self):
        with tempfile.TemporaryDirectory() as d:
            values=list(fixture(d));values[4]["600001"]["days"]["2026-09-29"]["high"]=1001
            result=evaluate(values)
            self.assertIsNone(result["promoted"])
            self.assertIn("NECESSARY_CONDITION_CONFLICT",result["reason"])

    def test_intraday_or_wrong_day_quote_is_unavailable(self):
        for timestamp in ["2026-09-30T14:59:59+08:00","2026-09-29T15:00:00+08:00"]:
            with tempfile.TemporaryDirectory() as d:
                values=list(fixture(d));values[4]["600001"]["quote"]["observed_at"]=timestamp
                self.assertIsNone(evaluate(values)["promoted"])

    def test_price_limit_disagreement_is_not_resolved_by_ratio_guess(self):
        with tempfile.TemporaryDirectory() as d:
            values=list(fixture(d));values[4]["600001"]["quote"]["values"]["upper_candidate"]=1101
            self.assertEqual(evaluate(values)["reason"],"TARGET_PRICE_REFERENCE_OR_LIMIT_SOURCE_CONFLICT")

    def test_wrong_consecutive_board_count_is_not_a_negative_label(self):
        with tempfile.TemporaryDirectory() as d:
            values=list(fixture(d));values[2]["rows"][0]["boards"]=4
            result=evaluate(values)
            self.assertIsNone(result["promoted"])
            self.assertIn("CONSECUTIVE_BOARD_MEMBERSHIP_CONFLICT",result["reason"])

    def test_missing_daily_reference_preserves_candidate_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            values=list(fixture(d));values[4]={}
            result=evaluate(values)
            self.assertEqual(result["code"],"600001")
            self.assertIsNone(result["promoted"])

    def test_exright_reference_mismatch_does_not_silently_use_yesterday_close(self):
        with tempfile.TemporaryDirectory() as d:
            values=list(fixture(d));values[4]["600001"]["quote"]["values"]["previous_close"]=999
            self.assertIsNone(evaluate(values)["promoted"])

    def test_real_raw_archive_reconstructs_and_keeps_date_conflicts(self):
        if not ARCHIVE.exists():self.skipTest("real receipt archive absent")
        manifest,pools,daily,failures=reconstruct(ARCHIVE)
        self.assertEqual(len(manifest["requested_codes"]),102)
        self.assertEqual(len(daily),95)
        self.assertNotIn(("2026-09-29","limit_up"),pools)
        self.assertEqual(len(pools[("2026-09-30","limit_up")]["rows"]),48)
        self.assertEqual(sum(f["reason"]=="pool returned date differs from the requested exact trading day" for f in failures),4)

    def test_real_replay_is_source_bound_and_never_forward(self):
        if not REPLAY.exists():self.skipTest("real replay archive absent")
        path=next((REPLAY/"replay_registry").glob("*.json"))
        registered=load_registered(path)
        self.assertEqual(registered["forecast_status"],"HISTORICAL_REPLAY")
        value=json.loads((REPLAY/"outcome_inbox/2026-09-30/source_bound.json").read_text(encoding="utf-8"))
        verify(registered,value)
        result=source_score(registered,value,"2026-10-02T23:59:00+08:00")
        promotion=[r for r in result["rows"] if r["layer"]=="promotion"]
        self.assertEqual(len(promotion),9)
        self.assertEqual(sum(r["correct"] is True for r in promotion),6)
        altered=deepcopy(value);altered["candidate_closing_truth"][0]["promoted"]=False
        with self.assertRaises(ValueError):verify(registered,altered)

    def test_real_original_snapshot_matches_all_nine_candidate_codes(self):
        snapshot=ROOT/"data/review_20260929_1557"
        if not snapshot.exists():self.skipTest("original snapshot absent")
        source,_=legacy_signal(snapshot,"2026-09-29")
        self.assertEqual(sum(r["boards"]>=2 for r in source["rows"]),9)
        with self.assertRaises(ValueError):legacy_signal(snapshot,"2026-09-30")


if __name__=="__main__":unittest.main()
