import json
from pathlib import Path
import tempfile
import unittest

from daily_research_contract_v292 import (
    PHASES, binary_metrics, classification_metrics, digest, result_row, rule,
    score, statistics, validate_extension,
)
from daily_research_template_v292 import draft_extension, empty_claim
from daily_research_runner_v292 import append, current_results, read_chain, run, write_new
from daily_research_report_v292 import render
from next_session_expectation_contract_v271 import register
from test_next_session_expectation_v271 import plan


def research_plan():
    p = plan()
    p["daily_research"] = draft_extension(p)
    item = empty_claim(call="弱修复", specification={"name": "TEST_PHASE", "partial_pairs": [["弱修复", "强修复"]]})
    item["basis"] = ["TEST_ONLY"]
    item["invalidation"] = ["TEST_ONLY"]
    p["daily_research"]["emotion"] = item
    return p


def outcome(registered=None):
    source = {"status": "verified", "source": "TEST_ONLY", "observed_at": "2026-09-22T15:00:00+08:00", "available_at": "2026-09-22T16:00:00+08:00"}
    return {"date": "2026-09-22", "assembled_at": "2026-09-22T16:01:00+08:00",
        "registration_hash": registered["sha256"] if registered else None,
        "market": {**source, "primary": "RISK_OFF"},
        "limit_pool": {**source, "scope": "MAINBOARD_NON_ST_CLOSING_LIMIT_UP", "complete": True, "rows": []}}


class DailyResearchTests(unittest.TestCase):
    def test_full_unknowns_are_explicit_and_denominator_cannot_shrink(self):
        p = research_plan()
        self.assertEqual(validate_extension(p, "2026-09-21T18:01:00+08:00")["status"], "EXTENSION_VALIDATED")
        p["daily_research"]["candidates"] = []
        with self.assertRaises(ValueError):
            validate_extension(p, "2026-09-21T18:01:00+08:00")

    def test_extension_cannot_change_original_prediction(self):
        p = research_plan()
        p["daily_research"]["candidates"][0]["events"]["promotion"]["call"] = False
        with self.assertRaises(ValueError):
            validate_extension(p, "2026-09-21T18:01:00+08:00")

    def test_break_cannot_guess_when_promotion_is_unknown(self):
        p = research_plan()
        p["candidates"][0]["promotion_call"] = "UNCERTAIN"
        p["daily_research"] = draft_extension(p)
        p["daily_research"]["candidates"][0]["events"]["break"]["call"] = True
        with self.assertRaises(ValueError):
            validate_extension(p, "2026-09-21T18:01:00+08:00")

    def test_next_auction_in_current_facts_is_forbidden(self):
        p = research_plan()
        p["daily_research"]["today"]["emotion"] = {"status": "verified", "value": "涨", "source": "TEST_ONLY",
            "observed_at": "2026-09-22T09:25:00+08:00", "available_at": "2026-09-22T09:25:00+08:00"}
        with self.assertRaises(ValueError):
            validate_extension(p, "2026-09-21T18:01:00+08:00")

    def test_uncalibrated_probability_rejected(self):
        p = research_plan()
        p["daily_research"]["candidates"][0]["events"]["promotion"]["probability"] = .8
        with self.assertRaises(ValueError):
            validate_extension(p, "2026-09-21T18:01:00+08:00")

    def test_fake_calibration_receipt_rejected(self):
        p = research_plan()
        item = p["daily_research"]["candidates"][0]["events"]["promotion"]
        item["probability"] = .7
        item["calibration"] = {"method": rule({"name": "TEST_ONLY"}), "target": "promotion", "scope": "STOCK:600001",
            "status": "OUT_OF_SAMPLE_CALIBRATED", "source": "DOES_NOT_EXIST.json", "source_sha256": "a"*64}
        with self.assertRaises(ValueError):
            validate_extension(p, "2026-09-21T18:01:00+08:00")

    def test_verified_calibration_is_target_specific_and_pre_freeze(self):
        p = research_plan()
        item = p["daily_research"]["candidates"][0]["events"]["promotion"]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"calibration.json"
            receipt = {"method": rule({"name": "PRIOR_OOS_ONLY"}), "target": "promotion", "scope": "STOCK:600001",
                "status": "OUT_OF_SAMPLE_CALIBRATED", "available_at": "2026-09-21T17:00:00+08:00", "max_label_date": "2026-09-18",
                "evaluation_n": 60, "brier": .2, "bins": [{"n": 60, "mean_probability": .6, "actual_rate": .6}]}
            write_new(path, receipt)
            from daily_research_runner_v292 import sha
            item["probability"] = .6
            item["probability_interval"] = [.5, .7]
            item["calibration"] = {**receipt, "source": str(path), "source_sha256": sha(path)}
            validate_extension(p, "2026-09-21T18:01:00+08:00")
            receipt["max_label_date"] = "2026-09-22"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            item["calibration"] = {**receipt, "source": str(path), "source_sha256": sha(path)}
            with self.assertRaises(ValueError):
                validate_extension(p, "2026-09-21T18:01:00+08:00")

    def test_missing_pool_is_not_all_failed_and_market_is_independent(self):
        with tempfile.TemporaryDirectory() as d:
            _, r = register(research_plan(), d, "2026-09-21T18:01:00+08:00")
            value = outcome(r); value.pop("limit_pool")
            s = score(r, value, "2026-09-22T18:00:00+08:00")
            self.assertIsNone(next(x for x in s["rows"] if x["layer"] == "promotion")["truth"])
            self.assertFalse(next(x for x in s["rows"] if x["layer"] == "market_primary")["correct"])

    def test_rule_mismatch_prevents_hindsight_labeling(self):
        with tempfile.TemporaryDirectory() as d:
            _, r = register(research_plan(), d, "2026-09-21T18:01:00+08:00")
            value = outcome(r)
            value["labels"] = [{**value["market"], "layer": "emotion", "key": "MARKET", "value": "强修复", "rule_sha256": "b"*64}]
            with self.assertRaises(ValueError):
                score(r, value, "2026-09-22T18:00:00+08:00")

    def test_partial_emotion_is_not_full_accuracy(self):
        with tempfile.TemporaryDirectory() as d:
            _, r = register(research_plan(), d, "2026-09-21T18:01:00+08:00")
            value = outcome(r)
            value["labels"] = [{**value["market"], "layer": "emotion", "key": "MARKET", "value": "强修复",
                "rule_sha256": r["plan"]["daily_research"]["emotion"]["label_rule"]["sha256"]}]
            s = score(r, value, "2026-09-22T18:00:00+08:00")
            emotion = [x for x in s["rows"] if x["layer"] == "emotion"]
            metrics = classification_metrics(emotion)
            self.assertEqual(metrics["accuracy"], 0)
            self.assertEqual(metrics["partial"], 1)

    def test_loss_metric_polarity_and_boundaries(self):
        p = research_plan()
        p["daily_research"]["metrics"]["failed_rate"]["band"] = {"metric": "failed_rate", "stage": "CLOSE", "scope": "MAINBOARD",
            "unit": "ratio", "low": .2, "high": .3, "direction": "LOWER_BETTER", "basis": "TEST_ONLY"}
        with tempfile.TemporaryDirectory() as d:
            _, r = register(p, d, "2026-09-21T18:01:00+08:00")
            value = outcome(r)
            obs = {**value["market"], "metric": "failed_rate", "scope": "MAINBOARD", "unit": "ratio", "value": .1}
            value["observations"] = [obs]
            s = score(r, value, "2026-09-22T18:00:00+08:00")
            self.assertEqual(s["bands"][0]["result"], "EXCEEDS")
            obs["value"] = .3
            self.assertEqual(score(r, value, "2026-09-22T18:00:00+08:00")["bands"][0]["result"], "MEETS")

    def test_legacy_layers_cannot_be_retroactively_predicted(self):
        with tempfile.TemporaryDirectory() as d:
            _, r = register(plan(), d, "2026-09-21T18:01:00+08:00")
            s = score(r, outcome(r), "2026-09-22T18:00:00+08:00")
            unknowns = [x for x in s["rows"] if x["layer"] in ["theme", "leader", "reseal"]]
            self.assertTrue(all(x["call"] is None and x["correct"] is None for x in unknowns))

    def test_wrong_target_and_preclose_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            _, r = register(research_plan(), d, "2026-09-21T18:01:00+08:00")
            value = outcome(r); value["date"] = "2026-09-23"
            with self.assertRaises(ValueError): score(r, value, "2026-09-23T18:00:00+08:00")
            with self.assertRaises(ValueError): score(r, outcome(r), "2026-09-22T14:00:00+08:00")

    def test_all_fail_accuracy_does_not_hide_zero_recall(self):
        rows = [result_row("promotion", str(i), False, i == 0) for i in range(10)]
        metrics = binary_metrics(rows)
        self.assertEqual(metrics["accuracy"], .9)
        self.assertEqual(metrics["all_known_positive_recall"], 0)
        self.assertEqual(metrics["f1"], 0)
        self.assertIsNone(metrics["precision"])

    def test_uncalled_positives_stay_in_total_recall(self):
        rows = [result_row("promotion", "A", True, True), result_row("promotion", "B", None, True)]
        metrics = binary_metrics(rows)
        self.assertEqual(metrics["recall"], 1)
        self.assertEqual(metrics["all_known_positive_recall"], .5)
        self.assertEqual(metrics["uncalled_known_positives"], 1)

    def test_brier_bin_includes_one_and_excludes_unknown(self):
        rows = [result_row("promotion", "A", True, True, probability=1), result_row("promotion", "B", True, False, probability=.7),
            result_row("promotion", "C", None, probability=None)]
        metrics = binary_metrics(rows)
        self.assertAlmostEqual(metrics["brier"], .245)
        self.assertEqual(metrics["calibration_bins"][9]["n"], 1)
        self.assertEqual(metrics["probability_scored_n"], 2)

    def test_missing_settlement_stays_in_window_and_height_tier(self):
        with tempfile.TemporaryDirectory() as d:
            _, r = register(research_plan(), d, "2026-09-21T18:01:00+08:00")
            report = statistics([], [r], "2026-09-23T18:00:00+08:00", r["plan"]["calendar"])
            view = report["windows"]["20"]
            self.assertEqual(view["registered_forward_dates"], 1)
            self.assertEqual(view["missing_registration_sessions"], 1)
            self.assertEqual(view["tiers"]["2进3"]["unknown_outcomes"], 1)
            self.assertEqual(view["tiers"]["最高板"]["total_frozen_rows"], 1)

    def test_ledger_idempotent_revisions_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _, r = register(research_plan(), d/"reg", "2026-09-21T18:01:00+08:00")
            source = d/"source.json"; value = outcome(r); write_new(source, value)
            s = score(r, value, "2026-09-22T18:00:00+08:00")
            ledger = d/"ledger.jsonl"
            self.assertEqual(append(ledger, s, source), "SETTLEMENT_APPENDED")
            self.assertEqual(append(ledger, s, source), "UNCHANGED")
            value["limit_pool"]["rows"] = [{"code": "600001", "boards": 3}]
            source2 = d/"new.json"; write_new(source2, value)
            newer = score(r, value, "2026-09-22T18:30:00+08:00")
            self.assertEqual(append(ledger, newer, source2), "REVISION_APPENDED")
            events = read_chain(ledger)
            self.assertEqual(len(events), 2)
            self.assertTrue(events[-1]["truth_changes"])
            self.assertEqual(current_results(events, "2026-09-22T18:10:00+08:00")[0]["outcome_hash"], s["outcome_hash"])
            ledger.write_text(ledger.read_text(encoding="utf-8").replace('"revision":1', '"revision":8'), encoding="utf-8")
            with self.assertRaises(ValueError): read_chain(ledger)

    def test_tick_waits_then_scores_then_does_not_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _, r = register(research_plan(), d/"reg", "2026-09-21T18:01:00+08:00")
            cal = d/"cal.json"; write_new(cal, r["plan"]["calendar"])
            inbox = d/"inbox"; ledger = d/"ledger.jsonl"
            first = run(d/"reg", inbox, ledger, cal, d/"run1", "2026-09-22T14:00:00+08:00")
            self.assertEqual(first["jobs"][0]["status"], "WAIT_TARGET_CLOSE")
            second = run(d/"reg", inbox, ledger, cal, d/"run2", "2026-09-22T16:00:00+08:00")
            self.assertEqual(second["jobs"][0]["status"], "WAIT_VERIFIED_TARGET_OUTCOMES")
            write_new(inbox/"2026-09-22"/"first.json", outcome(r))
            third = run(d/"reg", inbox, ledger, cal, d/"run3", "2026-09-22T18:00:00+08:00")
            self.assertEqual(third["jobs"][0]["status"], "SETTLEMENT_APPENDED")
            fourth = run(d/"reg", inbox, ledger, cal, d/"run4", "2026-09-22T18:30:00+08:00")
            self.assertEqual(fourth["jobs"][0]["status"], "UNCHANGED")
            self.assertEqual(len(read_chain(ledger)), 1)
            rendered = next((d/"run4").glob("*.md")).read_text(encoding="utf-8")
            self.assertEqual(rendered.count("【"), 11)
            self.assertIn("归因证据不足", rendered)

    def test_invalid_registration_withholds_aggregate(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _, r = register(plan(), d/"reg", "2026-09-21T18:01:00+08:00")
            cal = d/"cal.json"; write_new(cal, r["plan"]["calendar"])
            write_new(d/"reg"/"invalid.json", {"sha256": "bad", "plan": {}})
            report = run(d/"reg", d/"inbox", d/"ledger", cal, d/"run", "2026-09-22T18:00:00+08:00")
            self.assertEqual(report["statistics"]["status"], "WITHHELD_INVALID_INPUT")
            self.assertEqual(report["status"], "ATTENTION_REQUIRED")

    def test_calendar_expiration_fails_before_processing(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); p = plan(); calendar = p["calendar"]
            calendar["scope"] = ["2026-09-21", "2026-09-23"]
            cal = d/"cal.json"; write_new(cal, calendar)
            with self.assertRaises(ValueError):
                run(d/"reg", d/"inbox", d/"ledger", cal, d/"out", "2026-10-02T18:00:00+08:00")

    def test_source_edit_is_detected_even_when_ledger_is_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _, r = register(plan(), d/"reg", "2026-09-21T18:01:00+08:00")
            value = outcome(r); source = d/"outcome.json"; write_new(source, value)
            ledger = d/"ledger.jsonl"
            append(ledger, score(r, value, "2026-09-22T18:00:00+08:00"), source)
            value["market"]["primary"] = "RISK_ON"
            source.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ValueError): read_chain(ledger)

    def test_example_cannot_be_entered_into_real_scoring(self):
        with tempfile.TemporaryDirectory() as d:
            _, r = register(plan(), d, "2026-09-21T18:01:00+08:00")
            value = outcome(r); value["example_only"] = True
            with self.assertRaises(ValueError): score(r, value, "2026-09-22T18:00:00+08:00")

    def test_descriptive_interval_does_not_certify_stability(self):
        metrics = binary_metrics([result_row("promotion", str(i), True, i < 6) for i in range(10)])
        low, high = metrics["accuracy_wilson_95"]
        self.assertLess(low, .6)
        self.assertGreater(high, .6)


if __name__ == "__main__":
    unittest.main()
