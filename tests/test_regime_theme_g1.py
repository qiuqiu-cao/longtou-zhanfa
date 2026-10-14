"""Synthetic causal, denominator, membership and paired-validation checks."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cycle_regime_diagnostics as cycle
import theme_structure_g1 as theme
import theme_g1_validation as shadow


class CycleTests(unittest.TestCase):
    def test_metrics_independent_confusion_and_complete_recall(self):
        calls = [True, False, True, False, None, True]
        truths = [True, True, False, False, True, None]
        rows = [{"signal_date": "2026-10-08", "prediction": p, "truth": t} for p, t in zip(calls, truths)]
        m = cycle.metrics(rows)
        self.assertEqual(m["confusion"], {"TP": 1, "TN": 1, "FP": 1, "FN": 1})
        self.assertEqual(m["scoreable_count"], 4)
        self.assertEqual(m["original_registered_count"], 6)
        self.assertAlmostEqual(m["Accuracy"], .5)
        self.assertAlmostEqual(m["complete_positive_Recall"], 1/3)
        self.assertAlmostEqual(m["F1"], .4)
        self.assertIsNone(m["calibrated_Brier"])

    def test_fixed_native_mapping(self):
        self.assertEqual(cycle.HISTORY_MAP.get("启动", "UNKNOWN"), "UNKNOWN")
        self.assertEqual(cycle.HISTORY_MAP.get("大分歧"), "分歧")
        self.assertEqual(cycle.HISTORY_MAP.get("加速"), cycle.HISTORY_MAP.get("一致"))
        self.assertNotIn("冰点", cycle.HISTORY_MAP.values())

    def test_unknown_does_not_infer_from_target_scenario(self):
        self.assertEqual(cycle.validate_cycle(None, "2026-10-08", "2026-10-08T20:00:00+08:00"), "UNKNOWN")

    def known_context(self, directory):
        path = Path(directory)/"evidence.json"
        path.write_text(json.dumps({"generated_at": "2026-10-08T15:05:00+08:00", "feedback": -.1, "promotion": .3, "failure": .2}), encoding="utf-8")
        c = {"version": "signal-cycle-context-1", "signal_date": "2026-10-08", "state": "修复",
             "formed_at": "2026-10-08T16:00:00+08:00", "evidence": []}
        for key, value in (("feedback", -.1), ("promotion", .3), ("failure", .2)):
            c["evidence"].append({"id": key, "value": value, "signal_date": "2026-10-08",
                "observed_at": "2026-10-08T15:00:00+08:00", "available_at": "2026-10-08T15:05:00+08:00",
                "source_path": str(path), "source_sha256": cycle.sha(path), "source_pointer": [key]})
        return c

    def test_cycle_source_value_and_time_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            c = self.known_context(directory)
            self.assertEqual(cycle.validate_cycle(c, "2026-10-08", "2026-10-08T17:00:00+08:00"), "修复")
            c["evidence"][0]["value"] = 99
            with self.assertRaisesRegex(ValueError, "value differs"):
                cycle.validate_cycle(c, "2026-10-08", "2026-10-08T17:00:00+08:00")

    def test_cycle_missing_future_and_repeated_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            c = self.known_context(directory)
            c["evidence"] = c["evidence"][:2]
            with self.assertRaises(ValueError): cycle.validate_cycle(c, "2026-10-08", "2026-10-08T17:00:00+08:00")
            c = self.known_context(directory); c["formed_at"] = "2026-10-09T16:00:00+08:00"
            with self.assertRaises(ValueError): cycle.validate_cycle(c, "2026-10-08", "2026-10-08T17:00:00+08:00")
            c = self.known_context(directory); c["evidence"][1]["source_pointer"] = ["feedback"]; c["evidence"][1]["value"] = -.1
            with self.assertRaisesRegex(ValueError, "repeat"):
                cycle.validate_cycle(c, "2026-10-08", "2026-10-08T17:00:00+08:00")

    def test_official_windows_preserve_unknown_records(self):
        sessions = [f"2026-09-{i:02}" for i in range(1, 31)]
        rows = [{"signal_date": d, "boards": 2, "regime": "UNKNOWN", "prediction": False, "truth": None} for d in sessions]
        r = cycle.summarise(rows, sessions, sessions[-1])
        self.assertEqual(r["20"]["overall"]["original_registered_count"], 20)
        self.assertEqual(r["60"]["overall"]["original_registered_count"], 30)
        self.assertEqual(r["all"]["by_state"]["UNKNOWN"]["scoreable_count"], 0)

    def test_no_candidate_day_still_counts_state_support(self):
        r = cycle.summarise([], ["2026-10-08"], "2026-10-08", {"2026-10-08": "修复"})
        self.assertEqual(r["all"]["by_state"]["修复"]["signal_days"], 1)
        self.assertEqual(r["all"]["by_state"]["修复"]["original_registered_count"], 0)


class ThemeTests(unittest.TestCase):
    def fixture(self, n=5, failed=2):
        codes = [f"{i:06}" for i in range(1, n+1)]
        failures = [f"{100+i:06}" for i in range(failed)]
        facts = [{"code": c, "name": c, "boards": 2 if c in codes else None,
                  "pool_kind": "limit_up" if c in codes else "failed_board",
                  "industry_provider": "电池", "category_status": "PROVIDER_CURRENT_CATEGORY_ONLY"} for c in codes+failures]
        report = {"signal_date": "2026-10-08", "knowledge_cutoff": "2026-10-08T18:00:00+08:00", "all_candidates": facts,
                  "groups": [{"evidence_lane": "PROVIDER_CURRENT_CATEGORY", "group_name": "固态电池", "group_id": "A", "member_codes": codes+failures}],
                  "pool_counts": {"closing": n, "failed": failed, "relay": n, "first_board": 0}}
        base = {"signal_date": "2026-10-08", "target_date": "2026-10-09", "knowledge_cutoff": "2026-10-08T17:00:00+08:00",
                "relay": {"candidates": [{"code": c, "name": c, "boards": 2, "raw_score": .6,
                    "predicted_positive": False, "actual_one_word_daily": 0, "rank": i+1} for i, c in enumerate(codes)]}}
        policy = {"aliases": {}, "excluded_names": [], "excluded_patterns": [],
                  "families": {"battery": {"canonical": "电池", "source_lane": "PROVIDER_SEMANTIC_FAMILY", "group_id": "battery", "children": ["固态电池"]}}}
        return report, base, policy

    def test_strict_40pct_boundary_and_dedup(self):
        args = self.fixture(); args[0]["groups"][0]["member_codes"].append("000001")
        r = theme.analyse(*args)
        g = next(g for g in r["full_theme_risk_table"] if g["group_name"] == "固态电池")
        self.assertEqual(g["closing_count"], 5); self.assertEqual(g["failed_over_sealed_ratio"], .4)
        self.assertEqual(g["risk"], "未触发")
        args = self.fixture(failed=3)
        self.assertTrue(all(g["risk"] == "分歧风险" for g in theme.analyse(*args)["full_theme_risk_table"]))

    def test_zero_denominator_and_incomplete_membership(self):
        args = self.fixture(n=0); r = theme.analyse(*args)
        self.assertTrue(all(g["failed_over_sealed_ratio"] is None for g in r["full_theme_risk_table"]))
        args = self.fixture(); args[0]["all_candidates"][-1]["category_status"] = "UNKNOWN"
        self.assertTrue(all(g["risk"] == "UNKNOWN" for g in theme.analyse(*args)["full_theme_risk_table"]))

    def test_two_isolated_three_supported(self):
        for n, isolated in ((2, True), (3, False)):
            r = theme.analyse(*self.fixture(n=n))
            self.assertTrue(all(v["G1_excluded"] is isolated for v in r["all_candidates"]))
            self.assertEqual(len(r["G1_shadow_top5"]), 0 if isolated else n)
            self.assertEqual(len(r["all_candidates"]), n)

    def test_parent_support_overrides_small_child(self):
        args = self.fixture(n=3)
        args[0]["groups"][0]["member_codes"] = ["000001", "000100", "000101"]
        args[0]["groups"].append({"evidence_lane": "PROVIDER_SEMANTIC_FAMILY", "group_id": "battery", "group_name": "电池", "member_codes": ["000001", "000002", "000003"]})
        r = theme.analyse(*args)
        self.assertFalse(r["all_candidates"][0]["G1_excluded"])

    def test_unknown_classification_never_filters(self):
        args = self.fixture(n=2); args[0]["all_candidates"][0]["category_status"] = "UNKNOWN"
        r = theme.analyse(*args)
        self.assertTrue(all(v["structure_status"] == "UNKNOWN" for v in r["all_candidates"]))
        self.assertEqual(len(r["G1_shadow_top5"]), 2)

    def test_score_ties_all_candidates_and_negative_first(self):
        args = self.fixture(n=7); args[1]["relay"]["candidates"].reverse()
        before = copy.deepcopy(args); r = theme.analyse(*args)
        self.assertEqual(args, before)
        self.assertEqual([v["code"] for v in r["original_model_top5"]], [f"{i:06}" for i in range(1, 6)])
        self.assertEqual(len(r["all_candidates"]), 7)
        self.assertIsNone(r["original_selectors"]["non_one_word_positive_first"])
        self.assertEqual(r["original_selectors"]["non_one_word_rank_first"], "000001")

    def test_future_labels_cannot_change_signal_overlay(self):
        args = self.fixture(); first = theme.analyse(*args)
        args[0]["truth"] = [False]; args[1]["outcomes"] = [True]
        for row in args[0]["all_candidates"]:
            row["promoted_next_day"] = True; row["trade_close"] = 1e9
        self.assertEqual(first, theme.analyse(*args))

    def test_industry_and_generic_do_not_fake_theme_support(self):
        args = self.fixture(n=2)
        args[0]["groups"] = [{"evidence_lane": "PROVIDER_INDUSTRY_PROXY", "group_name": "电池", "group_id": "ind", "member_codes": ["000001", "000002"]}]
        r = theme.analyse(*args)
        self.assertTrue(all(v["structure_status"] == "UNKNOWN" for v in r["all_candidates"]))


class ValidationTests(unittest.TestCase):
    def days(self):
        rows = [{"code": "000001", "predicted_positive": False}, {"code": "000002", "predicted_positive": True}]
        selector = lambda code: {"non_one_word_rank_first": code, "non_one_word_positive_first": code}
        reg = {"all_candidates": rows, "BASE": {"selectors": selector("000001"), "list": ["000001", "000002"]},
               "G1": {"selectors": selector("000002"), "list": ["000001", "000002"]}}
        return [{"status": "PAIRED_CANDIDATE_DAY", "registration": copy.deepcopy(reg),
                 "truths": {"000001": i%2 == 0, "000002": i%2 == 0 or i%5 == 1}} for i in range(120)]

    def test_independent_paired_difference_blocks_and_gate(self):
        days = self.days(); r = shadow.evaluate(days)
        b = sum(d["truths"]["000001"] for d in days)/120
        g = sum(d["truths"]["000002"] for d in days)/120
        self.assertAlmostEqual(r["primary_delta"], g-b)
        self.assertEqual(r["paired_days"], 120)
        self.assertEqual(r["effective_coverage"], 1)
        self.assertEqual(r, shadow.evaluate(days))
        self.assertTrue(r["passed"])

    def test_unknown_labels_and_missing_slots_stay_in_coverage(self):
        days = self.days()
        for i in range(20): days[i] = {"status": "NO_REGISTRATION"}
        days[25]["truths"]["000001"] = None
        r = shadow.evaluate(days)
        self.assertEqual(r["paired_days"], 99)
        self.assertAlmostEqual(r["effective_coverage"], 99/120)
        self.assertFalse(r["passed"])

    def test_no_candidates_remain_distinct(self):
        days = self.days(); days[0]["registration"]["all_candidates"] = []
        for a in ("BASE", "G1"):
            days[0]["registration"][a]["selectors"] = {"non_one_word_rank_first": None, "non_one_word_positive_first": None}
        days[0]["truths"] = {}; days[0]["status"] = "NO_CANDIDATES"
        r = shadow.evaluate(days)
        self.assertEqual(r["day_status_counts"]["NO_CANDIDATES"], 1)
        self.assertEqual(r["coverage_counts"]["eligible"], 119)

    def test_partial_window_is_not_acceptance(self):
        with self.assertRaises(ValueError): shadow.evaluate(self.days()[:60])

    def test_abstention_not_a_success_or_a_failed_trade(self):
        days = self.days(); days[0]["registration"]["all_candidates"][0]["predicted_positive"] = None
        days[0]["status"] = "ABSTENTION"
        r = shadow.evaluate(days)
        self.assertEqual(r["paired_days"], 119)
        self.assertEqual(r["day_status_counts"]["ABSTENTION"], 1)

    def test_hash_chain_detects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"ledger.jsonl"
            shadow.append(path, {"event": "TEST", "immutable_sources": []})
            self.assertEqual(len(shadow.read_chain(path)), 1)
            path.write_text(path.read_text(encoding="utf-8").replace('"TEST"', '"EDIT"'), encoding="utf-8")
            with self.assertRaises(ValueError): shadow.read_chain(path)

    def test_register_settle_idempotence_and_no_retroactive_start(self):
        # Entirely synthetic registry/source files. No production ledger is written.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tr, base, policy = ThemeTests().fixture(n=3, failed=0)
            base["generated_at"] = "2026-10-08T17:00:00+08:00"
            (root/"policy.json").write_text(json.dumps(policy), encoding="utf-8")
            p = {"formed_at": "2026-10-08T16:30:00+08:00", "model_binding": {"model_sha256": "synthetic"},
                 "theme_policy": {"path": str(root/"policy.json")}}
            (root/"protocol.json").write_text(json.dumps(p), encoding="utf-8")
            base["model_bindings"] = {"relay": {}}
            (root/"base.json").write_text(json.dumps(base), encoding="utf-8")
            tr["original_base_prediction_binding"] = {"path": str(root/"base.json")}
            (root/"theme.json").write_text(json.dumps(tr), encoding="utf-8")
            view = theme.analyse(tr, base, policy)
            view.update(generated_at="2026-10-08T19:00:00+08:00", prospective_eligible=True, model_binding=p["model_binding"],
                protocol_binding={"sha256": theme.sha(root/"protocol.json")}, source_hashes={str(root/n): theme.sha(root/n) for n in ("theme.json", "base.json")})
            (root/"overlay.json").write_text(json.dumps(view), encoding="utf-8")
            registered = {"forecast_status": "FORWARD_REGISTERED", "sha256": "synthetic-reg", "knowledge_cutoff": "2026-10-08T20:00:00+08:00",
                "recorded_at": "2026-10-08T20:01:00+08:00", "plan": {"signal_date": "2026-10-08", "target_date": "2026-10-09", "candidates": base["relay"]["candidates"]}}
            (root/"registered.json").write_text(json.dumps(registered), encoding="utf-8")
            (root/"calendar.json").write_text(json.dumps({"sessions": ["2026-10-08", "2026-10-09"]}), encoding="utf-8")
            (root/"implementation.json").write_text('{}', encoding="utf-8")
            patches = [patch.object(shadow, "load_protocol", return_value=p), patch.object(shadow, "verify_implementation", return_value=shadow.binding(root/"implementation.json")),
                patch.object(shadow, "load_registered", return_value=registered), patch("current_model_diagnostics._calendar", return_value={"sessions": ["2026-10-08", "2026-10-09"]})]
            from contextlib import ExitStack
            with ExitStack() as stack:
                for mocked in patches: stack.enter_context(mocked)
                with patch.object(shadow, "now", return_value="2026-10-08T21:00:00+08:00"):
                    e = shadow.register(root/"protocol.json", root/"calendar.json", root/"registered.json", root/"overlay.json")
                    self.assertEqual(e["window_index"], 0)
                    with self.assertRaisesRegex(ValueError, "already registered"):
                        shadow.register(root/"protocol.json", root/"calendar.json", root/"registered.json", root/"overlay.json")
                with patch.object(shadow, "now", return_value="2026-10-09T16:00:00+08:00"):
                    with self.assertRaisesRegex(ValueError, "no retrospective"):
                        shadow.register(root/"protocol.json", root/"calendar.json", root/"registered.json", root/"overlay.json")
                    (root/"outcome.json").write_text('{}', encoding="utf-8")
                    labels = [{"layer": "promotion", "key": r["code"], "truth": i == 0, "outcome_known": True} for i, r in enumerate(base["relay"]["candidates"])]
                    result = {"registration_hash": "synthetic-reg", "recorded_at": "2026-10-09T15:30:00+08:00", "rows": labels, "outcome_hash": "synthetic-outcome"}
                    source_event = {"result": result, "source": str(root/"outcome.json"), "sha256": "synthetic-event"}
                    with patch("hypothesis_bound_daily_runner_v298.read_chain", return_value=[source_event]), patch("hypothesis_bound_daily_runner_v298.score", return_value=result):
                        first = shadow.settle(root/"protocol.json", root/"production.jsonl")
                        second = shadow.settle(root/"protocol.json", root/"production.jsonl")
                        self.assertEqual(first["new_settlements"], 1)
                        self.assertEqual(second["new_settlements"], 0)
                        self.assertEqual(len(shadow.read_chain(root/"shadow/settlements.jsonl")[0]["truths"]), 3)


if __name__ == "__main__":
    unittest.main()
