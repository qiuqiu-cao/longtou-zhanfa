"""Synthetic signal and mature-label diagnostics; no fitting or real output."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import pandas as pd

import current_model_diagnostics as diagnostics


class PoisonMapping(dict):
    """Fail if the signal-only API even attempts to read a forbidden field."""
    forbidden = {"target", "truth", "promoted_next_day", "trade_close", "trade_high", "next_open_gap", "outcomes"}

    def get(self, key, default=None):
        if key in self.forbidden:
            raise AssertionError("future/label field accessed: " + key)
        return super().get(key, default)


class CurrentModelDiagnosticsSynthetic(unittest.TestCase):
    def fixture(self):
        mm = {"module": "market", "arm": "BASE", "fields": ["yesterday_premium", "promotion_mid"]}
        rm = {"module": "relay", "arm": "BASE", "fields": ["same_height_rivals", "signal_one_word", "turnover"]}
        roster = [{"code": "000002", "name": "synthetic B", "boards": 2, "actual_one_word_daily": 0,
                   "features": {"same_height_rivals": 2., "signal_one_word": 0., "turnover": None}},
                  {"code": "000001", "name": "synthetic A", "boards": 2, "actual_one_word_daily": 1,
                   "features": {"same_height_rivals": 2., "signal_one_word": 0., "turnover": .04}},
                  {"code": "000003", "name": "synthetic C", "boards": 2, "actual_one_word_daily": 0,
                   "features": {"same_height_rivals": 2., "signal_one_word": 0., "turnover": .02}}]
        pack = {"signal_date": "2026-09-30", "target_date": "2026-10-08",
                "market": {"features": {"yesterday_premium": -.01, "promotion_mid": .3}}, "relay": roster}
        market = {"prediction": "IMPROVE", "raw_scores": {"IMPROVE": .6, "CONTINUE": .3, "WORSEN": .1}}
        relay = {"candidates": [{"code": r["code"], "boards": 2, "actual_one_word_daily": r["actual_one_word_daily"],
                "raw_score": .6 if r["code"] in ("000001", "000002") else .5,
                "predicted_positive": r["code"] == "000001"} for r in roster]}
        return pack, market, relay, mm, rm

    def call(self, args):
        return diagnostics.signal_diagnostics(*args, asof="2026-10-07T12:00:00+00:00")

    def test_ties_negative_nonword_and_inputs_unchanged(self):
        args = self.fixture()
        before = copy.deepcopy(args)
        observed = self.call(args)
        self.assertEqual(args, before)
        self.assertEqual([r["code"] for r in observed["candidates"]], ["000001", "000002", "000003"])
        self.assertEqual(observed["all_rank_first"], "000001")
        self.assertEqual(observed["positive_model_primary"], "000001")
        self.assertEqual(observed["non_one_word_rank_first"], "000002")
        self.assertIsNone(observed["non_one_word_positive_first"])
        self.assertEqual(observed["non_one_word_primary_status"], "EMPTY")
        second = observed["candidates"][1]
        self.assertEqual(second["promotion_call"], "FAIL")
        self.assertEqual(second["model_promotion_score"], 60.)
        self.assertIn("turnover", second["missing_model_fields"])
        self.assertIn("LEGACY_PROXY_DIFFERS_FROM_ACTUAL_OHLC_DEFINITION", observed["candidates"][0]["fact_prompts"])

    def test_unknown_word_invalidates_full_group_nonword_primary(self):
        args = self.fixture()
        args[0]["relay"][2]["actual_one_word_daily"] = None
        args[2]["candidates"][2]["actual_one_word_daily"] = None
        observed = self.call(args)
        self.assertFalse(observed["non_one_word_scope_complete"])
        self.assertEqual(observed["non_one_word_primary_status"], "UNKNOWN_IDENTITY")
        self.assertIsNone(observed["non_one_word_rank_first"])
        self.assertIsNone(observed["non_one_word_positive_first"])
        self.assertEqual(observed["known_nonword_subset_rank_first"], "000002")

    def test_future_labels_and_prices_cannot_change_signal_diagnostics(self):
        args = self.fixture()
        original = self.call(args)
        pack = args[0]
        pack["target"], pack["truth"], pack["trade_high"] = "arbitrary", True, 1e10
        for row in pack["relay"]:
            row["truth"], row["trade_close"], row["promoted_next_day"] = False, -1e10, 0
            row["features"] = PoisonMapping(row["features"] | {"target": "forbidden", "next_open_gap": 99.})
        pack["market"]["features"] = PoisonMapping(pack["market"]["features"] | {"target": "forbidden"})
        pack = PoisonMapping(pack)
        args = (pack, PoisonMapping(args[1]), PoisonMapping(args[2]), args[3], args[4])
        self.assertEqual(original, self.call(args))

    def test_incomplete_base_roster_and_future_feature_manifest_rejected(self):
        args = self.fixture()
        args[2]["candidates"].pop()
        with self.assertRaisesRegex(ValueError, "omits original"):
            self.call(args)
        args = self.fixture()
        args[4]["fields"].append("trade_close")
        with self.assertRaisesRegex(ValueError, "allowlist"):
            self.call(args)

    def test_historical_truth_requires_exact_mature_official_target(self):
        row = {"target_date": "2026-10-08", "label_valid": True, "target": 0}
        self.assertEqual(diagnostics._truth(row, "relay", "2026-09-30", "2026-10-08", "2026-10-07T12:00:00+00:00"),
                         (None, "TARGET_NOT_MATURED"))
        self.assertEqual(diagnostics._truth(row, "relay", "2026-09-30", "2026-10-08", "2026-10-08T15:00:00+08:00"),
                         (False, "KNOWN"))
        self.assertEqual(diagnostics._truth(row, "relay", "2026-09-30", "2026-10-09", "2026-10-10T15:00:00+08:00"),
                         (None, "TARGET_NOT_EXACT_OFFICIAL_NEXT_SESSION"))
        row["label_valid"] = False
        self.assertEqual(diagnostics._truth(row, "relay", "2026-09-30", "2026-10-08", "2026-10-10T15:00:00+08:00"),
                         (None, "LABEL_UNKNOWN"))

    def historical_fixture(self, root):
        sessions = ["2025-07-01", "2025-07-02", "2025-07-03", "2026-01-02", "2026-01-05", "2026-09-01", "2026-09-02"]
        calendar = root / "calendar.json"
        calendar.write_text(json.dumps({"status": "verified", "source": "SYNTHETIC_EXCHANGE", "sessions": sessions}), encoding="utf-8")
        _, _, _, mm, rm = self.fixture()
        market_rows = [{"fold": "2025H2", "date": "2025-07-01", "target_date": "2025-07-02", "arm": "BASE",
            "status": "PREDICTED_UNCALIBRATED", "predicted": "IMPROVE", "target": "WORSEN", "label_valid": True,
            "confidence": .6, "yesterday_premium": -.01, "promotion_mid": .3,
            "raw_score_IMPROVE": .6, "raw_score_CONTINUE": .3, "raw_score_WORSEN": .1}]
        relay_rows = [{"fold": "2025H2", "date": "2025-07-01", "target_date": "2025-07-02", "arm": "BASE",
            "code": "000001", "name": "synthetic A", "boards": 2, "actual_one_word_daily": None,
            "same_height_rivals": 0., "signal_one_word": 0., "turnover": .02,
            "status": "PREDICTED_UNCALIBRATED", "predicted": 0, "target": 1., "label_valid": True, "raw_score": .5}]
        for module, rows, manifest in [("market", market_rows, mm), ("relay", relay_rows, rm)]:
            history_dir = root / (module+"_history"); history_dir.mkdir()
            source = history_dir / "predictions.csv"
            pd.DataFrame(rows).to_csv(source, index=False)
            (history_dir / "report.json").write_text(json.dumps({"module": module, "folds": diagnostics.FOLDS,
                "arms": ["BASE", "C1", "C2"], "protocol_sha256": "a"*64,
                "predictions_sha256": diagnostics.sha(source), "formed_at": "2026-10-07T10:00:00+00:00"}), encoding="utf-8")
            model_dir = root / "models" / module / "BASE"; model_dir.mkdir(parents=True)
            model_path = model_dir / "model.joblib"; model_path.write_bytes(b"SYNTHETIC_NOT_A_MODEL_NO_LOADING")
            manifest = manifest | {"model_path": str(model_path), "sha256": diagnostics.sha(model_path)}
            (model_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return calendar

    def test_history_full_official_grid_unknowns_and_no_source_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calendar = self.historical_fixture(root)
            before = {str(p): diagnostics.sha(p) for p in root.rglob("*") if p.is_file()}
            output = root / "new_appendix"
            report = diagnostics.historical(root, calendar, output, asof="2026-10-07T12:00:00+00:00")
            rows = pd.read_csv(output / "full_error_rows.csv")
            candidates = pd.read_csv(output / "relay_candidate_diagnostics.csv")
            self.assertEqual(len(rows), 14)
            self.assertEqual(len(candidates), 1)
            self.assertTrue(pd.isna(candidates.loc[0, "actual_one_word_daily"]))
            self.assertTrue(pd.isna(candidates.loc[0, "non_one_word_rank_first"]) if "non_one_word_rank_first" in candidates else True)
            self.assertEqual(report["new_forward_samples"], 0)
            self.assertEqual(report["evidence_class"], "HISTORICAL_REPLAY_NOT_FORWARD")
            self.assertEqual(report["accuracy_change"], "NO_ACCURACY_CHANGE")
            self.assertEqual(report["folds"]["2025H2"]["official_signal_dates"], 3)
            self.assertEqual(report["folds"]["2025H2"]["market"]["unknown_or_pending_rows"], 2)
            self.assertEqual(report["folds"]["2025H2"]["relay_candidates"]["FN"], 1)
            self.assertEqual(before, {p: diagnostics.sha(p) for p in before})
            with self.assertRaises(FileExistsError):
                diagnostics.historical(root, calendar, output, asof="2026-10-07T12:00:00+00:00")

    def test_history_csv_hash_tampering_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calendar = self.historical_fixture(root)
            source = root / "relay_history/predictions.csv"
            source.write_text(source.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                diagnostics.historical(root, calendar, root / "must_not_exist", asof="2026-10-07T12:00:00+00:00")
            self.assertFalse((root / "must_not_exist").exists())

    def test_new_missing_score_full_rank_unknown_and_pure_clock(self):
        args = self.fixture()
        args[2]["candidates"][2]["raw_score"] = None
        observed = self.call(args)
        self.assertFalse(observed["raw_score_scope_complete"])
        for key in ("all_rank_first", "one_word_rank_first", "positive_model_primary",
                    "non_one_word_rank_first", "non_one_word_positive_first"):
            self.assertIsNone(observed[key])
        self.assertEqual(observed["non_one_word_primary_status"], "UNKNOWN_SCORE")
        self.assertEqual(observed["one_word_rank_status"], "UNKNOWN_SCORE")
        self.assertTrue(all(r["rank"] is None for r in observed["candidates"]))
        self.assertNotIn("generated_at", observed)
        self.assertEqual(observed["knowledge_cutoff"], "2026-10-07T12:00:00+00:00")
        self.assertIn("子集观察", observed["known_subset_scope"])
        self.assertIn("本附页不计算", observed["trading_primary_status"])

    def test_new_bound_rank_conflict_rejected_and_bool_word_supported(self):
        args = self.fixture()
        args[2]["candidates"][0]["rank"] = 1
        with self.assertRaisesRegex(ValueError, "bound original rank conflicts"):
            self.call(args)
        args = self.fixture()
        args[0]["relay"][1]["actual_one_word_daily"] = True
        args[2]["candidates"][1]["actual_one_word_daily"] = True
        observed = self.call(args)
        self.assertEqual(observed["one_word_rank_first"], "000001")
        self.assertEqual(observed["one_word_rank_status"], "KNOWN")
        self.assertEqual(observed["non_one_word_primary_status"], "EMPTY")
        self.assertIsNone(observed["non_one_word_positive_first"])

    def test_new_existing_facts_and_complete_recall_include_abstention(self):
        observed = self.call(self.fixture())
        self.assertEqual(observed["market"]["signal_facts"]["yesterday_premium"], -.01)
        self.assertEqual(observed["market"]["original_raw_scores"]["CONTINUE"], .3)
        self.assertEqual(observed["candidates"][0]["signal_facts"]["turnover"], .04)
        relay = [{"evaluation_status": "SCORED", "original_prediction": True, "truth": True, "error": False},
                 {"evaluation_status": "UNKNOWN", "original_prediction": None, "truth": True, "error": None}]
        metrics = diagnostics._classification(relay, "relay")
        self.assertEqual(metrics["scored_FN"], 0)
        self.assertEqual(metrics["complete_positive_misses"], 1)
        self.assertEqual(metrics["complete_positive_recall"], .5)
        self.assertAlmostEqual(metrics["F1"], 2/3)
        market = [{"evaluation_status": "SCORED", "original_prediction": "CONTINUE", "truth": "CONTINUE", "error": False},
                  {"evaluation_status": "UNKNOWN", "original_prediction": None, "truth": "CONTINUE", "error": None}]
        cls = diagnostics._classification(market, "market")["by_class"]["CONTINUE"]
        self.assertEqual(cls, {"truth_support": 2, "predicted_calls": 1, "TP": 1, "complete_recall": .5})

    def test_new_historical_facts_and_absent_group_unknown_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calendar = self.historical_fixture(root)
            output = root / "appendix"
            diagnostics.historical(root, calendar, output, asof="2026-10-07T12:00:00+00:00")
            rows = pd.read_csv(output / "full_error_rows.csv")
            market = rows[rows.module.eq("market") & rows.signal_date.eq("2025-07-01")].iloc[0]
            self.assertEqual(market.raw_score_CONTINUE, .3)
            self.assertEqual(market.yesterday_premium, -.01)
            absent = rows[rows.module.eq("relay") & rows.signal_date.eq("2025-07-02")].iloc[0]
            self.assertEqual(absent.non_one_word_primary_status, "UNKNOWN_SOURCE")
            self.assertEqual(absent.one_word_rank_status, "UNKNOWN_SOURCE")
            self.assertTrue(pd.isna(absent.non_one_word_scope_complete))
            candidates = pd.read_csv(output / "relay_candidate_diagnostics.csv")
            self.assertEqual(json.loads(candidates.signal_facts.iloc[0])["turnover"], .02)

    def test_new_native_registration_binding_cutoffs_and_conflicts(self):
        args = self.fixture()
        pack, market_base, relay_base, mm, rm = args
        pack["sources"] = {}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); archive = root / "archive"; archive.mkdir()
            mp, rp = root / "market.json", root / "relay.json"
            mp.write_text("{}", encoding="utf-8"); rp.write_text("{}", encoding="utf-8")
            projection = diagnostics.hashlib.sha256(diagnostics._canonical(pack).encode()).hexdigest()
            common = {"signal_date": pack["signal_date"], "target_date": pack["target_date"],
                      "feature_projection_sha256": projection, "protocol_sha256": "a"*64,
                      "feature_archive": str(archive)}
            def binding(manifest):
                return {"manifest": manifest, "path": "synthetic_manifest", "sha256": "b"*64,
                        "model_path": "synthetic_model", "model_sha256": "c"*64}
            market = common | {"module": "market", "knowledge_cutoff": "2026-10-07T11:00:00+00:00",
                "arms": {"BASE": market_base}, "model_bindings": {"BASE": binding(mm)}}
            relay = common | {"module": "relay", "knowledge_cutoff": "2026-10-07T11:00:00.100+00:00",
                "arms": {"BASE": relay_base}, "model_bindings": {"BASE": binding(rm)}}
            registered = types.SimpleNamespace(_registered=lambda p: (market if Path(p) == mp else relay,))
            loader = unittest.mock.Mock(return_value=pack)
            inputs = types.SimpleNamespace(load_feature_archive=loader)
            with patch.dict("sys.modules", {"model_upgrade_shadow": registered, "model_upgrade_inputs": inputs}), \
                 patch.object(diagnostics, "now", return_value="2026-10-07T13:00:00+00:00"):
                result = diagnostics.signal_from_registrations(mp, rp, archive, root / "signal_appendix")
                loader.assert_called_once_with(archive, cutoff="2026-10-07T11:00:00+00:00")
                self.assertEqual(result["generated_at"], "2026-10-07T13:00:00+00:00")
                self.assertEqual(result["knowledge_cutoff"], market["knowledge_cutoff"])
                self.assertEqual(result["original_module_cutoffs"]["relay"], relay["knowledge_cutoff"])
                relay["arms"]["BASE"]["candidates"][0]["boards"] = 3
                with self.assertRaisesRegex(ValueError, "metadata conflicts"):
                    diagnostics.signal_from_registrations(mp, rp, archive, root / "rejected")
                self.assertFalse((root / "rejected").exists())


class ThemeLadderDiagnosticsSynthetic(unittest.TestCase):
    day = "2026-10-08"
    cutoff = "2026-10-08T18:00:00+08:00"

    def fixture(self):
        def pool(kind, rows):
            return {"date": self.day, "kind": kind, "capture_complete_within_provider_response": True,
                    "available_at": "2026-10-08T17:00:00+08:00", "rows": rows}
        closing = pool("limit_up", [
            {"code": "000001", "name": "甲", "boards": 4, "industry": "电池"},
            {"code": "000002", "name": "乙", "boards": 3, "industry": "电子"},
            {"code": "000003", "name": "丙", "boards": 2, "industry": "塑料"},
            {"code": "000004", "name": "丁", "boards": 1, "industry": "电池"},
            {"code": "000005", "name": "戊", "boards": 1, "industry": "电力"}])
        failed = pool("failed_board", [{"code": "000006", "name": "己", "industry": "电池"},
                                         {"code": "000007", "name": "庚", "industry": "材料"}])
        def category(code, names):
            identities = {"锂电池": "1", "固态电池": "2", "储能概念": "3", "新能源": "4"}
            return {"code": code, "categories": {"available_at": "2026-10-08T17:10:00+08:00",
                "rows": [{"board_code": identities[name], "board_name": name} for name in names]}}
        view = {"knowledge_cutoff": self.cutoff, "candidates": [
            category("000001", ["锂电池", "固态电池"]), category("000002", ["固态电池"]),
            category("000003", ["锂电池"]), category("000004", ["固态电池", "锂电池"]),
            category("000005", ["储能概念", "新能源"])]}
        base = [{"code": str(i).zfill(6), "boards": 5-i, "predicted_positive": False,
                 "raw_score": .4, "rank": i} for i in (1, 2, 3)]
        return closing, failed, view, self.day, self.cutoff, base

    def call(self, args=None):
        return diagnostics.theme_ladder_diagnostics(*(args or self.fixture()))

    def test_first_board_failed_unknown_and_deduplication_without_scoring(self):
        args = self.fixture()
        before = copy.deepcopy(args)
        result = self.call(args)
        self.assertEqual(before, args)
        self.assertEqual(len(result["all_candidates"]), 7)
        self.assertEqual(result["pool_counts"], {"closing": 5, "failed": 2, "relay": 3, "first_board": 2})
        self.assertEqual(result["category_unknown_codes"], ["000006", "000007"])
        group = next(g for g in result["groups"] if g["evidence_lane"] == "PROVIDER_SEMANTIC_FAMILY")
        self.assertEqual(group["closing_count"], 4)
        self.assertEqual(group["relay_count"], 3)
        self.assertEqual(group["first_board_codes"], ["000004"])
        self.assertEqual(group["failed_codes"], ["000006"])
        self.assertNotIn("000005", group["member_codes"])
        self.assertTrue(group["ladder_contiguous"])
        self.assertTrue(group["all_known_original_relay_calls_negative"])
        self.assertFalse(group["failed_membership_coverage_complete"])
        self.assertFalse(group["mainline_confirmation_authorized"])
        self.assertNotIn("score", group)
        self.assertNotIn("rank", group)
        self.assertFalse(result["model_or_prediction_changed"])

    def test_late_stale_wrong_date_and_duplicate_inputs(self):
        args = self.fixture(); args[0]["available_at"] = "2026-10-08T18:01:00+08:00"
        with self.assertRaisesRegex(ValueError, "follows diagnostic cutoff"):
            self.call(args)
        args = self.fixture(); args[2]["candidates"][0]["categories"]["available_at"] = "2026-10-07T16:00:00+08:00"
        result = self.call(args)
        self.assertEqual(result["all_candidates"][0]["category_missing_reason"], "CAPTURE_OLDER_THAN_24H_POLICY")
        args = self.fixture(); args[2]["candidates"][0]["categories"]["available_at"] = "2026-10-07T19:00:00+08:00"
        self.assertEqual(self.call(args)["all_candidates"][0]["category_missing_reason"], "CATEGORY_NOT_OBSERVED_ON_SIGNAL_DATE")
        args = self.fixture(); args[1]["date"] = "2026-10-09"
        with self.assertRaisesRegex(ValueError, "exact-date"):
            self.call(args)
        args = self.fixture(); args[1]["rows"][0]["code"] = "000001"
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.call(args)
        args = self.fixture(); args[5].pop()
        with self.assertRaisesRegex(ValueError, "omits original"):
            self.call(args)

    def test_future_labels_prices_and_calls_not_read_or_used(self):
        args = self.fixture(); expected = self.call(args)
        closing, failed, view, day, cutoff, base = args
        for pool in (closing, failed):
            pool["rows"] = [PoisonMapping(r | {"truth": True, "trade_close": 999, "target": "future"}) for r in pool["rows"]]
        view["candidates"] = [PoisonMapping(r | {"promoted_next_day": True}) for r in view["candidates"]]
        base = [PoisonMapping(r | {"truth": True, "next_open_gap": 99}) for r in base]
        self.assertEqual(expected, self.call((PoisonMapping(closing), PoisonMapping(failed),
                                              PoisonMapping(view), day, cutoff, base)))

    def test_archive_rebuild_cutoff_binding_hash_and_immutable_output(self):
        args = self.fixture(); closing, failed, view, day, cutoff, base_rows = args
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); pool = root/"pool"; context = root/"context"
            pool.mkdir(); context.mkdir()
            for p in [pool/"manifest.json", pool/"normalized.json", context/"context.json"]:
                p.write_text("{}", encoding="utf-8")
            (pool/"manifest.json").write_text(json.dumps({"target_date": day}), encoding="utf-8")
            (context/"manifest.json").write_text('{"version":"synthetic"}', encoding="utf-8")
            source = root/"original_source"; source.write_text("synthetic", encoding="utf-8")
            original = {"signal_date": day, "target_date": "2026-10-09", "production_model_changed": False,
                "knowledge_cutoff": "2026-10-08T17:00:00+08:00", "generated_at": "2026-10-08T17:01:00+08:00",
                "relay": {"candidates": base_rows}, "source_hashes": {str(source): diagnostics.sha(source)}}
            bp = root/"base.json"; bp.write_text(json.dumps(original), encoding="utf-8")
            manifest = {"recorded_at": "2026-10-08T17:00:00+08:00", "calendar": {"sessions": [day]}, "requests": []}
            current = {"recorded_at": "2026-10-08T17:10:00+08:00", "scope": {"price_signal_date": day,
                "source_path": str(pool), "manifest_sha256": diagnostics.sha(pool/"manifest.json"),
                "codes": [r["code"] for r in closing["rows"]]}, "candidates": view["candidates"]}
            outcome = types.SimpleNamespace(reconstruct=lambda p: (manifest, {(day,"limit_up"): closing, (day,"failed_board"): failed}, {}, []))
            context_module = types.SimpleNamespace(reconstruct=lambda p: current, asof_view=lambda c,t: view)
            reconciled = types.SimpleNamespace(VERSION="reconciled")
            with patch.dict("sys.modules", {"public_outcome_sources_v293": outcome,
                "point_context_v295": context_module, "point_context_reconciled_v295": reconciled}), \
                patch.object(diagnostics, "now", return_value="2026-10-08T18:01:00+08:00"):
                output = root/"appendix"
                result = diagnostics.theme_from_archives(pool, context, day, output, cutoff, bp)
                self.assertEqual(result["original_base_prediction_binding"]["sha256"], diagnostics.sha(bp))
                self.assertEqual(diagnostics._read(output/"report.json")["all_candidates"], result["all_candidates"])
                with self.assertRaises(FileExistsError):
                    diagnostics.theme_from_archives(pool, context, day, output, cutoff, bp)
                (pool/"manifest.json").write_text('{"target_date":"2026-10-09"}', encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "after signal date"):
                    diagnostics.theme_from_archives(pool, context, day, root/"future_archive", cutoff, bp)
                (pool/"manifest.json").write_text(json.dumps({"target_date": day}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "in the future"):
                    diagnostics.theme_from_archives(pool, context, day, root/"future_clock", "2026-10-08T18:02:00+08:00", bp)
                current["recorded_at"] = "2026-10-08T18:00:01+08:00"
                with self.assertRaisesRegex(ValueError, "formed after"):
                    diagnostics.theme_from_archives(pool, context, day, root/"late", cutoff, bp)
                current["recorded_at"] = "2026-10-08T17:10:00+08:00"
                source.write_text("tampered", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    diagnostics.theme_from_archives(pool, context, day, root/"tampered", cutoff, bp)


if __name__ == "__main__":
    unittest.main()
