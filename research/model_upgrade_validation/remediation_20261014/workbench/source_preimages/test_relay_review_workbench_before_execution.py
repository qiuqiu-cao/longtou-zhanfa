"""Engineering contracts; fixtures/tests do not measure prediction improvement."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import stock_judgement_review as o
import relay_review_workbench as w


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.facts_path = self.root / "facts.json"
        self.draft_path = self.root / "draft.json"
        self.facts = {"signal_date": "2026-10-14", "target_date": "2026-10-15",
            "supplement_context_diagnostic_only": True, "supplement_model_exposure": "MODEL_OUTPUT_ALREADY_SEEN",
            "source_bindings": {}, "rows": [{"code": "000001", "name": "甲", "boards": 2, "same_height_peers": ["000002"]},
                {"code": "000002", "name": "乙", "boards": 2, "same_height_peers": ["000001"]}]}
        o.write_new(self.facts_path, self.facts)
        self.draft = {"status": "DIAGNOSTIC_TEMPLATE_NO_FORECAST", "signal_date": self.facts["signal_date"],
            "target_date": self.facts["target_date"], "facts_path": str(self.facts_path.resolve()),
            "facts_sha256": o.sha(self.facts_path), "model_exposure": "MODEL_OUTPUT_ALREADY_SEEN",
            "author_review_completed": False, "source_bindings": o.bind(self.facts_path),
            "rows": [{"code": r["code"], "name": r["name"], "boards": r["boards"],
                "promotion_judgement": None, "role_assessment": {"SUPPLEMENT": {"state": None, "reason": "父锚关系待核。"}},
                "forum_preread_assessment": {"reviewed": False, "reason": "父锚关系待核。"},
                "opening_confirmation": {"09:25": "实际竞价后核对。"},
                "short_term_assessment": {"confirmation_plan": [{"paths": {"maintained": {
                    "state": "UNKNOWN", "reason": "父锚关系待核。", "conditions": [{"metric": "price", "expected": "KNOWN_LITERAL"}]}}}]}}
                for r in self.facts["rows"]]}
        o.write_new(self.draft_path, self.draft)

    def compact(self):
        path = self.root / "compact.json"
        with patch.object(w.forum, "checked", return_value={}):
            result = w.compact(self.facts_path, self.draft_path, path)
        return path, result

    def replace(self, path, value):
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def test_prepare_cli_forwards_all_repeated_history_and_narrative_inputs(self):
        arguments = ["prepare"]
        for field in ("features", "theme-report", "mainline-report", "compatibility", "bank", "calendar", "pool", "relation-bank"):
            arguments.extend(["--" + field, "input-" + field])
        arguments += ["--output-dir", str(self.root / "run"), "--history-dataset", "a", "--history-dataset", "b",
            "--mainline-history", "c", "--mainline-history", "d", "--narrative-reviews", "e"]
        def prepare(**kwargs):
            o.write_new(Path(kwargs["output_dir"]) / "reasoning" / "facts_with_forum_reasoning.json", {"rows": []})
            return {"rows": []}
        with patch.object(w.forum, "prepare", side_effect=prepare) as call, patch.object(w, "render"):
            w.main(arguments)
        values = call.call_args.kwargs
        self.assertEqual(values["history_datasets"], ("a", "b"))
        self.assertEqual(values["mainline_history"], ("c", "d"))
        self.assertEqual(values["narrative_reviews"], "e")
        self.assertEqual(values["exposure"], "MODEL_OUTPUT_ALREADY_SEEN")

    def test_prepare_failure_preserves_pipeline_receipt_and_cause(self):
        out = self.root / "failed"
        with patch.object(w.forum, "prepare", side_effect=ValueError("source guard rejected")):
            with self.assertRaisesRegex(ValueError, "source guard"):
                w.prepare(*(["x"] * 6), out, "pool", "relations")
        receipt = o.read(out / "prepare_failure.json")
        self.assertEqual(receipt["error"], "source guard rejected")
        self.assertEqual(receipt["new_fits"], 0)

    def test_freeze_and_diagnostic_review_delegate_without_registration_bypass(self):
        for fn in ("freeze", "review_context"):
            with patch.object(w.forum, fn, side_effect=ValueError("original clock gate")) as call:
                with self.assertRaisesRegex(ValueError, "original clock gate"):
                    getattr(w, fn)("facts", "draft", "out")
                call.assert_called_once_with("facts", "draft", "out")

    def test_compare_feedback_failure_keeps_exact_original_comparison(self):
        out = self.root / "comparison.json"
        def comparison(*args):
            o.write_new(out, {"metrics": {"accuracy": .5}, "original": True})
            return {"status": "SAVED"}
        with patch.object(w.forum, "compare", side_effect=comparison), patch.object(w.feedback, "project", side_effect=ValueError("bound argument missing")):
            with self.assertRaisesRegex(ValueError, "bound argument"):
                w.compare("record", "base", "outcome", out)
        self.assertEqual(o.read(out), {"metrics": {"accuracy": .5}, "original": True})
        r = o.read(out.with_name("comparison.workbench.json"))
        self.assertEqual(r["comparison_sha256"], o.sha(out))
        self.assertEqual(r["status"], "COMPARISON_SAVED_ARGUMENT_FEEDBACK_FAILED")

    def test_compare_success_runs_feedback_once(self):
        out = self.root / "comparison.json"
        def comparison(*args):
            o.write_new(out, {"status": "DONE"}); return {"status": "DONE"}
        with patch.object(w.forum, "compare", side_effect=comparison), patch.object(w.feedback, "project", return_value={"status": "FEEDBACK"}) as call:
            w.compare("record", "base", "outcomes", out)
        self.assertEqual(call.call_args.args[:2], ("record", out))
        self.assertEqual(o.read(out.with_name("comparison.workbench.json"))["status"], "COMPARISON_AND_ARGUMENT_FEEDBACK_SAVED")

    def test_compact_deduplicates_prose_without_filling_states_or_directions(self):
        _, packet = self.compact()
        self.assertEqual(len(packet["reason_bank"]), 2)
        self.assertEqual(len(next(e for e in packet["reason_bank"].values() if e["text"] == "父锚关系待核。")["original_locations"]), 6)
        for row in packet["draft"]["rows"]:
            self.assertIsNone(row["promotion_judgement"])
            self.assertIsNone(row["role_assessment"]["SUPPLEMENT"]["state"])
            self.assertFalse(row["forum_preread_assessment"]["reviewed"])
            self.assertEqual(row["short_term_assessment"]["confirmation_plan"][0]["paths"]["maintained"]["conditions"][0]["expected"], "KNOWN_LITERAL")

    def test_roundtrip_exact_rows_and_binding_with_author_reason_edit(self):
        path, packet = self.compact()
        rid = packet["draft"]["rows"][0]["forum_preread_assessment"]["reason"]["$reason"]
        packet["reason_bank"][rid]["text"] = "作者实际复核后的共享理由。"
        self.replace(path, packet)
        with patch.object(w, "validate_completed", return_value="TEST_VALIDATED") as checked:
            result = w.expand(path, self.root / "expanded.json")
        for row in result["rows"]:
            self.assertEqual(row["forum_preread_assessment"]["reason"], "作者实际复核后的共享理由。")
            self.assertIsNone(row["promotion_judgement"])
        checked.assert_called_once()
        self.assertEqual(result["source_bindings"][str(path.resolve())], o.sha(path))

    def test_expand_pending_template_rejected_without_synthetic_unknown(self):
        path, _ = self.compact()
        with self.assertRaisesRegex(ValueError, "still pending"):
            w.expand(path, self.root / "expanded.json")
        self.assertFalse((self.root / "expanded.json").exists())

    def test_reference_cannot_replace_forecast_or_structured_condition(self):
        for locate in ("forecast", "expected"):
            path, packet = self.compact()
            rid = next(iter(packet["reason_bank"]))
            if locate == "forecast":
                packet["draft"]["rows"][0]["promotion_judgement"] = {"$reason": rid}
            else:
                packet["draft"]["rows"][0]["short_term_assessment"]["confirmation_plan"][0]["paths"]["maintained"]["conditions"][0]["expected"] = {"$reason": rid}
            self.replace(path, packet)
            with self.assertRaisesRegex(ValueError, "only valid"):
                w.expand(path, self.root / "expanded.json")
            path.unlink()

    def test_expand_missing_reason_and_changed_exposure_rejected(self):
        path, packet = self.compact()
        rid = packet["draft"]["rows"][0]["forum_preread_assessment"]["reason"]["$reason"]
        packet["reason_bank"].pop(rid)
        self.replace(path, packet)
        with self.assertRaisesRegex(ValueError, "missing or empty"):
            w.expand(path, self.root / "expanded.json")
        packet["draft"]["model_exposure"] = "MODEL_NOT_REVIEWED_SELF_ATTESTED"
        self.replace(path, packet)
        with self.assertRaisesRegex(ValueError, "metadata"):
            w.expand(path, self.root / "expanded.json")

    def test_changed_original_draft_rejected(self):
        path, _ = self.compact()
        self.draft_path.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source changed"):
            w.expand(path, self.root / "expanded.json")

    def test_exact_full_pool_and_model_exposure_checked(self):
        changed = copy.deepcopy(self.draft); changed["rows"].pop()
        with self.assertRaisesRegex(ValueError, "every original"):
            w._draft_binding(self.facts_path, self.facts, changed)
        changed = copy.deepcopy(self.draft); changed["model_exposure"] = "MODEL_NOT_REVIEWED_SELF_ATTESTED"
        with self.assertRaisesRegex(ValueError, "model exposure"):
            w._draft_binding(self.facts_path, self.facts, changed)

    def test_model_scores_and_baseline_calls_absent_from_view_projection(self):
        self.facts["rows"][0]["leader_role_evidence"] = {"score": 98.888, "baseline_call": "PROMOTE"}
        self.facts["rows"][0]["model_score"] = 98.888
        self.draft["rows"][0]["model_score"] = 98.888
        report = w.review_projection(self.facts, self.draft)
        encoded = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("98.888", encoded)
        self.assertNotIn("baseline_call", encoded)
        self.assertNotIn("leader_role_evidence", encoded)
        self.assertEqual(report["pool_size"], 2)
        self.assertEqual(report["cards"][0]["same_height_peers"], [{"code": "000002", "name": "乙"}])

    def test_html_escapes_forum_text_and_preserves_all_candidates(self):
        report = w.review_projection(self.facts)
        report["cards"][0]["opinions"] = [{"opinion_id": "1", "author": "作者", "source_section": "COMMENT", "use": "BACKGROUND",
            "discussion_published_at": None, "parent_post_published_at": "OLD_POST", "obtained_at": "NOW", "available_at": "NOW",
            "excerpt": "<script>attack()</script>", "url": "unknown"}]
        page = w.html_page(report)
        self.assertNotIn("<script>attack()", page)
        self.assertIn("&lt;script&gt;attack()", page)
        self.assertEqual(page.count("<article "), 2)
        self.assertIn("000001", page); self.assertIn("000002", page)
        self.assertIn("正文／评论各自发布时间：</strong>未知", page)
        self.assertIn("主帖：</strong>OLD_POST", page)


if __name__ == "__main__":
    unittest.main()
