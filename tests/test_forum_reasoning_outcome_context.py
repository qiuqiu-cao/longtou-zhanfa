"""Synthetic contracts only; no fitted model, fetched source or real promotion claim."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import forum_reasoning_outcome_context as feedback
import forum_reasoning_review as forum
import stock_judgement_review as original


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.text = self.root / "forum.txt"
        self.text.write_text("作者提出有条件的晋级和角色假设。", encoding="utf-8")
        self.signal, self.target = "2026-10-14", "2026-10-15"
        self.codes = [f"00000{i}" for i in range(1, 6)]
        self.calls = ["PROMOTE", "PROMOTE", "FAIL", "UNKNOWN", "PROMOTE"]
        self.labels = dict(zip(self.codes, [True, False, True, True, None]))
        self.effects = ["SUPPORT_PROMOTE", "SUPPORT_PROMOTE", "SUPPORT_FAIL", "ROLE_ONLY", "NEUTRAL"]
        self.feature_path = self.root / "features.json"
        original.write_new(self.feature_path, {"signal_date": self.signal, "target_date": self.target,
            "calendar": {"sessions": [self.signal, self.target]}, "relay": [
                {"code": code, "name": code, "boards": 2, "actual_one_word_daily": False,
                 "features": {k: None for k in original.PRICE_FIELDS}} for code in self.codes]})
        self.bindings = original.bind(self.text) | original.bind(self.feature_path)
        self.facts = {"signal_date": self.signal, "target_date": self.target,
            "feature_archive": str(self.root), "sessions": [self.signal, self.target],
            "pool_size": 5, "source_bindings": self.bindings,
            "supplement_model_exposure": "MODEL_OUTPUT_ALREADY_SEEN",
            "supplement_context_diagnostic_only": False, "rows": [
                {"code": code, "name": code, "boards": 2, "actual_one_word_daily": False,
                 "same_height_peers": [c for c in self.codes if c != code],
                 "daily_price_volume": {k: None for k in original.PRICE_FIELDS}} for code in self.codes]}
        self.context = {"version": forum.VERSION, "signal_date": self.signal,
            "target_date": self.target, "source_bindings": self.bindings,
            "discussion_only_paths": [str(self.text)],
            "verification_paths_by_scope": {scope: [] for scope in forum.SCOPES}, "rows": []}
        for i, code in enumerate(self.codes):
            p = {"source_id": "synthetic", "source_section": "AUTHOR_BODY", "offset": 0,
                "excerpt": "作者提出有条件的晋级和角色假设。", "text_path": str(self.text),
                "author": "测试作者", "use": "SIGNAL_DAY_DISCUSSION"}
            p["opinion_id"] = forum.opinion_id(p)
            self.context["rows"].append({"code": code, "name": code, "boards": 2,
                "coverage": "SYNTHETIC", "opinions": [p]})
        self.review = forum.draft_template(self.context)
        self.review.update(signal_date=self.signal, target_date=self.target,
            author_review_completed=True, model_exposure="MODEL_OUTPUT_ALREADY_SEEN")
        for i, row in enumerate(self.review["rows"]):
            row.update(promotion_judgement=self.calls[i], role_hypothesis="候选角色未知",
                relative_thesis="完整同位竞争", support="有条件推理", rebuttal="存在替代解释",
                cancel_condition="相对表现走弱", missing=[], comparison_peers=[c for c in self.codes if c != row["code"]],
                fact_references=[str(self.text)], opening_confirmation={k: "待实际观察" for k in ("09:25", "09:35", "10:00")})
            assessment = row["forum_reasoning_assessment"]
            assessment.update(reviewed=True, synthesis="逐观点独立判断", agreement="PARTIAL", agreement_reason="保留条件和反证")
            assessment["opinions"][0].update(author_claim="有条件的候选假设", author_reasoning="比较完整同位",
                testable_expectation="当相对表现维持时可能延续", author_claim_form="HYPOTHESIS",
                author_call="PROMOTE" if i == 0 else "UNSPECIFIED",
                author_expectation_target_date=self.target if i == 0 else None,
                verification={"status": "UNKNOWN", "scope": "ROLE_HYPOTHESIS", "reason": "未认证角色", "evidence_refs": [str(self.text)]},
                decision="ACCEPT", reason="采纳条件假设", counterevidence=["仍有其他同位候选"],
                independent_effect=self.effects[i], cancel_conditions=["相对表现走弱时取消"])
        self.facts_path = self.root / "facts.json"
        self.record_path = self.root / "record.json"
        self.comparison_path = self.root / "comparison.json"
        self._write(self.facts_path, self.facts)
        self.record = {**self.review, "version": original.VERSION,
            "status": "INDEPENDENT_PREAUCTION_JUDGEMENT_NOT_PRODUCTION_FREEZE",
            "formed_at": "2026-10-14T08:00:00+00:00", "knowledge_cutoff": "2026-10-14T08:00:00+00:00",
            "facts_path": str(self.facts_path), "facts_sha256": original.sha(self.facts_path),
            "feature_archive": str(self.root), "compatibility_policy": "synthetic_policy_only",
            "pool_size": 5, "source_bindings": self.bindings | original.bind(self.facts_path)}
        self._write(self.record_path, self.record)
        self.receipt = {"path": str(self.record_path), "sha256": original.sha(self.record_path),
            "signal_date": self.signal, "target_date": self.target, "pool_size": 5}
        self.baseline_path = self.root / "BASE.json"
        self.base_calls = dict(zip(self.codes, [True, True, False, False, False]))
        original.write_new(self.baseline_path, {"signal_date": self.signal, "target_date": self.target,
            "generated_at": "2026-10-14T08:30:00+00:00", "knowledge_cutoff": "2026-10-14T08:30:00+00:00",
            "feature_sha256": original.sha(self.feature_path), "source_hashes": self.bindings,
            "relay": {"candidates": [{"code": c, "predicted_positive": v} for c, v in self.base_calls.items()]}})
        calls = dict(zip(self.codes, [True, True, False, None, True]))
        metrics = {"independent": original.metrics(calls, self.labels), "BASE": original.metrics(self.base_calls, self.labels)}
        strata = {"2": metrics}
        self.comparison = {"version": original.VERSION, "status": "SAME_POOL_SOURCE_BOUND_COMPARISON_ONLY",
            "formed_at": original.now().isoformat(), "signal_date": self.signal, "target_date": self.target,
            "model_exposure": self.record["model_exposure"], "source_bindings": self.record["source_bindings"] | original.bind(self.record_path) | original.bind(self.baseline_path),
            "metrics": metrics, "breakdown": {"boards": strata, "actual_one_word_daily": {"False": metrics}},
            "rows": [{"code": c, "independent_call": calls[c], "BASE_call": self.base_calls[c], "actual_promoted": self.labels[c]} for c in self.codes]}
        self._write(self.comparison_path, self.comparison)
        for target, value in (("checked", self.context),):
            mocked = patch.object(forum, target, return_value=value)
            mocked.start(); self.addCleanup(mocked.stop)
        mocked = patch.object(forum.supplement, "validate_review", return_value={})
        mocked.start(); self.addCleanup(mocked.stop)
        mocked = patch.object(original, "registrations", return_value=[self.receipt])
        mocked.start(); self.addCleanup(mocked.stop)
        self.real_outcomes = feedback._read_outcomes
        mocked = patch.object(feedback, "_read_outcomes", return_value=self.labels)
        mocked.start(); self.addCleanup(mocked.stop)

    def _write(self, path, value):
        import json
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def project(self, name="feedback"):
        return feedback.project(self.record_path, self.comparison_path, self.root / name)

    def diagnostic(self):
        facts = copy.deepcopy(self.facts); facts["supplement_context_diagnostic_only"] = True
        self._write(self.facts_path, facts)
        review = copy.deepcopy(self.review)
        review.update(status="COMPLETED_APPEND_ONLY_SUPPLEMENT_REVIEW_NO_FORECAST", forecast_registered=False,
            facts_path=str(self.facts_path), facts_sha256=original.sha(self.facts_path), source_bindings=self.bindings | original.bind(self.facts_path))
        for row in review["rows"]:
            row["promotion_judgement"] = None
        path = self.root / "completed_review.json"; self._write(path, review)
        return path, review

    def test_full_pool_feedback_preserves_labels_errors_and_direction_only(self):
        result = self.project()
        self.assertEqual(result["pool_size"], 5)
        self.assertEqual(result["opinion_count"], 5)
        rows = {r["code"]: r for r in result["rows"]}
        self.assertEqual(rows["000001"]["opinions"][0]["direction_compatibility"], "DIRECTION_COMPATIBLE")
        self.assertTrue(rows["000002"]["independent_error_flags"]["false_positive"])
        self.assertTrue(rows["000003"]["independent_error_flags"]["false_negative"])
        self.assertTrue(rows["000004"]["independent_error_flags"]["positive_abstention"])
        self.assertIsNone(rows["000005"]["actual_promoted"])
        self.assertEqual(result["original_comparison_metrics"], self.comparison["metrics"])
        self.assertFalse(result["forum_causal_benefit_verified"])
        self.assertEqual(rows["000001"]["opinions"][0]["author_independent_direction_comparison"], "SAME_DIRECTION")

    def test_role_and_neutral_are_not_role_truth_and_conditions_never_inferred(self):
        result = self.project()
        for row in result["rows"]:
            opinion = row["opinions"][0]
            self.assertEqual(opinion["argument_truth"], "UNKNOWN")
            self.assertEqual(opinion["condition_effectiveness"], "UNKNOWN")
            self.assertEqual(opinion["cancellation_effectiveness"], "UNKNOWN")
            self.assertFalse(opinion["role_identity_from_promotion_label"])
        self.assertEqual(result["rows"][3]["opinions"][0]["direction_compatibility"], "NOT_APPLICABLE_ROLE_OR_NEUTRAL")

    def test_cross_date_comparison_is_rejected(self):
        self.comparison["target_date"] = "2026-10-16"; self._write(self.comparison_path, self.comparison)
        with self.assertRaisesRegex(ValueError, "dates"):
            self.project()

    def test_missing_and_duplicate_candidate_rejected(self):
        for duplicate in (False, True):
            changed = copy.deepcopy(self.comparison)
            changed["rows"].append(copy.deepcopy(changed["rows"][0])) if duplicate else changed["rows"].pop()
            self._write(self.comparison_path, changed)
            with self.assertRaises(ValueError): self.project()

    def test_forged_label_or_call_rejected_against_saved_sources(self):
        for field in ("actual_promoted", "BASE_call", "independent_call"):
            changed = copy.deepcopy(self.comparison); changed["rows"][0][field] = False
            self._write(self.comparison_path, changed)
            with self.assertRaisesRegex(ValueError, "direction or label"):
                self.project()

    def test_unknown_label_cannot_become_false(self):
        self.comparison["rows"][-1]["actual_promoted"] = False; self._write(self.comparison_path, self.comparison)
        with self.assertRaisesRegex(ValueError, "direction or label"):
            self.project()

    def test_wrong_hash_and_absent_registration_rejected(self):
        self.comparison["source_bindings"][str(self.text)] = "0" * 64; self._write(self.comparison_path, self.comparison)
        with self.assertRaises(ValueError): self.project()
        with patch.object(original, "registrations", return_value=[]):
            with self.assertRaisesRegex(ValueError, "registration"):
                self.project()

    def test_changed_facts_and_fake_metrics_rejected(self):
        self.comparison["metrics"]["independent"]["Accuracy"] = 1; self._write(self.comparison_path, self.comparison)
        with self.assertRaisesRegex(ValueError, "metrics"):
            self.project()
        self.facts["pool_size"] = 4; self._write(self.facts_path, self.facts)
        with self.assertRaisesRegex(ValueError, "facts"):
            self.project()

    def test_baseline_missing_or_ambiguous_rejected(self):
        changed = copy.deepcopy(self.comparison); del changed["source_bindings"][str(self.baseline_path)]
        self._write(self.comparison_path, changed)
        with self.assertRaisesRegex(ValueError, "one bound"):
            self.project()
        extra = self.root / "BASE2.json"; extra.write_bytes(self.baseline_path.read_bytes())
        self.comparison["source_bindings"].update(original.bind(extra)); self._write(self.comparison_path, self.comparison)
        with self.assertRaisesRegex(ValueError, "one bound"):
            self.project()

    def test_original_comparison_need_not_duplicate_baseline_extra_sources(self):
        extra = self.root / "extra_baseline_code.txt"; extra.write_text("synthetic source", encoding="utf-8")
        baseline = original.read(self.baseline_path)
        baseline["source_hashes"].update(original.bind(extra)); self._write(self.baseline_path, baseline)
        self.comparison["source_bindings"].update(original.bind(self.baseline_path))
        self._write(self.comparison_path, self.comparison)
        self.assertNotIn(str(extra), self.comparison["source_bindings"])
        result = self.project()
        self.assertEqual(result["source_bindings"][str(extra)], original.sha(extra))

    def test_late_or_future_knowledge_baseline_rejected(self):
        before = original.read(self.baseline_path)
        for field, value in (("generated_at", "2026-10-15T02:00:00+00:00"),
                             ("knowledge_cutoff", "2026-10-14T08:31:00+00:00")):
            baseline = copy.deepcopy(before); baseline[field] = value
            self._write(self.baseline_path, baseline)
            self.comparison["source_bindings"].update(original.bind(self.baseline_path)); self._write(self.comparison_path, self.comparison)
            with self.assertRaises(ValueError): self.project()

    def test_existing_outcome_adapter_requires_unique_bound_exact_manifest(self):
        from unittest.mock import Mock
        with patch("model_upgrade_compat.runtime") as runtime:
            with self.assertRaisesRegex(ValueError, "one bound"):
                self.real_outcomes(self.comparison, self.record)
            runtime.assert_not_called()
            manifest = self.root / "target" / "manifest.json"
            original.write_new(manifest, {"version": "v293-public-outcome-receipts-1", "signal_date": self.signal, "target_date": self.target})
            comparison = copy.deepcopy(self.comparison); comparison["source_bindings"].update(original.bind(manifest))
            inputs = Mock(); inputs.build_outcomes.return_value = {"candidates": self.labels, "sources": original.bind(manifest)}
            runtime.return_value = (inputs, None)
            self.assertEqual(self.real_outcomes(comparison, self.record), self.labels)
            inputs.build_outcomes.assert_called_once_with(self.record, manifest.parent)
            second = self.root / "second" / "manifest.json"; original.write_new(second, original.read(manifest))
            comparison["source_bindings"].update(original.bind(second))
            with self.assertRaisesRegex(ValueError, "one bound"):
                self.real_outcomes(comparison, self.record)

    def test_pending_review_keeps_complete_pool_without_directions_or_abstention_claim(self):
        path, _ = self.diagnostic()
        result = feedback.pending(self.facts_path, path, self.root / "pending")
        self.assertEqual(result["pool_size"], 5)
        self.assertTrue(result["outcome_pending"])
        for row in result["rows"]:
            self.assertIsNone(row["actual_promoted"]); self.assertIsNone(row["BASE_call"])
            self.assertIsNone(row["independent_call"]); self.assertIsNone(row["independent_error_flags"])
            self.assertIsNone(row["BASE_error_flags"]); self.assertEqual(row["error_evaluation_status"], "NOT_EVALUATED")
        self.assertEqual(result["new_fits"], 0)
        self.assertFalse(result["forecast_registered"])

    def test_pending_review_cannot_import_forecast_or_incomplete_review(self):
        path, review = self.diagnostic()
        review["rows"][0]["promotion_judgement"] = "PROMOTE"; self._write(path, review)
        with self.assertRaisesRegex(ValueError, "directions"):
            feedback.pending(self.facts_path, path, self.root / "pending")
        review["rows"][0]["promotion_judgement"] = None; review["author_review_completed"] = False; self._write(path, review)
        with self.assertRaisesRegex(ValueError, "completed"):
            feedback.pending(self.facts_path, path, self.root / "pending")

    def test_outputs_are_immutable_and_old_comparison_untouched(self):
        before = original.sha(self.comparison_path)
        self.project()
        self.assertEqual(before, original.sha(self.comparison_path))
        with self.assertRaises(FileExistsError): self.project()
        self.assertTrue((self.root / "feedback" / "review.md").is_file())


if __name__ == "__main__":
    unittest.main()
