"""Isolated contract fixtures only; never new financial predictions or scores."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import hypothesis_review_v298 as review
from daily_research_report_v292 import render as old_render
from daily_research_report_v298 import render, hypothesis_table
from daily_research_template_v298 import draft_extension
from point_context_v295 import sha

ROOT = Path(__file__).resolve().parent
RECEIPT = ROOT / "results/v297/experience_proxy_acceptance_20261002_v1/report.json"
ORIGINAL = ROOT / "data/v271/expectation_registry/2026-09-30_2c74dc406eef7dcf.json"


class HypothesisReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
        cls.loaded = review.load_review(RECEIPT)

    def fixture(self, available=True):
        plan = deepcopy(self.original["plan"])
        # Reusing factual fields is only a unit fixture. This is not registered,
        # not represented as an independent forecast, and not scored.
        if available:
            plan["knowledge_cutoff"] = datetime.now(timezone.utc).isoformat()
        plan["daily_research"] = draft_extension(plan)
        return plan

    def bound(self):
        return review.attach(self.fixture(), RECEIPT)

    def test_template_represents_all_fifteen_without_fabricated_evidence(self):
        plan = self.fixture()
        self.assertEqual([r["id"] for r in plan["daily_research"]["hypotheses"]], review.IDS)
        self.assertFalse(any(r["backtest_completed"] or r["core_rule"] for r in plan["daily_research"]["hypotheses"]))
        self.assertEqual(review.validate_hypotheses(plan, True)["status"], "FIFTEEN_UNTESTED_HYPOTHESES_REPRESENTED")

    def test_missing_duplicate_and_reordered_hypotheses_rejected(self):
        original = self.fixture()
        for mutation in [lambda x: x.pop(), lambda x: x.__setitem__(1, deepcopy(x[0])), lambda x: x.reverse()]:
            plan = deepcopy(original)
            mutation(plan["daily_research"]["hypotheses"])
            with self.assertRaisesRegex(ValueError, "all fifteen"):
                review.validate_hypotheses(plan, True)

    def test_old_grouped_template_cannot_enter_new_freeze(self):
        from daily_research_template_v292 import draft_extension as grouped
        plan = self.fixture(); plan["daily_research"] = grouped(plan)
        self.assertEqual(review.validate_hypotheses(plan)["status"], "LEGACY_HYPOTHESES_UNCHANGED")
        with self.assertRaisesRegex(ValueError, "new freezes require"):
            review.validate_hypotheses(plan, True)

    def test_binding_keeps_calls_probabilities_and_every_other_field(self):
        plan = self.fixture(); before = deepcopy(plan)
        attached = review.attach(plan, RECEIPT)
        for key in ["candidates", "market", "bands", "signal_date", "target_date", "knowledge_cutoff"]:
            self.assertEqual(attached[key], before[key])
        for key, value in before["daily_research"].items():
            if key != "hypotheses": self.assertEqual(attached["daily_research"][key], value)
        self.assertEqual(plan, before)
        validated = review.validate_hypotheses(attached, True)
        self.assertEqual(validated["tested_proxies"], 7)
        self.assertFalse(validated["probability_authorized"])

    def test_actual_later_research_cannot_backfill_original_cutoff(self):
        with self.assertRaisesRegex(ValueError, "arrived after knowledge cutoff"):
            review.attach(self.fixture(available=False), RECEIPT)

    def test_audited_availability_uses_receipt_not_earlier_bank_time(self):
        plan = self.fixture()
        bank = json.loads(Path(self.loaded["bank"]).read_text(encoding="utf-8"))
        plan["knowledge_cutoff"] = bank["available_at"]
        with self.assertRaisesRegex(ValueError, "arrived after knowledge cutoff"):
            review.attach(plan, RECEIPT)

    def test_signal_and_target_label_boundaries_rejected(self):
        for field, value in [("signal_date", "2026-09-29"), ("target_date", "2026-09-30")]:
            plan = self.fixture(); plan[field] = value
            with self.assertRaisesRegex(ValueError, "labels after signal date"):
                review.attach(plan, RECEIPT)

    def test_bound_grade_status_definition_and_permissions_tamper_rejected(self):
        original = self.bound()
        changes = [("grade", "A"), ("status", "CONFIRMED_EDGE"), ("definition", "selected winners"), ("core_rule", True), ("backtest_completed", False)]
        for field, value in changes:
            plan = deepcopy(original); plan["daily_research"]["hypotheses"][0][field] = value
            with self.assertRaisesRegex(ValueError, "differ from audited source"):
                review.validate_hypotheses(plan)
        plan = deepcopy(original); plan["hypothesis_binding"]["probability_authorized"] = True
        with self.assertRaisesRegex(ValueError, "permissions differ"):
            review.validate_hypotheses(plan)

    def test_binding_and_policy_removal_cannot_hide_statistical_claims(self):
        plan = self.bound(); plan.pop("hypothesis_binding")
        with self.assertRaisesRegex(ValueError, "every hypothesis must remain untested"):
            review.validate_hypotheses(plan)
        plan["daily_research"].pop("hypothesis_policy")
        with self.assertRaisesRegex(ValueError, "claims require"):
            review.validate_hypotheses(plan)

    def test_source_artifact_change_rejected_without_mutating_archive(self):
        original_sha = review.sha
        wanted = ROOT / "results/v297/experience_proxy_tests_20261002_v2/hypothesis_masks_NO_NEXT_SESSION_LABELS.csv"
        def changed(path):
            return "0"*64 if Path(path).resolve() == wanted.resolve() else original_sha(path)
        with patch("hypothesis_review_v298.sha", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "research source missing or changed"):
                review.validate_hypotheses(self.bound())

    def test_self_hashed_bank_cannot_override_experiment_results(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
            bank = json.loads(Path(self.loaded["bank"]).read_text(encoding="utf-8"))
            bank["hypotheses"][0]["grade"] = "A"
            bank_path = base / "hypothesis_bank_NOT_LIVE_RULES.json"
            bank_path.write_text(json.dumps(bank), encoding="utf-8")
            receipt["source_hashes"].pop(self.loaded["bank"])
            receipt["source_hashes"][str(bank_path)] = sha(bank_path)
            local = base / "report.json"; local.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bank entries differ"):
                review.load_review(local)

    def test_reviewed_or_already_bound_draft_cannot_be_overwritten(self):
        plan = self.fixture(); plan["daily_research"]["draft_status"] = "AUTHOR_REVIEWED"
        with self.assertRaisesRegex(ValueError, "only an unreviewed"):
            review.attach(plan, RECEIPT)
        with self.assertRaisesRegex(ValueError, "do not overwrite"):
            review.attach(self.bound(), RECEIPT)

    def test_legacy_report_is_exactly_unchanged(self):
        report = {"recorded_at": datetime.now(timezone.utc).isoformat(), "statistics": {"reason": "ISOLATED UNIT FIXTURE"}}
        self.assertEqual(render(self.original, report), old_render(self.original, report))
        self.assertEqual(sha(ORIGINAL), "bfbaeb295637451e396850ee583fcafa3eb0ab6a99fb1b154e4c46a8c23aa40c")

    def test_new_report_contains_each_hypothesis_with_unknowns_and_hashes(self):
        registered = deepcopy(self.original); registered["plan"] = self.bound()
        report = {"recorded_at": datetime.now(timezone.utc).isoformat(), "statistics": {"reason": "ISOLATED UNIT FIXTURE"}}
        output = render(registered, report)
        self.assertEqual(output.count("【"), 11)
        self.assertIn("未检验 / 时点输入不足", output)
        self.assertIn(self.loaded["available_at"], output)
        self.assertIn(self.loaded["hypotheses"][0]["statistical_evidence"]["source_sha256"], output)
        for item in self.loaded["hypotheses"]:
            self.assertIn("| " + item["name"] + " | " + item["grade"] + " |", output)

    def test_freeze_cli_rejects_incomplete_policy_before_any_registration(self):
        import hypothesis_bound_daily_runner_v298 as runner
        plan = self.fixture(); plan["daily_research"]["hypotheses"].pop()
        with patch("hypothesis_bound_daily_runner_v298.validate_extension", return_value={}), patch("pathlib.Path.read_text", return_value=json.dumps(plan)), patch("sys.argv", ["runner", "freeze", "--plan", "ISOLATED_NOT_FORECAST.json"]), patch("hypothesis_bound_daily_runner_v298.register") as register:
            with self.assertRaisesRegex(ValueError, "all fifteen"):
                runner.main()
            register.assert_not_called()


if __name__ == "__main__":
    unittest.main()
