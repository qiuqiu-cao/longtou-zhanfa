import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import stock_judgement_review as review


class JudgementIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.feature_dir = self.root / "features"
        self.feature_dir.mkdir()
        self.feature_file = self.feature_dir / "features.json"
        original = {"signal_date": "2026-10-08", "target_date": "2026-10-09",
            "calendar": {"sessions": ["2026-10-08", "2026-10-09", "2026-10-12"]},
            "relay": [{"code": code, "name": name, "boards": 2, "actual_one_word_daily": word,
                       "features": {"turnover": .1, "signal_amount": 1e8}}
                      for code, name, word in (("000001", "甲", 0.), ("000002", "乙", 1.))]}
        self.feature_file.write_text(json.dumps(original), encoding="utf-8")
        self.facts = {"version": review.VERSION, "prepared_at": "2026-10-08T12:00:00+00:00",
            "signal_date": "2026-10-08", "target_date": "2026-10-09", "sessions": original["calendar"]["sessions"],
            "feature_archive": str(self.feature_dir), "compatibility_policy": "unused_in_freeze",
            "source_bindings": review.bind(self.feature_file), "pool_size": 2,
            "rows": [{"code": c["code"], "name": c["name"], "boards": 2,
                      "actual_one_word_daily": c["actual_one_word_daily"],
                      "daily_price_volume": {k: c["features"].get(k) for k in review.PRICE_FIELDS},
                      "same_height_peers": ["000002" if c["code"] == "000001" else "000001"]}
                     for c in original["relay"]]}
        self.draft = {"model_exposure": "MODEL_OUTPUT_ALREADY_SEEN",
            "rows": [{"code": c["code"], "promotion_judgement": "PROMOTE", "role_hypothesis": "待验证角色",
                "relative_thesis": "同身位比较尚需竞价", "support": "当日收盘事实",
                "rebuttal": "目标竞价未发生", "cancel_condition": "竞争者胜出则取消关注",
                "missing": ["分钟承接"], "comparison_peers": c["same_height_peers"],
                "fact_references": [str(self.feature_file.resolve())],
                "opening_confirmation": {k: "待发生；只追加确认" for k in ("09:25", "09:35", "10:00")}}
                for c in self.facts["rows"]]}
        self.facts_path = self.root / "facts.json"
        self.draft_path = self.root / "draft.json"
        self.facts_path.write_text(json.dumps(self.facts), encoding="utf-8")
        self.draft_path.write_text(json.dumps(self.draft), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_candidates_and_complete_same_height_peers_required(self):
        draft = copy.deepcopy(self.draft)
        draft["rows"].pop()
        with self.assertRaisesRegex(ValueError, "all original candidates"):
            review.validate_judgements(self.facts, draft)
        draft = copy.deepcopy(self.draft)
        draft["rows"][0]["comparison_peers"] = []
        with self.assertRaisesRegex(ValueError, "same-height"):
            review.validate_judgements(self.facts, draft)

    def test_score_seen_cannot_be_declared_as_verified_blind(self):
        draft = copy.deepcopy(self.draft)
        draft["model_exposure"] = "BLIND_VERIFIED"
        with self.assertRaisesRegex(ValueError, "declare whether"):
            review.validate_judgements(self.facts, draft)

    def test_changed_fact_projection_rejected_even_if_draft_matches(self):
        facts = copy.deepcopy(self.facts)
        facts["rows"].pop()
        facts["pool_size"] = 1
        with self.assertRaisesRegex(ValueError, "lost original candidates"):
            review.validate_fact_projection(facts)
        facts = copy.deepcopy(self.facts)
        facts["rows"][0]["daily_price_volume"]["turnover"] = 0
        with self.assertRaisesRegex(ValueError, "price/volume changed"):
            review.validate_fact_projection(facts)

    def test_actual_preauction_gate_and_next_session(self):
        sessions = self.facts["sessions"]
        for target, moment in (("2026-10-12", "2026-10-08T13:00:00+00:00"),
                               ("2026-10-09", "2026-10-09T01:15:00+00:00"),
                               ("2026-10-09", "2026-10-08T06:00:00+00:00")):
            with self.assertRaises(ValueError):
                review.pre_auction("2026-10-08", target, sessions, review.stamp(moment))

    def test_unknown_not_failure_and_abstention_does_not_hide_recall(self):
        metrics = review.metrics({"a": None, "b": True, "c": False, "d": False},
                                 {"a": True, "b": True, "c": False, "d": None})
        self.assertEqual(metrics["Accuracy"], 1.)
        self.assertEqual(metrics["complete_positive_Recall"], .5)
        self.assertEqual(metrics["positive_abstentions"], 1)
        self.assertEqual(metrics["unknown_labels"], 1)
        self.assertEqual(metrics["scoring_coverage"], .5)
        self.assertEqual(metrics["correct_fraction_of_all_known_labels"], 2/3)
        self.assertIsNone(metrics["Brier"])

    def test_reference_and_reasoning_gaps_rejected(self):
        draft = copy.deepcopy(self.draft)
        draft["rows"][0]["fact_references"] = ["current_unsaved_web_page"]
        with self.assertRaisesRegex(ValueError, "outside checked"):
            review.validate_judgements(self.facts, draft)
        draft = copy.deepcopy(self.draft)
        draft["rows"][0]["rebuttal"] = ""
        with self.assertRaisesRegex(ValueError, "independent reasoning"):
            review.validate_judgements(self.facts, draft)

    def test_freeze_registers_once_and_does_not_claim_production_or_blinding(self):
        with patch.object(review, "LEDGER", self.root / "registry.jsonl"), patch.object(review, "now",
                return_value=datetime(2026, 10, 8, 13, tzinfo=timezone.utc)):
            record_path = self.root / "record.json"
            record = review.freeze(self.facts_path, self.draft_path, record_path)
            self.assertEqual(len(review.registrations()), 1)
            self.assertEqual(review.registrations()[0]["sha256"], review.sha(record_path))
            self.assertFalse(record["blinding_verified"])
            self.assertFalse(record["original_fixed_shadow_or_production_registration"])
            with self.assertRaisesRegex(ValueError, "already registered"):
                review.freeze(self.facts_path, self.draft_path, self.root / "another.json")

    def test_freeze_rejects_changed_sources(self):
        self.feature_file.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source changed"):
            review.freeze(self.facts_path, self.draft_path, self.root / "record.json")

    def test_write_new_preserves_original(self):
        target = self.root / "exclusive.json"
        review.write_new(target, {"original": True})
        with self.assertRaises(FileExistsError):
            review.write_new(target, {"original": False})
        self.assertEqual(review.read(target), {"original": True})

    def test_compare_pairs_full_pool_and_allows_baseline_formed_after_masked_judgement(self):
        with patch.object(review, "LEDGER", self.root / "registry.jsonl"), patch.object(review, "now",
                return_value=datetime(2026, 10, 8, 13, tzinfo=timezone.utc)):
            record_path = self.root / "record.json"
            review.freeze(self.facts_path, self.draft_path, record_path)
            baseline = {"signal_date": "2026-10-08", "target_date": "2026-10-09",
                "generated_at": "2026-10-08T13:01:00+00:00", "knowledge_cutoff": "2026-10-08T12:59:00+00:00",
                "feature_sha256": review.sha(self.feature_file), "source_hashes": review.bind(self.feature_file),
                "relay": {"candidates": [{"code": "000001", "predicted_positive": False},
                                          {"code": "000002", "predicted_positive": True}]}}
            baseline_path = self.root / "baseline.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            # Synthetic interface fixture only; actual labels always use build_outcomes.
            adapter = SimpleNamespace(build_outcomes=lambda record, archive: {
                "candidates": {"000001": True, "000002": False}, "sources": {}})
            with patch("model_upgrade_compat.runtime", return_value=(adapter, None)):
                result = review.compare(record_path, baseline_path, "synthetic_fixture", self.root / "comparison.json")
            self.assertEqual(result["metrics"]["independent"]["TP"], 1)
            self.assertEqual(result["metrics"]["BASE"]["FN"], 1)
            self.assertEqual(len(result["rows"]), 2)
            self.assertEqual(result["comparability"], "FULL_COVERAGE")
            self.assertFalse(result["automatic_model_or_top5_change_allowed"])
            record = review.read(record_path)
            record["rows"][0]["promotion_judgement"] = "FAIL"
            record_path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "original judgement changed"):
                review.compare(record_path, baseline_path, "synthetic_fixture", self.root / "changed.json")


if __name__ == "__main__":
    unittest.main()
