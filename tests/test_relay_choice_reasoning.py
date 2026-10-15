import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import relay_choice_reasoning as r


ROOT = Path(__file__).resolve().parents[1]


def fixture():
    names = r.RELAY_FIELDS
    codes = ("001001", "001002", "001003")
    model, details, facts, authored = [], [], [], []
    for i, code in enumerate(codes):
        logit = (1.0, .25, -.5)[i]
        score = 1/(1+math.exp(-logit))
        terms = [{"field": field, "signal_value": (1 if i == 0 else 0) if field == "signal_one_word" else 0,
                  "imputed_value": 0, "training_mean": 0,
                  "logit_contribution": logit if field == "signal_one_word" else 0} for field in names]
        actual = 1.0 if i < 2 else 0.0
        m = {"code": code, "name": "fixture"+code, "boards": 3, "actual_one_word_daily": actual,
             "raw_score": score, "predicted_positive": score > .5}
        model.append(m)
        details.append({**m, "logit": logit, "intercept": 0, "terms": terms, "missing_fields": []})
        facts.append({"code": code, "name": m["name"], "boards": 3, "actual_one_word_daily": actual,
                      "same_height_peers": [c for c in codes if c != code]})
        authored.append({"code": code, "promotion_judgement": "UNKNOWN" if i == 2 else "PROMOTE",
                         "role_hypothesis": "unknown role", "support": "observed fact", "rebuttal": "alternative",
                         "cancel_condition": "condition", "opening_confirmation": {k: "pending" for k in ("09:25", "09:35", "10:00")}, "missing": ["actual path"]})
    header = {"signal_date": "2026-10-13", "target_date": "2026-10-14"}
    base = {**header, "knowledge_cutoff": "2026-10-13T08:00:00+00:00", "model_version": "fixture",
            "market": {"prediction": "WORSEN"}, "relay": {"candidates": model, "ranking": list(codes)}}
    exp = {**header, "status": "SOURCE_REBUILT_EXACT_ORIGINAL_LOGITS", "original_knowledge_cutoff": base["knowledge_cutoff"], "rows": details}
    f = {**header, "rows": facts}
    a = {**header, "model_exposure": "MODEL_OUTPUT_ALREADY_SEEN", "rows": authored}
    return base, exp, f, a


class ReasoningContractTests(unittest.TestCase):
    def test_complete_pairs_reconstruct_each_model_gap(self):
        report = r.project(*fixture())
        self.assertEqual(report["candidate_count"], 3)
        self.assertEqual(sum(len(x["all_peer_model_comparisons"]) for x in report["rows"]), 6)
        for row in report["rows"]:
            for pair in row["all_peer_model_comparisons"]:
                self.assertAlmostEqual(sum(x["logit_difference"] for x in pair["all_term_differences"]), pair["logit_gap"])

    def test_actual_one_word_not_replaced_by_proxy(self):
        rows = r.project(*fixture())["rows"]
        self.assertTrue(rows[1]["proxy_actual_identity_disagrees"])
        pair = next(x for x in rows[0]["all_peer_model_comparisons"] if x["peer_code"] == rows[1]["code"])
        self.assertTrue(pair["proxy_differs_while_actual_one_word_same"])
        self.assertFalse(pair["market_core_absorption_or_entry_cause_confirmed"])

    def test_abstention_and_empty_nonword_primary_are_preserved(self):
        report = r.project(*fixture())
        self.assertIsNone(report["non_one_word_positive_first"])
        self.assertEqual(report["non_one_word_relative_first"], "001003")
        self.assertEqual(report["rows"][-1]["independent_call"], "UNKNOWN")
        self.assertIsNone(report["trade_primary"])

    def test_render_keeps_below_threshold_relative_watch_and_original_call(self):
        report = r.project(*fixture())
        text = r.render(report, {"sources": []})
        self.assertIn("非一字相对首位（待证据审阅）：fixture001003 001003", text)
        self.assertIn("非一字相对首位（待审）", text)
        self.assertIn("同板位非一字排第1", text)
        self.assertFalse(report["rows"][-1]["model_positive"])
        self.assertFalse(report["score_call_order_changed"])
        self.assertEqual(report["rows"][-1]["independent_call"], "UNKNOWN")

    def test_wrong_date_and_reordered_rank_are_rejected(self):
        for kind in ("date", "rank"):
            b, e, f, a = fixture()
            if kind == "date":
                e["target_date"] = "2026-10-15"
            else:
                b["relay"]["ranking"].reverse()
            with self.assertRaises(ValueError):
                r.project(b, e, f, a)

    def test_cannot_delete_candidate_or_same_height_peer(self):
        for kind in ("candidate", "peer"):
            b, e, f, a = fixture()
            if kind == "candidate":
                a["rows"].pop()
            else:
                f["rows"][0]["same_height_peers"].pop()
            with self.assertRaises(ValueError):
                r.project(b, e, f, a)

    def test_changed_call_or_logit_sum_cannot_be_presentation(self):
        for kind in ("call", "term"):
            b, e, f, a = fixture()
            if kind == "call":
                b["relay"]["candidates"][0]["predicted_positive"] = False
                e["rows"][0]["predicted_positive"] = False
            else:
                e["rows"][0]["terms"][0]["logit_contribution"] += .1
            with self.assertRaises(ValueError):
                r.project(b, e, f, a)

    def test_render_explains_positive_terms_as_score_direction(self):
        report = r.project(*fixture())
        text = r.render(report, {"sources": []})
        self.assertIn("把分数推向晋级侧", text)
        self.assertIn("共同失败", text)
        self.assertIn("真实一字身份相同", text)
        self.assertNotIn("晋级概率=", text)


class ActualFrozenEstimatorTests(unittest.TestCase):
    def setUp(self):
        self.base = ROOT/"results/daily_review_20261013T074936Z/model/base_predictions.json"
        self.exp = ROOT/"results/daily_review_20261013T074936Z/formal_completion_20261013T111804Z/original_full10_logit_explanations.json"
        self.record = ROOT/"results/daily_review_20261013T074936Z/independent_source_repaired/record_before_scores.json"
        self.bank = ROOT/"research/model_upgrade_validation/choice_reasoning_20261013T113350Z/checked_reasoning_expression_lessons.json"
        self.registered = ROOT/"data/v271/expectation_registry/2026-10-13_47f5c10d3a45999a.json"

    def test_self_consistent_forged_terms_rejected_by_real_estimator(self):
        data = json.loads(self.exp.read_text(encoding="utf-8"))
        data["rows"][0]["terms"][0]["logit_contribution"] += .1
        data["rows"][0]["terms"][1]["logit_contribution"] -= .1
        with tempfile.TemporaryDirectory() as d:
            changed = Path(d)/"fake_explanation.json"
            changed.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "individual model contribution changed"):
                r.build(self.base, changed, self.record, self.bank, Path(d)/"out", registered_path=self.registered)
            self.assertFalse((Path(d)/"out/report.json").exists())

    def test_forged_observed_value_rejected_even_if_score_unchanged(self):
        data = json.loads(self.exp.read_text(encoding="utf-8"))
        term = next(x for x in data["rows"][0]["terms"] if x["field"] == "turnover")
        term["signal_value"] += .5
        with tempfile.TemporaryDirectory() as d:
            changed = Path(d)/"fake_observation.json"
            changed.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "individual model contribution changed"):
                r.build(self.base, changed, self.record, self.bank, Path(d)/"out", registered_path=self.registered)

    def test_forum_quote_must_match_author_source_span(self):
        bank = json.loads(self.bank.read_text(encoding="utf-8"))
        bank["lessons"][0]["anchor"]["text"] = "unquoted absolute guarantee"
        with tempfile.TemporaryDirectory() as d:
            changed = Path(d)/"fake_bank.json"
            changed.write_text(json.dumps(bank, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact checked author span"):
                r.check_forum_bank(changed)


if __name__ == "__main__":
    unittest.main()
