"""Argument-review contracts; synthetic fixtures are not prediction evidence."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import stock_judgement_review as o
import forum_reasoning_review as f


class ArgumentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.text = self.out/"forum.txt";self.text.write_text("甲是补涨候选吗？甲需要换手后确认。",encoding="utf-8")
        self.market = self.out/"raw_daily.json";o.write_new(self.market,{"date":"2026-10-14","close":100})
        first = {"source_id":"s","source_section":"AUTHOR_BODY","offset":0,"excerpt":"甲是补涨候选吗？",
            "source_claim_scope":"AUTHOR_ASKS_ROLE_NOT_ASSERTED","text_path":str(self.text),"use":"OLDER_DISCUSSION_BACKGROUND"}
        second = {"source_id":"s","source_section":"AUTHOR_BODY","offset":9,"excerpt":"甲需要换手后确认。",
            "text_path":str(self.text),"use":"OLDER_DISCUSSION_BACKGROUND"}
        for p in (first,second):
            p["opinion_id"] = f.opinion_id(p)
        self.context = {"version":f.VERSION,"signal_date":"2026-10-14","target_date":"2026-10-15",
            "rows":[{"code":"000001","name":"甲","boards":2,"opinions":[first,second]},
                    {"code":"000002","name":"乙","boards":2,"opinions":[]}],
            "source_bindings":o.bind(self.text) | o.bind(self.market),"discussion_only_paths":[str(self.text)],
            "verification_paths_by_scope":{"MARKET_OBSERVATION":[str(self.market)],"COMPANY_BUSINESS":[],"EVENT_STATUS":[],"DISCUSSION_PRESENCE":[str(self.text)]}}
        self.draft = f.draft_template(self.context)
        for row in self.draft["rows"]:
            row["promotion_judgement"] = "PROMOTE"
            a = row["forum_reasoning_assessment"]
            a.update(reviewed=True,synthesis="结合已核价格及竞争独立判断，材料未知不是负面。",
                agreement="PARTIAL" if a["opinions"] else "UNDETERMINED",agreement_reason="采纳检查条件，身份因果保持未知。")
            for i,p in enumerate(a["opinions"]):
                p.update(author_claim="作者提出候选身份问题。" if i == 0 else "作者要求换手后确认。",
                    author_reasoning="检验相对承接，而非仅凭名称。",testable_expectation="观察完整竞争中是否延续并维持相对表现。",
                    author_claim_form="QUESTION" if i == 0 else "HYPOTHESIS",author_call="UNSPECIFIED",
                    verification={"status":"UNKNOWN","scope":"ROLE_HYPOTHESIS","reason":"身份缺独立证据，推理可检验。","evidence_refs":[str(self.text)]},
                    decision="PENDING" if i == 0 else "ACCEPT",reason="保留角色问题。" if i == 0 else "采纳确认条件而非认证身份。",
                    counterevidence=["完整同位还有其他强候选。"],independent_effect="UNCERTAIN" if i == 0 else "SUPPORT_PROMOTE",
                    cancel_conditions=["相对承接走弱或无法回封时取消。"])

    def validate(self,draft=None):
        return f.validate(self.context,self.draft if draft is None else draft)

    def opinion(self,index=1):
        return self.draft["rows"][0]["forum_reasoning_assessment"]["opinions"][index]

    def test_unknown_hypothesis_can_inform_independent_prediction_without_role_truth(self):
        self.validate()
        self.assertEqual(self.opinion()["verification"]["status"],"UNKNOWN")
        self.assertEqual(self.opinion()["independent_effect"],"SUPPORT_PROMOTE")
        self.assertEqual(self.draft["rows"][1]["promotion_judgement"],"PROMOTE")

    def test_pending_template_or_just_read_note_is_insufficient(self):
        with self.assertRaises(ValueError):
            self.validate(f.draft_template(self.context))
        self.draft["rows"][0]["forum_reasoning_assessment"] = {"reviewed":True,"reason":"已读"}
        with self.assertRaises(ValueError):
            self.validate()

    def test_missing_duplicate_opinion_and_missing_candidate_rejected(self):
        for change in ("missing","duplicate","candidate"):
            d = copy.deepcopy(self.draft)
            if change == "candidate":
                d["rows"].pop()
            elif change == "missing":
                d["rows"][0]["forum_reasoning_assessment"]["opinions"].pop()
            else:
                d["rows"][0]["forum_reasoning_assessment"]["opinions"].append(copy.deepcopy(self.opinion()))
            with self.assertRaises(ValueError):
                self.validate(d)

    def test_question_cannot_be_explicit_author_prediction(self):
        self.opinion(0).update(author_call="PROMOTE",author_expectation_target_date="2026-10-15")
        with self.assertRaisesRegex(ValueError,"question|forecast"):
            self.validate()

    def test_known_question_cannot_be_relabelled_to_bypass_guard(self):
        self.opinion(0).update(author_claim_form="OPINION",author_call="PROMOTE",author_expectation_target_date="2026-10-13")
        with self.assertRaisesRegex(ValueError,"question|Question"):
            self.validate()

    def test_old_tomorrow_is_not_current_target_prediction(self):
        self.opinion().update(author_call="PROMOTE",author_expectation_target_date="2026-10-15")
        with self.assertRaisesRegex(ValueError,"old tomorrow"):
            self.validate()

    def test_discussion_presence_can_be_verified_but_not_funds_cause(self):
        v = self.opinion()["verification"];v.update(status="VERIFIED",scope="DISCUSSION_PRESENCE")
        self.validate()
        v.update(scope="FUNDS_CAUSALITY",evidence_refs=[str(self.market)])
        with self.assertRaisesRegex(ValueError,"causal|role|fund|certif"):
            self.validate()

    def test_market_facts_need_market_source_not_forum(self):
        v = self.opinion()["verification"];v.update(status="VERIFIED",scope="MARKET_OBSERVATION")
        with self.assertRaises(ValueError):
            self.validate()
        v["evidence_refs"] = [str(self.market)]
        self.validate()

    def test_company_and_event_cannot_be_verified_from_daily_prices(self):
        v = self.opinion()["verification"];v.update(status="VERIFIED",evidence_refs=[str(self.market)])
        for scope in ("COMPANY_BUSINESS","EVENT_STATUS"):
            v["scope"] = scope
            with self.assertRaises(ValueError):
                self.validate()

    def test_pending_argument_not_definite_directional_support(self):
        self.opinion().update(decision="PENDING")
        with self.assertRaisesRegex(ValueError,"pending"):
            self.validate()

    def test_null_reason_outside_ref_and_empty_accepted_cancellation_rejected(self):
        for key in ("reason","refs","cancel"):
            d = copy.deepcopy(self.draft);p = d["rows"][0]["forum_reasoning_assessment"]["opinions"][1]
            if key == "reason":p["reason"] = None
            elif key == "refs":p["verification"]["evidence_refs"] = ["unbound.json"]
            else:p["cancel_conditions"] = []
            with self.assertRaises(ValueError):self.validate(d)

    def test_uncovered_stock_cannot_agree_with_nonexistent_author(self):
        self.draft["rows"][1]["forum_reasoning_assessment"]["agreement"] = "AGREEMENT"
        with self.assertRaisesRegex(ValueError,"uncovered"):
            self.validate()

    def test_diagnostic_freeze_cannot_register_forecast(self):
        facts_path = self.out/"facts.json"
        facts = {"supplement_context_diagnostic_only":True,"source_bindings":self.context["source_bindings"],"supplement_model_exposure":"MODEL_OUTPUT_ALREADY_SEEN"}
        o.write_new(facts_path,facts)
        draft = {**self.draft,"author_review_completed":True,"facts_path":str(facts_path),"facts_sha256":o.sha(facts_path),"model_exposure":"MODEL_OUTPUT_ALREADY_SEEN"}
        draft_path = self.out/"draft.json";o.write_new(draft_path,draft)
        with patch.object(f,"checked",return_value=self.context):
            with self.assertRaisesRegex(ValueError,"diagnostic"):
                f.freeze(facts_path,draft_path,self.out/"forecast.json")
        self.assertFalse((self.out/"forecast.json").exists())

    def test_completed_new_review_still_runs_original_freeze_gates(self):
        facts_path = self.out/"facts.json"
        o.write_new(facts_path,{"supplement_context_diagnostic_only":False,"source_bindings":self.context["source_bindings"],"supplement_model_exposure":"MODEL_OUTPUT_ALREADY_SEEN"})
        draft = {**self.draft,"author_review_completed":True,"facts_path":str(facts_path),"facts_sha256":o.sha(facts_path),"model_exposure":"MODEL_OUTPUT_ALREADY_SEEN"}
        draft_path = self.out/"draft.json";o.write_new(draft_path,draft)
        with patch.object(f,"checked",return_value=self.context),patch.object(f.supplement,"freeze",return_value={"status":"ORIGINAL_GATES"}) as freeze:
            self.assertEqual(f.freeze(facts_path,draft_path,self.out/"forecast.json")["status"],"ORIGINAL_GATES")
            freeze.assert_called_once_with(facts_path,draft_path,self.out/"forecast.json")


if __name__ == "__main__":
    unittest.main()
