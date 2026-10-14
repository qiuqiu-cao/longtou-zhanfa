import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import stock_judgement_review as o
import supplement_relation_review as relation
import supplement_role_review as review
from test_supplement_relation_review import fixture
REAL_RELATION_BUILD = relation.build


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.pool,self.bank,_ = fixture(self.out)
        self.calls = []
        self.calendar = self.out/"calendar.json"
        # Source qualification itself is exercised by the relation tests and
        # the real full15 replay. This stub isolates workflow ordering/gates.
        validator = patch.object(review.catalyst,"validate_bank",side_effect=self.checked_bank)
        validator.start();self.addCleanup(validator.stop)

    def checked_bank(self,*args):
        bank = o.read(self.bank)
        sources = {s["source_id"]:s for s in bank["sources"]}
        claims = [{**c,"scope":"STOCK","code":"000002","kind":"NARRATIVE"} for c in bank["claims"]]
        return bank,sources,claims,o.bind(self.bank) | o.bind(self.calendar)

    def forum(self,pool,bank,calendar,out,exposure):
        self.calls.append("forum")
        p = o.read(pool)
        _,sources,claims,proof = self.checked_bank()
        rows = review.preread.project(p["rows"],claims,sources,p["signal_date"])
        r = {"rows":rows,"source_bindings":o.bind(pool) | proof,"formed_at":o.now().isoformat(),"model_exposure":exposure}
        o.write_new(out/"report.json",r)
        return r

    def relations(self,pool,bank,out,exposure):
        self.calls.append("relations")
        return REAL_RELATION_BUILD(pool,bank,out,exposure=exposure)

    def run_context(self):
        with patch.object(review.preread,"build",side_effect=self.forum),patch.object(review.relation,"build",side_effect=self.relations):
            return review.context(self.pool,self.bank,self.bank,self.calendar,self.out/"run")

    def complete_draft(self,facts):
        path = self.out/"run"/"facts_with_supplement.json"
        draft = o.read(self.out/"run"/"author_supplement_PENDING.json")
        draft["author_review_completed"] = True
        for row in draft["rows"]:
            row["forum_preread_assessment"].update(reviewed=True,reason="原文覆盖有限，不以未知看空")
            a = row["supplement_assessment"];a.update(reviewed=True,reason="关系假设保留独立原因")
            for edge in a["relations"]:
                edge.update(decision="PENDING",reason="保留线索，因果未知")
        return path,draft

    def test_actual_sequence_full_pool_no_autofilled_direction(self):
        facts = self.run_context()
        self.assertEqual(self.calls,["forum","relations"])
        self.assertEqual(facts["pool_size"],2)
        draft = o.read(self.out/"run"/"author_supplement_PENDING.json")
        self.assertTrue(all(x["promotion_judgement"] is None for x in draft["rows"]))
        self.assertFalse(facts["forecast_registration_allowed"])

    def test_diagnostic_context_cannot_register_even_after_review(self):
        facts = self.run_context();path,draft = self.complete_draft(facts)
        draft_path = self.out/"completed.json";o.write_new(draft_path,draft)
        with patch.object(review.legacy,"freeze") as freeze:
            with self.assertRaisesRegex(ValueError,"diagnostic"):
                review.freeze(path,draft_path,self.out/"forecast.json")
            freeze.assert_not_called()
        result = review.review_context(path,draft_path,self.out/"review.json")
        self.assertFalse(result["forecast_registered"])

    def test_completed_diagnostic_cannot_rewrite_old_direction(self):
        facts = self.run_context();path,draft = self.complete_draft(facts)
        draft["rows"][0]["promotion_judgement"] = "PROMOTE"
        draft_path = self.out/"bad.json";o.write_new(draft_path,draft)
        with self.assertRaisesRegex(ValueError,"retroactive"):
            review.review_context(path,draft_path,self.out/"review.json")

    def test_pending_and_omitted_forum_review_rejected(self):
        facts = self.run_context();path,draft = self.complete_draft(facts)
        draft["rows"][1]["forum_preread_assessment"]["reviewed"] = False
        with self.assertRaisesRegex(ValueError,"forum pre-read review"):
            review.validate_review(facts,draft)
        draft["author_review_completed"] = False
        with self.assertRaisesRegex(ValueError,"pending"):
            review.validate_review(facts,draft)

    def test_row_context_tampering_and_missing_stage_rejected(self):
        facts = self.run_context()
        changed = copy.deepcopy(facts)
        changed["rows"][0]["supplement_relation_context"]["relations"] = []
        with self.assertRaisesRegex(ValueError,"projection"):
            review.checked(changed)
        changed = copy.deepcopy(facts);changed["preparation_sequence"].remove("FORUM_PREREAD")
        with self.assertRaisesRegex(ValueError,"stage"):
            review.checked(changed)

    def test_pre_read_failure_stops_before_relations_or_legacy(self):
        with patch.object(review.preread,"build",side_effect=ValueError("bad source")),patch.object(review.relation,"build") as rel,patch.object(review.legacy,"prepare") as legacy:
            with self.assertRaisesRegex(ValueError,"bad source"):
                review.prepare("f","t","m","c","b","calendar",self.out/"run",self.pool,self.bank)
            rel.assert_not_called();legacy.assert_not_called()

    def test_new_prepare_calls_legacy_after_source_stages(self):
        def legacy(*args):
            self.calls.append("legacy")
            return {"rows":o.read(self.pool)["rows"],"source_bindings":{},"signal_date":"2026-10-14","target_date":"2026-10-15"}
        def template(facts,path):
            return {"rows":[{"code":x["code"],"promotion_judgement":None} for x in facts["rows"]]}
        with patch.object(review.preread,"build",side_effect=self.forum),patch.object(review.relation,"build",side_effect=self.relations),patch.object(review.legacy,"prepare",side_effect=legacy),patch.object(review.legacy,"_draft_template",side_effect=template):
            facts = review.prepare("f","t","m","c",self.bank,self.calendar,self.out/"run",self.pool,self.bank)
        self.assertEqual(self.calls,["forum","relations","legacy"])
        self.assertFalse(facts["supplement_context_diagnostic_only"])


if __name__ == "__main__":
    unittest.main()
