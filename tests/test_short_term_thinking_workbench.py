import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import pandas as pd

import short_term_thinking_dataset_builder as b
import short_term_thinking_workbench as t
import relay_review_workbench as w


class ThinkingWorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.pack=self.root/"features.json"
        b.write_new(self.pack,dict(signal_date="2026-10-14",target_date="2026-10-15",relay=[dict(code="000001",name="甲",boards=2,features={"boards":2})]))
        self.facts=dict(signal_date="2026-10-14",target_date="2026-10-15",rows=[dict(code="000001",name="甲",boards=2)],source_bindings={})
        self.fp=self.root/"facts.json";b.write_new(self.fp,self.facts)
        self.bundle=self.root/"bundle.joblib"
        joblib.dump(dict(features=pd.DataFrame([dict(date="2026-10-14",code="000001",name="甲",boards=2,primary_role="RELAY_CONTENDER",roles=[],cycle_regime="BALANCED",feature_cutoff="2026-10-14")]),sources={str(self.pack.resolve()):b.sha(self.pack)}),self.bundle)

    def test_attached_context_preserves_entire_legacy_rows_and_masks_scores(self):
        actual=t.attach(self.fp,self.pack,self.bundle,self.root/"attached")
        self.assertEqual(actual["rows"],self.facts["rows"])
        projection=t.checked(actual)
        self.assertTrue(projection["model_scores_masked"])
        self.assertFalse(set(projection["rows"][0]) & {"score","target","raw_score","prediction","promotion_judgement"})

    def test_cache_cannot_be_reused_for_changed_price_pack(self):
        copy_path=self.root/"changed.json"
        pack=b.read(self.pack);pack["relay"][0]["features"]["price"]=999
        b.write_new(copy_path,pack)
        with self.assertRaisesRegex(ValueError,"exact source"):
            t.context(copy_path,self.bundle)

    def test_changed_context_is_rejected_before_freeze(self):
        actual=t.attach(self.fp,self.pack,self.bundle,self.root/"attached")
        path=Path(actual["causal_thinking_context_path"])
        path.write_text("{}",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"binding changed"):
            t.checked(actual)

    def test_pending_review_cannot_display_model_scores(self):
        t.attach(self.fp,self.pack,self.bundle,self.root/"attached")
        pending=self.root/"pending.json";b.write_new(pending,{"author_review_completed":False})
        with self.assertRaisesRegex(ValueError,"review must be archived"):
            t.score_context(self.root/"attached/facts_with_causal_thinking.json",pending,"unused-model",self.root/"scores.json")

    def test_prepare_forwards_every_history_input_to_new_engine(self):
        def original_prepare(**kwargs):
            b.write_new(Path(kwargs["output_dir"])/"reasoning/facts_with_forum_reasoning.json",{"rows":[]})
            return {"rows":[]}
        with patch.object(w.forum,"prepare",side_effect=original_prepare),patch.object(w.thinking,"attach",return_value={"rows":[]}) as attach,patch.object(w,"render"),patch.object(w.original,"sha",return_value="sha"):
            w.prepare(*(["input"]*6),self.root/"prepared","pool","bank",thinking_bundle="causal-history",history_datasets=["h1","h2"],mainline_history=["m1","m2"],narrative_reviews="narratives")
        self.assertEqual(attach.call_args.kwargs["history_datasets"],("h1","h2"))
        self.assertEqual(attach.call_args.kwargs["mainline_history"],("m1","m2"))
        self.assertEqual(attach.call_args.kwargs["narrative_reviews"],"narratives")


if __name__=="__main__":unittest.main()
