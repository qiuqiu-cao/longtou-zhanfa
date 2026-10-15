import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import historical_symbolic_context as h
import historical_symbolic_research as r
import relay_review_workbench as w
import short_term_thinking_dataset_builder as b


class SymbolicWorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def legacy_prepare(self,**kwargs):
        b.write_new(Path(kwargs["output_dir"])/"reasoning/facts_with_forum_reasoning.json",{"rows":[]})
        return {"rows":[]}

    def test_cli_keeps_all_histories_parent_evidence_and_masked_layers(self):
        args=["prepare"]
        for key in ["features","theme-report","mainline-report","compatibility","bank","calendar","pool","relation-bank"]:
            args.extend(["--"+key,"input-"+key])
        args += ["--output-dir",str(self.root/"out"),"--history-dataset","h1","--history-dataset","h2",
            "--mainline-history","m1","--mainline-history","m2","--narrative-reviews","narrative",
            "--identity-evidence","parent-evidence","--thinking-bundle","causal","--symbolic-context","symbolic"]
        with patch.object(w.forum,"prepare",side_effect=self.legacy_prepare) as original,patch.object(w.identity,"attach",return_value={"rows":[]}), \
             patch.object(w.thinking,"attach",return_value={"rows":[]}) as thinking,patch.object(w.symbolic,"attach_review",return_value={"rows":[]}) as symbolic, \
             patch.object(w,"render"),patch.object(w.original,"sha",return_value="digest"):
            w.main(args)
        self.assertEqual(original.call_args.kwargs["history_datasets"],("h1","h2"))
        self.assertEqual(thinking.call_args.kwargs["mainline_history"],("m1","m2"))
        self.assertEqual(thinking.call_args.kwargs["narrative_reviews"],"narrative")
        self.assertEqual(thinking.call_args.kwargs["identity_evidence"],"parent-evidence")
        self.assertEqual(symbolic.call_args.args[1:3],(Path("input-features"),"symbolic"))

    def test_symbolic_prepare_requires_original_masked_causal_layer(self):
        with patch.object(w.forum,"prepare",side_effect=self.legacy_prepare),patch.object(w,"render"):
            with self.assertRaisesRegex(ValueError,"masked causal"):
                w.prepare(*(["input"]*6),self.root/"failed","pool","bank",symbolic_context="artifact")
        self.assertTrue((self.root/"failed/prepare_failure.json").exists())

    def test_symbolic_guard_failure_precedes_original_freeze(self):
        facts=self.root/"facts.json";b.write_new(facts,{})
        with patch.object(w.symbolic,"checked_review",side_effect=ValueError("changed roles")),patch.object(w.forum,"freeze") as original:
            with self.assertRaisesRegex(ValueError,"changed roles"):
                w.freeze(facts,"review","output")
        original.assert_not_called()

    def test_rehashed_fabricated_count_is_rejected_by_source_reconstruction(self):
        rows=[dict(code="000001",name="甲",boards=2,symbolic_primary_first_support=99)]
        path=self.root/"context.json"
        b.write_new(path,dict(version=h.VERSION,policy=h.POLICY,signal_date="2026-10-15",rows=rows,
            features_path="original-pack",artifact_path="artifact",source_bindings={}))
        facts=dict(symbolic_context_required=True,symbolic_context_path=str(path),signal_date="2026-10-15",rows=rows,
            source_bindings={str(path.resolve()):b.sha(path)})
        actual=copy.deepcopy(rows);actual[0]["symbolic_primary_first_support"]=1
        with patch("short_term_thinking_workbench.checked",return_value={"rows":rows}),patch.object(h,"context_for_review",return_value={"rows":actual}):
            with self.assertRaisesRegex(ValueError,"source reconstruction"):
                h.checked_review(facts)

    def test_pending_review_cannot_read_a_saved_model_or_display_scores(self):
        facts=self.root/"facts.json";review=self.root/"pending.json"
        b.write_new(facts,{});b.write_new(review,{"author_review_completed":False})
        with patch.object(h,"checked_review",return_value={}),patch("short_term_thinking_workbench.checked",return_value={}),patch.object(r.joblib,"load") as model:
            with self.assertRaisesRegex(ValueError,"completed independent review"):
                r.score_review(facts,review,"model",self.root/"scores.json")
        model.assert_not_called()


if __name__=="__main__":unittest.main()
