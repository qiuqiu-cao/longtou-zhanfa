"""Read-only evidence views composed with the existing review workbench.

Run the original prepare/author/freeze/compare/execution gates as before. This
companion consumes their bound causal facts and full history without changing
original fact rows, directions, registries, model parameters or schedules.
"""
from pathlib import Path
import argparse
import joblib
import pandas as pd

import short_term_thinking_dataset_builder as b
import short_term_thinking_workbench as thinking
import relay_full_cohort_transition as transition
import relay_causal_review_context as semantic


def project(facts_path,bundle_path,review_path=None):
    facts=b.read(facts_path);original=thinking.checked(facts)
    if original is None:raise ValueError("original causal workbench context required")
    engine=joblib.load(bundle_path);b.verify(engine["sources"])
    day=facts["signal_date"];current=pd.DataFrame(original["rows"])
    if set(current.code)!=set(x["code"] for x in facts["rows"]):raise ValueError("complete current pool required")
    if day not in set(engine["attempts"].signal_date):raise ValueError("new date requires its own bound closing/failed source; old pool forbidden")
    # Keep all previous candidates, which initialize cycle and relative position.
    candidates=pd.concat([engine["features"][engine["features"].date.lt(day)],current],ignore_index=True)
    attempts=semantic.normalize_attempts(engine["attempts"],engine["bars"],engine["sessions"])
    transitions=transition.build_transition_context(candidates,attempts,engine["bars"],engine["sessions"])
    edges=[];bindings={}
    if review_path:
        import relay_review_workbench as existing
        record=b.read(review_path);existing.validate_completed(facts_path,record)
        if record.get("facts_sha256")!=b.sha(facts_path) or not record.get("author_review_completed"):raise ValueError("completed review must bind exact facts")
        b.verify(record.get("source_bindings",{}));bindings[str(Path(review_path).resolve())]=b.sha(review_path)
        for row in record["rows"]:
            for parent in row.get("identity_discovery_assessment",{}).get("accepted_anchor_codes",[]):
                edges.append(dict(date=day,code=row["code"],parent_code=parent,accepted=True,hypothesis_public_date=day,
                    source="ARCHIVED_AUTHOR_HYPOTHESIS_NOT_INDEPENDENT_IDENTITY_GOLD"))
    context=semantic.build_for_day(day,candidates,engine["prepared"]["market"],attempts,engine["bars"],engine["sessions"],transitions,author_parent_edges=edges)
    joined=context["rows"].merge(transitions["rows"].drop(columns=["boards"]),on=["date","code"],validate="one_to_one")
    if len(joined)!=len(facts["rows"]):raise ValueError("current row cannot be dropped")
    return dict(formed_at=b.now(),version=semantic.VERSION,signal_date=day,target_date=facts["target_date"],
        rows=b.clean(joined.to_dict("records")),all_active_parent_relations=b.clean(context["all_active_parent_relations"].to_dict("records")),
        policy=semantic.POLICY,model_scores_masked=True,facts_path=str(Path(facts_path).resolve()),facts_sha256=b.sha(facts_path),bundle_path=str(Path(bundle_path).resolve()),
        original_history_datasets=original["history_datasets"],original_mainline_history=original["mainline_history"],original_narrative_reviews=original["narrative_reviews"],
        original_parent_chain_evidence=original["parent_chain_evidence"],review_path=str(Path(review_path).resolve()) if review_path else None,
        role_hypotheses_not_auto_author_calls=True,original_rows_not_modified=True,new_fits=0,new_formal_predictions=0,
        source_bindings=original["source_bindings"]|engine["sources"]|bindings|
            {str(Path(p).resolve()):b.sha(p) for p in [facts_path,bundle_path,__file__,semantic.__file__,transition.__file__]})


def checked(context_path):
    context=b.read(context_path);b.verify(context["source_bindings"])
    if b.sha(context["facts_path"])!=context["facts_sha256"]:raise ValueError("original facts changed")
    actual=project(context["facts_path"],context["bundle_path"],context.get("review_path"))
    for field in ["version","signal_date","rows","all_active_parent_relations","policy","original_history_datasets","original_mainline_history","original_narrative_reviews","original_parent_chain_evidence"]:
        if context[field]!=actual[field]:raise ValueError("causal review context tampering or initialization mismatch: "+field)
    return context


def saved_score_view(facts_path,context_path,review_path,scores_path,output_path):
    import relay_review_workbench as existing
    facts=b.read(facts_path);record=b.read(review_path)
    existing.validate_completed(facts_path,record)
    if not record.get("author_review_completed") or record.get("facts_sha256")!=b.sha(facts_path):raise ValueError("completed original author review required before model view")
    context=checked(context_path)
    if context["facts_sha256"]!=b.sha(facts_path):raise ValueError("context belongs to different facts")
    scores=b.read(scores_path);b.verify(scores.get("source_bindings",{}))
    if scores["signal_date"]!=facts["signal_date"] or set(x["code"] for x in scores["rows"])!=set(x["code"] for x in context["rows"]):raise ValueError("same complete dated scores required")
    bycode={x["code"]:x for x in context["rows"]};rows=[]
    for original in scores["rows"]:
        row=bycode[original["code"]];value=original.get("channel_b_score",original.get("score"))
        rows.append(dict(code=original["code"],name=original["name"],frozen_research_score=value,
            legacy_direction=original.get("direction"),channel_review=row["channel_review"],
            B_direction=("PROMOTE" if value>.5 else "FAIL") if row["b_direction_allowed"] and value is not None else None,
            actionable=False,status="UNCALIBRATED_FAILED_RESEARCH_MODEL_WITH_SCOPE_REVIEW"))
    result=dict(formed_at=b.now(),signal_date=facts["signal_date"],rows=rows,new_fits=0,new_inferences=0,new_author_directions=0,new_formal_predictions=0,
        source_bindings=context["source_bindings"]|{str(Path(p).resolve()):b.sha(p) for p in [context_path,review_path,scores_path,__file__]})
    b.write_new(output_path,result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("context");p.add_argument("--facts",required=True);p.add_argument("--bundle",required=True);p.add_argument("--review");p.add_argument("--output",required=True)
    p=sub.add_parser("score-view");p.add_argument("--facts",required=True);p.add_argument("--context",required=True);p.add_argument("--review",required=True);p.add_argument("--scores",required=True);p.add_argument("--output",required=True)
    args=parser.parse_args()
    if args.command=="context":result=project(args.facts,args.bundle,args.review);b.write_new(args.output,result)
    else:result=saved_score_view(args.facts,args.context,args.review,args.scores,args.output)
    print(dict(rows=len(result["rows"]),new_fits=0,status="READ_ONLY_EXISTING_WORKBENCH_EVIDENCE_VIEW"))


if __name__=="__main__":main()
