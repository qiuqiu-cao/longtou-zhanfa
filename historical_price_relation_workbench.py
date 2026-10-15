"""Read-only research companion to the original review workbench.

Uses the same audited price-relation function on a complete, updated causal
engine bundle. It never bypasses or modifies the original forecast freeze.
"""
from pathlib import Path
import argparse

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import short_term_thinking_dataset_builder as b
import short_term_thinking_workbench as thinking
import historical_price_relation as relation


def project(facts_path, bundle_path):
    facts=b.read(facts_path);causal=thinking.checked(facts)
    if causal is None:
        raise ValueError("original causal workbench context required first")
    bundle=joblib.load(bundle_path);b.verify(bundle["sources"])
    day=facts["signal_date"]
    complete=bundle["attempts"].attrs.get("failed_scope_complete_dates",[])
    if day not in complete:
        raise ValueError("updated complete current closing/failed bundle required; no old pool substitution")
    candidates=pd.DataFrame(causal["rows"])
    if set(candidates.code)!=set(r["code"] for r in facts["rows"]):
        raise ValueError("complete original current candidate pool required")
    rows,pairs=relation.build_causal(candidates,bundle["attempts"],bundle["bars"],bundle["sessions"],complete)
    return dict(version=relation.VERSION,formed_at=b.now(),signal_date=day,rows=b.clean(rows.to_dict("records")),
        policy=relation.POLICY,all_current_pool_parent_pairs=len(pairs),model_scores_masked=True,
        history_datasets=causal["history_datasets"],mainline_history=causal["mainline_history"],
        narrative_reviews=causal["narrative_reviews"],parent_chain_evidence=causal["parent_chain_evidence"],
        facts_path=str(Path(facts_path).resolve()),bundle_path=str(Path(bundle_path).resolve()),
        facts_sha256=b.sha(facts_path),source_bindings=causal["source_bindings"]|bundle["sources"]|
        {str(Path(p).resolve()):b.sha(p) for p in [facts_path,bundle_path,__file__,relation.__file__]},
        new_fits=0,formal_forecast_freeze_unchanged=True)


def score(facts_path, context_path, review_path, model_path, output_path):
    import relay_review_workbench as workbench
    import lagged_price_relation_research as experiment
    facts=b.read(facts_path);review=b.read(review_path);context=b.read(context_path)
    # Run the complete existing diagnostic review gate, including forum/parent views.
    workbench.validate_completed(facts_path,review)
    if review.get("author_review_completed") is not True or review.get("facts_sha256")!=b.sha(facts_path):
        raise ValueError("completed review must bind these exact original facts")
    b.verify(review.get("source_bindings",{}));b.verify(context["source_bindings"])
    if context["facts_sha256"]!=b.sha(facts_path):
        raise ValueError("price context belongs to other facts")
    recomputed=project(facts_path,context["bundle_path"])
    if context["rows"]!=recomputed["rows"] or context["policy"]!=relation.POLICY:
        raise ValueError("statistical context tampering or incomplete pool")
    causal=thinking.checked(facts);packs=[]
    for path,sha in causal["source_bindings"].items():
        if Path(path).suffix==".json" and ("features" in Path(path).name or "feature_pack" in Path(path).name):
            pack=b.read(path)
            if pack.get("signal_date")==facts["signal_date"] and "relay" in pack and b.sha(path)==sha:
                packs.append(pack)
    if len(packs)!=1:
        raise ValueError("one exact source-bound score-free feature pack required")
    frame=pd.DataFrame([dict(r["features"],code=r["code"],name=r["name"],boards=r["boards"]) for r in packs[0]["relay"]])
    dynamic=pd.DataFrame(causal["rows"])[["code"]+b.DYNAMIC_FIELDS]
    frame=frame.drop(columns=list(set(b.DYNAMIC_FIELDS)&set(frame))).merge(dynamic,on="code",validate="one_to_one")
    frame=frame.merge(pd.DataFrame(context["rows"])[["code"]+relation.FIELDS],on="code",validate="one_to_one")
    saved=joblib.load(model_path)
    if saved.get("relation_version")!=relation.VERSION or saved["features"]!=experiment.FEATURES or saved["cutoff"]>facts["signal_date"]:
        raise ValueError("relation model version/feature/cutoff mismatch")
    with threadpool_limits(limits=2):
        values=saved["model"].predict_proba(frame[saved["features"]])[:,list(saved["model"].classes_).index(1)]
    rows=[dict(code=r.code,name=r.name,boards=int(r.boards),channel_b_score=float(values[i]),
        direction="PROMOTE" if values[i]>.5 else "FAIL",daily_one_word=r.daily_one_word,
        channel="B_NONWORD" if r.daily_one_word==0 else "A_ANCHOR_B_SCORE_NOT_TRADABLE" if r.daily_one_word==1 else "UNRESOLVED_IDENTITY") for i,r in enumerate(frame.itertuples())]
    result=dict(formed_at=b.now(),status="FAILED_RESEARCH_MODEL_SCORES_NOT_PRODUCTION_OR_NEW_INDEPENDENT_VIEW",signal_date=facts["signal_date"],rows=rows,
        model_cutoff=saved["cutoff"],uncalibrated=True,review_path=str(Path(review_path).resolve()),review_sha256=b.sha(review_path),
        source_bindings=context["source_bindings"]|{str(Path(p).resolve()):b.sha(p) for p in [context_path,review_path,model_path,__file__]},
        original_review_exposure=review.get("model_exposure"),blinding_independently_verified=False,
        new_fits=0,new_formal_predictions=0,new_actual_fills=0)
    b.write_new(output_path,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    add=sub.add_parser("context");add.add_argument("--facts",required=True);add.add_argument("--bundle",required=True);add.add_argument("--output",required=True)
    add=sub.add_parser("score");add.add_argument("--facts",required=True);add.add_argument("--context",required=True);add.add_argument("--review",required=True);add.add_argument("--model",required=True);add.add_argument("--output",required=True)
    args=parser.parse_args()
    if args.command=="context":
        result=project(args.facts,args.bundle);b.write_new(args.output,result)
    else:
        result=score(args.facts,args.context,args.review,args.model,args.output)
    print(dict(rows=len(result["rows"]),new_fits=0,status=result.get("status","MASKED_CONTEXT_BUILT")))


if __name__=="__main__":main()
