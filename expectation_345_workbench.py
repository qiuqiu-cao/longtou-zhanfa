"""Research-only saved-model view after the existing complete author review gate."""
from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import short_term_thinking_dataset_builder as b
import expectation_345_features as projection
import expectation_345_research as trial
import relay_causal_review_workbench as causal_workbench
import relay_review_workbench as existing


def comparable(value):
    if value is pd.NA or value is pd.NaT:return None
    if isinstance(value,dict):return {k:comparable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [comparable(v) for v in value]
    return b.clean(value)


def qualified_projection(artifact,relay,sessions):
    frame,fields=projection.project(artifact,relay,sessions)
    status=artifact["rows"].set_index(["date","code"]).loc[pd.MultiIndex.from_frame(frame[["date","code"]]),"full_cohort_status"].to_numpy()
    for old,new in [("PARTIAL_MEMBER_OUTCOMES","PARTIAL_MEMBERS_UNKNOWN"),("NOT_APPLICABLE_EMPTY_PREVIOUS_COHORT","NOT_APPLICABLE_EMPTY_COHORT")]:
        oldfield,newfield="causal_cohort_"+old,"causal_cohort_"+new
        if not frame[oldfield].eq(0).all():raise ValueError("source enum contract changed; explicit qualification required")
        frame[newfield]=(status==new).astype(float)
        fields[fields.index(oldfield)]=newfield;frame=frame.drop(columns=oldfield)
    return frame[["date","code"]+fields],fields


def validate_review(facts_path,review_path):
    record=b.read(review_path);existing.validate_completed(facts_path,record)
    if not record.get("author_review_completed") or record.get("facts_sha256")!=b.sha(facts_path):
        raise ValueError("completed whole-cohort author review bound to the original facts is required")
    b.verify(record.get("source_bindings",{}))
    return record


def score(facts_path,context_path,review_path,artifact_path,output_path):
    validate_review(facts_path,review_path)
    context=causal_workbench.checked(context_path)
    if context["facts_sha256"]!=b.sha(facts_path):raise ValueError("wrong complete current facts")
    result_path=trial.ROOT/"evaluation_scope_alignment/verification_and_corrected_evaluation.json"
    validation=b.read(result_path);b.verify(validation["source_bindings"])
    artifact=joblib.load(artifact_path)
    if artifact["version"]!=context["version"]:raise ValueError("causal semantic version mismatch")
    day=context["signal_date"];expected={x["code"] for x in context["rows"]}
    rows=artifact["rows"];original=rows[rows.date.eq(day)].set_index("code")
    if set(original.index)!=expected:raise ValueError("complete same-day causal bundle required")
    # The projection must consume the live API output, not silently bypass it.
    live=pd.DataFrame(context["rows"]).set_index("code")
    for field in set(projection.COHORT_FIELDS+["review_cycle_state","review_roles","current_price_shape","cycle_context_complete","current_price_path_status","active_anchor_discovery_scope","complete_active_parent_count","full_cohort_status"]):
        for code in expected:
            a,z=comparable(original.loc[code,field]),comparable(live.loc[code,field])
            if a!=z:raise ValueError("live/historical projection source mismatch: "+field+" "+code)
    savedparents=artifact["parents"][artifact["parents"].date.eq(day)].sort_values(["code","parent_code"]).to_dict("records")
    liveparents=sorted(context["all_active_parent_relations"],key=lambda x:(x["code"],x["parent_code"]))
    if comparable(savedparents)!=comparable(liveparents):raise ValueError("all live active parent relations must match bound causal bundle")
    engine=joblib.load(context["bundle_path"])
    frame,fields=qualified_projection(artifact,engine["prepared"]["relay"],engine["sessions"])
    current=frame[frame.date.eq(day)].copy()
    if set(current.code)!=expected:raise ValueError("current candidate lost in projection")
    trained=joblib.load(trial.ROOT/"trainable_corrected_dataset_QUALIFIED.joblib")
    if fields!=trained["fields"]:raise ValueError("training/inference feature contract differs")
    training_copy=trained["full"].set_index(["date","code"])
    actual=current.set_index(["date","code"])
    np.testing.assert_allclose(actual[fields],training_copy.loc[actual.index,fields],atol=0,rtol=0,equal_nan=True)
    comparison=pd.read_csv(trial.ROOT/"evaluation_scope_alignment/complete18885_predictions_ORIGINAL_FROZEN_COHORT.csv",dtype={"code":str},low_memory=False)
    output=[];models=[]
    for arm in trial.ARMS:
        model_path=trial.ROOT/"fixed_comparison"/(day[:7]+"_"+arm+"_model.joblib")
        saved=joblib.load(model_path)
        if saved["cutoff"]>day or saved["features"]!=fields:raise ValueError("model uses a future cutoff or incompatible features")
        with threadpool_limits(limits=2):
            scores=saved["model"].predict_proba(current[fields])[:,list(saved["model"].classes_).index(1)]
        expected_scores=comparison[comparison.date.eq(day)&comparison.arm.eq(arm)].set_index("code")
        np.testing.assert_allclose(scores,expected_scores.loc[current.code,"b_score"],atol=1e-15,rtol=0)
        models.append(model_path)
        for code,value in zip(current.code,scores):
            shape=original.loc[code,"current_price_shape"]
            oldrow=expected_scores.loc[code]
            output.append(dict(date=day,code=code,name=oldrow["name"],arm=arm,uncalibrated_raw_B_score=float(value),
                research_call="UNKNOWN" if pd.isna(shape) else ("PROMOTE" if value>.5 else "FAIL") if shape==0 else "A_FROZEN_BASE_ONLY",
                feature_count=len(fields),model_cutoff=saved["cutoff"],actionable=False,production_eligible=False))
    result=dict(formed_at=b.now(),signal_date=day,target_date=context["target_date"],rows=output,
        complete_current_candidates=len(expected),feature_count=len(fields),training_and_live_API_feature_values_exact=True,
        author_review_completed_before_view=True,latest_corrected_features_actually_fitted=True,
        corrected_validation_pass=validation["corrected_screen_pass"],cost_validation_pass=validation["cost_screen_pass"],
        status="UNCALIBRATED_FAILED_RESEARCH_COMPARISON_NOT_ADOPTED",new_fits=0,original_facts_changed=False,original_directions_changed=False,
        original_models_replaced=False,source_bindings=context["source_bindings"]|validation["source_bindings"]|
            {str(Path(p).resolve()):b.sha(p) for p in [facts_path,context_path,review_path,artifact_path,result_path,__file__,*models]})
    b.verify(result["source_bindings"]);b.write_new(output_path,result)
    return dict(candidates=len(expected),research_scores=len(output),features=len(fields),new_fits=0,adopted=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--facts",required=True);p.add_argument("--context",required=True);p.add_argument("--review",required=True)
    p.add_argument("--artifact",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();print(score(a.facts,a.context,a.review,a.artifact,a.output))
