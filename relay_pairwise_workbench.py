"""Display saved fixed-cohort ranking only after the original author review gate."""
from pathlib import Path
import argparse
import joblib
import pandas as pd
import numpy as np

import short_term_thinking_dataset_builder as b
import relay_review_workbench as original
import relay_context_pairwise_ranker as ranking
import relay_pairwise_ranking_research as research


def score_after_review(facts_path,review_path,output_path):
    facts=b.read(facts_path);review=b.read(review_path)
    original.validate_completed(facts_path,review)
    if not review.get("author_review_completed") or review.get("facts_sha256")!=b.sha(facts_path):
        raise ValueError("complete original author review must precede any model view")
    b.verify(review.get("source_bindings",{}))
    protocol=b.read(research.ROOT/"protocol_BEFORE_ANY_FIT.json");b.verify(protocol["sources"])
    proof_path=research.previous.ROOT/"current6_retrained_research_workbench_AFTER_AUTHOR_REVIEW.json"
    # This fixed-cohort companion uses the exact already verified live projection.
    if str(proof_path.resolve()) not in protocol["sources"]:raise ValueError("live projection not bound by experiment")
    proof=b.read(proof_path);b.verify(proof["source_bindings"])
    if proof["signal_date"]!=facts["signal_date"] or not proof["training_and_live_API_feature_values_exact"]:
        raise ValueError("new date needs its own verified complete live feature projection")
    day=facts["signal_date"]
    data=joblib.load(research.ROOT/"fixed_input_before_fit.joblib")
    full,fields=data["full"],data["fields"]
    current=full[full.date.eq(day)]
    if set(current.code)!=set(x["code"] for x in facts["rows"]):raise ValueError("same complete current candidate set required")
    model_path=research.ROOT/"fixed_comparison"/(day[:7]+"_model.joblib")
    saved=joblib.load(model_path)
    if saved["cutoff"]>day or saved["features"]!=fields:raise ValueError("future cutoff or different feature contract")
    prediction=ranking.predict_cohort(saved["model"],current[current.daily_one_word.eq(0)],fields)
    evidence_path=research.ROOT/"fixed_comparison/all22662_predictions_BEFORE_EVALUATION.csv"
    prior=pd.read_csv(evidence_path,dtype={"code":str},low_memory=False)
    expected=prior[prior.date.eq(day)&prior.arm.eq(research.ARM)].set_index("code")
    np.testing.assert_allclose(prediction.ranking_score,expected.loc[prediction.code,"ranking_score"],atol=1e-15,rtol=0,equal_nan=True)
    rows=current[["date","code","name","boards","daily_one_word","current_price_shape"]].merge(prediction,on=["date","code"],how="left",validate="one_to_one")
    rows["arm"]=research.ARM;rows["score"]=np.nan;rows["call"]="UNKNOWN"
    result=dict(formed_at=b.now(),signal_date=day,target_date=facts["target_date"],rows=b.clean(rows.to_dict("records")),
        author_review_completed_before_view=True,complete_cohort=len(current),base_features=92,pair_features=276,
        event_probability=None,new_absolute_predictions=0,new_formal_predictions=0,new_fits=0,
        source_bindings=protocol["sources"]|{str(Path(p).resolve()):b.sha(p) for p in [facts_path,review_path,proof_path,model_path,evidence_path,__file__]})
    b.verify(result["source_bindings"]);b.write_new(output_path,result)
    return dict(rows=len(rows),new_fits=0,absolute_event_head=False,formal_predictions_changed=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--facts",required=True);parser.add_argument("--review",required=True);parser.add_argument("--output",required=True)
    a=parser.parse_args();print(score_after_review(a.facts,a.review,a.output))
