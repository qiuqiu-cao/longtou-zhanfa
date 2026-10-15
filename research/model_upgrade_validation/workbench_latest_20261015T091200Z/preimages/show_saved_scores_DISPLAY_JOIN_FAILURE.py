"""Research comparison after the exact full6 diagnostic review, never a fit."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import short_term_thinking_dataset_builder as b
import relay_review_workbench as workbench
import short_term_thinking_research as old
import historical_symbolic_context as symbolic

ROOT=Path(__file__).resolve().parent


def main():
    validation=b.read(ROOT/"author_review_validation.json")
    b.verify(validation["source_bindings"])
    facts=Path(validation["facts_path"]);review=Path(validation["review_path"])
    draft=b.read(review)
    if validation["model_scores_displayed_before_review"] or not draft["author_review_completed"]:
        raise ValueError("actual full independent review must precede current scores")
    workbench.validate_completed(facts,draft)
    out=ROOT/"saved_scores_AFTER_COMPLETED_REVIEW"
    if out.exists():raise FileExistsError("immutable research score display already exists")
    out.mkdir()
    models={arm:b.DEFAULT_RUN/"execution_after_binning_repair"/("2026-10_"+arm+"_model.joblib") for arm in ["ST_DUAL","ST_B_ONLY"]}
    models["ST_DUAL_SYMBOLIC_DIFFUSION"]=symbolic.RUN/"fixed_comparison/2026-10_model.joblib"
    manifest_path=old.BASE_ROOT/"models/relay/BASE/manifest.json";manifest=b.read(manifest_path)
    models["BASE"]=Path(manifest["model_path"])
    b.verify({str(models["BASE"].resolve()):manifest["sha256"]})
    before=dict(formed_at=b.now(),review_path=str(review.resolve()),review_sha256=b.sha(review),facts_path=str(facts.resolve()),facts_sha256=b.sha(facts),
        model_sources={str(p.resolve()):b.sha(p) for p in models.values()},new_fits=0,new_formal_forecasts=0,
        conditional_outlooks_already_written_before_model_display={x["code"]:x["conditional_outlook"] for x in draft["rows"]},
        failed_models_remain_research_only=True)
    b.write_new(out/"inference_receipt_BEFORE_MODEL_LOAD.json",before)
    for arm in ["ST_DUAL","ST_B_ONLY"]:
        workbench.main(["thinking-score","--facts",str(facts),"--review",str(review),"--model",str(models[arm]),"--output",str(out/(arm+".json"))])
    workbench.main(["symbolic-score","--facts",str(facts),"--review",str(review),"--model",str(models["ST_DUAL_SYMBOLIC_DIFFUSION"]),"--output",str(out/"ST_DUAL_SYMBOLIC_DIFFUSION.json")])
    context=b.read(b.read(facts)["causal_thinking_context_path"])
    cp=b.read(symbolic.RUN/"latest15_prepared_v2/features_BEFORE_MODEL_DISPLAY.json")
    frame=pd.DataFrame([dict(code=x["code"],name=x["name"],boards=x["boards"],**{k:v for k,v in x["features"].items() if k!="boards"}) for x in cp["relay"]])
    model=joblib.load(models["BASE"])
    probabilities=model.predict_proba(frame[manifest["fields"]])[:,list(model.classes_).index(1)]
    rows=[];oldrows={x["code"]:x for x in draft["rows"]}
    for stock,value in zip(cp["relay"],probabilities):
        rows.append(dict(code=stock["code"],name=stock["name"],boards=stock["boards"],BASE=float(value),
            original_independent_formal_direction=oldrows[stock["code"]]["promotion_judgement"],
            prereview_conditional_outlook=oldrows[stock["code"]]["conditional_outlook"],
            conditional_reason=oldrows[stock["code"]]["conditional_outlook_reason"]))
    for arm in ["ST_DUAL","ST_B_ONLY","ST_DUAL_SYMBOLIC_DIFFUSION"]:
        values={x["code"]:x for x in b.read(out/(arm+".json"))["rows"]}
        if set(values)!={x["code"] for x in rows}:raise ValueError("full current6 research roster changed")
        for row in rows:row[arm]=values[row["code"]]["score"]
    result=pd.DataFrame(rows)
    for arm in models:
        result[arm+"_direction"]=np.where(result[arm]>.5,"PROMOTE","FAIL")
        result[arm+"_nonword_rank"]=result[arm].rank(method="min",ascending=False).astype(int)
    result.to_csv(out/"complete_current6_research_comparison.csv",index=False,encoding="utf-8-sig")
    result_rows=b.clean(result.to_dict("records"))
    b.write_new(out/"complete_current6_research_comparison.json",dict(formed_at=b.now(),signal_date=cp["signal_date"],target_date=cp["target_date"],rows=result_rows,
        status="RESEARCH_SCORES_AFTER_FULL_DIAGNOSTIC_REVIEW_NOT_FORMAL_FORECAST",uncalibrated_scores=True,promote_direction_is_model_threshold_only=True,
        original_full11_role_freeze_not_passed=True,alternative_source_not_original_V293=True,new_fits=0,new_formal_forecasts=0,actual_fills=0,
        source_bindings=validation["source_bindings"]|before["model_sources"]|{str(review.resolve()):b.sha(review)}))
    print(result[["code","name","prereview_conditional_outlook","BASE","ST_DUAL","ST_B_ONLY","ST_DUAL_SYMBOLIC_DIFFUSION"]].to_string(index=False))


if __name__=="__main__":main()
