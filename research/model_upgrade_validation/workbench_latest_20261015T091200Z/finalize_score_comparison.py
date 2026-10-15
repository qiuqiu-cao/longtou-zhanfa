"""Re-use all saved inference rows after a display-column join correction."""
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

import short_term_thinking_dataset_builder as b
import short_term_thinking_research as old
import historical_symbolic_context as symbolic

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"saved_scores_AFTER_COMPLETED_REVIEW"


def main():
    review_path=ROOT/"author_current6_COMPLETED_BEFORE_MODEL_DISPLAY.json";review=b.read(review_path)
    before=b.read(OUT/"inference_receipt_BEFORE_MODEL_LOAD.json")
    if before["review_sha256"]!=b.sha(review_path):raise ValueError("completed review changed")
    b.verify(before["model_sources"])
    pack_path=symbolic.RUN/"latest15_prepared_v2/features_BEFORE_MODEL_DISPLAY.json";pack=b.read(pack_path)
    original={x["code"]:x for x in review["rows"]}
    manifest_path=old.BASE_ROOT/"models/relay/BASE/manifest.json";manifest=b.read(manifest_path)
    b.verify({manifest["model_path"]:manifest["sha256"]})
    frame=pd.DataFrame([dict(x["features"],code=x["code"]) for x in pack["relay"]])
    model=joblib.load(manifest["model_path"]);scores=model.predict_proba(frame[manifest["fields"]])[:,list(model.classes_).index(1)]
    rows=[]
    for stock,score in zip(pack["relay"],scores):
        x=original[stock["code"]]
        rows.append(dict(code=x["code"],name=x["name"],boards=x["boards"],BASE=float(score),
            original_independent_formal_direction=x["promotion_judgement"],prereview_conditional_outlook=x["conditional_outlook"],conditional_reason=x["conditional_outlook_reason"]))
    b.write_new(OUT/"BASE_research_inference_AFTER_REVIEW.json",dict(formed_at=b.now(),signal_date=pack["signal_date"],rows=rows,model_path=manifest["model_path"],model_sha256=manifest["sha256"],
        status="UNMODIFIED_BASE_RESEARCH_INFERENCE_NOT_FORMAL",source_bindings={str(pack_path.resolve()):b.sha(pack_path),str(review_path.resolve()):b.sha(review_path)},new_fits=0,new_formal_forecast=0))
    sources={}
    for arm in ["ST_DUAL","ST_B_ONLY","ST_DUAL_SYMBOLIC_DIFFUSION"]:
        path=OUT/(arm+".json");saved=b.read(path)
        if saved["review_sha256"]!=b.sha(review_path) or datetime.fromisoformat(saved["formed_at"])<=datetime.fromisoformat(review["reviewed_at"]):
            raise ValueError("stock score exposure must follow exact author review")
        field="score" if arm=="ST_DUAL_SYMBOLIC_DIFFUSION" else "channel_b_score"
        values={x["code"]:x[field] for x in saved["rows"]}
        if set(values)!=set(original):raise ValueError("six-stock pool mismatch")
        for row in rows:row[arm]=values[row["code"]]
        sources[str(path.resolve())]=b.sha(path)
    result=pd.DataFrame(rows)
    for arm in ["BASE","ST_DUAL","ST_B_ONLY","ST_DUAL_SYMBOLIC_DIFFUSION"]:
        result[arm+"_direction"]=np.where(result[arm]>.5,"PROMOTE","FAIL")
        result[arm+"_nonword_rank"]=result[arm].rank(method="min",ascending=False).astype(int)
    result.to_csv(OUT/"complete_current6_research_comparison.csv",index=False,encoding="utf-8-sig")
    value=dict(formed_at=b.now(),signal_date=pack["signal_date"],target_date=pack["target_date"],rows=b.clean(result.to_dict("records")),
        status="RESEARCH_SCORES_AFTER_FULL_DIAGNOSTIC_REVIEW_NOT_FORMAL_FORECAST",uncalibrated_scores=True,original_full11_role_freeze_not_passed=True,
        alternative_source_not_original_V293=True,saved3inference_artifacts_reused_not_rerun=True,display_key_fix_only=True,new_fits=0,new_formal_forecasts=0,actual_fills=0,
        source_bindings=sources|b.read(ROOT/"author_review_validation.json")["source_bindings"]|before["model_sources"]|{str(Path(__file__).resolve()):b.sha(__file__)})
    b.write_new(OUT/"complete_current6_research_comparison.json",value)
    print(result[["code","name","prereview_conditional_outlook","BASE","ST_DUAL","ST_B_ONLY","ST_DUAL_SYMBOLIC_DIFFUSION"]].to_string(index=False))


if __name__=="__main__":main()
