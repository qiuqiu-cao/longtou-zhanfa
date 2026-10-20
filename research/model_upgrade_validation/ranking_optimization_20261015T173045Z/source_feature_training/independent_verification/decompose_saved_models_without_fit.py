"""Separate source eligibility, inference feature fill and fitted-weight effects.

This is a diagnostic attribution over fixed saved models, never another trial.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("source_independent_helpers", OUT/"verify_source_feature_independent.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
WORK, BATCH = helpers.WORK, helpers.BATCH


def main():
    final_path=WORK/"same_source_route_alignment/complete26439_predictions_WITH_SAME_SOURCE_ROUTES.csv"
    final=helpers.csv(final_path)
    q=final[final.arm.eq("CORRECTED_CAUSAL")].copy()
    learned=final[final.arm.eq(helpers.ARM)].copy()
    assert len(q)==len(learned)==3777
    data=joblib.load(WORK/"prepared.joblib")
    full,fields=data["full"],data["fields"]
    protocol=helpers.read(WORK/"protocol_BEFORE_ANY_FIT.json")
    learned_index=learned.set_index(["date","code"])
    original=q.copy();original["evaluation_call"]=original.call
    qualified=q.copy();qualified["evaluation_call"]=qualified.recovered_B_call
    direct_parts=[];replays=[];windows=[];source_models={}
    with patch.object(HistGradientBoostingClassifier,"fit",side_effect=AssertionError("attribution must not fit")):
        for check in protocol["training_checks"]:
            model_path=Path(check["original_model_path"])
            assert helpers.sha(model_path)==check["original_model_sha256"]
            source_models[str(model_path)]=check["original_model_sha256"]
            model=joblib.load(model_path)["model"]
            test=full[full.date.str[:7].eq(check["month"])]
            with threadpool_limits(limits=2):
                score=model.predict_proba(test[fields])[:,list(model.classes_).index(1)]
            raw=test[["date","code"]].copy();raw["old_model_patched_input_b_score"]=score
            frame=qualified[qualified.date.str[:7].eq(check["month"])].merge(raw,on=["date","code"],validate="one_to_one")
            frame["qualified_frozen_b_score"]=frame.b_score
            frame["frozen_qualification_call"]=frame.evaluation_call
            frame["b_score"]=frame.old_model_patched_input_b_score
            frame["ranking_score"]=frame.old_model_patched_input_b_score
            zero=frame.verified_current_price_shape.eq(0)
            frame.loc[zero,"score"]=frame.loc[zero,"old_model_patched_input_b_score"]
            frame.loc[zero,"evaluation_call"]=np.where(frame.loc[zero,"old_model_patched_input_b_score"].gt(.5),"PROMOTE","FAIL")
            # For all other routes inherit the already qualified reference.
            compare=frame[["date","code","qualified_frozen_b_score","old_model_patched_input_b_score"]].copy()
            compare["learned_model_patched_input_b_score"]=learned_index.loc[pd.MultiIndex.from_frame(compare[["date","code"]]),"b_score"].to_numpy()
            compare["inference_input_fill_score_change"]=compare.old_model_patched_input_b_score-compare.qualified_frozen_b_score
            compare["fitted_weight_score_change"]=compare.learned_model_patched_input_b_score-compare.old_model_patched_input_b_score
            compare["month"]=check["month"];compare["reused_model"]=check["reuse_original_model"]
            if check["reuse_original_model"]:
                assert compare.fitted_weight_score_change.abs().le(1e-12).all()
            windows.append(dict(month=check["month"],reused_model=check["reuse_original_model"],candidates=len(compare),
                input_fill_raw_score_changed=int(compare.inference_input_fill_score_change.abs().gt(1e-12).sum()),
                fitted_weights_raw_score_changed=int(compare.fitted_weight_score_change.abs().gt(1e-12).sum()),
                maximum_input_fill_score_change=float(compare.inference_input_fill_score_change.abs().max()),
                maximum_fitted_weight_score_change=float(compare.fitted_weight_score_change.abs().max())))
            direct_parts.append(frame);replays.append(compare)
    inference_only=pd.concat(direct_parts,ignore_index=True)
    # The callable final wrapper may store an explicit source-aligned event field.
    learned["evaluation_call"] = learned.evaluation_call if "evaluation_call" in learned and learned.evaluation_call.notna().all() else learned.recovered_B_call
    modes={"ORIGINAL_FROZEN_CALL":original,"QUALIFICATION_ONLY_FIXED_RAW_SCORE":qualified,
        "QUALIFICATION_AND_INFERENCE_FEATURE_ONLY_OLD_MODEL":inference_only,
        "QUALIFICATION_AND_INFERENCE_AND_TRAINED_FEATURE":learned}
    records=[];ranks=[]
    for period,first,last in helpers.PERIODS:
        for mode,frame in modes.items():
            f=frame[frame.date.between(first,last)]
            for scope,part in [("FULL_POOL",f),("ORIGINAL_NONWORD",f[f.daily_one_word.eq(0)])]:
                records.append(dict(period=period,mode=mode,scope=scope,**helpers.classify(part)))
            for k in [1,2,3]:
                summary,_=helpers.full_slot_rank(f,k)
                ranks.append(dict(period=period,mode=mode,**summary))
    records=pd.DataFrame(records);ranks=pd.DataFrame(ranks)
    contrasts=[]
    comparisons=[("QUALIFICATION", "ORIGINAL_FROZEN_CALL", "QUALIFICATION_ONLY_FIXED_RAW_SCORE"),
        ("INFERENCE_FEATURE_FILL", "QUALIFICATION_ONLY_FIXED_RAW_SCORE", "QUALIFICATION_AND_INFERENCE_FEATURE_ONLY_OLD_MODEL"),
        ("FITTED_WEIGHTS", "QUALIFICATION_AND_INFERENCE_FEATURE_ONLY_OLD_MODEL", "QUALIFICATION_AND_INFERENCE_AND_TRAINED_FEATURE")]
    indexed=records.set_index(["period","mode","scope"])
    ranked=ranks.set_index(["period","mode","k"])
    for period,_,_ in helpers.PERIODS:
        for name,left,right in comparisons:
            for scope in ["FULL_POOL","ORIGINAL_NONWORD"]:
                a=indexed.loc[(period,left,scope)];z=indexed.loc[(period,right,scope)]
                row=dict(period=period,attribution=name,scope=scope,left=left,right=right)
                for key in ["tp","fp","Recall","Precision","F1","abstentions"]:
                    row[key+"_left"]=a[key];row[key+"_right"]=z[key];row[key+"_difference"]=z[key]-a[key]
                if scope=="ORIGINAL_NONWORD":
                    for k in [1,2,3]:
                        aa=ranked.loc[(period,left,k)];zz=ranked.loc[(period,right,k)]
                        row[f"Top{k}_hits_difference"]=zz.hits-aa.hits
                        row[f"Top{k}_full_slots"]=aa.full_slots
                contrasts.append(row)
    outputs={"四层来源归因_完整指标.csv":records,"四层来源归因_固定Top1Top2Top3.csv":ranks,
        "来源资格_推断输入_重学权重三段增量.csv":pd.DataFrame(contrasts),
        "全部3777原模型patched输入与新模型分数归因.csv":pd.concat(replays,ignore_index=True),
        "16窗权重学习与推断填值分开.csv":pd.DataFrame(windows)}
    for name,frame in outputs.items():helpers.write_csv(OUT/name,frame)
    for path,wanted in source_models.items():assert helpers.sha(path)==wanted
    receipt=dict(formed_at=datetime.now(timezone.utc).isoformat(),passed=True,new_fits=0,old_saved_models_replayed=16,
        old_model_patched_inference_candidates=3777,reused_windows_zero_learning_effect=7,
        qualification_vs_inference_vs_fitted_weight_effects_separated=True,
        policy="Diagnostic decomposition of already fixed saved scores only; not another fitted candidate, threshold trial or selection for deployment",
        caveat="Previously seen history; input and weight changes are descriptive associations, not causal trading performance; all event scores uncalibrated",
        source_bindings=source_models|{str(p.resolve()):helpers.sha(p) for p in [Path(__file__),OUT/"verify_source_feature_independent.py",final_path,WORK/"protocol_BEFORE_ANY_FIT.json",WORK/"prepared.joblib",*[OUT/name for name in outputs]]})
    helpers.write_new(OUT/"三段增益归因独立核验收据.json",receipt)
    display=pd.DataFrame(contrasts)
    print(display[display.period.isin(["HISTORICAL_POOLED","2026Sep"])&display.scope.eq("ORIGINAL_NONWORD")][["period","attribution","tp_difference","fp_difference","Recall_difference","Precision_difference","F1_difference","Top3_hits_difference"]].to_string(index=False))


if __name__=="__main__": main()
