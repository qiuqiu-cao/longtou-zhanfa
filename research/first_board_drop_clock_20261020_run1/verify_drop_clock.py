"""Verify the single-field ablation, unchanged memberships and saved heads."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import joblib
import numpy as np
import pandas as pd
import first_board_leader_model as base
import first_second_stage_model as stage
import first_board_capital_risk_model as risk
import first_board_drop_clock as experiment

OUT=Path(__file__).resolve().parent
protocol=json.loads((OUT/"protocol_BEFORE_ANY_FIT.json").read_text("utf-8"))
report=json.loads((OUT/"report.json").read_text("utf-8"))
for path,digest in {**protocol["inputs"],**protocol["protected"],**protocol["raw_source_hashes"]}.items():assert base.sha(path)==digest,path
assert base.sha(experiment.__file__)==protocol["source_sha256"]
original_fields=stage.VARIANTS["STAGE_CYCLE_GENE"]
assert experiment.FIELDS==[f for f in original_fields if f!="listing_age_sessions"]
assert len(experiment.FIELDS)==61
rows=pd.read_csv(OUT/"ALL_FIRST_BOARD_UNCHANGED_FACTS_LABELS_AND_SCREEN.csv",dtype={"code":str},float_precision="round_trip")
original=pd.read_csv(experiment.REF/"all_first_and_second_boards_WITH_SEPARATE_LABELS.csv",dtype={"code":str},float_precision="round_trip")
original=original[original.stage.eq(1)].reset_index(drop=True)
assert list(zip(rows.signal_date,rows.code))==list(zip(original.signal_date,original.code))
np.testing.assert_allclose(rows[original_fields+[stage.TARGET]],original[original_fields+[stage.TARGET]],atol=0,rtol=0,equal_nan=True)
assert rows.label_end.fillna("UNKNOWN").tolist()==original.label_end.fillna("UNKNOWN").tolist()
screen=pd.read_csv(experiment.SCREEN/"COMPLETE_FIRST_BOARD_FACTS_FEATURES_LABELS_AND_SCREEN.csv",dtype={"code":str})
assert rows.research_selection_allowed.tolist()==screen.research_selection_allowed.tolist()
assert rows.exclusion_reason.fillna("").tolist()==screen.exclusion_reason.fillna("").tolist()
name="validation_"+experiment.VARIANT
bundle=joblib.load(OUT/"models"/(name+"_bundle.joblib"))
assert bundle["features"]==experiment.FIELDS
cal_info=json.loads((OUT/"models"/(name+"_calibration_receipt.json")).read_text("utf-8"))
base_cal_info=json.loads((experiment.REF/"models/validation_S1_STAGE_CYCLE_GENE_calibration_receipt.json").read_text("utf-8"))
assert cal_info["cutoff"]==base_cal_info["cutoff"]=="2024-11-01"
assert cal_info["inner_max_label_end"]<cal_info["cutoff"]
checks=[]
for suffix in ("_calibration_prefix","_full"):
    receipt=json.loads((OUT/"models"/(name+suffix+"_COMPLETED.json")).read_text("utf-8"))
    assert base.sha(OUT/"models"/(name+suffix+".joblib"))==receipt["sha256"]
    members=pd.read_csv(OUT/"models"/(name+suffix+"_members.csv"),dtype={"code":str},float_precision="round_trip")
    baseline_members=pd.read_csv(experiment.REF/"models"/("validation_S1_STAGE_CYCLE_GENE"+suffix+"_members.csv"),dtype={"code":str},float_precision="round_trip")
    pd.testing.assert_frame_equal(members,baseline_members)
    assert not members.duplicated(["signal_date","code"]).any() and members.label_end.max()<"2025-01-01"
    if suffix=="_calibration_prefix":assert members.label_end.max()<cal_info["cutoff"]
    model=joblib.load(OUT/"models"/(name+suffix+".joblib"))
    fields=model.named_steps["observed_columns"].transformers_[0][2]
    assert "listing_age_sessions" not in fields
    assert len(fields)==60 and set(fields)==set(experiment.FIELDS)-{"opened_after_one_price"}
    params=model.named_steps["hist"].get_params()
    expected=base.make_model("hist_gradient").get_params()
    assert params==expected
    used=set()
    for trees in model.named_steps["hist"]._predictors:
        for tree in trees:
            indices=tree.nodes[tree.nodes["is_leaf"]==0]["feature_idx"]
            used.update(int(i) for i in indices)
    assert all(i<len(fields) for i in used)
    assert all(fields[i]!="listing_age_sessions" for i in used)
    checks.append(dict(core=name+suffix,members=len(members),max_label_end=members.label_end.max(),
        original_member_rows_exact=True,model_input_fields=len(fields),archive_clock_absent=True))
pred=pd.read_csv(OUT/"validation_ALL_BEFORE_METRICS.csv",dtype={"code":str},float_precision="round_trip")
x=rows.merge(pred[["signal_date","code"]],on=["signal_date","code"],validate="one_to_one")
assert len(pred)==4724 and pred[stage.TARGET].isna().sum()==10
raw=bundle["model"].predict_proba(x[bundle["features"]])[:,1]
np.testing.assert_allclose(raw,pred.model_score,atol=1e-12,rtol=1e-12)
if bundle["calibrator"] is not None:
    logits=np.log(np.clip(raw,1e-6,1-1e-6)/np.clip(1-raw,1e-6,1-1e-6))
    calibrated=bundle["calibrator"].predict_proba(logits.reshape(-1,1))[:,1]
    np.testing.assert_allclose(calibrated,pred.calibrated_score,atol=1e-12,rtol=1e-12)
cal=pd.read_csv(OUT/"models"/(name+"_calibration_rows.csv"),dtype={"code":str},float_precision="round_trip")
base_cal=pd.read_csv(experiment.REF/"models/validation_S1_STAGE_CYCLE_GENE_calibration_rows.csv",dtype={"code":str},float_precision="round_trip")
pd.testing.assert_frame_equal(cal[["signal_date","code","stage","label_end",stage.TARGET]],base_cal[["signal_date","code","stage","label_end",stage.TARGET]])
prefix=joblib.load(OUT/"models"/(name+"_calibration_prefix.joblib"))
cx=rows.merge(cal[["signal_date","code"]],on=["signal_date","code"],validate="one_to_one")
np.testing.assert_allclose(prefix.predict_proba(cx[experiment.FIELDS])[:,1],cal.uncalibrated_oof,atol=1e-12,rtol=1e-12)

def picked(frame,apply_screen=True):
    keep=frame[frame.research_selection_allowed] if apply_screen else frame
    indices=[]
    for day,part in keep.groupby("signal_date",sort=True):
        indices.append(part.index[np.lexsort((part.code.to_numpy(),-part.model_score.to_numpy()))[0]])
    return keep.loc[indices]

chosen=picked(pred);metric=report["decision"]["primary"]
assert len(chosen)==metric["selected"]==117 and chosen[stage.TARGET].notna().all()
assert chosen[stage.TARGET].eq(1).sum()==metric["hits"]==46
assert chosen[stage.TARGET].eq(0).sum()==metric["false_positives"]==71
assert pred[stage.TARGET].eq(1).sum()-46==metric["false_negatives"]==817
assert picked(pred[pred.one_price.eq(0)])[stage.TARGET].eq(1).sum()==41
assert picked(pred,False)[stage.TARGET].eq(1).sum()==report["decision"]["unfiltered_diagnostic"]["hits"]==52
frozen=risk.attach_frozen(rows,experiment.REF/"validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv")
old=picked(frozen)
pair=old[["signal_date","code","name",stage.TARGET]].merge(chosen[["signal_date","code","name",stage.TARGET]],on="signal_date",validate="one_to_one",suffixes=("_BASE","_DROP_CLOCK"))
pair.to_csv(OUT/"COMPLETE117_BASE_VS_DROP_CLOCK_LABEL_ONLY.csv",index=False,encoding="utf-8-sig")
before=pair[stage.TARGET+"_BASE"];after=pair[stage.TARGET+"_DROP_CLOCK"]
assert report["core_fit_count"]==len(list((OUT/"models").glob("*_COMPLETED.json")))==2
assert report["decision"]["selected"] is None and not list((OUT/"models").glob("2025-*.joblib"))
result=dict(status="PASS",verified_at=base.now(),complete_source_rows=len(rows),original62_fact_fields_and_labels_preserved=True,
    new_estimator61_fields_only_archive_clock_removed=True,cores=checks,saved_core_and_calibration_replay=True,
    complete_validation_rows=4724,validation_unknown_pool=10,selected_dates=117,selected_unknown=0,hits=46,
    changed_picks=int(pair.code_BASE.ne(pair.code_DROP_CLOCK).sum()),gain=int((before.eq(0)&after.eq(1)).sum()),loss=int((before.eq(1)&after.eq(0)).sum()),
    monthly_challenge_fits=0,new_fits_during_verification=0,new_forward_results=0,actual_trades=0,goal_60_passed=False,
    source_sha256=base.sha(experiment.__file__),verifier_sha256=base.sha(__file__))
base.write_json(OUT/"verification.json",result)
print(json.dumps(result,ensure_ascii=True,indent=2))
