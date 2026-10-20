"""Audit the appended ordered arm using both native and saved Python models."""
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
import first_board_regularized_ordered as experiment

OUT=Path(__file__).resolve().parent
protocol=json.loads((OUT/"protocol_BEFORE_ANY_FIT.json").read_text("utf-8"))
extension=json.loads((OUT/"ordered_extension_BEFORE_ANY_ORDERED_FIT.json").read_text("utf-8"))
report=json.loads((OUT/"combined_report_APPEND_ONLY.json").read_text("utf-8"))
for path,digest in {**protocol["inputs"],**protocol["protected"],**protocol["raw_source_hashes"],**extension["inputs"]}.items():assert base.sha(path)==digest,path
assert base.sha(experiment.__file__)==protocol["source_sha256"]
assert extension["params"]==protocol["deferred_arm"]["params"]==experiment.CAT_PARAMS
assert base.sha(OUT/"continue_frozen_ordered_arm.py")==extension["runner_sha256"]
rows=pd.read_csv(OUT/"ALL_FIRST_BOARD_UNCHANGED_FEATURES_LABELS_SCREEN.csv",dtype={"code":str},float_precision="round_trip")
name="validation_CHRONO_ORDERED"
cal_info=json.loads((OUT/"models"/(name+"_calibration_receipt.json")).read_text("utf-8"))
full=rows[rows[stage.TARGET].notna()&rows.signal_date.lt("2025-01-01")&rows.label_end.lt("2025-01-01")]
assert len(full)==10586 and cal_info["cutoff"]==sorted(full.signal_date.unique())[-42]=="2024-11-01"
core_checks=[]
for suffix in ("_calibration_prefix","_full"):
    receipt=json.loads((OUT/"models"/(name+suffix+"_COMPLETED.json")).read_text("utf-8"))
    file=OUT/"models"/(name+suffix+".joblib")
    native_file=file.with_suffix(".cbm")
    assert base.sha(file)==receipt["sha256"] and base.sha(native_file)==receipt["native_sha256"]
    assert receipt["params"]==experiment.CAT_PARAMS
    assert receipt["test_data_used_for_fit"] is False and receipt["no_test_dependent_early_stopping"] is True
    members=pd.read_csv(OUT/"models"/(name+suffix+"_members.csv"),dtype={"code":str},float_precision="round_trip")
    expected=full[full.label_end.lt(cal_info["cutoff"])] if suffix=="_calibration_prefix" else full
    assert set(zip(members.signal_date,members.code))==set(zip(expected.signal_date,expected.code))
    assert list(zip(members.signal_date,members.code))==sorted(zip(members.signal_date,members.code))
    linear_members=pd.read_csv(OUT/"models"/("validation_REGULARIZED_LINEAR"+suffix+"_members.csv"),dtype={"code":str},float_precision="round_trip")
    pd.testing.assert_frame_equal(members,linear_members)
    assert members.label_end.lt(receipt["knowledge_cutoff"]).all()
    assert members.label_end.max()<(cal_info["cutoff"] if suffix=="_calibration_prefix" else "2025-01-01")
    data=rows.set_index(["signal_date","code"]).loc[pd.MultiIndex.from_frame(members[["signal_date","code"]])]
    np.testing.assert_allclose(members[stage.TARGET],data[stage.TARGET],atol=0,rtol=0)
    sizes=members.groupby("signal_date").size()
    weights=members.signal_date.map(sizes).rdiv(1);weights=weights/weights.mean()
    counts=members[stage.TARGET].value_counts()
    classes={int(y):len(members)/(2*n) for y,n in counts.items()}
    np.testing.assert_allclose(members.sample_weight,weights,atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(members.class_multiplier,members[stage.TARGET].map(classes),atol=1e-12,rtol=1e-12)
    model=joblib.load(file)
    native=experiment.cat_class()();native.load_model(str(native_file))
    params=model.get_all_params()
    assert params["boosting_type"]=="Ordered" and params["has_time"] is True
    assert params["grow_policy"]=="SymmetricTree" and params["bootstrap_type"]=="No"
    assert params["depth"]==4 and model.tree_count_==160 and params["iterations"]==160
    assert abs(params["learning_rate"]-.06)<1e-7 and params["l2_leaf_reg"]==8
    assert params["random_seed"]==42 and params["allow_writing_files"] is False
    assert params["use_best_model"] is False
    np.testing.assert_allclose(params["class_weights"],[classes[0],classes[1]],atol=1e-6,rtol=1e-6)
    observed=[f for f in experiment.FIELDS if data[f].notna().any()]
    assert observed==receipt["observed_fields"]==model.feature_names_
    np.testing.assert_allclose(model.predict_proba(data[observed]),native.predict_proba(data[observed]),atol=1e-12,rtol=1e-12)
    core_checks.append(dict(core=name+suffix,rows=len(members),max_label_end=members.label_end.max(),observed_fields=len(observed),
        native_python_replay_equal=True,chronological_order_verified=True,actual_trees=model.tree_count_,warnings=receipt["warnings"]))

bundle=joblib.load(OUT/"models"/(name+"_bundle.joblib"))
pred=pd.read_csv(OUT/"validation_CHRONO_ORDERED_ALL_BEFORE_METRICS.csv",dtype={"code":str},float_precision="round_trip")
assert len(pred)==4724 and pred[stage.TARGET].isna().sum()==10
expected=rows[rows.signal_date.between("2025-01-01","2025-06-30")]
assert set(zip(pred.signal_date,pred.code))==set(zip(expected.signal_date,expected.code))
x=rows.set_index(["signal_date","code"]).loc[pd.MultiIndex.from_frame(pred[["signal_date","code"]])]
raw=bundle["model"].predict_proba(x[bundle["observed_fields"]])[:,1]
np.testing.assert_allclose(pred.model_score,raw,atol=1e-12,rtol=1e-12)
np.testing.assert_allclose(pred[stage.TARGET],x[stage.TARGET],atol=0,rtol=0,equal_nan=True)
assert pred.research_selection_allowed.tolist()==x.research_selection_allowed.tolist()
if bundle["calibrator"] is not None:
    logits=np.log(np.clip(raw,1e-6,1-1e-6)/np.clip(1-raw,1e-6,1-1e-6))
    calibrated=bundle["calibrator"].predict_proba(logits.reshape(-1,1))[:,1]
    np.testing.assert_allclose(pred.calibrated_score,calibrated,atol=1e-12,rtol=1e-12)
cal=pd.read_csv(OUT/"models"/(name+"_calibration_rows.csv"),dtype={"code":str},float_precision="round_trip")
prefix=joblib.load(OUT/"models"/(name+"_calibration_prefix.joblib"))
info=json.loads((OUT/"models"/(name+"_calibration_prefix_COMPLETED.json")).read_text("utf-8"))
cx=rows.set_index(["signal_date","code"]).loc[pd.MultiIndex.from_frame(cal[["signal_date","code"]])]
np.testing.assert_allclose(cal.uncalibrated_oof,prefix.predict_proba(cx[info["observed_fields"]])[:,1],atol=1e-12,rtol=1e-12)

def pick(frame):
    indices=[]
    for day,group in frame[frame.research_selection_allowed].groupby("signal_date",sort=True):
        indices.append(group.index[np.lexsort((group.code.to_numpy(),-group.model_score.to_numpy()))[0]])
    return frame.loc[indices]

selected=pick(pred);metric=report["ordered"]["primary"]
assert len(selected)==metric["selected"]==117 and selected[stage.TARGET].notna().all()
assert selected[stage.TARGET].eq(1).sum()==metric["hits"]==43
assert selected[stage.TARGET].eq(0).sum()==metric["false_positives"]==74
assert pred[stage.TARGET].eq(1).sum()-43==metric["false_negatives"]==820
assert pick(pred[pred.one_price.eq(0)])[stage.TARGET].eq(1).sum()==39
reference=risk.attach_frozen(rows,experiment.REF/"validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv")
baseline=pick(reference)
linear=pd.read_csv(OUT/"validation_REGULARIZED_LINEAR_ALL_BEFORE_METRICS.csv",dtype={"code":str},float_precision="round_trip")
paired=baseline[["signal_date","code","name",stage.TARGET]].rename(columns={c:c+"_BASE" for c in ("code","name",stage.TARGET)})
for arm,data in [("LINEAR",pick(linear)),("ORDERED",selected)]:
    paired=paired.merge(data[["signal_date","code","name",stage.TARGET]].rename(columns={c:c+"_"+arm for c in ("code","name",stage.TARGET)}),on="signal_date",validate="one_to_one")
paired.to_csv(OUT/"COMPLETE117_BASE_LINEAR_ORDERED_LABEL_ONLY.csv",index=False,encoding="utf-8-sig")
old=paired[stage.TARGET+"_BASE"];new=paired[stage.TARGET+"_ORDERED"]
assert len(list((OUT/"models").glob("*_COMPLETED.json")))==report["total_core_fit_count"]==4
assert len(list((OUT/"models").glob("*CHRONO_ORDERED*_COMPLETED.json")))==report["ordered_core_fit_count"]==2
assert not list((OUT/"models").glob("2025-*.joblib")) and report["ordered"]["pass_validation"] is False
result=dict(status="PASS",verified_at=base.now(),cores=core_checks,complete_validation_rows=4724,validation_unknown_pool=10,
    selected_dates=117,hits=43,unknown_selected=0,gain=int((old.eq(0)&new.eq(1)).sum()),loss=int((old.eq(1)&new.eq(0)).sum()),
    all_frozen_parameters_and_members_verified=True,original_linear_artifacts_unchanged=True,
    ordered_core_fits=2,total_batch_core_fits=4,calibrators=2,monthly_challenge_fits=0,
    new_fits_during_verification=0,new_forward_results=0,actual_trades=0,goal_60_passed=False,
    verifier_sha256=base.sha(__file__),runner_sha256=base.sha(OUT/"continue_frozen_ordered_arm.py"))
for path,digest in extension["inputs"].items():assert base.sha(path)==digest,path
base.write_json(OUT/"ordered_verification_APPEND_ONLY.json",result)
print(json.dumps(result,ensure_ascii=True,indent=2))
