"""Verify linear pipeline statistics, exact temporal members and saved scores."""
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
report=json.loads((OUT/"report.json").read_text("utf-8"))
for path,digest in {**protocol["inputs"],**protocol["protected"],**protocol["raw_source_hashes"]}.items():assert base.sha(path)==digest,path
assert base.sha(experiment.__file__)==protocol["source_sha256"]
rows=pd.read_csv(OUT/"ALL_FIRST_BOARD_UNCHANGED_FEATURES_LABELS_SCREEN.csv",dtype={"code":str},float_precision="round_trip")
original=pd.read_csv(experiment.REF/"all_first_and_second_boards_WITH_SEPARATE_LABELS.csv",dtype={"code":str},float_precision="round_trip")
original=original[original.stage.eq(1)].reset_index(drop=True)
assert list(zip(rows.signal_date,rows.code))==list(zip(original.signal_date,original.code))
np.testing.assert_allclose(rows[experiment.FIELDS+[stage.TARGET]],original[experiment.FIELDS+[stage.TARGET]],atol=0,rtol=0,equal_nan=True)
name="validation_REGULARIZED_LINEAR"
cal_info=json.loads((OUT/"models"/(name+"_calibration_receipt.json")).read_text("utf-8"))
full=rows[rows[stage.TARGET].notna()&rows.signal_date.lt("2025-01-01")&rows.label_end.lt("2025-01-01")]
assert len(full)==10586 and cal_info["cutoff"]==sorted(full.signal_date.unique())[-42]
checks=[]
for suffix in ("_calibration_prefix","_full"):
    receipt=json.loads((OUT/"models"/(name+suffix+"_COMPLETED.json")).read_text("utf-8"))
    assert base.sha(OUT/"models"/(name+suffix+".joblib"))==receipt["sha256"]
    members=pd.read_csv(OUT/"models"/(name+suffix+"_members.csv"),dtype={"code":str},float_precision="round_trip")
    expected=full[full.label_end.lt(cal_info["cutoff"])] if suffix=="_calibration_prefix" else full
    assert set(zip(members.signal_date,members.code))==set(zip(expected.signal_date,expected.code))
    assert list(zip(members.signal_date,members.code))==sorted(zip(members.signal_date,members.code))
    assert members.label_end.lt(receipt["knowledge_cutoff"]).all()
    counts=members[stage.TARGET].value_counts()
    class_weights={int(y):len(members)/(2*n) for y,n in counts.items()}
    weights=members.signal_date.map(members.groupby("signal_date").size()).rdiv(1)
    weights=weights/weights.mean()
    np.testing.assert_allclose(weights,members.sample_weight,atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(members[stage.TARGET].map(class_weights),members.class_multiplier,atol=1e-12,rtol=1e-12)
    source=rows.set_index(["signal_date","code"]).loc[pd.MultiIndex.from_frame(members[["signal_date","code"]])]
    np.testing.assert_allclose(members[stage.TARGET],source[stage.TARGET],atol=0,rtol=0)
    model=joblib.load(OUT/"models"/(name+suffix+".joblib"))
    observed=receipt["observed_fields"]
    assert observed==[f for f in experiment.FIELDS if source[f].notna().any()]
    imputer=model.named_steps["simpleimputer"];scaler=model.named_steps["standardscaler"];linear=model.named_steps["logisticregression"]
    values=source[observed].to_numpy(float)
    medians=np.nanmedian(values,axis=0)
    np.testing.assert_allclose(imputer.statistics_,medians,atol=1e-12,rtol=1e-12)
    missing=np.isnan(values)
    filled=np.where(missing,medians[None,:],values)
    expected_indicators=np.flatnonzero(missing.any(axis=0))
    assert imputer.indicator_.features_.tolist()==expected_indicators.tolist()
    transformed=np.column_stack([filled,missing[:,expected_indicators].astype(float)])
    np.testing.assert_allclose(scaler.mean_,transformed.mean(axis=0),atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose(scaler.var_,transformed.var(axis=0),atol=1e-9,rtol=1e-10)
    assert linear.C==.25 and linear.class_weight=="balanced" and linear.max_iter==1000
    checks.append(dict(core=name+suffix,members=len(members),observed_fields=len(observed),indicator_fields=len(expected_indicators),solver_iterations=linear.n_iter_.tolist(),warnings=receipt["warnings"]))

bundle=joblib.load(OUT/"models"/(name+"_bundle.joblib"))
pred=pd.read_csv(OUT/"validation_REGULARIZED_LINEAR_ALL_BEFORE_METRICS.csv",dtype={"code":str},float_precision="round_trip")
assert len(pred)==4724 and pred[stage.TARGET].isna().sum()==10
source=rows.set_index(["signal_date","code"]).loc[pd.MultiIndex.from_frame(pred[["signal_date","code"]])]
model=bundle["model"];imputer=model.named_steps["simpleimputer"];scaler=model.named_steps["standardscaler"];linear=model.named_steps["logisticregression"]
x=source[bundle["observed_fields"]].to_numpy(float);missing=np.isnan(x)
filled=np.where(missing,imputer.statistics_[None,:],x)
x=np.column_stack([filled,missing[:,imputer.indicator_.features_].astype(float)])
z=(x-scaler.mean_)/scaler.scale_
logit=z@linear.coef_.ravel()+linear.intercept_[0]
probability=1/(1+np.exp(-logit))
np.testing.assert_allclose(probability,pred.model_score,atol=1e-12,rtol=1e-12)
if bundle["calibrator"] is not None:
    clipped=np.clip(probability,1e-6,1-1e-6)
    calibrated=bundle["calibrator"].predict_proba(np.log(clipped/(1-clipped)).reshape(-1,1))[:,1]
    np.testing.assert_allclose(calibrated,pred.calibrated_score,atol=1e-12,rtol=1e-12)
cal=pd.read_csv(OUT/"models"/(name+"_calibration_rows.csv"),dtype={"code":str},float_precision="round_trip")
prefix=joblib.load(OUT/"models"/(name+"_calibration_prefix.joblib"))
info=json.loads((OUT/"models"/(name+"_calibration_prefix_COMPLETED.json")).read_text("utf-8"))
cx=rows.set_index(["signal_date","code"]).loc[pd.MultiIndex.from_frame(cal[["signal_date","code"]])]
np.testing.assert_allclose(prefix.predict_proba(cx[info["observed_fields"]])[:,1],cal.uncalibrated_oof,atol=1e-12,rtol=1e-12)

def pick(frame):
    keep=[]
    for day,g in frame[frame.research_selection_allowed].groupby("signal_date",sort=True):
        keep.append(g.index[np.lexsort((g.code.to_numpy(),-g.model_score.to_numpy()))[0]])
    return frame.loc[keep]

selected=pick(pred);metric=report["decision"]["arms"]["REGULARIZED_LINEAR"]["primary"]
assert len(selected)==metric["selected"]==117 and selected[stage.TARGET].notna().sum()==117
assert selected[stage.TARGET].eq(1).sum()==metric["hits"]==40
assert selected[stage.TARGET].eq(0).sum()==metric["false_positives"]==77
assert pred[stage.TARGET].eq(1).sum()-40==metric["false_negatives"]==823
assert pick(pred[pred.one_price.eq(0)])[stage.TARGET].eq(1).sum()==38
frozen=risk.attach_frozen(rows,experiment.REF/"validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv")
old=pick(frozen)
paired=old[["signal_date","code",stage.TARGET]].merge(selected[["signal_date","code",stage.TARGET]],on="signal_date",validate="one_to_one",suffixes=("_baseline","_linear"))
paired.to_csv(OUT/"COMPLETE117_BASE_VS_LINEAR_LABEL_ONLY.csv",index=False,encoding="utf-8-sig")
old_y=paired[stage.TARGET+"_baseline"];new_y=paired[stage.TARGET+"_linear"]
assert len(list((OUT/"models").glob("*_COMPLETED.json")))==report["core_fit_count"]==2
assert not list((OUT/"models").glob("*CHRONO_ORDERED*")) and not list((OUT/"models").glob("2025-*.joblib"))
assert report["deferred_ordered_not_fitted"] is True
result=dict(status="PASS",verified_at=base.now(),unchanged_source_rows=len(rows),unchanged62_fields_labels_screen=True,
    replayed_linear_cores=2,training_only_imputation_scaling_and_weights_verified=True,manual_linear_logit_predictions_verified=True,
    complete_validation_rows=4724,selected=117,hits=40,gain=int((old_y.eq(0)&new_y.eq(1)).sum()),loss=int((old_y.eq(1)&new_y.eq(0)).sum()),
    cores=checks,ordered_fits=0,ordered_status="DEFERRED_DEPENDENCY_NOT_A_REJECTED_MODEL",monthly_fits=0,
    new_fits_during_verification=0,new_forward_results=0,actual_trades=0,goal_60_passed=False,
    source_sha256=base.sha(experiment.__file__),verifier_sha256=base.sha(__file__))
base.write_json(OUT/"verification.json",result)
print(json.dumps(result,ensure_ascii=True,indent=2))
