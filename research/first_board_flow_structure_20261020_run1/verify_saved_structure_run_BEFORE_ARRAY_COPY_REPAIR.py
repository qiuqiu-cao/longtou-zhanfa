"""Independent direct-window11-feature audit plus saved model replays; no fits."""
from pathlib import Path
import sys
import json

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import first_board_flow_structure as flow
import first_board_leader_model as base
import first_second_stage_model as stage
import first_board_capital_risk_model as risk
import numpy as np
import pandas as pd
import joblib

OUT=Path(__file__).resolve().parent
protocol=json.loads((OUT/"protocol_BEFORE_ANY_FIT.json").read_text("utf-8"))
report=json.loads((OUT/"report.json").read_text("utf-8"))
for key in ("protected","inputs","raw_sources"):
    for path,digest in protocol[key].items():assert base.sha(path)==digest,(key,path)
assert protocol["source_sha256"]==base.sha(flow.__file__)==base.sha(OUT/"source_before_first_fit.py")
rows=pd.read_csv(OUT/"ALL_FIRST_BOARD_OBSERVABLE_STRUCTURE_AND_UNCHANGED_LABELS.csv",dtype={"code":str},float_precision="round_trip")
original=pd.read_csv(flow.REF/"all_first_and_second_boards_WITH_SEPARATE_LABELS.csv",dtype={"code":str},float_precision="round_trip")
original=original[original.stage.eq(1)].reset_index(drop=True)
assert list(zip(rows.signal_date,rows.code))==list(zip(original.signal_date,original.code))
np.testing.assert_allclose(rows[stage.VARIANTS["STAGE_CYCLE_GENE"]+[stage.TARGET]],original[stage.VARIANTS["STAGE_CYCLE_GENE"]+[stage.TARGET]],atol=1e-12,rtol=1e-12,equal_nan=True)
_,bars,sessions,audit=base.build_dataset()
codes={code:part.set_index("p") for code,part in bars.groupby("code",sort=False)}
expected=[]
for row in rows.itertuples():
    stock=codes[row.code]
    positions=np.arange(row.p-40,row.p)
    history=stock.reindex(positions)
    observed=history.code.notna().to_numpy()
    valid=(history.eligible.fillna(False)&history.valid_price.fillna(False)&history.regular_price.fillna(False)&history.amount.gt(0)&history.volume.gt(0)).to_numpy(bool)
    vol=history.volume.to_numpy(float);ret=(history.close/history.pre_close-1).to_numpy(float)
    amp=((history.high-history.low)/history.pre_close).to_numpy(float)
    loc=((2*history.close-history.high-history.low)/(history.high-history.low).where(history.high.gt(history.low))).to_numpy(float)
    amount=history.amount.to_numpy(float)
    gap=(history.open/history.pre_close-1).abs().to_numpy(float)
    inside=(history.close/history.open-1).abs().to_numpy(float)
    loc[~valid]=np.nan
    values=[np.nan]*11
    if observed[20:].all():
        n=int(np.isfinite(loc[20:]).sum());values[1]=n/20
        if n>=15:
            mask=np.isfinite(loc[20:]);weights=vol[20:][mask]
            values[0]=float(np.sum(loc[20:][mask]*weights)/np.sum(weights))
        if valid[20:].all():
            v=vol[20:];r=ret[20:];a=amp[20:];tr=history.turnover.iloc[20:].to_numpy(float)
            values[2]=float(v[r>0].sum()/v.sum())
            values[3]=float(np.square(v).sum()/v.sum()**2)
            if np.isfinite(tr).all():values[4]=float(tr.sum())
            denominator=np.abs(r).sum()
            if denominator>0:values[5]=float(abs(r.sum())/denominator)
            im=np.abs(r)/(amount[20:]/1e8)
            values[6]=float(np.median(im))
            move=gap[20:].sum()+inside[20:].sum()
            if move>0:values[7]=float(gap[20:].sum()/move)
            current_valid=bool(row.eligible and row.valid_price and row.regular_price and row.amount>0 and row.volume>0)
            if values[6]>0 and current_valid:values[9]=float(abs(row.close/row.pre_close-1)/(row.amount/1e8)/values[6])
            median_amp=float(np.median(a))
            if median_amp>0 and current_valid:values[10]=float(((row.high-row.low)/row.pre_close)/median_amp)
    if observed.all() and valid.all():
        flags=[vol[j]>np.median(vol[j-20:j]) and amp[j]<np.median(amp[j-20:j]) for j in range(20,40)]
        values[8]=float(np.mean(flags))
    expected.append(values)
np.testing.assert_allclose(np.array(expected),rows[flow.PAST+flow.SIGNAL].to_numpy(float),atol=1e-10,rtol=1e-10,equal_nan=True)

model_checks=[]
for path in sorted((OUT/"models").glob("*_bundle.joblib")):
    name=path.name[:-len("_bundle.joblib")];bundle=joblib.load(path)
    file=OUT/(name+"_ALL_BEFORE_METRICS.csv") if name.startswith("validation_") else OUT/(name[:7]+"_ALL_BEFORE_METRICS.csv")
    saved=pd.read_csv(file,dtype={"code":str},float_precision="round_trip")
    inputs=rows.merge(saved[["signal_date","code"]],on=["signal_date","code"],validate="one_to_one")
    pred=stage.predict(bundle,inputs)
    np.testing.assert_allclose(pred[["model_score","calibrated_score"]],saved[["model_score","calibrated_score"]],atol=1e-12,rtol=1e-12,equal_nan=True)
    cal=pd.read_csv(OUT/"models"/(name+"_calibration_rows.csv"),dtype={"code":str},float_precision="round_trip")
    cal_info=json.loads((OUT/"models"/(name+"_calibration_receipt.json")).read_text("utf-8"))
    assert cal_info["inner_max_label_end"]<cal_info["cutoff"]
    cal_x=rows.merge(cal[["signal_date","code"]],on=["signal_date","code"],validate="one_to_one")
    prefix=joblib.load(OUT/"models"/(name+"_calibration_prefix.joblib"))
    np.testing.assert_allclose(prefix.predict_proba(cal_x[bundle["features"]])[:,1],cal.uncalibrated_oof,atol=1e-12,rtol=1e-12)
    for suffix in ("_calibration_prefix","_full"):
        info=json.loads((OUT/"models"/(name+suffix+"_COMPLETED.json")).read_text("utf-8"))
        assert base.sha(OUT/"models"/(name+suffix+".joblib"))==info["sha256"]
        member=pd.read_csv(OUT/"models"/(name+suffix+"_members.csv"),dtype={"code":str},float_precision="round_trip")
        cutoff="2025-01-01" if name.startswith("validation_") else name[:7]+"-01"
        assert member.label_end.max()<cutoff
        raw=member.merge(original[["signal_date","code",stage.TARGET]],on=["signal_date","code"],validate="one_to_one",suffixes=("_member","_original"))
        np.testing.assert_allclose(raw[stage.TARGET+"_member"],raw[stage.TARGET+"_original"],atol=0,rtol=0)
    model_checks.append(dict(name=name,rows=len(saved),calibration_rows=len(cal)))

for variant,stats in report["decision"]["variants"].items():
    pred=pd.read_csv(OUT/("validation_"+variant+"_ALL_BEFORE_METRICS.csv"),dtype={"code":str},float_precision="round_trip")
    selected=pred[pred.research_selection_allowed].sort_values(["signal_date","model_score","code"],ascending=[True,False,True]).groupby("signal_date",sort=False).head(1)
    m=stats["primary"]
    assert len(selected)==m["selected"]==117
    assert int(selected[stage.TARGET].eq(1).sum())==m["hits"]
    assert int(selected[stage.TARGET].eq(0).sum())==m["false_positives"]
    assert int(pred[stage.TARGET].eq(1).sum()-selected[stage.TARGET].eq(1).sum())==m["false_negatives"]
case=pd.read_csv(OUT/"HISTORICAL_20260828_COMPLETE61_STRUCTURE_CARDS_NO_SCORES_NO_TARGETS.csv",dtype={"code":str},float_precision="round_trip")
assert len(case)==61 and not any("LABEL_ONLY" in c for c in case)
assert "model_score" not in case and "calibrated_score" not in case
assert case.program_fund_identity.eq("UNKNOWN").all() and case.absorption_fund_identity.eq("UNKNOWN").all()
assert set(case.code)==set(original[original.signal_date.eq("2026-08-28")].code)
if report["decision"]["selected"]:
    pred=pd.read_csv(OUT/"EVALUATION_ALL_PREDICTIONS.csv",dtype={"code":str},float_precision="round_trip")
    ref=risk.attach_frozen(rows,flow.REF/"S1_ALL_PREDICTIONS.csv")
    assert base.clean(flow.previous.period_metrics(pred,ref))==report["evaluation"]
    allowed=pred[pred.research_selection_allowed]
    second=pd.read_csv(flow.REF/"S2_ALL_PREDICTIONS.csv",dtype={"code":str},float_precision="round_trip")
    chain,stats=stage.chain_recheck(allowed,second)
    assert stats==report["same_stock_second_recheck"]
else:
    assert len(model_checks)==2 and report["core_fit_count"]==4
    assert not list((OUT/"models").glob("2025-*.joblib"))
base.write_json(OUT/"verification.json",dict(status="PASS",verified_at=base.now(),rows=len(rows),all11_features_independently_recomputed=True,
    unchanged62_features_labels_and_protected_hashes=True,model_replays=model_checks,case_full_pool=61,
    cases_are_retrospective=True,minute_sampling_not_used_for_training=True,fund_identity_validated=False,
    new_fits=0,goal_60_passed=report["goal_60_passed"],source_sha256=base.sha(flow.__file__),verifier_sha256=base.sha(__file__)))
print(json.dumps(dict(status="PASS",rows=len(rows),heads=len(model_checks),full_case_pool=61),ensure_ascii=True,indent=2))
