"""Independent event-window accounting and saved-model replay; no fitting."""
from pathlib import Path
import sys
import json

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import first_board_matured_event_model as memory
import first_board_leader_model as base
import first_second_stage_model as stage
import first_board_capital_risk_model as risk
import numpy as np
import pandas as pd
import joblib

OUT=Path(__file__).resolve().parent
protocol=json.loads((OUT/"protocol_BEFORE_ANY_FIT.json").read_text("utf-8"))
report=json.loads((OUT/"report.json").read_text("utf-8"))
for name in ("protected","inputs","original_raw_hashes"):
    for path,digest in protocol[name].items():
        assert base.sha(path)==digest,(name,path)
assert base.sha(memory.__file__)==protocol["source_sha256"]==base.sha(OUT/"source_before_first_fit.py")
rows=pd.read_csv(OUT/"ALL_FIRST_BOARDS_WITH_MATURED_MEMORY_AND_UNCHANGED_SCREEN.csv",dtype={"code":str},float_precision="round_trip")
original=pd.read_csv(memory.REF/"all_first_and_second_boards_WITH_SEPARATE_LABELS.csv",dtype={"code":str},float_precision="round_trip")
original=original[original.stage.eq(1)].reset_index(drop=True)
paths=pd.read_csv(memory.LONG,dtype={"code":str},float_precision="round_trip")
sessions=json.loads(base.INPUTS["calendar"].read_text("utf-8"))["sessions"]
position={d:i for i,d in enumerate(sessions)}
assert list(zip(rows.signal_date,rows.code))==list(zip(original.signal_date,original.code))
np.testing.assert_allclose(rows[stage.VARIANTS["STAGE_CYCLE_GENE"]+[stage.TARGET]],original[stage.VARIANTS["STAGE_CYCLE_GENE"]+[stage.TARGET]],atol=1e-12,rtol=1e-12,equal_nan=True)
old_screen=pd.read_csv(memory.SCREEN/"COMPLETE_FIRST_BOARD_FACTS_FEATURES_LABELS_AND_SCREEN.csv",dtype={"code":str},float_precision="round_trip")
assert rows.research_selection_allowed.tolist()==old_screen.research_selection_allowed.tolist()
known=original[original[stage.TARGET].notna()].copy()
known["ready"]=known.label_end.map(position)
long=paths[paths.max_height_5_LABEL_ONLY.notna()].copy()
long["ready"]=long.label_end.map(position)
short_codes={c:g for c,g in known.groupby("code")}
long_codes={c:g for c,g in long.groupby("code")}
expected=[]
for p,part in original.groupby("p",sort=True):
    eligible=known[known.p.between(p-252,p-1)&known.ready.le(p)]
    eligible_long=long[long.p.between(p-252,p-1)&long.ready.le(p)]
    priors=[eligible[stage.TARGET].mean(),
        eligible_long.loc[eligible_long.reach2_LABEL_ONLY.eq(1),"reach3_LABEL_ONLY"].mean(),
        eligible_long.loc[eligible_long.reach3_LABEL_ONLY.eq(1),"reach4_LABEL_ONLY"].mean(),
        eligible_long.reach4_LABEL_ONLY.mean()]
    for row in part.itertuples():
        past=short_codes.get(row.code,known.iloc[:0])
        past=past[past.p.between(p-252,p-1)&past.ready.le(p)].sort_values("p")
        history=long_codes.get(row.code,long.iloc[:0])
        history=history[history.p.between(p-252,p-1)&history.ready.le(p)].sort_values("p")
        last=history.max_height_5_LABEL_ONLY
        den=[len(past),int(history.reach2_LABEL_ONLY.eq(1).sum()),int(history.reach3_LABEL_ONLY.eq(1).sum()),len(history)]
        num=[past[stage.TARGET].sum(),history.reach3_LABEL_ONLY.sum(),history.reach4_LABEL_ONLY.sum(),history.reach4_LABEL_ONLY.sum()]
        rates=[(n+5*q)/(d+5) if pd.notna(q) else np.nan for n,d,q in zip(num,den,priors)]
        values=[len(past),rates[0],rates[0]-priors[0],past[stage.TARGET].iloc[-1] if len(past) else np.nan,
                p-past.p.iloc[-1] if len(past) else np.nan,int(past.p.ge(p-20).sum()),len(history),rates[1],rates[2],rates[3],
                int(last.eq(2).sum()),int(last.eq(3).sum()),last.max() if len(last) else np.nan,last.iloc[-1] if len(last) else np.nan]
        expected.append([row.signal_date,row.code]+values)
check=pd.DataFrame(expected,columns=["signal_date","code"]+memory.FIRST+memory.PATH)
check=rows[["signal_date","code"]].merge(check,on=["signal_date","code"],validate="one_to_one")
np.testing.assert_allclose(check[memory.FIRST+memory.PATH],rows[memory.FIRST+memory.PATH],atol=1e-12,rtol=1e-12,equal_nan=True)

core_checks=[]
for receipt_path in sorted((OUT/"models").glob("*_COMPLETED.json")):
    receipt=json.loads(receipt_path.read_text("utf-8"));name=receipt["name"]
    model_path=OUT/"models"/(name+".joblib")
    assert base.sha(model_path)==receipt["sha256"]
    members=pd.read_csv(OUT/"models"/(name+"_members.csv"),dtype={"code":str},float_precision="round_trip")
    assert members.label_end.max()< ("2025-01-01" if name.startswith("validation_") else name[:7]+"-01")
    merged=members.merge(rows[["signal_date","code",stage.TARGET]],on=["signal_date","code"],validate="one_to_one",suffixes=("_saved","_source"))
    np.testing.assert_allclose(merged[stage.TARGET+"_saved"],merged[stage.TARGET+"_source"],atol=0,rtol=0)
    core_checks.append(dict(name=name,rows=len(members),max_label_end=members.label_end.max()))

replay_checks=[]
for bundle_path in sorted((OUT/"models").glob("*_bundle.joblib")):
    name=bundle_path.name[:-len("_bundle.joblib")]
    bundle=joblib.load(bundle_path)
    if name.startswith("validation_"):
        predictions_path=OUT/(name+"_ALL_BEFORE_METRICS.csv")
    else:
        predictions_path=OUT/(name[:7]+"_ALL_BEFORE_METRICS.csv")
    saved=pd.read_csv(predictions_path,dtype={"code":str},float_precision="round_trip")
    inputs=rows.merge(saved[["signal_date","code"]],on=["signal_date","code"],validate="one_to_one")
    replay=stage.predict(bundle,inputs)
    np.testing.assert_allclose(replay[["model_score","calibrated_score"]],saved[["model_score","calibrated_score"]],atol=1e-12,rtol=1e-12,equal_nan=True)
    # Prefix predictions and Platt outputs must also reproduce from current features.
    cal=pd.read_csv(OUT/"models"/(name+"_calibration_rows.csv"),dtype={"code":str},float_precision="round_trip")
    cal_receipt=json.loads((OUT/"models"/(name+"_calibration_receipt.json")).read_text("utf-8"))
    assert cal_receipt["inner_max_label_end"]<cal_receipt["cutoff"]
    cal_x=rows.merge(cal[["signal_date","code"]],on=["signal_date","code"],validate="one_to_one")
    prefix=joblib.load(OUT/"models"/(name+"_calibration_prefix.joblib"))
    np.testing.assert_allclose(prefix.predict_proba(cal_x[bundle["features"]])[:,1],cal.uncalibrated_oof,atol=1e-12,rtol=1e-12)
    replay_checks.append(dict(name=name,rows=len(saved),calibration_rows=len(cal)))

for variant,result in report["decision"]["variants"].items():
    saved=pd.read_csv(OUT/("validation_"+variant+"_ALL_BEFORE_METRICS.csv"),dtype={"code":str},float_precision="round_trip")
    eligible=saved[saved.research_selection_allowed]
    take=eligible.sort_values(["signal_date","model_score","code"],ascending=[True,False,True]).groupby("signal_date").head(1)
    assert int(take[stage.TARGET].eq(1).sum())==result["primary"]["hits"]
    assert int(take[stage.TARGET].eq(0).sum())==result["primary"]["false_positives"]
    assert int(saved[stage.TARGET].eq(1).sum()-take[stage.TARGET].eq(1).sum())==result["primary"]["false_negatives"]
    assert result["primary"]["full_signal_dates"]==117

if report["decision"]["selected"]:
    pred=pd.read_csv(OUT/"EVALUATION_ALL_PREDICTIONS.csv",dtype={"code":str},float_precision="round_trip")
    ref=risk.attach_frozen(rows,memory.REF/"S1_ALL_PREDICTIONS.csv")
    recalculated=memory.period_metrics(pred,ref)
    assert base.clean(recalculated)==report["evaluation"]
    allowed=pred[pred.research_selection_allowed]
    second=pd.read_csv(memory.REF/"S2_ALL_PREDICTIONS.csv",dtype={"code":str},float_precision="round_trip")
    chain,stats=stage.chain_recheck(allowed,second)
    assert stats==report["same_stock_second_recheck"]
    # All official source rows and false/excluded positives remain retained.
    assert len(pred)==int(rows.signal_date.ge("2025-07-01").sum())
else:
    assert len(core_checks)==4
    assert not list((OUT/"models").glob("2025-*.joblib"))
result=dict(status="PASS",verified_at=base.now(),complete_rows=len(rows),independent_matured_event_rows=len(check),
    all14_fields_independently_equal=True,original62_features_and_next_session_labels_unchanged=True,
    risk_screen_unchanged=True,core_models=core_checks,bundles_replayed=replay_checks,
    new_fits=0,current_fund_identity_unknown=True,goal_60_passed=report["goal_60_passed"],
    source_sha256=base.sha(memory.__file__),verifier_sha256=base.sha(__file__))
base.write_json(OUT/"verification.json",result)
print(json.dumps(dict(status="PASS",rows=len(rows),saved_cores=len(core_checks),replayed_heads=len(replay_checks)),ensure_ascii=True,indent=2))
