"""Direct per-date peer enumeration and saved model replay. Never fits."""
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
import first_board_legacy_peer as experiment
import legacy_industry_snapshot as legacy

OUT=Path(__file__).resolve().parent
protocol=json.loads((OUT/"protocol_BEFORE_ANY_FIT.json").read_text("utf-8"))
report=json.loads((OUT/"report.json").read_text("utf-8"))
for path,digest in {**protocol["inputs"],**protocol["protected"],**protocol["raw_source_hashes"]}.items():
    assert base.sha(path)==digest,path
assert base.sha(experiment.__file__)==protocol["source_sha256"]
rows=pd.read_csv(OUT/"ALL_FIRST_BOARD_LEGACY_PEER_FEATURES_UNCHANGED_LABELS.csv",dtype={"code":str},float_precision="round_trip")
original=pd.read_csv(experiment.REF/"all_first_and_second_boards_WITH_SEPARATE_LABELS.csv",dtype={"code":str},float_precision="round_trip")
original=original[original.stage.eq(1)].reset_index(drop=True)
assert list(zip(rows.signal_date,rows.code))==list(zip(original.signal_date,original.code))
np.testing.assert_allclose(rows[stage.VARIANTS["STAGE_CYCLE_GENE"]+[stage.TARGET]],original[stage.VARIANTS["STAGE_CYCLE_GENE"]+[stage.TARGET]],atol=0,rtol=0,equal_nan=True)
assert rows.label_end.fillna("UNKNOWN").tolist()==original.label_end.fillna("UNKNOWN").tolist()
screen=pd.read_csv(experiment.SCREEN/"COMPLETE_FIRST_BOARD_FACTS_FEATURES_LABELS_AND_SCREEN.csv",dtype={"code":str})
assert rows.research_selection_allowed.tolist()==screen.research_selection_allowed.tolist()
assert rows.exclusion_reason.fillna("").tolist()==screen.exclusion_reason.fillna("").tolist()
mapping,manifest=legacy.load_snapshot()
assert not manifest["usable_as_historical_current_industry"] and not manifest["strict_original_publication_verified"]
_,raw,_,_=base.build_dataset()
b=raw.sort_values(["code","p"]).reset_index(drop=True)
lookup=mapping.set_index("code").legacy_group_PROXY_ALLOWED
groups=b.code.map(lookup).to_numpy(object)
codes=b.code.to_numpy();positions=b.p.to_numpy()
price=b.close.to_numpy(float);reference=b.pre_close.to_numpy(float)
amount=b.amount.to_numpy(float);volume=b.volume.to_numpy(float)
flag=(b.eligible&b.valid_price&b.regular_price).to_numpy(bool)
previous=np.roll(price,1)
adjacent=(codes==np.roll(codes,1))&(positions-np.roll(positions,1)==1)
adjacent[0]=False
observed=pd.notna(groups)&flag&np.roll(flag,1)&adjacent&(amount>0)&(volume>0)&(np.abs(reference-previous)<=.011)
limit=b.limit.to_numpy(bool);first=b["first"].to_numpy(bool);height=b.height.to_numpy(float);failed=b.failed.to_numpy(bool)
previous_first=np.roll(first,1)&adjacent
previous_high=(np.roll(height,1)>=3)&adjacent
count=np.r_[0,np.cumsum(observed.astype(int))]
idx=np.arange(len(b));current5=np.zeros(len(b),bool);prior5=np.zeros(len(b),bool)
valid=idx>=4
current5[valid]=(count[idx[valid]+1]-count[idx[valid]-4]==5)&(codes[valid]==codes[idx[valid]-4])
valid=idx>=5
prior5[valid]=(count[idx[valid]]-count[idx[valid]-5]==5)&(codes[valid]==codes[idx[valid]-5])
with np.errstate(divide="ignore",invalid="ignore"):
    ret=price/previous-1
    return5=np.where(observed&current5&(idx>=5)&(codes==np.roll(codes,5))&(positions-np.roll(positions,5)==5),price/np.roll(price,5)-1,np.nan)
    prior_amount=sum(np.roll(amount,k) for k in range(1,6))/5
    amount5=np.where(observed&prior5,amount/prior_amount,np.nan)
turnover=b.turnover.to_numpy(float);volume_ratio=b.volume_ratio_5.to_numpy(float)
catalog=mapping[mapping.code.str.startswith(("000","001","002","003","600","601","603","605"))&mapping.legacy_group_RAW.notna()].groupby("legacy_group_RAW").size().to_dict()
source_index=pd.MultiIndex.from_frame(b[["date","code"]])
signal_index=pd.MultiIndex.from_frame(rows[["signal_date","code"]].rename(columns={"signal_date":"date"}))
own_positions=source_index.get_indexer(signal_index)
assert (own_positions>=0).all()
expected=np.full((len(rows),len(experiment.CONTROL+experiment.PEERS)),np.nan)
days=b.groupby("date",sort=False).indices

def field_rank(group_indices,values,own):
    if not observed[own] or not first[own] or not np.isfinite(values[own]):return np.nan
    chosen=group_indices[first[group_indices]&np.isfinite(values[group_indices])]
    if not len(chosen):return np.nan
    v=values[chosen];value=values[own]
    return (int((v<value).sum())+(int((v==value).sum())+1)/2)/len(v)

for day,signals in rows.groupby("signal_date",sort=True):
    day_indices=days[day]
    usable=day_indices[observed[day_indices]]
    cohorts={g:usable[groups[usable]==g] for g in pd.unique(groups[usable])}
    for ri in signals.index:
        own=own_positions[ri];group=groups[own]
        expected[ri,0]=int(pd.notna(group))
        if pd.isna(group):continue
        complete=cohorts.get(group,np.array([],dtype=int))
        peers=complete[complete!=own];n=len(peers)
        values=np.full(17,np.nan);values[0]=n
        denominator=catalog[group]-1
        if denominator>0:values[1]=n/denominator
        values[11]=int(first[peers].sum())
        if n:
            values[2]=limit[peers].mean();values[3]=first[peers].mean();values[4]=(height[peers]>=2).mean()
            f=int(failed[peers].sum());l=int(limit[peers].sum())
            if f+l:values[5]=f/(f+l)
            values[6]=ret[peers].mean();values[7]=(ret[peers]>0).mean();values[8]=height[peers].max()
            prior=peers[previous_first[peers]]
            if len(prior):values[9]=limit[prior].mean()
            prior=peers[previous_high[peers]]
            if len(prior):values[10]=(~limit[prior]).mean()
            qualified=peers[np.isfinite(return5[peers])]
            if np.isfinite(return5[own]) and len(qualified):values[15]=return5[own]-return5[qualified].mean()
            qualified=peers[np.isfinite(amount5[peers])&(amount5[peers]>0)]
            if np.isfinite(amount5[own]) and amount5[own]>0 and len(qualified):values[16]=np.log(amount5[own]/amount5[qualified].mean())
        values[12]=field_rank(complete,turnover,own)
        values[13]=field_rank(complete,amount,own)
        values[14]=field_rank(complete,volume_ratio,own)
        expected[ri,1:]=values
np.testing.assert_allclose(expected,rows[experiment.CONTROL+experiment.PEERS],atol=1e-9,rtol=1e-9,equal_nan=True)
assert rows.legacy_group_known.eq(1).sum()==25808
assert rows.loc[rows.legacy_group_known.eq(0),experiment.PEERS].isna().all().all()
del raw,b,expected

frozen=risk.attach_frozen(rows,experiment.REF/"validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv")
def choose(frame):
    return frame[frame.research_selection_allowed].sort_values(["signal_date","model_score","code"],ascending=[True,False,True]).groupby("signal_date",sort=False).head(1)
baseline=choose(frozen)
paired=baseline[["signal_date","code","name",stage.TARGET]].rename(columns={c:c+"_BASE" for c in ("code","name",stage.TARGET)})
checks=[]
for variant in experiment.VARIANTS:
    name="validation_"+variant
    bundle=joblib.load(OUT/"models"/(name+"_bundle.joblib"))
    pred=pd.read_csv(OUT/(name+"_ALL_BEFORE_METRICS.csv"),dtype={"code":str},float_precision="round_trip")
    assert len(pred)==4724 and pred[stage.TARGET].isna().sum()==10
    assert set(zip(pred.signal_date,pred.code))==set(zip(frozen.signal_date,frozen.code))
    x=rows.merge(pred[["signal_date","code"]],on=["signal_date","code"],validate="one_to_one")
    raw_score=bundle["model"].predict_proba(x[bundle["features"]])[:,1]
    np.testing.assert_allclose(raw_score,pred.model_score,atol=1e-12,rtol=1e-12)
    if bundle["calibrator"] is not None:
        logit=np.log(np.clip(raw_score,1e-6,1-1e-6)/np.clip(1-raw_score,1e-6,1-1e-6))
        probability=bundle["calibrator"].predict_proba(logit.reshape(-1,1))[:,1]
        np.testing.assert_allclose(probability,pred.calibrated_score,atol=1e-12,rtol=1e-12)
    info=json.loads((OUT/"models"/(name+"_calibration_receipt.json")).read_text("utf-8"))
    assert info["inner_max_label_end"]<info["cutoff"]
    mature=rows[rows[stage.TARGET].notna()&rows.signal_date.lt("2025-01-01")&rows.label_end.lt("2025-01-01")]
    assert info["cutoff"]==sorted(mature.signal_date.unique())[-42]
    cal=pd.read_csv(OUT/"models"/(name+"_calibration_rows.csv"),dtype={"code":str},float_precision="round_trip")
    expected_cal=mature[mature.signal_date.ge(info["cutoff"])]
    assert set(zip(cal.signal_date,cal.code))==set(zip(expected_cal.signal_date,expected_cal.code))
    cx=rows.merge(cal[["signal_date","code"]],on=["signal_date","code"],validate="one_to_one")
    prefix=joblib.load(OUT/"models"/(name+"_calibration_prefix.joblib"))
    np.testing.assert_allclose(prefix.predict_proba(cx[bundle["features"]])[:,1],cal.uncalibrated_oof,atol=1e-12,rtol=1e-12)
    for suffix in ("_calibration_prefix","_full"):
        receipt=json.loads((OUT/"models"/(name+suffix+"_COMPLETED.json")).read_text("utf-8"))
        assert base.sha(OUT/"models"/(name+suffix+".joblib"))==receipt["sha256"]
        members=pd.read_csv(OUT/"models"/(name+suffix+"_members.csv"),dtype={"code":str},float_precision="round_trip")
        expected_members=mature[mature.label_end.lt(info["cutoff"])] if suffix=="_calibration_prefix" else mature
        assert set(zip(members.signal_date,members.code))==set(zip(expected_members.signal_date,expected_members.code))
        assert members.label_end.max()<"2025-01-01" and not members.duplicated(["signal_date","code"]).any()
        targets=members.merge(original[["signal_date","code",stage.TARGET]],on=["signal_date","code"],validate="one_to_one",suffixes=("_fit","_original"))
        np.testing.assert_allclose(targets[stage.TARGET+"_fit"],targets[stage.TARGET+"_original"],atol=0,rtol=0)
    picked=choose(pred);non=choose(pred[pred.one_price.eq(0)])
    metric=report["decision"]["variants"][variant]["primary"]
    assert len(picked)==metric["selected"]==117
    assert picked[stage.TARGET].notna().sum()==metric["known"]==117
    assert picked[stage.TARGET].eq(1).sum()==metric["hits"]
    assert picked[stage.TARGET].eq(0).sum()==metric["false_positives"]
    assert pred[stage.TARGET].eq(1).sum()-metric["hits"]==metric["false_negatives"]
    assert non[stage.TARGET].eq(1).sum()==report["decision"]["variants"][variant]["non_one_price"]["hits"]
    part=picked[["signal_date","code","name",stage.TARGET]].rename(columns={c:c+"_"+variant for c in ("code","name",stage.TARGET)})
    paired=paired.merge(part,on="signal_date",validate="one_to_one")
    old=paired[stage.TARGET+"_BASE"];new=paired[stage.TARGET+"_"+variant]
    checks.append(dict(variant=variant,features=len(bundle["features"]),training_rows=len(mature),validation_rows=len(pred),
        hits=int(new.eq(1).sum()),gain=int((old.eq(0)&new.eq(1)).sum()),loss=int((old.eq(1)&new.eq(0)).sum()),
        changed_picks=int(paired.code_BASE.ne(paired["code_"+variant]).sum()),
        scores_equal_frozen=bool(np.allclose(pred.model_score,frozen.model_score,atol=1e-12,rtol=1e-12))))
paired.to_csv(OUT/"COMPLETE117_BASE_CONTROL_AND_PEER_PICKS_LABEL_ONLY.csv",index=False,encoding="utf-8-sig")
assert report["decision"]["selected"] is None and report["core_fit_count"]==4
assert len(list((OUT/"models").glob("*_COMPLETED.json")))==4 and not list((OUT/"models").glob("2025-*.joblib"))
result=dict(status="PASS",verified_at=base.now(),full_rows=len(rows),all18_new_fields_independently_recomputed=True,
    method="Direct same-date observed peer enumeration,NumPy prior-adjacency/5window checks,no use of production group transforms",
    unchanged62_fields_labels_and_screen=True,source_unknown_rows=2695,core_models_replayed=4,calibration_heads_replayed=2,
    validation=checks,monthly_challenge_fits=0,new_fits_during_verification=0,new_forward_results=0,actual_trades=0,
    current_industry_truth=False,strict_original_publication_verified=False,goal_60_passed=False,
    source_sha256=base.sha(experiment.__file__),verifier_sha256=base.sha(__file__))
base.write_json(OUT/"verification.json",result)
print(json.dumps(result,ensure_ascii=True,indent=2))
