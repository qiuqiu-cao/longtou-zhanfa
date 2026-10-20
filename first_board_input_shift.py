"""Outcome-blind distribution audit of the unchanged first-board 62 fields."""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
import pandas as pd
import first_board_leader_model as base
import first_second_stage_model as stage
import first_board_capital_risk_model as risk
import first_board_matured_event_model as prior

FIELDS=list(stage.VARIANTS["STAGE_CYCLE_GENE"])
REF=risk.REFERENCE
QUANTILES=[.01,.05,.1,.25,.5,.75,.9,.95,.99]

def finite(series):
    values=pd.to_numeric(series,errors="coerce").to_numpy(dtype=float,copy=True)
    values[~np.isfinite(values)]=np.nan
    return values

def reference_bins(values):
    observed=values[np.isfinite(values)]
    if not len(observed):return np.array([-np.inf,np.inf])
    if np.unique(observed).size==1:
        c=observed[0]
        return np.array([-np.inf,c,np.nextafter(c,np.inf),np.inf])
    inside=np.unique(np.quantile(observed,np.arange(.1,1.,.1)))
    return np.r_[-np.inf,inside,np.inf]

def bin_counts(values,edges):
    known=np.isfinite(values)
    # Bins are left-inclusive,right-exclusive except infinite edges. A separate
    # missing bin participates in the denominator and in drift calculation.
    slots=np.searchsorted(edges[1:-1],values[known],side="right")
    result=np.bincount(slots,minlength=len(edges)-1)
    return np.r_[result,int((~known).sum())]

def psi(reference,current):
    if len(reference)!=len(current):raise ValueError("Mismatched fixed bins")
    a=(reference+.5)/(reference.sum()+.5*len(reference))
    b=(current+.5)/(current.sum()+.5*len(current))
    return float(np.sum((b-a)*np.log(b/a)))

def summarize(series,reference,edges):
    values=finite(series);obs=values[np.isfinite(values)];ref=reference[np.isfinite(reference)]
    result=dict(rows=len(values),finite=len(obs),missing=int(np.isnan(values).sum()),missing_fraction=float(np.isnan(values).mean()),
        unique_finite=int(len(np.unique(obs))),reference_has_observations=bool(len(ref)),reference_min=float(ref.min()) if len(ref) else None,
        reference_max=float(ref.max()) if len(ref) else None,mean=float(obs.mean()) if len(obs) else None,std=float(obs.std()) if len(obs) else None,
        min=float(obs.min()) if len(obs) else None,max=float(obs.max()) if len(obs) else None,
        below_reference_min_count=int((obs<ref.min()).sum()) if len(ref) else None,
        above_reference_max_count=int((obs>ref.max()).sum()) if len(ref) else None,
        counts=bin_counts(values,edges).tolist(),reference_counts=bin_counts(reference,edges).tolist())
    result["psi_with_missing_bin"]=psi(np.array(result["reference_counts"]),np.array(result["counts"]))
    for q in QUANTILES:result["q"+str(int(q*100)).zfill(2)]=float(np.quantile(obs,q)) if len(obs) else None
    return result

def run(output,resume=False):
    out=Path(output)
    if out.exists():
        expected={"protocol_BEFORE_AUDIT.json","source_before_audit.py","source_before_array_copy_repair.py","read_only_array_failure.json"}
        if not resume or {p.name for p in out.iterdir()}!=expected:raise FileExistsError("Preserve fixed outcome-blind audit")
        original_protocol=json.loads((out/"protocol_BEFORE_AUDIT.json").read_text("utf-8"))
        for path,digest in original_protocol["inputs"].items():assert base.sha(path)==digest
    else:
        if resume:raise ValueError("No recorded technical failure to resume")
        out.mkdir(parents=True)
    source=REF/"all_first_and_second_boards_WITH_SEPARATE_LABELS.csv"
    screen_file=prior.SCREEN/"COMPLETE_FIRST_BOARD_FACTS_FEATURES_LABELS_AND_SCREEN.csv"
    prefix_file=REF/"models/validation_S1_STAGE_CYCLE_GENE_calibration_prefix_members.csv"
    full_file=REF/"models/validation_S1_STAGE_CYCLE_GENE_full_members.csv"
    files=[source,screen_file,prefix_file,full_file,Path(base.__file__),Path(stage.__file__),Path(risk.__file__),
        REF/"report.json",REF/"S1_ALL_PREDICTIONS.csv",REF/"S2_ALL_PREDICTIONS.csv"]
    protocol=dict(formed_at=base.now(),batch_started_at="2026-10-20T15:04:52+00:00",budget_minutes=30,
        previous_goal_turn="PROGRESS:regularizedlinear/chronologicalordered4coresfixedcomparisonfinishedandrejected,nativeandPythonreplayspassed",
        task="Outcome-blind covariate/missingness and extreme-value audit before another learning hypothesis",
        fields=FIELDS,input_columns=["signal_date","code","stage","p"]+FIELDS,
        boundary_policy="Reference bins learned only from complete candidate pools on saved prefix-training dates;no outcomes read. Final42dates are distinct;fulltraining and2025H1 separate",
        metrics="Fixed prefix deciles with unique boundaries,constant-specific bins and explicit missing bin;PSIadd0.5eachbin;quantiles1/5/10/25/50/75/90/95/99;reference min/max exceedance;both fixed screened and unscreened pools",
        feature_semantics="Additionally quantify how each field varies within a signal day and whether original listing_age_sessions encodes the truncated source calendar;do not infer causal effect from drift alone",
        outcome_or_score_columns_read=False,new_fits=0,new_forward_results=0,actual_trades=0,goal_60_passed=False,
        selection_policy="No model/feature/threshold selection using the reported later-period statistics;hypothesis only after evidence;all62fields reported",
        stop="Complete fixed audit andverification or30minutes,no fitsthisbatch",
        inputs={str(p.resolve()):base.sha(p) for p in files},source_sha256=base.sha(__file__),production_changes=False)
    if resume:
        assert protocol["fields"]==original_protocol["fields"] and protocol["inputs"]==original_protocol["inputs"]
        protocol["parent_protocol_sha256"]=base.sha(out/"protocol_BEFORE_AUDIT.json")
        protocol["technical_repair"]="Copy NumPy values before replacing nonfinite;initialzero-metricfailure/source retained,no field/period/bin changes"
        base.write_json(out/"protocol_AFTER_ARRAY_COPY_REPAIR.json",protocol)
    else:
        base.write_json(out/"protocol_BEFORE_AUDIT.json",protocol);shutil.copy2(__file__,out/"source_before_audit.py")
    rows=pd.read_csv(source,usecols=protocol["input_columns"],dtype={"code":str},float_precision="round_trip")
    rows=rows[rows.stage.eq(1)].copy()
    screen=pd.read_csv(screen_file,usecols=["signal_date","code","research_selection_allowed","exclusion_reason"],dtype={"code":str})
    rows=rows.merge(screen,on=["signal_date","code"],validate="one_to_one")
    prefix_dates=sorted(pd.read_csv(prefix_file,usecols=["signal_date"]).signal_date.unique())
    train_dates=sorted(pd.read_csv(full_file,usecols=["signal_date"]).signal_date.unique())
    cal_dates=train_dates[-42:]
    assert prefix_dates[-1]<cal_dates[0] and cal_dates[0]=="2024-11-01"
    masks={"PREFIX_REFERENCE":rows.signal_date.isin(prefix_dates),"FULL_DEVELOPMENT":rows.signal_date.isin(train_dates),
        "CAL42":rows.signal_date.isin(cal_dates),"VALIDATION2025H1":rows.signal_date.between("2025-01-01","2025-06-30")}
    for month in sorted(rows.loc[masks["VALIDATION2025H1"],"signal_date"].str[:7].unique()):masks[month]=rows.signal_date.str.startswith(month)
    records=[];boundaries={};counts=[]
    for universe,allowed in [("complete_pool",pd.Series(True,index=rows.index)),("fixed_risk_pool",rows.research_selection_allowed)]:
        reference=rows[masks["PREFIX_REFERENCE"]&allowed]
        boundaries[universe]={}
        for field in FIELDS:
            rv=finite(reference[field]);edges=reference_bins(rv)
            boundaries[universe][field]=["-inf" if np.isneginf(x) else ("inf" if np.isposinf(x) else float(x)) for x in edges]
            for period,mask in masks.items():
                part=rows[mask&allowed]
                metric=summarize(part[field],rv,edges)
                records.append(dict(universe=universe,period=period,feature=field,**{k:v for k,v in metric.items() if k not in ("counts","reference_counts")}))
                for j,(observed,expected) in enumerate(zip(metric["counts"],metric["reference_counts"])):
                    counts.append(dict(universe=universe,period=period,feature=field,bin_id=j,missing_bin=j==len(edges)-1,
                        observed_count=observed,reference_count=expected,observed_total=len(part),reference_total=len(reference)))
    pd.DataFrame(records).to_csv(out/"ALL62_FIXED_BOUNDARY_SHIFT_METRICS_NO_OUTCOMES.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(counts).to_csv(out/"ALL_FIXED_BIN_COUNTS_NO_OUTCOMES.csv",index=False,encoding="utf-8-sig")
    base.write_json(out/"PREFIX_ONLY_FIXED_BOUNDARIES.json",boundaries)
    scopes={name:dict(rows=int(mask.sum()),dates=int(rows.loc[mask,"signal_date"].nunique()),
        risk_allowed=int((mask&rows.research_selection_allowed).sum()),excluded=int((mask&~rows.research_selection_allowed).sum()),
        first_date=rows.loc[mask,"signal_date"].min(),last_date=rows.loc[mask,"signal_date"].max()) for name,mask in masks.items()}
    variability=[]
    for name,mask in masks.items():
        part=rows[mask]
        for field in FIELDS:
            count=part.groupby("signal_date")[field].nunique(dropna=True)
            variability.append(dict(period=name,field=field,signal_dates=len(count),all_missing_dates=int(count.eq(0).sum()),
                one_finite_value_dates=int(count.eq(1).sum()),varying_within_day_dates=int(count.gt(1).sum())))
    pd.DataFrame(variability).to_csv(out/"ALL62_WITHIN_DAY_VARIABILITY_NO_OUTCOMES.csv",index=False,encoding="utf-8-sig")
    age_rows=rows[["signal_date","code","p","listing_age_sessions","research_selection_allowed"]].copy()
    age_rows["age_equals_source_calendar_position"]=age_rows.listing_age_sessions.eq(age_rows.p)
    age_rows.to_csv(out/"COMPLETE_AGE_CLOCK_DIAGNOSTIC_NO_OUTCOMES.csv",index=False,encoding="utf-8-sig")
    age={name:dict(rows=int(mask.sum()),equals_source_position=int((mask&age_rows.age_equals_source_calendar_position).sum()),
        equal_fraction=float(age_rows.loc[mask,"age_equals_source_calendar_position"].mean()),
        max_source_position=int(age_rows.loc[mask,"p"].max()),min_source_position=int(age_rows.loc[mask,"p"].min()),
        unique_age=int(age_rows.loc[mask,"listing_age_sessions"].nunique()),
        median_age=float(age_rows.loc[mask,"listing_age_sessions"].median())) for name,mask in masks.items()}
    metrics=pd.DataFrame(records)
    tops={u:metrics[(metrics.universe.eq(u))&metrics.period.eq("VALIDATION2025H1")].sort_values(["psi_with_missing_bin","feature"],ascending=[False,True]).head(12).to_dict("records") for u in boundaries}
    for path,digest in protocol["inputs"].items():assert base.sha(path)==digest,path
    result=dict(formed_at=base.now(),status="OUTCOME_BLIND_AUDIT_FINISHED",scopes=scopes,age_clock=age,
        top_distribution_changes_DIAGNOSTIC_ONLY=tops,reported_metric_rows=len(records),reported_bin_rows=len(counts),
        outcome_columns_read=False,model_fits=0,new_forward_results=0,actual_trades=0,goal_60_passed=False)
    base.write_json(out/"report.json",result)
    print(json.dumps(dict(scopes=scopes,age_clock=age,top_shift_features={u:[(v["feature"],v["psi_with_missing_bin"],v["above_reference_max_count"],v["missing_fraction"]) for v in a] for u,a in tops.items()}),ensure_ascii=True,indent=2))
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True);parser.add_argument("--resume",action="store_true")
    args=parser.parse_args();run(args.output,args.resume)
