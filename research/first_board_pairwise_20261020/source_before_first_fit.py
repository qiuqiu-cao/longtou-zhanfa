"""One preregistered first-board-close pairwise ranking challenge."""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
from pathlib import Path
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
import first_board_leader_model as base
import first_second_stage_model as stage

OUT=base.ROOT/"research/first_board_pairwise_20261020"


def pair_inputs(a,b):
    return np.concatenate([a,b,a-b],axis=1)


class PairRanker:
    def __init__(self,model,observed,fields):
        self.model=model;self.observed=observed;self.fields=fields

    def predict(self,frame):
        result=[]
        for day,cohort in frame.groupby("signal_date",sort=True):
            cohort=cohort.sort_values("code");n=len(cohort)
            if n==1:
                result.append(dict(signal_date=day,code=cohort.code.iloc[0],ranking_score=np.nan,opponents=0,ranking_status="SINGLETON_NO_COMPARATIVE_SCORE"));continue
            x=cohort[self.fields].to_numpy(float)
            left,right=np.where(~np.eye(n,dtype=bool))
            pairs=pair_inputs(x[left],x[right])
            with threadpool_limits(limits=2):p=self.model.predict_proba(pairs[:,self.observed])[:,1]
            matrix=np.full((n,n),np.nan);matrix[left,right]=p
            symmetric=.5*(matrix+1-matrix.T)
            assert np.nanmax(np.abs(symmetric+symmetric.T-1))<1e-12
            for i,code in enumerate(cohort.code):
                result.append(dict(signal_date=day,code=code,ranking_score=float(np.nanmean(symmetric[i])),opponents=n-1,ranking_status="COHORT_PREFERENCE_NOT_EVENT_PROBABILITY"))
        return pd.DataFrame(result)


def build_pairs(frame,fields):
    parts=[];targets=[];weights=[];groups=[]
    for day,cohort in frame.groupby("signal_date",sort=True):
        yes=cohort[cohort[stage.TARGET].eq(1)];no=cohort[cohort[stage.TARGET].eq(0)]
        n=len(yes)*len(no)
        groups.append(dict(signal_date=day,complete=len(cohort),positive=len(yes),negative=len(no),unknown=int(cohort[stage.TARGET].isna().sum()),directed_pairs=2*n))
        if not n:continue
        a=np.repeat(yes[fields].to_numpy(float),len(no),axis=0)
        b=np.tile(no[fields].to_numpy(float),(len(yes),1))
        parts.extend([pair_inputs(a,b),pair_inputs(b,a)])
        targets.extend([np.ones(n,dtype=np.int8),np.zeros(n,dtype=np.int8)])
        weights.extend([np.full(n,1/(2*n)),np.full(n,1/(2*n))])
    if not parts:raise ValueError("no mixed cohort can supervise ranking")
    x=np.concatenate(parts);y=np.concatenate(targets);w=np.concatenate(weights)
    w*=int(frame[stage.TARGET].notna().sum())/w.sum()
    return x,y,w,pd.DataFrame(groups)


def fit_ranker(frame,fields,params,name):
    directory=OUT/"models";directory.mkdir(exist_ok=True)
    complete=directory/f"{name}_COMPLETED.json";model_path=directory/f"{name}.joblib"
    if complete.exists():
        receipt=json.loads(complete.read_text("utf-8"));assert base.sha(model_path)==receipt["sha256"]
        return joblib.load(model_path)
    before=directory/f"{name}_BEFORE_FIT.json"
    if before.exists():raise RuntimeError("unresolved fit attempt: inspect original process before any resumption")
    x,y,w,groups=build_pairs(frame,fields)
    observed=~np.isnan(x).all(axis=0)
    receipt=dict(name=name,started_at=base.now(),frame_rows=len(frame),known_rows=int(frame[stage.TARGET].notna().sum()),directed_pairs=len(y),mixed_dates=int(groups.directed_pairs.gt(0).sum()),
                 features=fields,observed_pair_columns=np.flatnonzero(observed).tolist(),max_label_end=frame.loc[frame[stage.TARGET].notna(),"label_end"].max(),weight_mass=float(w.sum()))
    base.write_json(before,receipt)
    frame[["signal_date","code","p","label_end",stage.TARGET]].to_csv(directory/f"{name}_members_WITH_UNKNOWNS.csv",index=False)
    groups.to_csv(directory/f"{name}_complete_cohorts.csv",index=False)
    model=HistGradientBoostingClassifier(**params)
    with threadpool_limits(limits=2):model.fit(x[:,observed],y,sample_weight=w)
    ranker=PairRanker(model,observed,fields);joblib.dump(ranker,model_path)
    receipt.update(completed_at=base.now(),sha256=base.sha(model_path))
    base.write_json(complete,receipt)
    return ranker


def score(ranker,frame):
    return frame.merge(ranker.predict(frame),on=["signal_date","code"],how="left",validate="one_to_one")


def summarize(frame,column):
    return {str(k):stage.metric(frame,score=column,k=k) for k in (1,3,5)}


def train_window(rows,sessions,boundary):
    pos=int(np.searchsorted(sessions,boundary))
    frame=rows[rows.p.ge(pos-252)&rows.signal_date.lt(boundary)].copy()
    frame.loc[frame.label_end.ge(boundary),stage.TARGET]=np.nan
    return frame


def run():
    started=time.monotonic()
    if (OUT/"report.json").exists():raise FileExistsError("completed comparison is immutable")
    protocol=json.loads((OUT/"protocol_BEFORE_ANY_FIT.json").read_text("utf-8"))
    assert base.sha(protocol["reference"])==protocol["reference_sha256"]
    assert base.sha(protocol["input"])==protocol["input_sha256"]
    rows=pd.read_csv(protocol["input"],dtype={"code":str},float_precision="round_trip");rows=rows[rows.stage.eq(1)].copy()
    reference=pd.read_csv(protocol["reference"],dtype={"code":str},float_precision="round_trip")
    sessions=json.loads(base.INPUTS["calendar"].read_text("utf-8"))["sessions"]
    fields=protocol["fields"];params=protocol["core_parameters"]
    train=rows[rows.signal_date.lt("2025-01-01")].copy();train.loc[train.label_end.ge("2025-01-01"),stage.TARGET]=np.nan
    ranker=fit_ranker(train,fields,params,"validation")
    val=score(ranker,rows[rows.signal_date.between("2025-01-01","2025-06-30")])
    val.to_csv(OUT/"validation_ALL_PREDICTIONS_BEFORE_METRICS.csv",index=False)
    refpath=Path(protocol["input"]).parent/"validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv"
    valref=pd.read_csv(refpath,dtype={"code":str},float_precision="round_trip")
    validation=dict(pairwise=summarize(val,"ranking_score"),reference=summarize(valref,"model_score"))
    proceed=validation["pairwise"]["1"]["hits"]>=validation["reference"]["1"]["hits"]
    base.write_json(OUT/"validation_DECISION_BEFORE_TEST.json",dict(formed_at=base.now(),continue_to_test=proceed,metrics=validation))
    if not proceed:
        report=dict(completed_at=base.now(),status="REJECTED_VALIDATION",validation=validation,fits=len(list((OUT/"models").glob("*_COMPLETED.json"))),test_fits=0,goal_60_passed=False,goal_status="ACTIVE",code_sha256=base.sha(__file__))
        base.write_json(OUT/"report.json",report);print(json.dumps(base.clean(report),ensure_ascii=False,indent=2));return report
    parts=[]
    for month,frame in rows[rows.signal_date.ge("2025-07-01")].groupby(rows[rows.signal_date.ge("2025-07-01")].signal_date.str[:7],sort=True):
        if time.monotonic()-started>=protocol["fit_budget_seconds"]:
            base.write_json(OUT/"budget_stop_PARTIAL_NOT_COMPLETE.json",dict(at=base.now(),completed_months=[x.model_month.iloc[0] for x in parts],elapsed=time.monotonic()-started));return
        trained=fit_ranker(train_window(rows,sessions,month+"-01"),fields,params,month)
        result=score(trained,frame);result["model_month"]=month
        result.to_csv(OUT/f"{month}_ALL_PREDICTIONS_BEFORE_METRICS.csv",index=False)
        parts.append(result)
    predicted=pd.concat(parts,ignore_index=True)
    predicted=predicted.merge(reference[["signal_date","code","model_score","calibrated_score"]],on=["signal_date","code"],validate="one_to_one")
    predicted.to_csv(OUT/"ALL_PREDICTIONS.csv",index=False,encoding="utf-8-sig")
    periods={}
    for name,mask in [("all",pd.Series(True,index=predicted.index)),("2025H2",predicted.signal_date.lt("2026-01-01")),("2026",predicted.signal_date.ge("2026-01-01")),("non_one_price",predicted.one_price.eq(0))]:
        subset=predicted[mask]
        paired=subset[["signal_date","code",stage.TARGET,"ranking_score","model_score"]].rename(columns={stage.TARGET:base.TARGET,"model_score":"baseline_score","ranking_score":"model_score"})
        periods[name]=dict(pairwise=summarize(subset,"ranking_score"),reference=summarize(subset,"model_score"),paired_top1=base.paired_interval(paired,k=1))
    first_for_chain=predicted.drop(columns=["model_score","calibrated_score"]).rename(columns={"ranking_score":"model_score"});first_for_chain["calibrated_score"]=np.nan
    second_path=Path(protocol["input"]).parent/"S2_ALL_PREDICTIONS.csv"
    second=pd.read_csv(second_path,dtype={"code":str},float_precision="round_trip")
    chain,chain_metrics=stage.chain_recheck(first_for_chain,second);chain.to_csv(OUT/"FIRST_PICK_TO_UNCHANGED_SECOND_HEAD.csv",index=False,encoding="utf-8-sig")
    metric=periods["all"]["pairwise"]["1"]
    passed=bool(metric["hit_rate"]>=.6 and metric["known"]>=200 and metric["outcome_coverage"]>=.95 and all(periods[p]["pairwise"]["1"]["hit_rate"]>=.55 for p in ("2025H2","2026")))
    last=rows.signal_date.max();final=fit_ranker(train_window(rows,sessions,last),fields,params,"latest_research")
    old=joblib.load(Path(protocol["input"]).parent/"first_second_stage_bundle.joblib")
    joblib.dump(dict(first_ranker=final,second_head=old["heads"][2],knowledge_cutoff=last,first_score_is_probability=False,research_only=True),OUT/"first_ranker_second_recheck_bundle.joblib")
    report=dict(completed_at=base.now(),status="FULL_COMPARISON_COMPLETE",evaluation=periods,chain=chain_metrics,validation=validation,
                goal_60_passed=passed,goal_status="TARGET_POINT_METRIC_REACHED_RESEARCH_ONLY" if passed else "ACTIVE_60_NOT_REACHED",
                fits=len(list((OUT/"models").glob("*_COMPLETED.json"))),elapsed_seconds=time.monotonic()-started,code_sha256=base.sha(__file__),
                reference_sha256=protocol["reference_sha256"],input_sha256=protocol["input_sha256"],first_score_is_probability=False,actual_trades=0,production_allowed=False)
    base.write_json(OUT/"report.json",report)
    print(json.dumps(base.clean(dict(status=report["status"],top1=metric,reference=periods["all"]["reference"]["1"],paired=periods["all"]["paired_top1"],goal60=passed,fits=report["fits"])),ensure_ascii=False,indent=2))
    return report


if __name__=="__main__":
    # Persist classes under their importable module name, not __main__.
    from first_board_pairwise_model import run as run_importable
    run_importable()
