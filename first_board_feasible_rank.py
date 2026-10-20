"""Fixed comparison: admissible first-board classification and daily ranking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import first_board_leader_model as base
import first_second_stage_model as stage
import first_board_capital_risk_model as risk
import first_board_matured_event_model as previous
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

REF=risk.REFERENCE
SCREEN=previous.SCREEN
DEP=base.ROOT/"research/first_board_feasible_rank_dependency_20261020"
sys.path.insert(0,str(DEP/"runtime"))
import lightgbm as lgb

FIELDS=list(stage.VARIANTS["STAGE_CYCLE_GENE"])
ARMS=("ADMISSIBLE_HGB","ADMISSIBLE_LAMBDARANK")
RANK_PARAMS=dict(objective="lambdarank",boosting_type="gbdt",n_estimators=160,num_leaves=15,
    min_child_samples=60,learning_rate=.06,reg_lambda=8.,max_depth=-1,
    class_weight=None,subsample=1.,subsample_freq=0,colsample_bytree=1.,
    random_state=42,n_jobs=2,verbosity=-1,max_bin=255,deterministic=True,force_col_wise=True,
    lambdarank_truncation_level=4,lambdarank_norm=True,label_gain=[0,1],metric="ndcg")


def training_rows(rows,boundary,start_p=None):
    selected=rows.research_selection_allowed & rows[stage.TARGET].notna() & rows.signal_date.lt(boundary) & rows.label_end.lt(boundary)
    if start_p is not None:selected &= rows.p.ge(start_p)
    return rows[selected].sort_values(["signal_date","code"]).copy()


def daily_queries(train):
    if train.duplicated(["signal_date","code"]).any():raise ValueError("Duplicate signal/code query entry")
    if not train[stage.TARGET].isin([0,1]).all():raise ValueError("Ranking labels must be known binary promotions")
    if not train.research_selection_allowed.all():raise ValueError("Excluded stock in admissible training query")
    ordered=train.sort_values(["signal_date","code"]).copy()
    sizes=ordered.groupby("signal_date",sort=False).size()
    weights=1/ordered.groupby("signal_date",sort=False).code.transform("size")
    weights=weights/weights.mean()
    if int(sizes.sum())!=len(ordered):raise ValueError("Query sizes do not match contiguous input rows")
    return ordered,sizes.to_numpy(int),weights.to_numpy(float)


def fit_rank_core(train,out,name):
    directory=out/"models";directory.mkdir(exist_ok=True)
    before=directory/(name+"_BEFORE_FIT.json")
    if before.exists():raise FileExistsError("Fixed rank core already attempted")
    ordered,groups,weights=daily_queries(train)
    observed=[f for f in FIELDS if ordered[f].notna().any()]
    receipt=dict(name=name,started_at=base.now(),rows=len(ordered),dates=ordered.signal_date.nunique(),
        positives=int(ordered[stage.TARGET].sum()),max_label_end=ordered.label_end.max(),
        min_signal=ordered.signal_date.min(),max_signal=ordered.signal_date.max(),features=FIELDS,
        observed_fields=observed,params=RANK_PARAMS,library_version=lgb.__version__,
        sample_weights="Same equal-total-day row weights,normalized to mean1;native LambdaRank query normalization also enabled",
        objective_semantics="Binary label gains0/1,NDCG-oriented swaps focused near the top;raw score is not a probability",
        no_early_stopping=True,no_validation_test_used_for_fit=True)
    base.write_json(before,receipt)
    members=ordered[["signal_date","code","stage","label_end",stage.TARGET]].copy()
    members["sample_weight"]=weights
    members.to_csv(directory/(name+"_members.csv"),index=False)
    ledger=members.groupby("signal_date",sort=False).agg(query_size=("code","size"),positive_labels=(stage.TARGET,"sum"),weight_sum=("sample_weight","sum"),max_label_end=("label_end","max"))
    ledger.to_csv(directory/(name+"_queries.csv"))
    model=lgb.LGBMRanker(**RANK_PARAMS)
    model.fit(ordered[observed],ordered[stage.TARGET].astype(int),group=groups,sample_weight=weights,eval_at=(1,))
    joblib.dump(model,directory/(name+".joblib"))
    receipt.update(completed_at=base.now(),sha256=base.sha(directory/(name+".joblib")),
        actual_trees=model.n_estimators_,queries_all_negative=int(ledger.positive_labels.eq(0).sum()),
        queries_all_positive=int(ledger.positive_labels.eq(ledger.query_size).sum()))
    base.write_json(directory/(name+"_COMPLETED.json"),receipt)
    return model,observed


def fit_head(train,arm,out,name):
    if arm==ARMS[0]:
        stage.VARIANTS[arm]=FIELDS
        return stage.fit_head(train,arm,out,name)
    dates=sorted(train.signal_date.unique())
    if len(dates)<100:raise ValueError("Insufficient temporal training days")
    cutoff=dates[-42]
    inner=train[train.label_end.lt(cutoff)]
    cal=train[train.signal_date.ge(cutoff)]
    prefix,observed=fit_rank_core(inner,out,name+"_calibration_prefix")
    raw=prefix.predict(cal[observed])
    calibrator=LogisticRegression(C=1.,max_iter=1000,random_state=42).fit(raw.reshape(-1,1),cal[stage.TARGET].astype(int))
    positive=bool(calibrator.coef_[0,0]>0)
    if not positive:calibrator=None
    cal[["signal_date","code","stage","label_end",stage.TARGET]].assign(raw_rank_oof=raw).to_csv(out/"models"/(name+"_calibration_rows.csv"),index=False)
    base.write_json(out/"models"/(name+"_calibration_receipt.json"),dict(formed_at=base.now(),cutoff=cutoff,
        inner_max_label_end=inner.label_end.max(),calibration_rows=len(cal),positive_slope=positive,
        raw_transform="Raw LambdaRank margin,direct PlattC1;no sigmoid or probability interpretation before calibration"))
    model,observed=fit_rank_core(train,out,name+"_full")
    bundle=dict(version="first-board-admissible-rank-20261020",stage=1,variant=arm,model=model,features=FIELDS,
        observed_fields=observed,calibrator=calibrator,train_prior=float(train[stage.TARGET].mean()),
        train_max_label_end=train.label_end.max(),max_signal=train.signal_date.max(),
        raw_rank_score=True,calibration_status="HISTORICAL_QUALIFICATION_PENDING",research_only=True,
        dependency_runtime=str((DEP/"runtime").resolve()),dependency_version=lgb.__version__)
    joblib.dump(bundle,out/"models"/(name+"_bundle.joblib"))
    return bundle


def predict(bundle,rows):
    if not bundle.get("raw_rank_score",False):return stage.predict(bundle,rows)
    result=rows.copy()
    raw=bundle["model"].predict(result[bundle["observed_fields"]])
    result["model_score"]=raw
    result["calibrated_score"]=bundle["calibrator"].predict_proba(raw.reshape(-1,1))[:,1] if bundle["calibrator"] is not None else np.nan
    result["train_prior"]=bundle["train_prior"]
    return result


def run(output,resume_prefit=False):
    out=Path(output)
    if out.exists():
        allowed={"source_before_dll_path_repair.py","PREFIT_DEPENDENCY_PATH_FAILURE.json"}
        if not resume_prefit or {p.name for p in out.iterdir()}!=allowed:
            raise FileExistsError("One fixed comparison,refuse overwrite/repeat")
        failure=json.loads((out/"PREFIT_DEPENDENCY_PATH_FAILURE.json").read_text("utf-8"))
        assert failure["completed_core_fits"]==0 and failure["fit_started"] is False
    else:
        if resume_prefit:raise ValueError("No preserved pre-fit failure to resume")
        out.mkdir(parents=True)
    source=REF/"all_first_and_second_boards_WITH_SEPARATE_LABELS.csv"
    screen_file=SCREEN/"COMPLETE_FIRST_BOARD_FACTS_FEATURES_LABELS_AND_SCREEN.csv"
    validation_file=REF/"validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv"
    protected=[Path(base.__file__),Path(stage.__file__),Path(risk.__file__),Path(previous.__file__),
        REF/"report.json",REF/"S1_ALL_PREDICTIONS.csv",REF/"S2_ALL_PREDICTIONS.csv",SCREEN/"report.json"]
    sources=[source,screen_file,validation_file,DEP/"pypi_metadata_request.json",DEP/"wheel_acquisition.json",DEP/"official_sources.json",
        DEP/"lightgbm-4.7.0-py3-none-win_amd64.whl",DEP/"runtime/lightgbm/bin/lib_lightgbm.dll",DEP/"runtime/lightgbm/sklearn.py",DEP/"runtime/lightgbm/basic.py",
        base.ROOT/"research/first_board_saved_rank_audit_20261020/report.json"]
    protocol=dict(formed_at=base.now(),batch_started_at="2026-10-20T13:55:13+00:00",budget_minutes=30,
        previous_goal_turn="PROGRESS:complete17873rows/403days audit verified fixed-screen106/284and280/286days with available known winners;therefore test feasible daily-query training,not more failed indicators",
        hypothesis="Conditioning training on the unchanged admissible universe,and optimizing daily near-top ordering,improves complete-pool first-close Top1 promotion versus the saved full-universe classifier",
        baseline="Exact frozen original62-field scores plus unchanged prior fixed capital/thin-turnover screen;no baseline/stage2retraining",
        arms={ARMS[0]:"Original fixed HGB/class-balanced/equal-day weighting on only admissible known matured training rows",ARMS[1]:"Same admissible rows/62fields with daily grouped LambdaRank;raw margins rank stocks,not probabilities"},
        fields=FIELDS,rank_params=RANK_PARAMS,rank_truncation_rationale="Official default guidance k+3;primaryk1,hence fixed4,beforeanyfit;no search",
        training="Pre2025signals and labels for validation;selected arm only original14monthly preceding252official-position windows,known label_end<monthstart;admissible membership uses only original frozen screen",
        calibration="Each arm fixed prefix before final42trainingdates,strict maturity,PlattC1positive slope;HGBlogit versus directrankmargin;test qualification reported separately",
        validation="Complete original4724rows/117dates2025H1. Require strictly>52Top1hits,>=48non-one-price Top1hits,>=95%decision-datecoverage. Baseline wins ties;passing arms mosthits,HGBwinsarmties",
        evaluation="Only the selected passing arm,fixed14monthly windows;save full pool scores beforemetrics;compare original screened and unscreened references and same-stock savedsecondhead",
        target60="Top1knownhitrate>=60%,>=200known,>=95%outcomecoverage,noabstentiondates;2025H2and2026each>=55%.Same-stocksecondclose independentrecheck. No auction input",
        group_policy="Sort date/code beforegroup;all known binary labels;all-negative/all-positive/singleton queries retained with receipts;equal total day rowweights;nativequerynormalization recorded",
        distinction_from_prior_pairwise="No newleft/right pairs or pairwise Borda aggregation. Existing failed335016pair HGBandfeatures remain untouched;new grouped objective and admissible membership explicitly differ",
        stop="Two arms once;no new windows/parameters/thresholds afterresults;no nextmonthlyfits ifneitherpasses. Stop fixedcomparison or30minutes",
        library_version=lgb.__version__,python_version=sys.version,source_sha256=base.sha(__file__),
        inputs={str(p.resolve()):base.sha(p) for p in sources},protected={str(p.resolve()):base.sha(p) for p in protected},
        raw_source_hashes={str(p.resolve()):base.sha(p) for p in base.INPUTS.values()},
        historical_research_not_pristine_blind=True,new_forward_results=0,actual_trades=0,production_changes=False,
        prefit_technical_repair="Official wheel DLL lives underbin,notlib;zero models attempted;original failure/source retained" if resume_prefit else None)
    assert lgb.__version__=="4.7.0"
    base.write_json(out/"protocol_BEFORE_ANY_FIT.json",protocol);shutil.copy2(__file__,out/"source_before_first_fit.py")
    original=pd.read_csv(source,dtype={"code":str},float_precision="round_trip")
    rows=original[original.stage.eq(1)].copy()
    screen=pd.read_csv(screen_file,dtype={"code":str},float_precision="round_trip")
    rows=rows.merge(screen[["signal_date","code","research_selection_allowed","exclusion_reason","threeboard_evidence_status"]],on=["signal_date","code"],validate="one_to_one")
    assert len(rows)==28503
    rows.to_csv(out/"ALL_FIRST_BOARD_UNCHANGED_FEATURES_LABELS_AND_SCREEN.csv",index=False,encoding="utf-8-sig")
    train=training_rows(rows,"2025-01-01")
    val=rows[rows.signal_date.between("2025-01-01","2025-06-30")]
    frozen=risk.attach_frozen(rows,validation_file)
    baseline=risk.screen_metric(frozen);baseline_non=risk.screen_metric(frozen[frozen.one_price.eq(0)])
    assert baseline["hits"]==52 and baseline_non["hits"]==48
    base.write_json(out/"training_membership_BEFORE_FITS.json",dict(original_mature_training=len(rows[rows[stage.TARGET].notna()&rows.label_end.lt("2025-01-01")] ),
        admissible_training=len(train),dates=train.signal_date.nunique(),positives=int(train[stage.TARGET].sum()),max_label_end=train.label_end.max(),
        complete_validation_rows=len(val),complete_validation_unknown=int(val[stage.TARGET].isna().sum()),
        validation_days=val.signal_date.nunique(),fields_unchanged=True,unknowns_not_negative=True))
    decisions={}
    for arm in ARMS:
        bundle=fit_head(train,arm,out,"validation_"+arm)
        score=predict(bundle,val)
        score.to_csv(out/("validation_"+arm+"_ALL_BEFORE_METRICS.csv"),index=False)
        primary=risk.screen_metric(score);non=risk.screen_metric(score[score.one_price.eq(0)])
        decisions[arm]=dict(primary=primary,non_one_price=non,unfiltered_diagnostic=stage.metric(score),
            pass_validation=bool(primary["hits"]>baseline["hits"] and non["hits"]>=baseline_non["hits"] and primary["decision_date_coverage"]>=.95))
        print(json.dumps(base.clean(dict(arm=arm,validation=decisions[arm])),ensure_ascii=True),flush=True)
    passing=[arm for arm in ARMS if decisions[arm]["pass_validation"]]
    selected=max(passing,key=lambda a:decisions[a]["primary"]["hits"]) if passing else None
    decision=dict(formed_at=base.now(),selected=selected,baseline=baseline,baseline_non_one_price=baseline_non,arms=decisions)
    base.write_json(out/"selection_BEFORE_EVALUATION.json",decision)
    report=dict(formed_at=None,decision=decision,goal_60_passed=False,new_forward_results=0,actual_trades=0)
    if selected:
        sessions=json.loads(base.INPUTS["calendar"].read_text("utf-8"))["sessions"]
        parts=[]
        for month,part in rows[rows.signal_date.ge("2025-07-01")].groupby(rows.signal_date.str[:7],sort=True):
            boundary=month+"-01";p=int(np.searchsorted(sessions,boundary))
            train=training_rows(rows,boundary,p-252)
            head=fit_head(train,selected,out,month+"_"+selected)
            score=predict(head,part);score["model_month"]=month
            score.to_csv(out/(month+"_ALL_BEFORE_METRICS.csv"),index=False);parts.append(score)
            print(month+" saved",flush=True)
        scores=pd.concat(parts,ignore_index=True)
        scores.to_csv(out/"EVALUATION_ALL_PREDICTIONS.csv",index=False,encoding="utf-8-sig")
        frozen=risk.attach_frozen(rows,REF/"S1_ALL_PREDICTIONS.csv")
        report["evaluation"]=previous.period_metrics(scores,frozen)
        allowed=scores[scores.research_selection_allowed]
        second=pd.read_csv(REF/"S2_ALL_PREDICTIONS.csv",dtype={"code":str},float_precision="round_trip")
        chain,report["same_stock_second_recheck"]=stage.chain_recheck(allowed,second)
        chain.to_csv(out/"SAME_STOCK_SECOND_RECHECK_ALL.csv",index=False,encoding="utf-8-sig")
        report["historical_calibration_allowed_pool"]=stage.calibration_metrics(allowed)
        m=report["evaluation"]["all"]["new_screened"]
        report["goal_60_passed"]=bool(m["hit_rate"]>=.60 and m["known"]>=200 and m["outcome_coverage"]>=.95 and m["abstention_dates"]==0 and all(report["evaluation"][period]["new_screened"]["hit_rate"]>=.55 for period in ("2025H2","2026")))
    for path,digest in protocol["protected"].items():assert base.sha(path)==digest
    report.update(formed_at=base.now(),core_fit_count=len(list((out/"models").glob("*_COMPLETED.json"))),
        status="FIXED_MONTHLY_EVALUATION_FINISHED" if selected else "BOTH_ARMS_REJECTED_NO_MONTHLY_FITS")
    base.write_json(out/"report.json",report)
    print(json.dumps(base.clean(report),ensure_ascii=True,indent=2),flush=True)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True);parser.add_argument("--resume-prefit",action="store_true")
    args=parser.parse_args();run(args.output,args.resume_prefit)
