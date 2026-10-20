"""Fixed regularized-linear versus chronological-ordered first-board learning."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import first_board_leader_model as base
import first_second_stage_model as stage
import first_board_capital_risk_model as risk
import first_board_matured_event_model as prior

REF=risk.REFERENCE
SCREEN=prior.SCREEN
DEP=base.ROOT/"research/first_board_ordered_learning_dependency_20261020"
sys.path.insert(0,str(DEP/"runtime"))
FIELDS=list(stage.VARIANTS["STAGE_CYCLE_GENE"])
ARMS=("REGULARIZED_LINEAR","CHRONO_ORDERED")
CAT_PARAMS=dict(iterations=160,depth=4,learning_rate=.06,l2_leaf_reg=8.,loss_function="Logloss",
    boosting_type="Ordered",has_time=True,grow_policy="SymmetricTree",bootstrap_type="No",
    border_count=255,nan_mode="Min",random_strength=1.,random_seed=42,thread_count=2,
    task_type="CPU",allow_writing_files=False,use_best_model=False,verbose=False)
LINEAR_PARAMS=dict(C=.25,class_weight="balanced",max_iter=1000,random_state=42,
    preprocessing="Training-only median imputation plus missing indicators and StandardScaler,original recipe")


def prepare_train(train,knowledge_cutoff):
    if train.duplicated(["signal_date","code"]).any():raise ValueError("Duplicate training key")
    if not train[stage.TARGET].isin([0,1]).all():raise ValueError("Only known binary matured targets allowed")
    if not train.stage.eq(1).all():raise ValueError("First-board head requires stage1")
    if not (train.signal_date.lt(knowledge_cutoff)&train.label_end.lt(knowledge_cutoff)).all():
        raise ValueError("Training signal or target not strictly mature before knowledge cutoff")
    ordered=train.sort_values(["signal_date","code"]).copy()
    weights=1/ordered.groupby("signal_date").code.transform("size")
    weights=(weights/weights.mean()).to_numpy(float)
    counts=np.bincount(ordered[stage.TARGET].to_numpy(int),minlength=2)
    if (counts==0).any():raise ValueError("Both training classes required")
    class_weights=(len(ordered)/(2*counts)).tolist()
    return ordered,weights,class_weights


def cat_class():
    import catboost
    if catboost.__version__!="1.2.10":raise ValueError("Pinned CatBoost version mismatch")
    return catboost.CatBoostClassifier


def fit_core(train,arm,out,name,knowledge_cutoff):
    directory=out/"models";directory.mkdir(exist_ok=True)
    before=directory/(name+"_BEFORE_FIT.json")
    if before.exists():raise FileExistsError("Core already attempted;no automatic refit")
    train,weights,class_weights=prepare_train(train,knowledge_cutoff)
    observed=[f for f in FIELDS if train[f].notna().any()]
    receipt=dict(name=name,arm=arm,started_at=base.now(),rows=len(train),dates=train.signal_date.nunique(),
        positives=int(train[stage.TARGET].sum()),min_signal=train.signal_date.min(),max_signal=train.signal_date.max(),max_label_end=train.label_end.max(),
        fields=FIELDS,observed_fields=observed,dropped_all_nan_training_only=sorted(set(FIELDS)-set(observed)),
        class_weights=class_weights,weight_policy="Same normalized equal-total-day weights;class multipliersn/(2classcount)computedfromthistrainingset",
        input_order="signal_date,code ascending",params=LINEAR_PARAMS if arm==ARMS[0] else CAT_PARAMS,
        knowledge_cutoff=knowledge_cutoff,
        test_data_used_for_fit=False,no_test_dependent_early_stopping=True)
    base.write_json(before,receipt)
    members=train[["signal_date","code","stage","label_end",stage.TARGET]].copy()
    members["sample_weight"]=weights
    members["class_multiplier"]=members[stage.TARGET].map({0:class_weights[0],1:class_weights[1]})
    members.to_csv(directory/(name+"_members.csv"),index=False)
    if arm==ARMS[0]:model=base.make_model("logistic")
    else:model=cat_class()(**CAT_PARAMS,class_weights=class_weights)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if arm==ARMS[0]:model.fit(train[observed],train[stage.TARGET].astype(int),logisticregression__sample_weight=weights)
        else:model.fit(train[observed],train[stage.TARGET].astype(int),sample_weight=weights)
    joblib.dump(model,directory/(name+".joblib"))
    if arm==ARMS[1]:
        model.save_model(str(directory/(name+".cbm")))
        receipt["native_sha256"]=base.sha(directory/(name+".cbm"));receipt["actual_tree_count"]=model.tree_count_
        receipt["actual_params"]=model.get_all_params()
    else:receipt["actual_solver_iterations"]=model.named_steps["logisticregression"].n_iter_.tolist()
    receipt.update(completed_at=base.now(),sha256=base.sha(directory/(name+".joblib")),warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in caught])
    base.write_json(directory/(name+"_COMPLETED.json"),receipt)
    return model,observed


def fit_head(train,arm,out,name,knowledge_cutoff):
    dates=sorted(train.signal_date.unique())
    if len(dates)<100:raise ValueError("Insufficient temporal training dates")
    cutoff=dates[-42]
    inner=train[train.label_end.lt(cutoff)]
    cal=train[train.signal_date.ge(cutoff)].sort_values(["signal_date","code"])
    prefix,observed=fit_core(inner,arm,out,name+"_calibration_prefix",cutoff)
    raw=prefix.predict_proba(cal[observed])[:,1]
    logits=np.log(np.clip(raw,1e-6,1-1e-6)/np.clip(1-raw,1e-6,1-1e-6))
    calibrator=LogisticRegression(C=1.,max_iter=1000,random_state=42).fit(logits.reshape(-1,1),cal[stage.TARGET].astype(int))
    positive=bool(calibrator.coef_[0,0]>0)
    if not positive:calibrator=None
    cal[["signal_date","code","stage","label_end",stage.TARGET]].assign(uncalibrated_oof=raw).to_csv(out/"models"/(name+"_calibration_rows.csv"),index=False)
    base.write_json(out/"models"/(name+"_calibration_receipt.json"),dict(formed_at=base.now(),cutoff=cutoff,inner_max_label_end=inner.label_end.max(),calibration_rows=len(cal),positive_slope=positive))
    model,observed=fit_core(train,arm,out,name+"_full",knowledge_cutoff)
    bundle=dict(version="first-board-regularized-ordered-20261020",arm=arm,variant=arm,stage=1,features=FIELDS,
        observed_fields=observed,model=model,calibrator=calibrator,train_prior=float(train[stage.TARGET].mean()),
        train_max_label_end=train.label_end.max(),max_signal=train.signal_date.max(),research_only=True,
        calibration_status="HISTORICAL_QUALIFICATION_PENDING",dependency_runtime=str((DEP/"runtime").resolve()))
    joblib.dump(bundle,out/"models"/(name+"_bundle.joblib"))
    return bundle


def predict(bundle,rows):
    out=rows.copy()
    raw=bundle["model"].predict_proba(out[bundle["observed_fields"]])[:,1]
    out["model_score"]=raw
    if bundle["calibrator"] is not None:
        logits=np.log(np.clip(raw,1e-6,1-1e-6)/np.clip(1-raw,1e-6,1-1e-6))
        out["calibrated_score"]=bundle["calibrator"].predict_proba(logits.reshape(-1,1))[:,1]
    else:out["calibrated_score"]=np.nan
    out["train_prior"]=bundle["train_prior"]
    return out


def run(output,linear_only=False):
    active_arms=ARMS[:1] if linear_only else ARMS
    if not linear_only:
        acquisition=json.loads((DEP/"wheel_acquisition.json").read_text("utf-8"))
        if acquisition.get("import_status")!="PASS":raise ValueError("CatBoost dependency not qualified before fitting")
        cat_class()
        import catboost._catboost as native_module
    out=Path(output)
    if out.exists():raise FileExistsError("Preserve prior attempts;no repeated fitting")
    out.mkdir(parents=True)
    source=REF/"all_first_and_second_boards_WITH_SEPARATE_LABELS.csv"
    screen_file=SCREEN/"COMPLETE_FIRST_BOARD_FACTS_FEATURES_LABELS_AND_SCREEN.csv"
    protected=[Path(base.__file__),Path(stage.__file__),Path(risk.__file__),Path(prior.__file__),
        REF/"report.json",REF/"S1_ALL_PREDICTIONS.csv",REF/"S2_ALL_PREDICTIONS.csv",SCREEN/"report.json"]
    sources=[source,screen_file,DEP/"pypi_metadata_request.json",DEP/"official_sources.json",
        base.ROOT/"research/first_board_generalization_gap_20261020/report.json",base.ROOT/"research/first_board_generalization_gap_20261020/verification.json"]
    if not linear_only:sources.extend([DEP/"wheel_acquisition.json",DEP/"catboost-1.2.10-cp313-cp313-win_amd64.whl",DEP/"runtime/catboost/core.py",Path(native_module.__file__)])
    protocol=dict(formed_at=base.now(),batch_started_at="2026-10-20T14:44:53+00:00",budget_minutes=30,
        previous_goal_turn="PROGRESS:matched3heads and6cores acrosscompletecandidatepools showed same42day insample80.95%versusheldout45to52%;newtest addresseslearnergeneralization,not morefailedfeatures",
        hypothesis="Regularizedlinear orchronologicallyOrdered boosting on original62fields transfers first-close ranking better than frozenHGB",
        arms={arm:(LINEAR_PARAMS if arm==ARMS[0] else CAT_PARAMS) for arm in active_arms},fields=FIELDS,
        deferred_arm=dict(arm=ARMS[1],params=CAT_PARAMS,reason="100MBofficialwheeltransferconfirmedlivebutnotyetcomplete;deferredbeforeanyfit,notarejectedmodel") if linear_only else None,
        baseline="Frozen62fieldHGBscores plusunchangedoriginalcapital/thinturnover risk screen;no baseline orstage2refit",
        training="Samefulloriginalknownmaturemembers;date/codeordered,equaldayweights,balancedrawclasscounts;no variable feature windows or risk thresholds",
        calibration="Exactlast42trainingdates;prefixlabel_end<cutoff;fixedPlattC1positiveslope;fullcoretrainedaftercalibration. Internal ordered prefixes are not certified forward predictions",
        validation="Complete2025H1:4724rows/117dates. Need strictly>52Top1hits,>=48non-one-pricehits,>=95%decisiondatecoverage;baselinewinsties;passingarmsmosthits,linearwinsarmties",
        evaluation="Onlyselectedpasser,original14monthly252officialpositionwindows,label_end<monthstart;savefullpredictionsbeforemetrics;same-codesecondcloserecheckwithsavedsecondhead",
        target60="DailyTop1knownhitrate>=60%,>=200known,>=95%outcomecoverage,noabstentiondates,both2025H2and2026>=55%;firstcloseonly,no nextauctioninput",
        no_validation_eval_set_or_best_iteration=True,catboost_version="1.2.10",all_missing_column_policy="Training-only omit;sourceNaNfacts unchanged",
        stop="At2armsfixedcomparison or30minutes;no testtuning/earlystopping/windowchanges. Ifneitherpasses,no monthlyfits",
        inputs={str(p.resolve()):base.sha(p) for p in sources},protected={str(p.resolve()):base.sha(p) for p in protected},
        raw_source_hashes={str(p.resolve()):base.sha(p) for p in base.INPUTS.values()},source_sha256=base.sha(__file__),
        historical_research_not_pristine_blind=True,new_forward_results=0,actual_trades=0,production_changes=False)
    base.write_json(out/"protocol_BEFORE_ANY_FIT.json",protocol);shutil.copy2(__file__,out/"source_before_first_fit.py")
    rows=pd.read_csv(source,dtype={"code":str},float_precision="round_trip");rows=rows[rows.stage.eq(1)].copy()
    screen=pd.read_csv(screen_file,dtype={"code":str},float_precision="round_trip")
    rows=rows.merge(screen[["signal_date","code","research_selection_allowed","exclusion_reason","threeboard_evidence_status"]],on=["signal_date","code"],validate="one_to_one")
    rows.to_csv(out/"ALL_FIRST_BOARD_UNCHANGED_FEATURES_LABELS_SCREEN.csv",index=False,encoding="utf-8-sig")
    known=rows[rows[stage.TARGET].notna()]
    train=known[known.signal_date.lt("2025-01-01")&known.label_end.lt("2025-01-01")]
    val=rows[rows.signal_date.between("2025-01-01","2025-06-30")]
    frozen=risk.attach_frozen(rows,REF/"validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv")
    baseline=risk.screen_metric(frozen);nonbase=risk.screen_metric(frozen[frozen.one_price.eq(0)])
    assert baseline["hits"]==52 and nonbase["hits"]==48
    decisions={}
    for arm in active_arms:
        bundle=fit_head(train,arm,out,"validation_"+arm,"2025-01-01")
        score=predict(bundle,val)
        score.to_csv(out/("validation_"+arm+"_ALL_BEFORE_METRICS.csv"),index=False)
        primary=risk.screen_metric(score);non=risk.screen_metric(score[score.one_price.eq(0)])
        decisions[arm]=dict(primary=primary,non_one_price=non,unfiltered_diagnostic=stage.metric(score),
            pass_validation=bool(primary["hits"]>52 and non["hits"]>=48 and primary["decision_date_coverage"]>=.95))
        print(json.dumps(base.clean(dict(arm=arm,validation=decisions[arm])),ensure_ascii=True),flush=True)
    passing=[a for a in active_arms if decisions[a]["pass_validation"]]
    selected=max(passing,key=lambda a:decisions[a]["primary"]["hits"]) if passing else None
    decision=dict(formed_at=base.now(),selected=selected,baseline=baseline,baseline_non_one_price=nonbase,arms=decisions)
    base.write_json(out/"selection_BEFORE_EVALUATION.json",decision)
    report=dict(formed_at=None,decision=decision,goal_60_passed=False,new_forward_results=0,actual_trades=0,
        deferred_ordered_not_fitted=linear_only)
    if selected:
        sessions=json.loads(base.INPUTS["calendar"].read_text("utf-8"))["sessions"];parts=[]
        for month,part in rows[rows.signal_date.ge("2025-07-01")].groupby(rows.signal_date.str[:7],sort=True):
            boundary=month+"-01";p=int(np.searchsorted(sessions,boundary))
            tr=known[known.p.ge(p-252)&known.label_end.lt(boundary)]
            bundle=fit_head(tr,selected,out,month+"_"+selected,boundary)
            score=predict(bundle,part);score["model_month"]=month
            score.to_csv(out/(month+"_ALL_BEFORE_METRICS.csv"),index=False);parts.append(score)
            print(month+" saved",flush=True)
        scores=pd.concat(parts,ignore_index=True)
        scores.to_csv(out/"EVALUATION_ALL_PREDICTIONS.csv",index=False,encoding="utf-8-sig")
        frozen=risk.attach_frozen(rows,REF/"S1_ALL_PREDICTIONS.csv")
        report["evaluation"]=prior.period_metrics(scores,frozen)
        allowed=scores[scores.research_selection_allowed]
        second=pd.read_csv(REF/"S2_ALL_PREDICTIONS.csv",dtype={"code":str},float_precision="round_trip")
        chain,report["same_stock_second_recheck"]=stage.chain_recheck(allowed,second)
        chain.to_csv(out/"SAME_STOCK_SECOND_RECHECK_ALL.csv",index=False,encoding="utf-8-sig")
        report["historical_calibration_allowed_pool"]=stage.calibration_metrics(allowed)
        m=report["evaluation"]["all"]["new_screened"]
        report["goal_60_passed"]=bool(m["hit_rate"]>=.60 and m["known"]>=200 and m["outcome_coverage"]>=.95 and m["abstention_dates"]==0 and all(report["evaluation"][x]["new_screened"]["hit_rate"]>=.55 for x in ("2025H2","2026")))
    for path,digest in protocol["protected"].items():assert base.sha(path)==digest
    report.update(formed_at=base.now(),core_fit_count=len(list((out/"models").glob("*_COMPLETED.json"))),
        status="FIXED_MONTHLY_EVALUATION_FINISHED" if selected else ("LINEAR_REJECTED_ORDERED_NOT_FITTED" if linear_only else "BOTH_ARMS_REJECTED_NO_MONTHLY_FITS"))
    base.write_json(out/"report.json",report)
    print(json.dumps(base.clean(report),ensure_ascii=True,indent=2),flush=True)
    return report

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True);parser.add_argument("--linear-only",action="store_true")
    args=parser.parse_args();run(args.output,args.linear_only)
