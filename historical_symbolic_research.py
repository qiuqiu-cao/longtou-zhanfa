"""One new feature hypothesis on top of the saved ST_DUAL, no baseline refit.

The only changed estimator input is17 dated-name diffusion fields. Parameters,
monthly clocks, training members, day/class weights and0.5 threshold stay fixed.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

import short_term_thinking_dataset_builder as b
import short_term_thinking_research as old
from short_term_thinking_preprocessing import EmptyTrainingColumnGuard
import historical_symbolic_context as symbolic

ROOT=symbolic.RUN
DATA=ROOT/"qualified_final"
ARM="ST_DUAL_SYMBOLIC_DIFFUSION"
FIELDS=old.FEATURES+symbolic.FIELDS
SCREEN=dict(nonword_recall_not_lower_each_original_fold=True,nonword_precision_not_lower_each_original_fold=True,
    nonword_f1_not_lower_each_original_fold=True,top1_and_top2_not_lower_each_original_fold=True,
    september_precision_floor_vs_original_BASE=.55,september_recall_floor=.50,
    september_precision_or_top1_strict_gain=True,all_pool_accuracy_plus3pp_veto=False,
    execution_and_identity_validation_separate_from_prediction_screen=True,no_deployment_from_this_retrospective=True)


def load_frames():
    original=joblib.load(b.DEFAULT_RUN/"scope_correction/dataset.joblib")
    full=old.merged(original)
    current=joblib.load(b.DEFAULT_RUN/"scope_correction/current_extension.joblib")["full"]
    rows=joblib.load(DATA/"symbolic_context.joblib")["rows"]
    def join(frame):
        result=frame.merge(rows[["date","code"]+symbolic.FIELDS],on=["date","code"],validate="one_to_one")
        if len(result)!=len(frame):raise ValueError("original candidate loss in symbolic join")
        pd.testing.assert_frame_equal(result[list(frame.columns)].reset_index(drop=True),frame.reset_index(drop=True))
        return result
    return join(full),join(current)


def freeze():
    if (ROOT/"protocol_BEFORE_ANY_FIT.json").exists():raise FileExistsError("fixed symbolic protocol already frozen")
    qualification=b.read(DATA/"source_qualification_and_coverage.json")
    b.verify(qualification["source_bindings"])
    for w in qualification["windows"].values():
        if w["full_archive_name_diffusion_computable"]<.95 or w["dated_name_interval_coverage"]<.95:
            raise ValueError("new historical source coverage did not pass preregistered95% source floor")
    full,current=load_frames()
    checks=[]
    for month in pd.period_range("2025-07","2026-10",freq="M"):
        cutoff=str(month.start_time.date());train=old.history.training_rows(full,cutoff,"relay")
        if train.date.nunique()!=252 or train.target_date.max()>=cutoff:raise ValueError("exact252 mature prior-date clock required")
        baseline=b.DEFAULT_RUN/"execution_after_binning_repair"/(str(month)+"_ST_DUAL_model.joblib")
        receipt=b.read(baseline.with_name(str(month)+"_ST_DUAL_fit_STARTED.json"))
        if old.history._frame_digest(train,old.FEATURES)!=receipt["training_digest"]:raise ValueError("original training inputs changed")
        checks.append(dict(month=str(month),cutoff=cutoff,training_rows=len(train),training_days=252,max_target=train.target_date.max(),
            original_training_digest=receipt["training_digest"],new_training_digest=old.history._frame_digest(train,FIELDS),
            baseline_model_path=str(baseline.resolve()),baseline_model_sha256=b.sha(baseline)))
    sources=dict(qualification["source_bindings"])
    for path in [DATA/"symbolic_context.joblib",DATA/"source_qualification_and_coverage.json",ROOT/"execution_context.joblib",ROOT/"execution_source_coverage_report.json",
        b.DEFAULT_RUN/"delivery/complete_paired_predictions_VALIDATED.csv",b.DEFAULT_RUN/"scope_correction/dataset.joblib",b.DEFAULT_RUN/"scope_correction/current_extension.joblib",
        Path(__file__),b.ROOT/"short_term_thinking_preprocessing.py",b.ROOT/"historical_execution_context.py"]:
        sources[str(path.resolve())]=b.sha(path)
    protocol=dict(version="symbolic-single-incremental-feature-test-1",formed_at=b.now(),hypothesis="Dated-name first-board diffusion and same-symbol ladder context may reduce ST_DUAL false positives while preserving its nonword recall and ranking.",
        distinction_from_failed_prior="Append17 newly source-qualified structural fields only; do not refit or retune ST_DUAL/ST_B_ONLY or old H/E/price/VWAP packages.",
        challenger=ARM,fields=FIELDS,estimator=old.PARAMS,weights="exact same ST_DUAL train-only inverse day size and equal total class mass",threshold="strictly>0.5 unchanged",
        role_policy=symbolic.POLICY,names_are_not_business_or_active_theme=True,original_role_and_label_rows_unchanged=True,
        training_checks=checks,max_fits=16,fit_budget_seconds=1800,one_attempt_per_month=True,stop_on_any_failure=True,no_effect_based_retuning=True,
        screen=SCREEN,folds=old.history.FOLDS,execution_context=str((ROOT/"execution_context.joblib").resolve()),
        source_policy="Historical dated-name reconstruction and exact source scope, no current-industry backfill; actual source receipts remain late.",
        historical_prior_data_already_seen=True,blind_test=False,source_bindings=sources,
        runtime_versions={k:importlib.metadata.version(k) for k in ["numpy","pandas","scikit-learn","joblib"]})
    b.write_new(ROOT/"protocol_BEFORE_ANY_FIT.json",protocol)
    return protocol


def fit_one(train,month,cutoff,output):
    prefix=Path(output)/month
    b.write_new(str(prefix)+"_fit_STARTED.json",dict(formed_at=b.now(),month=month,arm=ARM,cutoff=cutoff,training_rows=len(train),
        max_target=train.target_date.max(),training_digest=old.history._frame_digest(train,FIELDS),fit_calls=1))
    model=Pipeline([("empty_training_columns",EmptyTrainingColumnGuard()),("hgb",HistGradientBoostingClassifier(**old.PARAMS))])
    with threadpool_limits(limits=2):model.fit(train[FIELDS],train.target.astype(int),hgb__sample_weight=old.training_weights(train))
    saved=dict(model=model,features=FIELDS,arm=ARM,cutoff=cutoff,training_rows=len(train),training_digest=old.history._frame_digest(train,FIELDS),
        trained_at=b.now(),symbolic_version=symbolic.VERSION,role_version=b.VERSION,estimator=old.PARAMS,
        empty_training_columns=model.named_steps["empty_training_columns"].empty_columns_)
    joblib.dump(saved,str(prefix)+"_model.joblib")
    return model


def evaluate(pred,execution_context):
    result={}
    executions=[]
    for fold in old.history.FOLDS+[dict(id="2026Oct",start="2026-10-01",end="2026-10-31")]:
        subset=pred[pred.date.between(fold["start"],fold["end"])]
        parts={arm:subset[subset.arm.eq(arm)].copy() for arm in ["BASE","ST_DUAL",ARM]}
        metrics={}
        for arm,f in parts.items():
            nonword=f[f.daily_one_word.eq(0)]
            chosen=old.top_selection(f);log=old.proxy_returns(chosen,execution_context);log["arm"]=arm;log["fold"]=fold["id"];executions.append(log)
            multi=nonword.groupby(["date","boards"]).filter(lambda p:len(p)>1)
            metrics[arm]=dict(full=old.metrics(f),nonword=old.metrics(nonword),top1=old.top_metrics(f),top2=old.top_metrics(f,2),
                same_height_top1=old.top_metrics(f,1,True),same_height_top2=old.top_metrics(f,2,True),
                multi_candidate_same_height_top1=old.top_metrics(multi,1,True),height_baseline_top1=old.top_metrics(f,height=True),execution_proxy=old.execution_metrics(log))
        guards={}
        bm,cm=metrics["ST_DUAL"],metrics[ARM]
        for field in ["Recall","Precision","F1"]:
            a,z=bm["nonword"][field],cm["nonword"][field]
            guards[field+"_not_lower"]=a is not None and z is not None and z>=a-1e-14
        for key in ["top1","top2"]:
            a,z=bm[key]["candidate_hit_rate"],cm[key]["candidate_hit_rate"]
            guards[key+"_not_lower"]=a is not None and z is not None and z>=a-1e-14
        if fold["id"]=="2026Sep":
            guards["precision_floor_original_BASE"]=cm["nonword"]["Precision"] is not None and cm["nonword"]["Precision"]>=.55
            guards["recall_floor"]=cm["nonword"]["Recall"] is not None and cm["nonword"]["Recall"]>=.5
            guards["precision_or_top1_strict_gain"]=cm["nonword"]["Precision"]>bm["nonword"]["Precision"] or cm["top1"]["candidate_hit_rate"]>bm["top1"]["candidate_hit_rate"]
        result[fold["id"]]=dict(metrics=metrics,incremental_guards=guards,
            recall_interval_vs_ST_DUAL=old.paired_block(parts["ST_DUAL"],parts[ARM]),top1_interval_vs_ST_DUAL=old.paired_block(parts["ST_DUAL"],parts[ARM],"top1"))
    historical_guards=[v for fold,value in result.items() if fold!="2026Oct" for v in value["incremental_guards"].values()]
    return dict(formed_at=b.now(),comparisons=result,prediction_screen_pass=all(historical_guards),production_deployment_authorized=False,
        independent_identity_accuracy=None,new_actual_fills=0,execution_gain_confirmed=False),pd.concat(executions,ignore_index=True)


def run():
    protocol=b.read(ROOT/"protocol_BEFORE_ANY_FIT.json")
    if protocol["fields"]!=FIELDS or protocol["estimator"]!=old.PARAMS or protocol["screen"]!=SCREEN:raise ValueError("fixed hypothesis changed")
    b.verify(protocol["source_bindings"])
    b.write_new(ROOT/"experiment_STARTED.json",dict(formed_at=b.now(),protocol_sha256=b.sha(ROOT/"protocol_BEFORE_ANY_FIT.json")))
    started=datetime.now(timezone.utc);output=ROOT/"fixed_comparison";output.mkdir(exist_ok=False)
    full,current=load_frames()
    originals=pd.read_csv(b.DEFAULT_RUN/"delivery/complete_paired_predictions_VALIDATED.csv",dtype={"code":str},low_memory=False)
    base_parts=originals[originals.arm.isin(["BASE","ST_DUAL"])].copy()
    baseline=base_parts[base_parts.arm.eq("BASE")].set_index(["date","code"]).score
    rows=[]
    for check in protocol["training_checks"]:
        if (datetime.now(timezone.utc)-started).total_seconds()>protocol["fit_budget_seconds"]:raise TimeoutError("fixed30min fit budget")
        month,cutoff=check["month"],check["cutoff"]
        train=old.history.training_rows(full,cutoff,"relay")
        if old.history._frame_digest(train,FIELDS)!=check["new_training_digest"] or train.target_date.max()>=cutoff:raise ValueError("frozen mature training input changed")
        test=current.copy() if month=="2026-10" else full[full.date.str[:7].eq(month)].copy()
        try:
            model=fit_one(train,month,cutoff,output)
            with threadpool_limits(limits=2):score=model.predict_proba(test[FIELDS])[:,list(model.classes_).index(1)]
            test["b_score"]=score;test["score"]=[baseline.loc[(x.date,x.code)] if x.daily_one_word==1 else score[i] for i,x in enumerate(test.itertuples())]
            test["arm"]=ARM;test["model_cutoff"]=cutoff
            keep=["date","code","name","boards","target_date","target","label_valid","daily_one_word","actual_one_word_daily","primary_role","score","b_score","arm","model_cutoff"]
            test[keep].to_csv(output/(month+"_predictions_BEFORE_EVALUATION.csv"),index=False,encoding="utf-8-sig")
            rows.append(test[keep])
            print(month,"single challenger saved",len(test),flush=True)
        except Exception as error:
            b.write_new(ROOT/"FAILURE_STOP.json",dict(formed_at=b.now(),month=month,error=repr(error),refit_allowed=False));raise
    predictions=pd.concat([base_parts]+rows,ignore_index=True)
    if predictions.duplicated(["date","code","arm"]).any():raise ValueError("duplicate complete comparison row")
    predictions.to_csv(output/"complete_paired_predictions.csv",index=False,encoding="utf-8-sig")
    context=joblib.load(ROOT/"execution_context.joblib")
    result,logs=evaluate(predictions,context)
    result.update(successful_challenger_fits=16,old_ST_DUAL_refits=0,original_BASE_refits=0,new_forward_samples=0,retrospective_not_blind=True)
    logs.to_csv(output/"all_top1_execution_scenarios.csv",index=False,encoding="utf-8-sig")
    b.write_new(output/"model_validation_results.json",result)
    return result


def score_review(facts_path,review_path,model_path,output):
    """Explicit research view after original full independent review validation."""
    import stock_judgement_review as o
    import short_term_thinking_workbench as thinking
    import relay_review_workbench as workbench
    facts=o.read(facts_path);context=symbolic.checked_review(facts);causal=thinking.checked(facts)
    if context is None or causal is None:raise ValueError("both masked causal and symbolic context required")
    review=o.read(review_path)
    if not (review.get("author_review_completed") is True or review.get("status")=="INDEPENDENT_PREAUCTION_JUDGEMENT_NOT_PRODUCTION_FREEZE"):
        raise ValueError("completed independent review required before score display")
    if review.get("facts_sha256")!=b.sha(review["facts_path"]):raise ValueError("original independent facts changed")
    if review.get("status")=="INDEPENDENT_PREAUCTION_JUDGEMENT_NOT_PRODUCTION_FREEZE":
        registry=[json.loads(x) for x in (b.ROOT/"data/independent_stock_judgement/registry.jsonl").read_text('utf-8-sig').splitlines()]
        if not any(x["path"]==str(Path(review_path).resolve()) and x["sha256"]==b.sha(review_path) for x in registry):raise ValueError("unregistered independent record")
    else:workbench.validate_completed(review["facts_path"],review)
    b.verify(review["source_bindings"])
    if review["signal_date"]!=facts["signal_date"] or {x["code"] for x in review["rows"]}!={x["code"] for x in facts["rows"]}:raise ValueError("independent review full pool mismatch")
    model=joblib.load(model_path)
    if model["arm"]!=ARM or model["symbolic_version"]!=symbolic.VERSION or model["cutoff"]>facts["signal_date"]:raise ValueError("symbolic model clock or schema mismatch")
    pack=b.read(context["features_path"])
    base=pd.DataFrame([dict(x["features"],code=x["code"]) for x in pack["relay"]])
    dynamic=pd.DataFrame(causal["rows"]);overlap=(set(base)&set(dynamic))-{ "code" }
    frame=base.drop(columns=list(overlap)).merge(dynamic,on="code",validate="one_to_one")
    frame=frame.merge(pd.DataFrame(context["rows"])[["code"]+symbolic.FIELDS],on="code",validate="one_to_one")
    with threadpool_limits(limits=2):score=model["model"].predict_proba(frame[model["features"]])[:,list(model["model"].classes_).index(1)]
    rows=[]
    for i,x in enumerate(frame.itertuples()):
        word=x.daily_one_word
        rows.append(dict(code=x.code,name=x.name,boards=x.boards,channel="B_NONWORD" if word==0 else "A_EMOTION_ONLY" if word==1 else "UNRESOLVED_IDENTITY",
            score=float(score[i]) if word==0 else None,direction=("PROMOTE" if score[i]>.5 else "FAIL") if word==0 else "EMOTION_MONITOR_ONLY" if word==1 else "UNKNOWN_IDENTITY"))
    result=dict(formed_at=b.now(),status="SYMBOLIC_FEATURE_RESEARCH_SCORES_AFTER_REVIEW_NOT_PRODUCTION",signal_date=facts["signal_date"],rows=rows,
        model_path=str(Path(model_path).resolve()),model_sha256=b.sha(model_path),review_sha256=b.sha(review_path),uncalibrated=True,
        new_fits=0,new_formal_forecast=0,actual_fills=0)
    b.write_new(output,result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=["freeze","run"]);a=p.parse_args(argv);result=globals()[a.command]()
    print(b.clean({key:result.get(key) for key in ["formed_at","successful_challenger_fits","prediction_screen_pass"]}))


if __name__=="__main__":main()
