"""One fixed lagged-price-relation arm, appended to corrected causal inputs.

Existing BASE, ST_DUAL, symbolic and exact-attempt experiments are immutable.
No threshold, learner, member, target, or training-weight search is permitted.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import importlib.metadata

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

import short_term_thinking_dataset_builder as b
import short_term_thinking_research as old
import exact_attempt_response_research as previous
import historical_price_relation as relation
from short_term_thinking_preprocessing import EmptyTrainingColumnGuard

ROOT=b.ROOT/"research/model_upgrade_validation/lagged_price_relation_20261015T103000Z"
CONTEXT=ROOT/"causal_context_v2/price_relation_context.joblib"
ARM="ST_DUAL_LAGGED_PRICE_RELATION"
REFERENCE=previous.ARM
FEATURES=old.FEATURES+relation.FIELDS
SCREEN=dict(reference_arm=REFERENCE,each_historical_fold_nonword_recall_precision_f1_not_lower=True,
    each_historical_fold_top1_top2_not_lower=True,september_recall_min=.5,september_precision_min=.55,
    september_f1_not_lower_original_ST_DUAL=True,september_top1_over_height_and_BASE=True,
    significant_recall_gain_over_BASE_required_for_success_claim=True,
    all_pool_accuracy_plus3pp_veto=False,execution_and_identity_validation_separate=True,
    no_production_deployment_from_retrospective=True)


def load():
    full,current=previous.load()
    context=joblib.load(CONTEXT)
    extra=context["rows"][["date","code"]+relation.FIELDS]
    def join(frame):
        result=frame.merge(extra,on=["date","code"],validate="one_to_one")
        pd.testing.assert_frame_equal(result[list(frame)].reset_index(drop=True),frame.reset_index(drop=True))
        return result
    return join(full),join(current)


def freeze():
    if (ROOT/"protocol_BEFORE_ANY_FIT.json").exists():
        raise FileExistsError("one fixed price-relation protocol already exists")
    audit=b.read(ROOT/"source_qualification_BEFORE_NEW_FIT.json")
    b.verify(audit["source_bindings"])
    if audit["new_fits"]!=0 or audit["independent_pair_checks"]!=9555:
        raise ValueError("source-only independent qualification required")
    full,current=load();before,_=previous.load();checks=[]
    for month in pd.period_range("2025-07","2026-10",freq="M"):
        cutoff=str(month.start_time.date())
        train=old.history.training_rows(full,cutoff,"relay")
        original=old.history.training_rows(before,cutoff,"relay")
        pd.testing.assert_frame_equal(train[list(original)],original)
        np.testing.assert_array_equal(old.training_weights(train),old.training_weights(original))
        if train.date.nunique()!=252 or train.target_date.max()>=cutoff:
            raise ValueError("exact252 prior matured member dates required")
        checks.append(dict(month=str(month),cutoff=cutoff,rows=len(train),days=252,max_target=train.target_date.max(),
            original_digest=old.history._frame_digest(original,old.FEATURES),
            training_digest=old.history._frame_digest(train,FEATURES)))
    paths=[Path(__file__),Path(relation.__file__),Path(old.__file__),Path(previous.__file__),CONTEXT,
        ROOT/"source_qualification_BEFORE_NEW_FIT.json",previous.ROOT/"completion_verified.json",
        previous.ROOT/"fixed_comparison/complete_paired_predictions.csv",
        b.ROOT/"research/model_upgrade_validation/historical_symbolic_context_20261015T081200Z/execution_context.joblib"]
    protocol=dict(formed_at=b.now(),version="lagged-price-relation-one-arm-1",arm=ARM,reference_arm=REFERENCE,
        distinct_evidence="Lagged cross-stock residual return covariance across all current first/relay/failed stocks and active historical anchors; not a replay of name diffusion or a failed intraday threshold",
        source_phase_fits=0,features=FEATURES,added_fields=relation.FIELDS,relation_policy=relation.POLICY,
        estimator=old.PARAMS,training_checks=checks,weights="Original inverse-date and equal total class mass, exact values unchanged",
        threshold="strict >0.5",dual_channel="Observed true one-word reuses frozen original BASE; remaining rows use challenger B",
        fits_max=16,fit_budget_seconds=1800,one_attempt_per_month=True,stop_on_failure=True,
        input_or_outcome_based_retune_allowed=False,prior_failed_packages_repeated=False,
        screen=SCREEN,historical_data_previously_seen=True,blind=False,
        current_missing_relation_rows_retained=True,actual_fills=0,
        source_bindings=audit["source_bindings"]|{str(p.resolve()):b.sha(p) for p in paths},
        versions={k:importlib.metadata.version(k) for k in ["numpy","pandas","scikit-learn","joblib"]})
    b.write_new(ROOT/"protocol_BEFORE_ANY_FIT.json",protocol)
    return protocol


def evaluate(pred):
    context=joblib.load(b.ROOT/"research/model_upgrade_validation/historical_symbolic_context_20261015T081200Z/execution_context.joblib")
    result={};executions=[]
    for fold in old.history.FOLDS+[dict(id="2026Oct",start="2026-10-01",end="2026-10-31")]:
        group=pred[pred.date.between(fold["start"],fold["end"])];metrics={};parts={}
        for arm in ["BASE","ST_DUAL",REFERENCE,ARM]:
            frame=group[group.arm.eq(arm)].copy();parts[arm]=frame
            nonword=frame[frame.daily_one_word.eq(0)]
            log=old.proxy_returns(old.top_selection(frame),context);log["arm"]=arm;log["fold"]=fold["id"];executions.append(log)
            metrics[arm]=dict(full=old.metrics(frame),nonword=old.metrics(nonword),top1=old.top_metrics(frame),top2=old.top_metrics(frame,2),
                same_height_top1=old.top_metrics(frame,1,True),same_height_top2=old.top_metrics(frame,2,True),
                height_baseline_top1=old.top_metrics(frame,height=True),execution_proxy=old.execution_metrics(log))
        ref,new=metrics[REFERENCE],metrics[ARM];guards={}
        for field in ["Recall","Precision","F1"]:
            a,z=ref["nonword"][field],new["nonword"][field]
            guards[field+"_not_lower"]=a is not None and z is not None and z>=a-1e-14
        for key in ["top1","top2"]:
            a,z=ref[key]["candidate_hit_rate"],new[key]["candidate_hit_rate"]
            guards[key+"_not_lower"]=a is not None and z is not None and z>=a-1e-14
        if fold["id"]=="2026Sep":
            guards.update(recall_floor=new["nonword"]["Recall"]>=.5,
                precision_floor=new["nonword"]["Precision"]>=.55,
                f1_vs_original_dual=new["nonword"]["F1"]>=metrics["ST_DUAL"]["nonword"]["F1"],
                top1_over_height_and_BASE=new["top1"]["candidate_hit_rate"]>max(metrics["BASE"]["top1"]["candidate_hit_rate"],new["height_baseline_top1"]["candidate_hit_rate"]))
        result[fold["id"]]=dict(metrics=metrics,incremental_guards=guards,
            recall_interval_vs_corrected_reference=old.paired_block(parts[REFERENCE],parts[ARM]),
            recall_interval_vs_BASE=old.paired_block(parts["BASE"],parts[ARM]),
            top1_interval_vs_corrected_reference=old.paired_block(parts[REFERENCE],parts[ARM],"top1"))
    passed=all(v for key,fold in result.items() if key!="2026Oct" for v in fold["incremental_guards"].values())
    return dict(formed_at=b.now(),comparisons=result,prediction_screen_pass=passed,
        new_challenger_fits=16,BASE_refits=0,ST_DUAL_refits=0,exact_attempt_reference_refits=0,
        independent_identity_accuracy=None,execution_gain_confirmed=False,new_mature_samples=0,
        actual_fills=0,production_deployment_authorized=False),pd.concat(executions,ignore_index=True)


def run():
    protocol=b.read(ROOT/"protocol_BEFORE_ANY_FIT.json");b.verify(protocol["source_bindings"])
    if protocol["features"]!=FEATURES or protocol["estimator"]!=old.PARAMS or protocol["screen"]!=SCREEN:
        raise ValueError("fixed protocol changed")
    b.write_new(ROOT/"experiment_STARTED.json",dict(formed_at=b.now(),protocol_sha256=b.sha(ROOT/"protocol_BEFORE_ANY_FIT.json")))
    output=ROOT/"fixed_comparison";output.mkdir(exist_ok=False)
    full,current=load()
    existing=pd.read_csv(previous.ROOT/"fixed_comparison/complete_paired_predictions.csv",dtype={"code":str},low_memory=False)
    base=existing[existing.arm.eq("BASE")].set_index(["date","code"]).score
    started=datetime.now(timezone.utc);parts=[]
    for check in protocol["training_checks"]:
        month,cutoff=check["month"],check["cutoff"]
        if (datetime.now(timezone.utc)-started).total_seconds()>protocol["fit_budget_seconds"]:
            raise TimeoutError("fixed30minute fitting budget")
        train=old.history.training_rows(full,cutoff,"relay")
        if old.history._frame_digest(train,FEATURES)!=check["training_digest"]:
            raise ValueError("frozen training input changed")
        test=current.copy() if month=="2026-10" else full[full.date.str[:7].eq(month)].copy()
        b.write_new(output/(month+"_fit_STARTED.json"),dict(formed_at=b.now(),arm=ARM,cutoff=cutoff,train_rows=len(train),fit_calls=1,training_digest=check["training_digest"]))
        try:
            model=Pipeline([("empty_training_columns",EmptyTrainingColumnGuard()),("hgb",HistGradientBoostingClassifier(**old.PARAMS))])
            with threadpool_limits(limits=2):
                model.fit(train[FEATURES],train.target.astype(int),hgb__sample_weight=old.training_weights(train))
                values=model.predict_proba(test[FEATURES])[:,list(model.classes_).index(1)]
            joblib.dump(dict(model=model,features=FEATURES,cutoff=cutoff,arm=ARM,trained_at=b.now(),estimator=old.PARAMS,
                training_rows=len(train),training_digest=check["training_digest"],relation_version=relation.VERSION),output/(month+"_model.joblib"))
            test["b_score"]=values
            test["score"]=[base.loc[(x.date,x.code)] if x.daily_one_word==1 else values[i] for i,x in enumerate(test.itertuples())]
            test["arm"]=ARM;test["model_cutoff"]=cutoff
            columns=["date","code","name","boards","target_date","target","label_valid","daily_one_word","actual_one_word_daily","primary_role","score","b_score","arm","model_cutoff"]
            test[columns].to_csv(output/(month+"_predictions_BEFORE_EVALUATION.csv"),index=False,encoding="utf-8-sig")
            parts.append(test[columns]);print(month,"new relation predictions saved",len(test),flush=True)
        except Exception as error:
            b.write_new(ROOT/"FAILURE_STOP.json",dict(formed_at=b.now(),month=month,error=repr(error),refit_allowed=False))
            raise
    pred=pd.concat([existing]+parts,ignore_index=True)
    if len(pred)!=15084 or pred.duplicated(["date","code","arm"]).any():
        raise ValueError("complete3771each of4arms required")
    pred.to_csv(output/"complete_paired_predictions.csv",index=False,encoding="utf-8-sig")
    result,logs=evaluate(pred)
    logs.to_csv(output/"all_top1_execution_scenarios.csv",index=False,encoding="utf-8-sig")
    b.write_new(output/"model_validation_results.json",result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("command",choices=["freeze","run"])
    args=parser.parse_args();result=globals()[args.command]()
    print({k:result.get(k) for k in ["formed_at","new_challenger_fits","prediction_screen_pass"]})
