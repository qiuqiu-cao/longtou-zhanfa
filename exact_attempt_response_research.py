"""One fixed corrected-input challenge after362verified failed-row omissions.

Same ST_DUAL estimator, features, mature members, weights and0.5 decision.
Only independently verified attempt counts/source qualification may differ.
"""
from datetime import datetime,timezone
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
from short_term_thinking_preprocessing import EmptyTrainingColumnGuard
import historical_symbolic_research as evaluator
import model_upgrade_compat as compat

ROOT=b.ROOT/"research/model_upgrade_validation/exact_attempt_response_20261015T100800Z"
SOURCE=b.ROOT/"research/model_upgrade_validation/workbench_latest_20261015T091200Z/attempt_reclassification_correction"
DATA=SOURCE/"corrected_dataset_SCHEMA_COMPLETE_RESEARCH_NOT_FITTED.joblib"
ENGINE=SOURCE/"corrected_engine_context_THROUGH_OCT15_RESEARCH_NOT_FITTED.joblib"
ARM="ST_DUAL_EXACT_FAILED_ATTEMPTS"
FEATURES=old.FEATURES
ALLOWED={"same_height_failed_count","same_height_elimination_rate","is_divergence_survivor","role_USURP_DRAGON"}


def load():
    corrected=joblib.load(DATA);original=joblib.load(b.DEFAULT_RUN/"scope_correction/dataset.joblib")
    full=old.merged(corrected);before=old.merged(original)
    assert list(full)==list(before)
    pd.testing.assert_frame_equal(full.drop(columns=list(ALLOWED|{"primary_role","roles"})),before.drop(columns=list(ALLOWED|{"primary_role","roles"})))
    old_current=joblib.load(b.DEFAULT_RUN/"scope_correction/current_extension.joblib")["full"]
    engine=joblib.load(ENGINE);current=old_current.drop(columns=list(set(b.DYNAMIC_FIELDS)&set(old_current))).merge(
        engine["features"][["date","code"]+b.DYNAMIC_FIELDS],on=["date","code"],validate="one_to_one")
    pd.testing.assert_frame_equal(current[list(old_current)].reset_index(drop=True),old_current.reset_index(drop=True),check_dtype=False)
    if len(full)!=9245 or len(current)!=64:raise ValueError("fixed full pools required")
    return full,current


def freeze():
    if (ROOT/"protocol_BEFORE_ANY_FIT.json").exists():raise FileExistsError("fixed source-correction protocol exists")
    proof=b.read(SOURCE/"independent_verification.json");b.verify(proof["source_bindings"])
    if proof["restored_failed_rows"]!=362 or b.sha(DATA)!=proof["schema_complete_dataset_sha256"]:raise ValueError("exact362verified correction required")
    if b.sha(ENGINE)!=proof["current_workbench_engine_sha256"]:raise ValueError("corrected current capsule changed")
    full,current=load();checks=[]
    original=joblib.load(b.DEFAULT_RUN/"scope_correction/dataset.joblib");oldfull=old.merged(original)
    for month in pd.period_range("2025-07","2026-10",freq="M"):
        cutoff=str(month.start_time.date());train=old.history.training_rows(full,cutoff,"relay");previous=old.history.training_rows(oldfull,cutoff,"relay")
        keys=["date","code","target_date","target","label_valid"]
        pd.testing.assert_frame_equal(train[keys],previous[keys]);np.testing.assert_array_equal(old.training_weights(train),old.training_weights(previous))
        if train.date.nunique()!=252 or train.target_date.max()>=cutoff:raise ValueError("exact252 prior mature dates required")
        path=b.DEFAULT_RUN/"execution_after_binning_repair"/(str(month)+"_ST_DUAL_model.joblib")
        receipt=b.read(path.with_name(str(month)+"_ST_DUAL_fit_STARTED.json"))
        if old.history._frame_digest(previous,FEATURES)!=receipt["training_digest"]:raise ValueError("old training reference changed")
        checks.append(dict(month=str(month),cutoff=cutoff,rows=len(train),days=252,max_target=train.target_date.max(),
            old_training_digest=receipt["training_digest"],new_training_digest=old.history._frame_digest(train,FEATURES),
            existing_model_path=str(path.resolve()),existing_model_sha256=b.sha(path)))
    paths=[DATA,ENGINE,SOURCE/"independent_verification.json",SOURCE/"correction_and_impact_report.json",Path(__file__),Path(old.__file__),Path(evaluator.__file__),
        b.ROOT/"short_term_attempt_reconstruction.py",b.ROOT/"short_term_thinking_preprocessing.py",b.DEFAULT_RUN/"delivery/complete_paired_predictions_VALIDATED.csv",
        b.ROOT/"research/model_upgrade_validation/historical_symbolic_context_20261015T081200Z/execution_context.joblib",
        Path(r"C:\Users\Administrator\.codex\attachments\c84f9d98-65db-439c-be78-35c4eb490852\goal-objective.md")]
    sources=proof["source_bindings"]|{str(p.resolve()):b.sha(p) for p in paths}
    protocol=dict(formed_at=b.now(),version="exact-attempt-input-correction-fixed-comparison-1",arm=ARM,
        hypothesis="Restoring verified touched-upper-but-not-closed failed attempts and qualifying the unresolved over-cap date may improve same-height elimination evidence without altering learner/weights.",
        new_evidence=dict(restored_failed_rows=362,affected_original_candidates=684,verified_source_rows=58772,over_cap_scope_unknown_date="2026-08-07"),
        old_failed_input_packages_not_replayed=True,original_static_fields_targets_and_members_unchanged=True,fields=FEATURES,allowed_changed_numeric_fields=sorted(ALLOWED),
        parameters=old.PARAMS,weights="Original ST_DUAL inverse-date count and equal total class mass, exact values unchanged",threshold="strictly>0.5 unchanged",
        fits_max=16,fit_budget_seconds=1800,one_attempt_per_month=True,stop_on_failure=True,no_result_based_retune=True,training_checks=checks,
        screen=evaluator.SCREEN,folds=old.history.FOLDS,old_models_reused_not_refitted=True,old_original_BASE_unchanged=True,
        historical_data_previously_seen=True,blind=False,source_receipts_not_backfilled=True,no_trades=True,source_bindings=sources,
        versions={k:importlib.metadata.version(k) for k in ["numpy","pandas","scikit-learn","joblib"]})
    b.write_new(ROOT/"protocol_BEFORE_ANY_FIT.json",protocol);return protocol


def run():
    protocol=b.read(ROOT/"protocol_BEFORE_ANY_FIT.json");b.verify(protocol["source_bindings"])
    if protocol["fields"]!=FEATURES or protocol["parameters"]!=old.PARAMS or protocol["screen"]!=evaluator.SCREEN:raise ValueError("fixed protocol changed")
    b.write_new(ROOT/"experiment_STARTED.json",dict(formed_at=b.now(),protocol_sha256=b.sha(ROOT/"protocol_BEFORE_ANY_FIT.json")))
    out=ROOT/"fixed_comparison";out.mkdir(exist_ok=False)
    full,current=load();saved=pd.read_csv(b.DEFAULT_RUN/"delivery/complete_paired_predictions_VALIDATED.csv",dtype={"code":str},low_memory=False)
    original=saved[saved.arm.isin(["BASE","ST_DUAL"])].copy();base=original[original.arm.eq("BASE")].set_index(["date","code"]).score
    started=datetime.now(timezone.utc);parts=[]
    for check in protocol["training_checks"]:
        month,cutoff=check["month"],check["cutoff"]
        if (datetime.now(timezone.utc)-started).total_seconds()>protocol["fit_budget_seconds"]:raise TimeoutError("fixed30min fit budget")
        train=old.history.training_rows(full,cutoff,"relay")
        if old.history._frame_digest(train,FEATURES)!=check["new_training_digest"]:raise ValueError("corrected input changed after preregistration")
        test=current.copy() if month=="2026-10" else full[full.date.str[:7].eq(month)].copy()
        b.write_new(out/(month+"_fit_STARTED.json"),dict(formed_at=b.now(),cutoff=cutoff,arm=ARM,train_rows=len(train),training_digest=check["new_training_digest"],max_target=train.target_date.max(),fit_calls=1))
        try:
            model=Pipeline([("empty_training_columns",EmptyTrainingColumnGuard()),("hgb",HistGradientBoostingClassifier(**old.PARAMS))])
            with threadpool_limits(limits=2):
                model.fit(train[FEATURES],train.target.astype(int),hgb__sample_weight=old.training_weights(train))
                values=model.predict_proba(test[FEATURES])[:,list(model.classes_).index(1)]
            joblib.dump(dict(model=model,features=FEATURES,cutoff=cutoff,arm=ARM,trained_at=b.now(),estimator=old.PARAMS,role_version=b.VERSION,
                training_rows=len(train),training_digest=check["new_training_digest"],correction_version="exact-price-attempt-reclassification-1"),out/(month+"_model.joblib"))
            test["b_score"]=values;test["score"]=[base.loc[(x.date,x.code)] if x.daily_one_word==1 else values[i] for i,x in enumerate(test.itertuples())]
            test["arm"]=ARM;test["model_cutoff"]=cutoff
            columns=["date","code","name","boards","target_date","target","label_valid","daily_one_word","actual_one_word_daily","primary_role","score","b_score","arm","model_cutoff"]
            test[columns].to_csv(out/(month+"_predictions_BEFORE_EVALUATION.csv"),index=False,encoding="utf-8-sig");parts.append(test[columns])
            print(month,"corrected-input challenger saved",len(test),flush=True)
        except Exception as error:
            b.write_new(ROOT/"FAILURE_STOP.json",dict(formed_at=b.now(),month=month,error=repr(error),refit_allowed=False));raise
    pred=pd.concat([original]+parts,ignore_index=True)
    if len(pred)!=11313 or pred.duplicated(["date","code","arm"]).any():raise ValueError("full3771per-arm cohort required")
    pred.to_csv(out/"complete_paired_predictions.csv",index=False,encoding="utf-8-sig")
    eval_module=compat.clone_module(evaluator,{"ARM":ARM})
    context=joblib.load(b.ROOT/"research/model_upgrade_validation/historical_symbolic_context_20261015T081200Z/execution_context.joblib")
    result,logs=eval_module.evaluate(pred,context)
    result.update(successful_new_corrected_input_fits=16,old_BASE_refits=0,old_ST_DUAL_refits=0,historical_not_blind=True,new_mature_samples=0,actual_fills=0,
        source_correction_not_algorithm_search=True,original_before_correction_results_untouched=True)
    logs.to_csv(out/"all_top1_execution_scenarios.csv",index=False,encoding="utf-8-sig")
    b.write_new(out/"model_validation_results.json",result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("command",choices=["freeze","run"]);args=parser.parse_args();result=globals()[args.command]()
    print({k:result.get(k) for k in ["formed_at","successful_new_corrected_input_fits","prediction_screen_pass"]})
