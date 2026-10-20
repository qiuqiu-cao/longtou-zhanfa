"""One new two-arm corrected-feature/cost comparison, never rerun old trials."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

import short_term_thinking_dataset_builder as b
import short_term_thinking_research as previous
import model_upgrade_history as history
from short_term_thinking_preprocessing import EmptyTrainingColumnGuard
import expectation_345_evaluation as evaluation

ROOT = b.ROOT / "research/model_upgrade_validation/expectation_345_20261015T163226Z"
OLD = b.ROOT / "research/model_upgrade_validation/causal_scope_alignment_20261015T121000Z"
ARMS = ("CORRECTED_CAUSAL", "CORRECTED_CAUSAL_COST1P5")
NEGATIVE_COST = {ARMS[0]:1., ARMS[1]:1.5}
PERIODS = history.FOLDS + [dict(id="2026Oct", start="2026-10-01", end="2026-10-31")]


def weights(train, negative_cost):
    if train.target.nunique() != 2: raise ValueError("two classes required")
    w = 1 / train.groupby("date").date.transform("size")
    mass = w.groupby(train.target).sum()
    w = w * train.target.map({0:negative_cost/mass[0],1:1./mass[1]})
    return w / w.sum() * train.date.nunique()


def qualify():
    prep=b.read(ROOT / "preparation_BEFORE_ANY_NEW_FIT.json");b.verify(prep["source_bindings"])
    source=joblib.load(ROOT / "trainable_corrected_dataset.joblib")
    full=source["full"].copy();fields=list(source["fields"])
    sem=joblib.load(OLD / "context/review_scope_context.joblib")["rows"].set_index(["date","code"])
    status=sem.loc[pd.MultiIndex.from_frame(full[["date","code"]]),"full_cohort_status"].to_numpy()
    repairs={"PARTIAL_MEMBER_OUTCOMES":"PARTIAL_MEMBERS_UNKNOWN", "NOT_APPLICABLE_EMPTY_PREVIOUS_COHORT":"NOT_APPLICABLE_EMPTY_COHORT"}
    changes=[]
    for old,new in repairs.items():
        a,c="causal_cohort_"+old,"causal_cohort_"+new
        assert full[a].eq(0).all()
        full[c]=(status==new).astype(float)
        full=full.drop(columns=a);fields[fields.index(a)]=c
        changes.append(dict(old_name=a,new_name=c,observed_count=int(full[c].sum()),technical_schema_fix_before_fitting=True))
    assert len(full)==9315 and len(fields)==92 and not full.duplicated(["date","code"]).any()
    full=full[[c for c in full if c not in fields]+fields]
    data=dict(full=full,fields=fields,sessions=source["sessions"],version="corrected-causal-trainable-345-v2-source-enum-qualified")
    joblib.dump(data,ROOT / "trainable_corrected_dataset_QUALIFIED.joblib")
    full.to_csv(ROOT / "all9315_trainable_features_QUALIFIED.csv",index=False,encoding="utf-8-sig")
    b.write_new(ROOT / "source_enum_qualification_BEFORE_FIT.json",dict(formed_at=b.now(),repairs=changes,new_fits=0,original_dataset_retained=True,
        source_bindings={str(p.resolve()):b.sha(p) for p in [ROOT/"trainable_corrected_dataset.joblib",ROOT/"trainable_corrected_dataset_QUALIFIED.joblib",Path(__file__)]}))
    return data


def references(full):
    frozen=pd.read_csv(OLD/"evaluation/all15084_frozen_predictions_with_explicit_channels.csv",dtype={"code":str},low_memory=False)
    metadata=full[full.date.between("2025-07-01","2026-10-31")][["date","code","name","boards","target_date","target","label_valid","daily_one_word","current_price_shape","review_cycle_state","current_price_path_status"]]
    parts=[]
    current=b.read(OLD/"workbench/完整81只_全部证据与原审阅.json")["rows"]
    for arm in ["BASE","ST_DUAL","ST_DUAL_LAGGED_PRICE_RELATION"]:
        source=frozen[frozen.arm.eq(arm)]
        score=source[["date","code","score"]].copy()
        for row in current:
            if row["date"]!="2026-10-15":continue
            key="current15_after_review_BASE" if arm=="BASE" else "current15_after_review_ST_DUAL" if arm=="ST_DUAL" else "lagged_price_relation_research_score"
            score=pd.concat([score,pd.DataFrame([dict(date=row["date"],code=row["code"],score=row[key])])],ignore_index=True)
        frame=metadata.merge(score,on=["date","code"],validate="one_to_one")
        assert len(frame)==3777 and frame.score.notna().all()
        truth=source[["date","code","target","label_valid"]].merge(frame[["date","code","target","label_valid"]],on=["date","code"],suffixes=("_old","_new"),validate="one_to_one")
        for key in ["target","label_valid"]:
            assert ((truth[key+"_old"]==truth[key+"_new"])|(truth[key+"_old"].isna()&truth[key+"_new"].isna())).all(),key
        frame["arm"]=arm;frame["ranking_score"]=frame.score
        frame["call"]=np.where(frame.score.gt(.5),"PROMOTE","FAIL")
        frame.loc[frame.daily_one_word.isna(),"call"]="UNKNOWN"
        parts.append(frame)
    return pd.concat(parts,ignore_index=True)


def freeze():
    if (ROOT/"protocol_BEFORE_ANY_FIT.json").exists():raise FileExistsError("fixed batch already frozen")
    data=qualify();full=data["full"];fields=data["fields"]
    ref=references(full);ref.to_csv(ROOT/"frozen_references_3777_each.csv",index=False,encoding="utf-8-sig")
    checks=[]
    for month in pd.period_range("2025-07","2026-10",freq="M"):
        cutoff=str(month.start_time.date());train=history.training_rows(full,cutoff,"relay")
        assert train.date.nunique()==252 and train.target_date.max()<cutoff
        checks.append(dict(month=str(month),cutoff=cutoff,rows=len(train),days=252,max_target=train.target_date.max(),training_digest=history._frame_digest(train,fields)))
    prep=b.read(ROOT/"preparation_BEFORE_ANY_NEW_FIT.json")
    paths=[Path(__file__),Path(evaluation.__file__),ROOT/"trainable_corrected_dataset_QUALIFIED.joblib",ROOT/"source_enum_qualification_BEFORE_FIT.json",ROOT/"frozen_references_3777_each.csv",ROOT/"preflight_tests.xml",b.ROOT/"tests/test_expectation_345.py"]
    protocol=dict(formed_at=b.now(),version="345-corrected-features-and-fixed-cost-comparison-v1",arms=list(ARMS),reference="frozen ST_DUAL; corrected arm is cost-arm reference",
        hypothesis="Source-qualified complete previous cohort, all observed active anchors and corrected multi-state rules may change discrimination; fixed negative cost1.5 may reduce false alarms with explicit recall cost",
        distinct_from_failed_trials=True,old_trials_not_repeated=True,features=fields,estimator=previous.PARAMS,negative_training_cost=NEGATIVE_COST,
        weights="Same complete252 matured days, inverse full-candidate count then class-mass balance; negative class mass multiplied by fixed1 or1.5 and normalized to252. Never selected from evaluation results",
        decision="strict score>0.5, never tuned; new unresolved shape => UNKNOWN B direction; A retains frozen BASE",
        ranking="All original nonword candidates retained; raw B scores only relative research ranking, UNKNOWN route never automatically positive",
        training_checks=checks,fit_max=32,fit_budget_seconds=1800,threshold_search=False,weight_search=False,
        preprocessing="Training-only all-empty-column constant adapter, native partial NaNs; unknown source cells unchanged",
        acceptance_corrected=dict(pooled_historical_F1_not_lower_than_dual=True,each_fold_recall_loss_max=.05,pooled_top1_not_lower=True),
        acceptance_cost=dict(pooled_FP_reduction_min=.20,each_fold_FP_not_higher=True,each_fold_recall_loss_max=.10,pooled_precision_higher=True,pooled_F1_loss_max=.02,pooled_TP_minus1p5FP_higher=True),
        tradeoff="A fixed false alarm penalty is a research preference, not realized monetary loss; do not require every metric to improve, report all losses and complete calls/coverage",
        periods=PERIODS,uncertainty=dict(block_sessions=5,draws=1500,seed=20261015,within_batch_bonferroni_tests=24,prior_sequential_trials_not_corrected=True),
        historical_previously_seen=True,blind=False,role_identity_gold=0,role_identity_accuracy=None,actual_return=None,production_eligible=False,
        LLM_evaluation="Three explicitly authorized isolated Codex reviewers; inherited AGENTS/results and later-signal exposure disclosed; historical diagnostic only, current15 target16 still future. No API required",
        source_bindings=prep["source_bindings"]|{str(p.resolve()):b.sha(p) for p in paths})
    b.verify(protocol["source_bindings"]);b.write_new(ROOT/"protocol_BEFORE_ANY_FIT.json",protocol)
    return dict(features=len(fields),months=16,maximum_fits=32,new_fits=0)


def evaluate(pred):
    summaries={};strata=[];rankings=[];changes=[];selections=[]
    for period in PERIODS:
        p=pred[pred.date.between(period["start"],period["end"])];arms={}
        for arm,f in p.groupby("arm",sort=True):
            nonword=f[f.daily_one_word.eq(0)]
            arms[arm]=dict(full=evaluation.classification(f),nonword=evaluation.classification(nonword),top1=evaluation.ranking(f),top2=evaluation.ranking(f,2))
            for dim in ["boards","daily_one_word","current_price_shape","review_cycle_state","current_price_path_status"]:
                for value,part in f.groupby(dim,dropna=False):strata.append(dict(period=period["id"],arm=arm,dimension=dim,stratum=str(value),**evaluation.classification(part)))
            for same in [False,True]:
                for height in [False,True]:
                    for k in [1,2]:
                        rankings.append(dict(period=period["id"],arm=arm,height_baseline=height,**evaluation.ranking(f,k,same,height),k=k))
                        s=evaluation.selection(f,k,same,height).copy();s["k"]=k;s["same_height"]=same;s["height_baseline"]=height;s["period"]=period["id"];selections.append(s)
        intervals={}
        for ref,new in [("ST_DUAL",ARMS[0]),(ARMS[0],ARMS[1])]:
            a=p[p.arm.eq(ref)&p.daily_one_word.eq(0)];z=p[p.arm.eq(new)&p.daily_one_word.eq(0)]
            intervals[new]=evaluation.block_intervals(a,z)
            changed=a[["date","code","target","label_valid","call"]].merge(z[["date","code","call"]],on=["date","code"],suffixes=("_ref","_new"),validate="one_to_one")
            changed=changed[changed.call_ref.ne(changed.call_new)].copy();changed["period"]=period["id"];changed["reference"]=ref;changed["challenger"]=new;changes.append(changed)
        summaries[period["id"]]=dict(metrics=arms,intervals=intervals)
    historical=pred[pred.date.le("2026-09-30")]
    pooled={arm:evaluation.classification(f[f.daily_one_word.eq(0)]) for arm,f in historical.groupby("arm")}
    top={arm:evaluation.ranking(f) for arm,f in historical.groupby("arm")}
    a,z,ref=pooled[ARMS[0]],pooled[ARMS[1]],pooled["ST_DUAL"]
    corrected_guards=dict(pooled_F1_not_lower=a["F1"]>=ref["F1"],pooled_top1_not_lower=top[ARMS[0]]["candidate_hit_rate"]>=top["ST_DUAL"]["candidate_hit_rate"])
    cost_guards=dict(pooled_FP_reduction=z["fp"]<=.8*a["fp"],pooled_precision_higher=z["Precision"] is not None and z["Precision"]>a["Precision"],pooled_F1_loss_within2pp=z["F1"] is not None and z["F1"]>=a["F1"]-.02,pooled_utility_higher=z["utility_TP_minus_1p5FP"]>a["utility_TP_minus_1p5FP"])
    for period in history.FOLDS:
        m=summaries[period["id"]]["metrics"];x=m[ARMS[0]]["nonword"];y=m[ARMS[1]]["nonword"];v=m["ST_DUAL"]["nonword"]
        corrected_guards[period["id"]+"_recall_loss_within5pp"]=x["Recall"]>=v["Recall"]-.05
        cost_guards[period["id"]+"_recall_loss_within10pp"]=y["Recall"]>=x["Recall"]-.10
        cost_guards[period["id"]+"_FP_not_higher"]=y["fp"]<=x["fp"]
    result=dict(formed_at=b.now(),comparisons=summaries,pooled_historical_nonword=pooled,corrected_guards=corrected_guards,cost_guards=cost_guards,
        corrected_screen_pass=all(corrected_guards.values()),cost_screen_pass=all(cost_guards.values()),
        threshold_or_weights_retuned=False,historical_diagnostic_not_blind=True,role_accuracy=None,actual_profitability=None,production_deployed=False)
    return result,pd.DataFrame(strata),pd.DataFrame(rankings),pd.concat(changes,ignore_index=True),pd.concat(selections,ignore_index=True)


def run():
    p=b.read(ROOT/"protocol_BEFORE_ANY_FIT.json");b.verify(p["source_bindings"])
    assert p["estimator"]==previous.PARAMS and p["negative_training_cost"]==NEGATIVE_COST
    b.write_new(ROOT/"experiment_STARTED.json",dict(formed_at=b.now(),protocol_sha256=b.sha(ROOT/"protocol_BEFORE_ANY_FIT.json")))
    output=ROOT/"fixed_comparison";output.mkdir(exist_ok=False)
    data=joblib.load(ROOT/"trainable_corrected_dataset_QUALIFIED.joblib");full,fields=data["full"],data["fields"]
    refs=pd.read_csv(ROOT/"frozen_references_3777_each.csv",dtype={"code":str},low_memory=False)
    base=refs[refs.arm.eq("BASE")].set_index(["date","code"]).score
    parts=[];start=time.monotonic();fits=0
    for check in p["training_checks"]:
        train=history.training_rows(full,check["cutoff"],"relay")
        assert train.date.nunique()==252 and history._frame_digest(train,fields)==check["training_digest"]
        test=full[full.date.str[:7].eq(check["month"])].copy()
        for arm in ARMS:
            if time.monotonic()-start>p["fit_budget_seconds"]:raise TimeoutError("fixed30minute fitting budget")
            prefix=output/(check["month"]+"_"+arm)
            b.write_new(str(prefix)+"_fit_STARTED.json",dict(formed_at=b.now(),arm=arm,month=check["month"],training_digest=check["training_digest"],rows=len(train),days=252,negative_cost=NEGATIVE_COST[arm],attempt=1))
            try:
                model=Pipeline([("empty_training_columns",EmptyTrainingColumnGuard()),("hgb",HistGradientBoostingClassifier(**p["estimator"]))])
                with threadpool_limits(limits=2):
                    model.fit(train[fields],train.target.astype(int),hgb__sample_weight=weights(train,NEGATIVE_COST[arm]));fits+=1
                    score=model.predict_proba(test[fields])[:,list(model.classes_).index(1)]
                joblib.dump(dict(model=model,features=fields,cutoff=check["cutoff"],training_rows=len(train),training_days=252,training_digest=check["training_digest"],negative_cost=NEGATIVE_COST[arm],estimator=p["estimator"],arm=arm,trained_at=b.now()),str(prefix)+"_model.joblib")
                result=test[["date","code","name","boards","target_date","target","label_valid","daily_one_word","current_price_shape","review_cycle_state","current_price_path_status"]].copy()
                result["ranking_score"]=score;result["b_score"]=score;result["score"]=score;result["call"]=np.where(score>.5,"PROMOTE","FAIL")
                for i,row in enumerate(result.itertuples()):
                    if row.current_price_shape==1:
                        value=base.loc[(row.date,row.code)];result.iloc[i,result.columns.get_loc("score")]=value;result.iloc[i,result.columns.get_loc("call")]="PROMOTE" if value>.5 else "FAIL"
                    elif pd.isna(row.current_price_shape):
                        result.iloc[i,result.columns.get_loc("call")]="UNKNOWN";result.iloc[i,result.columns.get_loc("score")]=np.nan
                result["arm"]=arm;result["model_cutoff"]=check["cutoff"]
                result.to_csv(str(prefix)+"_predictions_BEFORE_EVALUATION.csv",index=False,encoding="utf-8-sig");parts.append(result)
                b.write_new(str(prefix)+"_fit_COMPLETED.json",dict(formed_at=b.now(),arm=arm,training_digest=check["training_digest"],model_sha256=b.sha(str(prefix)+"_model.joblib"),prediction_sha256=b.sha(str(prefix)+"_predictions_BEFORE_EVALUATION.csv"),rows=len(result)))
                print(check["month"],arm,"saved",len(result),flush=True)
            except Exception as error:
                b.write_new(ROOT/"FAILURE_STOP.json",dict(formed_at=b.now(),error_type=type(error).__name__,error=str(error),completed_fits=fits,retry_authorized=False));raise
    assert fits==32
    pred=pd.concat([refs]+parts,ignore_index=True)
    assert len(pred)==3777*5 and not pred.duplicated(["date","code","arm"]).any()
    pred.to_csv(output/"complete_paired_predictions_BEFORE_EVALUATION.csv",index=False,encoding="utf-8-sig")
    result,strata,ranks,changes,selections=evaluate(pred)
    for name,f in [("all_strata_WITH_UNKNOWN",strata),("all_daily_and_sameheight_Top1_Top2_WITH_HEIGHT",ranks),("all_direction_changes_WITH_UNKNOWN",changes),("all_selected_rows_WITH_UNKNOWN",selections)]:f.to_csv(output/(name+".csv"),index=False,encoding="utf-8-sig")
    result.update(new_fits=fits,BASE_refits=0,prior_challenger_refits=0,elapsed_fit_seconds=time.monotonic()-start,
        source_bindings=p["source_bindings"]|{str(q.resolve()):b.sha(q) for q in [ROOT/"protocol_BEFORE_ANY_FIT.json",*output.glob("*")]})
    b.verify(result["source_bindings"]);b.write_new(output/"model_validation_results.json",result)
    return dict(new_fits=fits,corrected_screen_pass=result["corrected_screen_pass"],cost_screen_pass=result["cost_screen_pass"],elapsed_seconds=result["elapsed_fit_seconds"])


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=["freeze","run"]);args=parser.parse_args()
    print(json.dumps(globals()[args.command](),ensure_ascii=True))
