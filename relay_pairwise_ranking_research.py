"""One fixed conditional-ranking challenge, preserving all prior models and failures."""
from pathlib import Path
import argparse
import time
import json
import joblib
import numpy as np
import pandas as pd

import short_term_thinking_dataset_builder as b
import short_term_thinking_research as old
import model_upgrade_history as history
import expectation_345_research as previous
import expectation_345_evaluation as metrics
import relay_context_pairwise_ranker as ranking

ROOT=b.ROOT/"research/model_upgrade_validation/ranking_optimization_20261015T173045Z"
ARM="PAIRWISE_DAY_RANK"


def prepare():
    ROOT.mkdir(exist_ok=True)
    if (ROOT/"protocol_BEFORE_ANY_FIT.json").exists():raise FileExistsError("new fixed ranking protocol already exists")
    receipt=b.read(previous.ROOT/"completion_345_verified.json");b.verify(receipt["source_bindings"])
    data=joblib.load(previous.ROOT/"trainable_corrected_dataset_QUALIFIED.joblib")
    full=data["full"].copy();fields=data["fields"]
    original_path=b.ROOT/"research/model_upgrade_validation/short_term_thinking_20261015T071000Z/dataset.joblib"
    identity=joblib.load(original_path)["features"][["date","code","daily_one_word"]]
    frozen_path=previous.ROOT/"evaluation_scope_alignment/complete18885_predictions_ORIGINAL_FROZEN_COHORT.csv"
    references=pd.read_csv(frozen_path,dtype={"code":str},low_memory=False)
    refid=references[references.arm.eq("BASE")][["date","code","daily_one_word"]]
    identity=pd.concat([identity,refid]).drop_duplicates(["date","code"],keep="last")
    full=full.rename(columns={"daily_one_word":"latest_engine_daily_identity"}).merge(identity,on=["date","code"],how="left",validate="one_to_one")
    assert len(full)==9315 and full[full.date.eq("2026-10-15")].target.isna().all()
    common=references[references.arm.eq("BASE")].merge(full[["date","code","daily_one_word"]],on=["date","code"],suffixes=("_ref","_new"),validate="one_to_one")
    assert ((common.daily_one_word_ref==common.daily_one_word_new)|(common.daily_one_word_ref.isna()&common.daily_one_word_new.isna())).all()
    joblib.dump(dict(full=full,fields=fields,sessions=data["sessions"]),ROOT/"fixed_input_before_fit.joblib")
    references.to_csv(ROOT/"all18885_frozen_reference_predictions.csv",index=False,encoding="utf-8-sig")
    checks=[];group_parts=[]
    for month in pd.period_range("2025-07","2026-10",freq="M"):
        cutoff=str(month.start_time.date());train=history.training_rows(full,cutoff,"relay")
        assert train.date.nunique()==252 and train.target_date.max()<cutoff
        complete=full[full.date.isin(train.date.unique()) & full.daily_one_word.eq(0)].copy()
        # Every used label must already be mature, even when it was not in the filtered train adapter.
        complete.loc[complete.target_date.ge(cutoff),["target","label_valid"]]=[np.nan,False]
        _,_,w,groups=ranking.build_pairs(complete,fields)
        groups["month"]=str(month);group_parts.append(groups)
        checks.append(dict(month=str(month),cutoff=cutoff,window_days=252,complete_window_rows=len(train),complete_nonword_rows=len(complete),
            valid_training_rows=len(complete[complete.target.notna()&complete.label_valid.fillna(False).astype(bool)]),
            mixed_days=int(groups.directed_pairs.gt(0).sum()),directed_pairs=int(groups.directed_pairs.sum()),weight_mass=float(w.sum()),
            input_digest=history._frame_digest(complete,fields)))
    pd.concat(group_parts,ignore_index=True).to_csv(ROOT/"all16_complete_training_group_dispositions.csv",index=False,encoding="utf-8-sig")
    paths=[Path(__file__),Path(ranking.__file__),ROOT/"fixed_input_before_fit.joblib",ROOT/"all18885_frozen_reference_predictions.csv",ROOT/"all16_complete_training_group_dispositions.csv",original_path,
        b.ROOT/"nonword_pairwise_ranker_v282.py",b.ROOT/"results/v282/nonword_pairwise_20261002_v1/report.json",b.ROOT/"tests/test_relay_context_pairwise_ranker.py",ROOT/"preflight_tests.xml"]
    protocol=dict(formed_at=b.now(),objective="Continue model research, retain old modes and improve the scoring system",
        hypothesis="Fixed nonlinear conditional pairwise comparison over corrected92 causal inputs may improve complete-day nonword Top3 ordering relative to frozen ST_DUAL",
        new_arm=ARM,one_challenger=True,maximum_fits=16,fit_budget_seconds=1800,params=old.PARAMS,
        base_features=fields,pair_inputs="left92, right92, left-minus-right92 =276; HGB native NaNs retain both sides and shared market context; no whole-data imputation",
        labels="Same-day known matured positive versus known matured negative, both orientations; unknown labels never become negative",
        weights="Each mixed day directed pairs have total weight1; target signs balance equally. Empty, singleton, all-positive and all-negative days retained in dispositions but have no supervised pair",
        window="Use same full252 matured signal dates; restore complete original nonword cohort for those dates; only mature labels can form pairs",
        inference="Complete original nonword day, all directed opponents; .5*(P(left>right)+1-P(right>left)), average over opponents. Singleton score null, rank retained. No threshold or absolute event probability",
        identity="Original historical daily identity from old dataset, exact frozen reference identity for all3777 evaluation rows; latest shape qualification remains separate, never drops a ranking candidate",
        absolute_direction="No new direction head. Retain frozen corrected classifier direction in separate field; pairwise arm call UNKNOWN, ranking alone cannot assert PROMOTE",
        primary_metric="Pooled historical nonword Top3 hits/all preselected slots (unknown counts as not-yet-hit), complete first selection before outcome lookup",
        historical_screen=dict(pooled_top3_full_slot_gain_min=.02,pooled_top1_full_slot_not_lower=True,each_fold_top3_known_hit_loss_max=.02,each_fold_top3_known_coverage_loss_max=.05),
        uncertainty=dict(draws=2000,consecutive_signal_day_block=5,seed=20261015,family_tests=8,prior_experiments_not_corrected=True),
        periods=previous.PERIODS,training_checks=checks,
        distinct_from_v282="V282 used29 fields, scaled linear pairwise Logistic and window/C search. This is corrected92 conditional nonlinear context, fixed252 and params, no grid; old models/results untouched. No claim this is first ranking experiment",
        method_reference="https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/",
        method_scope="General pairwise-preference idea only; this HGB comparator is not RankNet reproduction or a guarantee of improvement",
        threshold_search=False,parameter_search=False,old_model_refits=0,historical_seen=True,blind=False,
        role_identity_accuracy=None,calibrated_event_probability=False,production_eligible=False,
        sources=receipt["source_bindings"]|{str(p.resolve()):b.sha(p) for p in paths})
    b.verify(protocol["sources"]);b.write_new(ROOT/"protocol_BEFORE_ANY_FIT.json",protocol)
    print(json.dumps(dict(features=len(fields),pair_inputs=len(fields)*3,fits=16,training_group_rows=sum(len(x) for x in group_parts))))


def chosen_summary(f,k=3,same=False,height=False):
    s=metrics.selection(f,k,same,height)
    known=s[s.target.notna()&s.label_valid.fillna(False).astype(bool)]
    hits=int(known.target.eq(1).sum());unknown=len(s)-len(known)
    return dict(selected=len(s),known=len(known),unknown=unknown,hits=hits,false_selected=len(known)-hits,
        hits_per_all_slots=hits/len(s) if len(s) else None,hits_per_known_slots=hits/len(known) if len(known) else None,
        unknown_best_case=(hits+unknown)/len(s) if len(s) else None,known_coverage=len(known)/len(s) if len(s) else None,
        selected_unknown_direction=int(s.call.eq("UNKNOWN").sum()),
        selected_missing_score=int(s.ranking_score.isna().sum()),groups_fewer_than_k=metrics.ranking(f,k,same,height)["groups_fewer_than_k"])


def paired_intervals(ref,new):
    result={};days=sorted(ref.date.unique())
    n=len(days);rng=np.random.default_rng(20261015);length=min(5,n)
    starts=rng.integers(0,n,size=(2000,(n+length-1)//length));indices=((starts[:,:,None]+np.arange(length))%n).reshape(2000,-1)[:,:n]
    for k in (1,3):
        values=[]
        for frame in (ref,new):
            chosen=metrics.selection(frame,k).copy()
            chosen["hit"]=(chosen.target.eq(1)&chosen.label_valid.fillna(False).astype(bool)).astype(int)
            a=chosen.groupby("date").agg(hits=("hit","sum"),slots=("code","size")).reindex(days,fill_value=0).to_numpy(float)
            sums=a[indices].sum(axis=1);values.append(sums[:,0]/sums[:,1])
        alpha=.05/8;delta=values[1]-values[0]
        result[str(k)]=dict(lower=float(np.quantile(delta,alpha/2)),upper=float(np.quantile(delta,1-alpha/2)),draws=2000,family_tests=8,
            scope="Full-slot hit difference, unknown remains in slots; prior experiments not corrected, retrospective descriptive")
    return result


def evaluate(pred):
    summaries={};selected=[];table=[]
    periods=previous.PERIODS+[dict(id="HISTORICAL_POOLED",start="2025-07-01",end="2026-09-30")]
    for period in periods:
        p=pred[pred.date.between(period["start"],period["end"])];summary={}
        for arm,f in p.groupby("arm"):
            summary[arm]={}
            for k in (1,2,3):
                summary[arm][str(k)]=chosen_summary(f,k)
                for same in (False,True):
                    for height in (False,True):
                        table.append(dict(period=period["id"],arm=arm,k=k,same_height=same,height_baseline=height,**chosen_summary(f,k,same,height)))
                        z=metrics.selection(f,k,same,height).copy();z["period"]=period["id"];z["k"]=k;z["same_height"]=same;z["height_baseline"]=height;selected.append(z)
        intervals=paired_intervals(p[p.arm.eq("ST_DUAL")],p[p.arm.eq(ARM)])
        summaries[period["id"]]=dict(metrics=summary,intervals=intervals)
    pooled=summaries["HISTORICAL_POOLED"]["metrics"]
    checks=dict(pooled_top3_full_slot_gain2pp=pooled[ARM]["3"]["hits_per_all_slots"]>=pooled["ST_DUAL"]["3"]["hits_per_all_slots"]+.02,
        pooled_top1_full_slot_not_lower=pooled[ARM]["1"]["hits_per_all_slots"]>=pooled["ST_DUAL"]["1"]["hits_per_all_slots"])
    for period in history.FOLDS:
        s=summaries[period["id"]]["metrics"];a,z=s["ST_DUAL"]["3"],s[ARM]["3"]
        checks[period["id"]+"_Top3_known_loss_within2pp"]=z["hits_per_known_slots"]>=a["hits_per_known_slots"]-.02
        checks[period["id"]+"_known_coverage_loss_within5pp"]=z["known_coverage"]>=a["known_coverage"]-.05
    return dict(formed_at=b.now(),summaries=summaries,historical_guards=checks,historical_screen_pass=all(checks.values()),
        deployed=False,new_absolute_head=False,event_probability_calibrated=False),pd.DataFrame(table),pd.concat(selected,ignore_index=True)


def run():
    p=b.read(ROOT/"protocol_BEFORE_ANY_FIT.json");b.verify(p["sources"])
    b.write_new(ROOT/"experiment_STARTED.json",dict(formed_at=b.now(),protocol_sha256=b.sha(ROOT/"protocol_BEFORE_ANY_FIT.json")))
    out=ROOT/"fixed_comparison";out.mkdir(exist_ok=False)
    data=joblib.load(ROOT/"fixed_input_before_fit.joblib");full,fields=data["full"],data["fields"]
    refs=pd.read_csv(ROOT/"all18885_frozen_reference_predictions.csv",dtype={"code":str},low_memory=False)
    parts=[];start=time.monotonic();fits=0
    for check in p["training_checks"]:
        train=history.training_rows(full,check["cutoff"],"relay")
        complete=full[full.date.isin(train.date.unique())&full.daily_one_word.eq(0)].copy()
        complete.loc[complete.target_date.ge(check["cutoff"]),["target","label_valid"]]=[np.nan,False]
        assert history._frame_digest(complete,fields)==check["input_digest"]
        prefix=out/check["month"]
        if time.monotonic()-start>1800:raise TimeoutError("fixed fitting budget")
        b.write_new(str(prefix)+"_fit_STARTED.json",dict(formed_at=b.now(),attempt=1,training_check=check))
        try:
            model,groups,details=ranking.fit_ranker(complete,fields,p["params"]);fits+=1
            path=str(prefix)+"_model.joblib"
            joblib.dump(dict(model=model,features=fields,cutoff=check["cutoff"],arm=ARM,representation=ranking.VERSION,training_check=check,details=details),path)
            test=full[full.date.str[:7].eq(check["month"])]
            metadata=refs[refs.arm.eq("CORRECTED_CAUSAL")&refs.date.str[:7].eq(check["month"])].copy()
            scores=ranking.predict_cohort(model,test[test.daily_one_word.eq(0)],fields)
            metadata=metadata.rename(columns={"score":"absolute_classifier_score","call":"absolute_classifier_call","ranking_score":"absolute_classifier_ranking_score","b_score":"absolute_classifier_b_score"})
            metadata=metadata.merge(scores,on=["date","code"],how="left",validate="one_to_one")
            metadata["score"]=np.nan;metadata["call"]="UNKNOWN";metadata["arm"]=ARM
            metadata.loc[~metadata.daily_one_word.eq(0),"ranking_status"]="OUTSIDE_NONWORD_RANKING_CHANNEL"
            metadata.to_csv(str(prefix)+"_predictions_BEFORE_EVALUATION.csv",index=False,encoding="utf-8-sig");parts.append(metadata)
            groups.to_csv(str(prefix)+"_training_groups.csv",index=False,encoding="utf-8-sig")
            b.write_new(str(prefix)+"_fit_COMPLETED.json",dict(formed_at=b.now(),model_sha256=b.sha(path),prediction_sha256=b.sha(str(prefix)+"_predictions_BEFORE_EVALUATION.csv"),**details))
            print(check["month"],details,flush=True)
        except Exception as error:
            b.write_new(ROOT/"FAILURE_STOP.json",dict(formed_at=b.now(),completed_fits=fits,error=repr(error),refit_allowed=False));raise
    pred=pd.concat([refs]+parts,ignore_index=True)
    assert len(pred)==3777*6 and not pred.duplicated(["date","code","arm"]).any()
    pred.to_csv(out/"all22662_predictions_BEFORE_EVALUATION.csv",index=False,encoding="utf-8-sig")
    result,table,selections=evaluate(pred)
    table.to_csv(out/"all_Top1_Top2_Top3_daily_sameheight_height_baselines.csv",index=False,encoding="utf-8-sig")
    selections.to_csv(out/"all_selected_rows_WITH_UNKNOWN.csv",index=False,encoding="utf-8-sig")
    result.update(new_fits=fits,old_model_refits=0,elapsed_seconds=time.monotonic()-start,sources=p["sources"]|{str(x.resolve()):b.sha(x) for x in [ROOT/"protocol_BEFORE_ANY_FIT.json",*out.glob("*")]})
    b.verify(result["sources"]);b.write_new(out/"ranking_validation_results.json",result)
    print(json.dumps(dict(new_fits=fits,old_refits=0,screen_pass=result["historical_screen_pass"],elapsed=result["elapsed_seconds"])))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=["prepare","run"]);args=parser.parse_args();globals()[args.command]()
