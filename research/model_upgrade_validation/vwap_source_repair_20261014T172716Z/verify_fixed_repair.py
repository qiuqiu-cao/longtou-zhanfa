"""Read-only independent numerical audit; no fit APIs are called."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import traceback
import sys
import numpy as np
import pandas as pd
import joblib

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
import model_upgrade_history as history
ARM="DAILY_VWAP_SOURCE_COVERAGE_REPAIR"

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def load(path):return pd.read_csv(path,dtype={"code":str},float_precision="round_trip",low_memory=False)
def numeric(x):return pd.to_numeric(x,errors="coerce")
def bools(x):return x.map(lambda y: np.nan if pd.isna(y) else 1. if str(y).lower() in ["true","1","1.0"] else 0. if str(y).lower() in ["false","0","0.0"] else np.nan)
def metrics(f):
    y,p=bools(f.truth),bools(f.prediction)
    ok=y.notna()&p.notna(); n=int(ok.sum())
    tp=int((ok&y.eq(1)&p.eq(1)).sum());tn=int((ok&y.eq(0)&p.eq(0)).sum())
    fp=int((ok&y.eq(0)&p.eq(1)).sum());fn=int((ok&y.eq(1)&p.eq(0)).sum())
    positive=int(y.eq(1).sum()); precision=tp/(tp+fp) if tp+fp else None
    recall=tp/positive if positive else None
    f1=None if precision is None or recall is None else 2*precision*recall/(precision+recall) if precision+recall else 0.
    return {"original_registered_count":len(f),"scoreable_count":n,"coverage":n/len(f) if len(f) else None,
            "Accuracy":(tp+tn)/n if n else None,"Precision":precision,"complete_positive_Recall":recall,"F1":f1,
            "confusion":{"TP":tp,"TN":tn,"FP":fp,"FN":fn},"complete_positive_misses":positive-tp,
            "positive_support":positive,"signal_days":int(f.signal_date.nunique())}
def assert_metrics(a,b):
    for k,v in a.items():
        if isinstance(v,dict):assert v==b[k],(k,v,b[k])
        elif v is None:assert b[k] is None,(k,b[k])
        elif isinstance(v,float):assert abs(v-b[k])<1e-12,(k,v,b[k])
        else:assert v==b[k],(k,v,b[k])
def token(x):return None if pd.isna(x) else str(x)
def same(a,b):np.testing.assert_allclose(numeric(a),numeric(b),rtol=1e-12,atol=1e-12,equal_nan=True)
def interval(base,challenge,sessions,fold,policy):
    days=[x for x in sessions if fold["start"]<=x<=fold["end"]]
    paired=base[["signal_date","code","truth","prediction"]].merge(challenge[["signal_date","code","prediction"]],on=["signal_date","code"],suffixes=("_b","_c"),validate="one_to_one")
    y,pb,pc=bools(paired.truth),bools(paired.prediction_b),bools(paired.prediction_c)
    ok=y.notna()&pb.notna()&pc.notna()
    paired["count"]=ok.astype(int);paired["net"]=(pc.eq(y).astype(int)-pb.eq(y).astype(int)).where(ok,0)
    tally=paired.groupby("signal_date")[["count","net"]].sum().reindex(days,fill_value=0)
    # Independently enumerate circular blocks and weight their day tallies.
    size=len(days);block=policy["official_session_block"]
    rng=np.random.default_rng(policy["seed"])
    starts=rng.integers(0,size,size=(policy["draws"],(size+block-1)//block))
    draws=[];undefined=0
    for seeds in starts:
        selected=[]
        for start in seeds:selected.extend([(int(start)+j)%size for j in range(block)])
        counts=np.bincount(selected[:size],minlength=size)
        denominator=int(np.dot(counts,tally["count"].to_numpy()))
        if denominator==0:undefined+=1
        else:draws.append(float(np.dot(counts,tally["net"].to_numpy())/denominator))
    return {"lower":float(np.quantile(draws,policy["lower_tail"])) if not undefined else None,
            "upper":float(np.quantile(draws,policy["upper_tail"])) if not undefined else None,
            "undefined_draws":undefined,"draws":policy["draws"],"official_days":size,"block_days":block}

def main():
    protocol=read(HERE/"protocol_BEFORE_ANY_FIT.json");report=read(HERE/"report.json")
    assert sha(HERE/"protocol_BEFORE_ANY_FIT.json")==report["protocol_sha256"]
    assert sha(HERE/"full_candidate_comparison.csv")==report["predictions_sha256"]
    for p,d in protocol["sources"].items():assert sha(p)==d,p
    data=joblib.load(protocol["prepared_path"]);original=data["relay"]
    layer=load(protocol["vwap_layer"])
    assert list(zip(original.date,original.code))==list(zip(layer.date,layer.code))
    source=original.copy();source["daily_close_vs_vwap"]=layer.daily_close_vs_vwap.to_numpy()
    base_saved=load(protocol["baseline_path"])
    comparison=load(HERE/"full_candidate_comparison.csv")
    assert len(comparison)==7414 and not comparison.duplicated(["arm","fold","signal_date","code"]).any()
    assert set(comparison.arm)=={"BASE",ARM}
    assert len(list(HERE.glob("*_fit_STARTED.json")))==3
    assert report["actual_predictive_fits"]==3 and report["BASE_refits"]==0
    checks=[];changed=[];score_errors=[]
    for fold in protocol["definition"]["folds"]:
        fid=fold["id"];result=report["comparisons"][fid]
        test=source[source.date.between(fold["start"],fold["end"])].copy()
        baseline=base_saved[base_saved.arm.eq("BASE")&base_saved.fold.eq(fid)].copy()
        br=comparison[comparison.arm.eq("BASE")&comparison.fold.eq(fid)].copy()
        cr=comparison[comparison.arm.eq(ARM)&comparison.fold.eq(fid)].copy()
        for frame in [br,cr]:
            assert list(zip(frame.signal_date,frame.code))==list(zip(test.date,test.code))
            assert [token(x) for x in frame.target_date]==[token(x) for x in test.target_date]
            same(bools(frame.truth),test.target.where(test.label_valid))
            same(frame.actual_one_word_daily,test.actual_one_word_daily)
            assert np.array_equal(bools(frame.prediction).to_numpy(),frame.raw_score.gt(.5).astype(float).to_numpy())
        same(br.raw_score,baseline.raw_score);same(bools(br.prediction),baseline.predicted)
        assert len(br)==len(cr)==len(test)
        pred=load(HERE/(fid+"_predictions_BEFORE_VALIDATION.csv"))
        assert list(zip(pred.date,pred.code))==list(zip(test.date,test.code))
        for field in protocol["definition"]["fields"]:same(pred[field],test[field])
        same(pred.target,test.target);assert pred.label_valid.tolist()==test.label_valid.tolist()
        same(pred.raw_score,cr.raw_score);same(pred.predicted,bools(cr.prediction))
        # Fixed membership, mature cutoff and original field digest; never refit.
        valid=source.label_valid.fillna(False).astype(bool)&source.target.notna()&source.date.lt(fold["start"])&source.target_date.lt(fold["start"])
        eligible=source.loc[valid];train=eligible[eligible.date.isin(sorted(eligible.date.unique())[-252:])]
        wanted=next(x for x in protocol["training_checks"] if x["fold"]==fid)
        assert len(train)==wanted["train_n"] and train.date.nunique()==252
        assert history._frame_digest(train,data["relay_fields"])==wanted["training_digest"]
        assert train.target_date.max()==wanted["training_max_label_date"]
        assert train.daily_close_vs_vwap.notna().sum()==wanted["newfield_train_known"]
        receipt=read(HERE/(fid+"_fit_STARTED.json"))
        assert receipt["fit_count"]==1 and receipt["training_digest"]==wanted["training_digest"]
        assert datetime.fromisoformat(receipt["formed_at"])>datetime.fromisoformat(protocol["formed_at"])
        modelpath=HERE/(fid+"_research_only.joblib")
        log=next(x for x in report["fit_logs"] if x["fold"]==fid)
        assert sha(modelpath)==log["model_sha256"]
        model=joblib.load(modelpath)
        params=model.named_steps["logisticregression"].get_params()
        for k,v in protocol["definition"]["parameters"].items():assert params[k]==v
        score=model.predict_proba(test[protocol["definition"]["fields"]])[:,1]
        error=float(np.max(abs(score-cr.raw_score.to_numpy())))
        assert error<1e-12;score_errors.append({"fold":fid,"max_abs_error":error})
        bm,cm=metrics(br),metrics(cr);assert_metrics(bm,result["BASE"]);assert_metrics(cm,result[ARM])
        stratum={}
        for arm,frame in [("BASE",br),(ARM,cr)]:
            perboard={}
            for height,g in frame.groupby("boards"):
                mm=metrics(g);assert_metrics(mm,result[arm]["by_boards"][str(int(height))]);perboard[str(int(height))]=mm
            identities={"true_nonword":frame.actual_one_word_daily.eq(0),"true_oneword":frame.actual_one_word_daily.eq(1),"unknown_true_identity":frame.actual_one_word_daily.isna(),
                "proxy0":test.signal_one_word.eq(0).to_numpy(),"proxy1":test.signal_one_word.eq(1).to_numpy()}
            identitymetrics={}
            for key,mask in identities.items():
                mm=metrics(frame.loc[mask]);assert_metrics(mm,result["subgroups"][arm][key]);identitymetrics[key]=mm
            stratum[arm]={"boards":perboard,"identity_and_proxy":identitymetrics}
        ci=interval(br,cr,protocol["sessions"],fold,protocol["definition"]["block_interval"])
        for k,v in ci.items():
            if isinstance(v,float):assert abs(v-result["fixed_interval"][k])<1e-12
            else:assert v==result["fixed_interval"][k]
        paired=br.merge(cr,on=["signal_date","code","fold"],suffixes=("_BASE","_REPAIR"),validate="one_to_one")
        delta=cm["Accuracy"]-bm["Accuracy"]
        assert abs(delta-result["Accuracy_delta"])<1e-12 and not result["passed"]
        flips=paired[bools(paired.prediction_BASE).ne(bools(paired.prediction_REPAIR))].copy()
        changed.append(flips)
        checks.append({"fold":fid,"full_candidates":len(test),"unknown_labels":int(test.target.isna().sum()),"BASE":bm,"REPAIR":cm,
                       "Accuracy_delta":delta,"strata":stratum,"interval":ci,"direction_changes":len(flips)})
    allchanges=pd.concat(changed,ignore_index=True)
    assert sum(x["full_candidates"] for x in checks)==3707
    assert sum(x["unknown_labels"] for x in checks)==67
    for p,d in protocol["sources"].items():assert sha(p)==d,p
    allchanges.to_csv(HERE/"full_direction_changes.csv",index=False)
    verified={"formed_at":datetime.now(timezone.utc).isoformat(),"status":"INDEPENDENT_RECOMPUTATION_PASSED_REPAIR_REJECTED",
        "checks":checks,"full_original_pool":3707,"paired_rows":7414,"unknown_labels_preserved":67,
        "all_direction_changes":len(allchanges),"saved_model_score_reproduction":score_errors,
        "verified_recorded_predictive_fit_receipts":3,"verified_recorded_BASE_refits":0,"new_verifier_fits":0,
        "source_hashes_unchanged":True,"strict_score_gt_0_5_checked":True,"registered_blocks_recomputed_without_shared_interval_helper":True,
        "protocol_sha256":sha(HERE/"protocol_BEFORE_ANY_FIT.json"),"full_direction_changes_sha256":sha(HERE/"full_direction_changes.csv"),
        "verification_script_sha256":sha(__file__),"gain_verified":False,"deployment_authorized":False,"failures":[]}
    with (HERE/"independent_verification.json").open("x",encoding="utf-8") as handle:json.dump(verified,handle,indent=2,allow_nan=False)
    print(json.dumps({"status":verified["status"],"changes":len(allchanges),"folds":[{"fold":x["fold"],"delta":x["Accuracy_delta"],"interval":x["interval"]} for x in checks]},indent=2))

if __name__=="__main__":
    try:main()
    except Exception:
        error=traceback.format_exc()
        failure={"formed_at":datetime.now(timezone.utc).isoformat(),"status":"INDEPENDENT_VERIFY_FAILED","error":error,"new_verifier_fits":0}
        target=HERE/"independent_verification.json"
        if not target.exists():
            with target.open("x",encoding="utf-8") as handle:json.dump(failure,handle,indent=2)
        raise
