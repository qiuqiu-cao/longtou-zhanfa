"""Independent replay of saved models, counts, clocks and execution arithmetic.

No fit or feature-search operation is available in this verification entry.
"""
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import historical_symbolic_research as r
import short_term_thinking_dataset_builder as b


def assert_values(actual, expected):
    if isinstance(expected,dict):
        for key,value in expected.items():
            assert_values(actual[key],value)
    elif expected is None:
        assert actual is None
    elif isinstance(expected,(int,float)):
        assert np.isclose(actual,expected,atol=1e-13,rtol=0), (actual,expected)
    else:
        assert actual == expected,(actual,expected)


def classification(f):
    known=f[f.label_valid.eq(True)&f.target.notna()&f.score.notna()]
    y=known.target.to_numpy()==1;p=known.score.to_numpy()>.5
    tp=int(np.sum(y&p));fp=int(np.sum(~y&p));fn=int(np.sum(y&~p));tn=int(np.sum(~y&~p))
    return dict(candidates=len(f),scored=len(known),tp=tp,fp=fp,fn=fn,tn=tn,
        Accuracy=(tp+tn)/len(known) if len(known) else None,
        Recall=tp/(tp+fn) if tp+fn else None,Precision=tp/(tp+fp) if tp+fp else None,
        F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)


def selection(f,k=1,cohort=False,height=False):
    keys=["date","boards"] if cohort else ["date"]
    order=keys+(["boards"] if height and not cohort else [])+([] if height else ["score"])+["code"]
    ascending=[True]*len(keys)+([False] if height and not cohort else [])+([] if height else [False])+[True]
    return f[f.daily_one_word.eq(0)].sort_values(order,ascending=ascending).groupby(keys,sort=True).head(k)


def ranking(f,k=1,cohort=False,height=False):
    s=selection(f,k,cohort,height);keys=["date","boards"] if cohort else ["date"]
    known=s[s.target.notna()&s.label_valid.eq(True)]
    return dict(selected=len(s),known=len(known),groups=s.groupby(keys).ngroups,
        candidate_hit_rate=float(known.target.mean()) if len(known) else None,
        group_any_hit_rate=float(known.groupby(keys).target.max().mean()) if len(known) else None,
        groups_with_unresolved_truth=int(s.groupby(keys).target.apply(lambda x:x.isna().any()).sum()))


def verify():
    out=r.ROOT/"independent_verification.json"
    if out.exists():raise FileExistsError("immutable independent verification already exists")
    protocol=b.read(r.ROOT/"protocol_BEFORE_ANY_FIT.json")
    b.verify(protocol["source_bindings"])
    protected=b.read(r.ROOT/"execution_start.json")["protected_before"];b.verify(protected)
    full,current=r.load_frames()
    predictions=pd.read_csv(r.ROOT/"fixed_comparison/complete_paired_predictions.csv",dtype={"code":str},low_memory=False)
    original=pd.read_csv(b.DEFAULT_RUN/"delivery/complete_paired_predictions_VALIDATED.csv",dtype={"code":str},low_memory=False)
    assert len(predictions)==11313 and not predictions.duplicated(["date","code","arm"]).any()
    for arm in ["BASE","ST_DUAL"]:
        left=predictions[predictions.arm.eq(arm)].sort_values(["date","code"])
        right=original[original.arm.eq(arm)].sort_values(["date","code"])
        pd.testing.assert_frame_equal(left[list(right.columns)].reset_index(drop=True),right.reset_index(drop=True))
    months=[]
    base=predictions[predictions.arm.eq("BASE")].set_index(["date","code"]).score
    for check in protocol["training_checks"]:
        month,cutoff=check["month"],check["cutoff"]
        train=r.old.history.training_rows(full,cutoff,"relay")
        assert train.date.nunique()==252 and train.target_date.max()<cutoff
        assert r.old.history._frame_digest(train,r.FIELDS)==check["new_training_digest"]
        assert r.old.history._frame_digest(train,r.old.FEATURES)==check["original_training_digest"]
        weights=(1/train.groupby("date").date.transform("size")).to_numpy(copy=True)
        y=train.target.to_numpy()
        for label in [0,1]:weights[y==label]*=.5/weights[y==label].sum()
        weights*=252
        np.testing.assert_allclose(weights,r.old.training_weights(train),atol=1e-13,rtol=0)
        receipt=b.read(r.ROOT/"fixed_comparison"/(month+"_fit_STARTED.json"))
        saved=joblib.load(r.ROOT/"fixed_comparison"/(month+"_model.joblib"))
        assert protocol["formed_at"]<receipt["formed_at"]<=saved["trained_at"]
        assert saved["training_digest"]==check["new_training_digest"]==receipt["training_digest"]
        assert saved["features"]==protocol["fields"] and saved["estimator"]==protocol["estimator"]
        test=current.copy() if month=="2026-10" else full[full.date.str[:7].eq(month)].copy()
        with threadpool_limits(limits=2):score=saved["model"].predict_proba(test[saved["features"]])[:,list(saved["model"].classes_).index(1)]
        seen=pd.read_csv(r.ROOT/"fixed_comparison"/(month+"_predictions_BEFORE_EVALUATION.csv"),dtype={"code":str})
        assert list(zip(test.date,test.code))==list(zip(seen.date,seen.code))
        np.testing.assert_allclose(score,seen.b_score,atol=1e-13,rtol=0)
        expected=np.array([base.loc[(x.date,x.code)] if x.daily_one_word==1 else score[i] for i,x in enumerate(test.itertuples())])
        np.testing.assert_allclose(expected,seen.score,atol=1e-13,rtol=0)
        pd.testing.assert_series_equal(test.target.reset_index(drop=True),seen.target.reset_index(drop=True),check_dtype=False)
        months.append(dict(month=month,training_rows=len(train),training_dates=252,scored_rows=len(test),score_max_abs_error=float(np.max(np.abs(score-seen.b_score)))))
    report=b.read(r.ROOT/"fixed_comparison/model_validation_results.json")
    metric_checks=0
    for fold in r.old.history.FOLDS+[dict(id="2026Oct",start="2026-10-01",end="2026-10-31")]:
        for arm in ["BASE","ST_DUAL",r.ARM]:
            f=predictions[predictions.arm.eq(arm)&predictions.date.between(fold["start"],fold["end"])]
            saved=report["comparisons"][fold["id"]]["metrics"][arm]
            assert_values(saved["full"],classification(f));assert_values(saved["nonword"],classification(f[f.daily_one_word.eq(0)]))
            for key,k,cohort,height in [("top1",1,False,False),("top2",2,False,False),("same_height_top1",1,True,False),("same_height_top2",2,True,False),("height_baseline_top1",1,False,True)]:
                assert_values(saved[key],ranking(f,k,cohort,height));metric_checks+=1
    logs=pd.read_csv(r.ROOT/"fixed_comparison/all_top1_execution_scenarios.csv",dtype={"code":str})
    capsule=joblib.load(r.ROOT/"execution_context.joblib");sessions=capsule["sessions"]
    keys=set(zip(logs.date,logs.code));codes=set(logs.code)
    bars={(x.date,x.code):x for x in capsule["bars"][capsule["bars"].code.isin(codes)].itertuples()}
    cashchecks=0
    for x in logs.itertuples():
        if x.status!="INFERRED_OPEN_PROXY_NOT_ACTUAL_FILL":continue
        i=sessions.index(x.date);entry=bars[(sessions[i+1],x.code)]
        assert x.B==sessions[i+1] and x.entry_open==entry.open
        for h in [1,2]:
            net=getattr(x,"b"+str(h)+"_net");mark=getattr(x,"b"+str(h)+"_mark")
            if pd.isna(mark):continue
            exitbar=bars[(sessions[i+1+h],x.code)]
            assert np.isclose(mark,exitbar.close/(entry.open*1.001)-1,atol=1e-13)
            if pd.notna(net):
                assert np.isclose(net,exitbar.close*.999/(entry.open*1.001)-1-.003,atol=1e-13);cashchecks+=1
            if getattr(x,"status_b"+str(h),None)=="SELL_BLOCKED_ONE_PRICE_DOWN_LIMIT":assert pd.isna(net)
    assert report["prediction_screen_pass"] is False and report["production_deployment_authorized"] is False
    model_paths=list((r.ROOT/"fixed_comparison").glob("*_model.joblib"))
    receipts=list((r.ROOT/"fixed_comparison").glob("*_fit_STARTED.json"))
    assert len(model_paths)==len(receipts)==16
    bindings=protected|protocol["source_bindings"]
    for path in [Path(__file__),r.ROOT/"fixed_comparison/complete_paired_predictions.csv",r.ROOT/"fixed_comparison/model_validation_results.json",*model_paths,*receipts]:bindings[str(path.resolve())]=b.sha(path)
    result=dict(formed_at=b.now(),status="PASS_REPLAY_NOT_PREDICTIVE_ACCEPTANCE",protected_hashes=len(protected),source_hashes=len(protocol["source_bindings"]),
        paired_rows=11313,original_candidate_rows_per_arm=3771,saved_models_replayed=16,months=months,
        original_BASE_and_ST_DUAL_rows_exact=True,classification_and_ranking_checks=metric_checks+24,execution_cashflows_recomputed=cashchecks,
        one_word_retains_saved_BASE=True,new_fits=0,new_formal_maturity=0,actual_fills=0,prediction_screen_pass=False,source_bindings=bindings)
    b.write_new(out,result);return result


if __name__=="__main__":
    result=verify();print(json.dumps({k:v for k,v in result.items() if k not in {"source_bindings","months"}},ensure_ascii=False))
