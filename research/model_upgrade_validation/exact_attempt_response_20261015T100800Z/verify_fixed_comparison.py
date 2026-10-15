"""Saved-model and metric replay for the one corrected-input experiment."""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import exact_attempt_response_research as r
import short_term_thinking_dataset_builder as b
from historical_symbolic_verify import classification,ranking,assert_values


def main():
    root=r.ROOT;protocol=b.read(root/"protocol_BEFORE_ANY_FIT.json");b.verify(protocol["source_bindings"])
    protected=b.read(root/"execution_start.json")["protected_before"];b.verify(protected)
    pred_path=root/"fixed_comparison/complete_paired_predictions.csv";pred=pd.read_csv(pred_path,dtype={"code":str},low_memory=False)
    baseline=pd.read_csv(b.DEFAULT_RUN/"delivery/complete_paired_predictions_VALIDATED.csv",dtype={"code":str},low_memory=False)
    for arm in ["BASE","ST_DUAL"]:
        expected=baseline[baseline.arm.eq(arm)].sort_values(["date","code"]).reset_index(drop=True)
        actual=pred[pred.arm.eq(arm)].sort_values(["date","code"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(expected,actual[list(expected)])
    full,current=r.load();base=pred[pred.arm.eq("BASE")].set_index(["date","code"]).score
    months=[];checks=0
    for item in protocol["training_checks"]:
        month,cutoff=item["month"],item["cutoff"]
        train=r.old.history.training_rows(full,cutoff,"relay")
        if train.date.nunique()!=252 or train.target_date.max()>=cutoff:raise ValueError("mature member clock changed")
        assert r.old.history._frame_digest(train,r.FEATURES)==item["new_training_digest"]
        weights=(1/train.groupby("date").date.transform("size")).to_numpy(copy=True);targets=train.target.to_numpy()
        for label in [0,1]:weights[targets==label]*=.5/weights[targets==label].sum()
        weights*=252;np.testing.assert_allclose(weights,r.old.training_weights(train),atol=1e-13,rtol=0)
        receipt=b.read(root/"fixed_comparison"/(month+"_fit_STARTED.json"));saved=joblib.load(root/"fixed_comparison"/(month+"_model.joblib"))
        assert protocol["formed_at"]<receipt["formed_at"]<=saved["trained_at"]
        assert saved["training_digest"]==receipt["training_digest"]==item["new_training_digest"]
        assert saved["estimator"]==r.old.PARAMS and saved["features"]==r.FEATURES
        test=current if month=="2026-10" else full[full.date.str[:7].eq(month)]
        with threadpool_limits(limits=2):values=saved["model"].predict_proba(test[r.FEATURES])[:,list(saved["model"].classes_).index(1)]
        original=pd.read_csv(root/"fixed_comparison"/(month+"_predictions_BEFORE_EVALUATION.csv"),dtype={"code":str})
        assert list(zip(test.date,test.code))==list(zip(original.date,original.code))
        np.testing.assert_allclose(values,original.b_score,atol=1e-13,rtol=0)
        expected=[base.loc[(x.date,x.code)] if x.daily_one_word==1 else values[i] for i,x in enumerate(test.itertuples())]
        np.testing.assert_allclose(expected,original.score,atol=1e-13,rtol=0)
        pd.testing.assert_series_equal(test.target.reset_index(drop=True),original.target.reset_index(drop=True),check_dtype=False)
        months.append(dict(month=month,train_rows=len(train),training_days=252,prediction_rows=len(test),score_replay_exact=True))
    result=b.read(root/"fixed_comparison/model_validation_results.json")
    for fold in r.old.history.FOLDS+[dict(id="2026Oct",start="2026-10-01",end="2026-10-31")]:
        for arm in ["BASE","ST_DUAL",r.ARM]:
            frame=pred[pred.arm.eq(arm)&pred.date.between(fold["start"],fold["end"])]
            actual=result["comparisons"][fold["id"]]["metrics"][arm]
            assert_values(actual["full"],classification(frame));assert_values(actual["nonword"],classification(frame[frame.daily_one_word.eq(0)]));checks+=2
            for key,k,cohort,height in [("top1",1,False,False),("top2",2,False,False),("same_height_top1",1,True,False),("same_height_top2",2,True,False),("height_baseline_top1",1,False,True)]:
                assert_values(actual[key],ranking(frame,k,cohort,height));checks+=1
    assert len(pred)==11313 and len(months)==16
    paths=list((root/"fixed_comparison").glob("*_model.joblib"));starts=list((root/"fixed_comparison").glob("*_fit_STARTED.json"))
    assert len(paths)==len(starts)==16
    record=dict(formed_at=b.now(),status="PASS_SAVED_MODEL_CLOCK_INPUT_OUTPUT_AND_METRIC_REPLAY_NOT_PREDICTIVE_ACCEPTANCE",model_replays=16,months=months,
        original_BASE_ST_DUAL_rows_exact=True,complete_paired_rows=len(pred),classification_and_ranking_checks=checks,protected_files_exact=len(protected),
        new_fits_in_verifier=0,new_formal_forecasts=0,new_mature_samples=0,actual_fills=0,prediction_screen_pass=result["prediction_screen_pass"],
        source_bindings=protected|protocol["source_bindings"]|{str(p.resolve()):b.sha(p) for p in [Path(__file__),pred_path,root/"fixed_comparison/model_validation_results.json",*paths,*starts]})
    b.write_new(root/"independent_verification.json",record);print({k:v for k,v in record.items() if k not in {"source_bindings","months"}})


if __name__=="__main__":main()
