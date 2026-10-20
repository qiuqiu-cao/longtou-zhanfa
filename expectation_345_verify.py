"""Replay saved models and correct evaluation membership without fitting anything."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import short_term_thinking_dataset_builder as b
import model_upgrade_history as history
import expectation_345_research as trial

ROOT = trial.ROOT


def scalar_metrics(frame):
    known = frame[frame.target.notna() & frame.label_valid.fillna(False).astype(bool)]
    cells = {(y, c): 0 for y in (0, 1) for c in ("PROMOTE", "FAIL", "UNKNOWN")}
    for row in known.itertuples():
        cells[int(row.target), row.call] += 1
    tp, fp, tn = cells[1, "PROMOTE"], cells[0, "PROMOTE"], cells[0, "FAIL"]
    positives = sum(cells[1, c] for c in ("PROMOTE", "FAIL", "UNKNOWN"))
    missed = positives-tp
    return dict(known=len(known), tp=tp, fp=fp, tn=tn, positives=positives,
        missed_positives=missed, abstained_positives=cells[1, "UNKNOWN"],
        abstained_negatives=cells[0, "UNKNOWN"],
        Recall=tp/positives if positives else None,
        Precision=tp/(tp+fp) if tp+fp else None,
        F1=2*tp/(2*tp+fp+missed) if 2*tp+fp+missed else None,
        full_known_accuracy=(tp+tn)/len(known) if len(known) else None)


def independent_selected(frame, k, same_height, height):
    pools = {}
    for row in frame.itertuples():
        if pd.isna(row.daily_one_word) or row.daily_one_word != 0:
            continue
        key = (row.date, row.boards) if same_height else (row.date,)
        pools.setdefault(key, []).append(row)
    selected=[]
    for key, rows in sorted(pools.items()):
        rows.sort(key=lambda row: ((-row.boards if height and not same_height else 0),
            (0 if height else -row.ranking_score if pd.notna(row.ranking_score) else float("inf")), row.code))
        selected.extend((row.date, row.code) for row in rows[:k])
    return selected


def align_evaluation_scope(pred):
    old = pd.read_csv(trial.OLD/"evaluation/all15084_frozen_predictions_with_explicit_channels.csv",
        dtype={"code":str}, low_memory=False)
    old = old[old.arm.eq("BASE")][["date", "code", "daily_one_word"]]
    work = b.read(trial.OLD/"workbench/完整81只_全部证据与原审阅.json")["rows"]
    current = pd.DataFrame([{k:row[k] for k in ("date","code","daily_one_word")} for row in work if row["date"]=="2026-10-15"])
    identity = pd.concat([old,current],ignore_index=True).rename(columns={"daily_one_word":"frozen_original_nonword_identity"})
    result=pred.merge(identity,on=["date","code"],how="left",validate="many_to_one")
    assert len(result)==18885 and len(identity)==3777
    result["engine_metadata_daily_one_word"]=result.daily_one_word
    result["daily_one_word"]=result.frozen_original_nonword_identity
    # References must use their own frozen routing/identity, not the latest metadata.
    mask=result.arm.isin(["BASE","ST_DUAL","ST_DUAL_LAGGED_PRICE_RELATION"])
    result.loc[mask,"call"]=np.where(result.loc[mask,"score"].gt(.5),"PROMOTE","FAIL")
    result.loc[mask & result.daily_one_word.isna(),"call"]="UNKNOWN"
    return result


def main():
    protocol=b.read(ROOT/"protocol_BEFORE_ANY_FIT.json")
    original=b.read(ROOT/"fixed_comparison/model_validation_results.json")
    b.verify(original["source_bindings"])
    data=joblib.load(ROOT/"trainable_corrected_dataset_QUALIFIED.joblib")
    full, fields=data["full"],data["fields"]
    pred=pd.read_csv(ROOT/"fixed_comparison/complete_paired_predictions_BEFORE_EVALUATION.csv",dtype={"code":str},low_memory=False)
    assert len(pred)==18885 and pred.groupby("arm").size().eq(3777).all()
    checks=[]
    for window in protocol["training_checks"]:
        train=history.training_rows(full,window["cutoff"],"relay")
        assert history._frame_digest(train,fields)==window["training_digest"]
        assert train.date.nunique()==252 and train.target_date.max()<window["cutoff"]
        test=full[full.date.str[:7].eq(window["month"])].copy()
        for arm in trial.ARMS:
            prefix=ROOT/"fixed_comparison"/(window["month"]+"_"+arm)
            saved=joblib.load(str(prefix)+"_model.joblib")
            start=b.read(str(prefix)+"_fit_STARTED.json"); end=b.read(str(prefix)+"_fit_COMPLETED.json")
            assert start["attempt"]==1 and saved["features"]==fields
            assert start["training_digest"]==end["training_digest"]==window["training_digest"]
            assert b.sha(str(prefix)+"_model.joblib")==end["model_sha256"]
            assert b.sha(str(prefix)+"_predictions_BEFORE_EVALUATION.csv")==end["prediction_sha256"]
            w=trial.weights(train,saved["negative_cost"])
            assert abs(w.sum()-252)<1e-10
            masses=w.groupby(train.target).sum()
            assert abs(masses.loc[0]/masses.loc[1]-saved["negative_cost"])<1e-12
            with threadpool_limits(limits=2):
                replay=saved["model"].predict_proba(test[fields])[:,list(saved["model"].classes_).index(1)]
            expected=pred[pred.arm.eq(arm)&pred.date.str[:7].eq(window["month"])].set_index(["date","code"])
            scores=expected.loc[pd.MultiIndex.from_frame(test[["date","code"]]),"b_score"].to_numpy()
            assert np.allclose(replay,scores,atol=1e-15,rtol=0)
            checks.append(dict(month=window["month"],arm=arm,rows=len(test),max_replay_error=float(np.max(np.abs(replay-scores))),
                matured_days=252,training_digest=window["training_digest"],weight_mass=float(w.sum()),negative_positive_mass_ratio=float(masses.loc[0]/masses.loc[1])))
    assert len(checks)==32
    output=ROOT/"evaluation_scope_alignment";output.mkdir(exist_ok=False)
    aligned=align_evaluation_scope(pred)
    differences=aligned[(aligned.engine_metadata_daily_one_word.ne(aligned.daily_one_word)) &
        ~(aligned.engine_metadata_daily_one_word.isna()&aligned.daily_one_word.isna())].copy()
    differences.to_csv(output/"all_identity_scope_differences_NO_SCORE_CHANGES.csv",index=False,encoding="utf-8-sig")
    aligned.to_csv(output/"complete18885_predictions_ORIGINAL_FROZEN_COHORT.csv",index=False,encoding="utf-8-sig")
    result,strata,ranks,changes,selections=trial.evaluate(aligned)
    for period in trial.PERIODS:
        group=aligned[aligned.date.between(period["start"],period["end"])]
        for arm, rows in group.groupby("arm"):
            for scope, part in [("full",rows),("nonword",rows[rows.daily_one_word.eq(0)])]:
                independent=scalar_metrics(part)
                reported=result["comparisons"][period["id"]]["metrics"][arm][scope]
                for key,value in independent.items():
                    assert value==reported[key],(period["id"],arm,scope,key)
            for k in (1,2):
                for same in (False,True):
                    for height in (False,True):
                        wanted=independent_selected(rows,k,same,height)
                        stored=selections[(selections.period.eq(period["id"])) & selections.arm.eq(arm) & selections.k.eq(k) & selections.same_height.eq(same) & selections.height_baseline.eq(height)]
                        assert sorted(wanted)==sorted(zip(stored.date,stored.code))
    for name,frame in [("all_strata_WITH_UNKNOWN",strata),("all_daily_and_sameheight_Top1_Top2_WITH_HEIGHT",ranks),
        ("all_direction_changes_WITH_UNKNOWN",changes),("all_selected_rows_WITH_UNKNOWN",selections)]:
        frame.to_csv(output/(name+".csv"),index=False,encoding="utf-8-sig")
    # Also publish the strict current shape cohort as sensitivity, never silently switch denominator.
    sensitivity=[]
    for period in trial.PERIODS:
        group=aligned[aligned.date.between(period["start"],period["end"]) & aligned.current_price_shape.eq(0)]
        for arm, part in group.groupby("arm"):
            sensitivity.append(dict(period=period["id"],arm=arm,scope="LATEST_STRICT_PRICE_SHAPE_0",**trial.evaluation.classification(part)))
    pd.DataFrame(sensitivity).to_csv(output/"latest_strict_shape_SENSITIVITY_SEPARATE.csv",index=False,encoding="utf-8-sig")
    result.update(new_fits=0,original_new_fits=32,original_result_retained=True,
        evaluation_scope_repair="Protocol requires original frozen nonword denominator. First evaluation accidentally used newer engine identity metadata. All 18885 model scores and labels untouched. Original identities and reference calls restored from frozen sources; new unknown routes remain abstentions.",
        identity_scope_changed_unique_rows=int(differences[["date","code"]].drop_duplicates().shape[0]),
        latest_shape_sensitivity_separate=True,original_frozen_identity_not_role_or_execution_truth=True,
        replayed_saved_models=checks,independent_scalar_metrics_and_selected_keys_verified=True)
    result["source_bindings"]=original["source_bindings"]|{str(q.resolve()):b.sha(q) for q in [Path(__file__),ROOT/"fixed_comparison/model_validation_results.json",*output.glob("*")]}
    b.verify(result["source_bindings"])
    b.write_new(output/"verification_and_corrected_evaluation.json",result)
    print(json.dumps(dict(replayed_saved_models=len(checks),new_fits=0,rows=len(aligned),
        identity_scope_changes=result["identity_scope_changed_unique_rows"],corrected_pass=result["corrected_screen_pass"],cost_pass=result["cost_screen_pass"])))


if __name__=="__main__":
    main()
