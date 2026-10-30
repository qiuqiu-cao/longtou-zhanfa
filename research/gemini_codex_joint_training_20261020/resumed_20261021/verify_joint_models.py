"""No-fit replay of the actual joint models, source bindings and complete-day settlements."""
from pathlib import Path
import sys
import json
import joblib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import first_board_joint_seal_model as m

def verify():
    freeze=json.loads((m.JOINT/"JOINT_FROZEN_PROTOCOL.json").read_text("utf-8"))
    draft=json.loads((Path(__file__).resolve().parent/"PROTOCOL_DRAFT_BEFORE_GEMINI_ACK.json").read_text("utf-8"))
    assert freeze["status"]=="AGREED_BEFORE_NEW_FITS"
    assert set(freeze)-set(draft)<=set(["agreed_at","acknowledgment_note"])
    for key in draft:
        if key!="status":assert freeze[key]==draft[key],key
    assert freeze["source_sha256"]==m.base.sha(m.__file__)
    for path,digest in freeze["bindings"].items():assert m.base.sha(path)==digest,path
    sources=json.loads((m.SOURCES/"ALL668_SOURCE_RECEIPTS.json").read_text("utf-8"))
    for source in sources:
        if source.get("qualified_date_envelope"):assert m.base.sha(source["source_path"])==source["sha256"]
    rows=m.load_rows()
    val=rows[rows.signal_date.between("2025-01-01","2025-06-30")].reset_index(drop=True)
    assert len(rows)==28503 and rows[m.stage.TARGET].isna().sum()==126
    assert len(val)==4724 and val[m.stage.TARGET].isna().sum()==10
    frozen=m.risk.attach_frozen(rows,m.REFERENCE/"validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv")
    compact=val[m.KEYS+[m.stage.TARGET,"label_end","one_price","research_selection_allowed"]+m.TIME+m.FUND].copy()
    compact=compact.merge(frozen[m.KEYS+["model_score","calibrated_score"]].rename(columns={"model_score":"BASE_raw_score","calibrated_score":"BASE_calibrated_score"}),on=m.KEYS,validate="one_to_one")
    all_pairs=[];audits={}
    for arm,entry in freeze["arms"].items():
        out=m.OUTPUT/arm
        report=json.loads((out/"report.json").read_text("utf-8"))
        assert report["trainer"]==entry["trainer"] and report["arm"]==arm
        scores=pd.read_csv(out/"VALIDATION_ALL_PREDICTIONS_BEFORE_METRICS.csv",dtype={"code":str},float_precision="round_trip")
        pd.testing.assert_frame_equal(scores[val.columns],val,check_dtype=False,check_exact=True)
        name="validation_"+arm
        head=joblib.load(out/"models"/(name+"_bundle.joblib"))
        assert head["features"]==entry["fields"]==m.ARMS[arm]
        replay=m.stage.predict(head,val)
        for field in ("model_score","calibrated_score","train_prior"):
            np.testing.assert_allclose(replay[field],scores[field],rtol=1e-12,atol=1e-12,equal_nan=True)
        for suffix in ("calibration_prefix","full"):
            core=joblib.load(out/"models"/(name+"_"+suffix+".joblib"))
            receipt=json.loads((out/"models"/(name+"_"+suffix+"_COMPLETED.json")).read_text("utf-8"))
            assert m.base.sha(out/"models"/(name+"_"+suffix+".joblib"))==receipt["sha256"]
            params=core.named_steps["hist"].get_params()
            for key,value in dict(max_iter=160,max_leaf_nodes=15,min_samples_leaf=60,learning_rate=.06,l2_regularization=8.,class_weight="balanced",random_state=42,early_stopping=False).items():assert params[key]==value
            actual=pd.read_csv(out/"models"/(name+"_"+suffix+"_members.csv"),dtype={"code":str},float_precision="round_trip")
            old=pd.read_csv(m.REFERENCE/"models"/("validation_S1_STAGE_CYCLE_GENE_"+suffix+"_members.csv"),dtype={"code":str},float_precision="round_trip")
            fields=m.KEYS+["stage","label_end",m.stage.TARGET]
            pd.testing.assert_frame_equal(actual[fields].sort_values(m.KEYS).reset_index(drop=True),old[fields].sort_values(m.KEYS).reset_index(drop=True),check_dtype=False,check_exact=True)
            assert actual.label_end.lt("2025-01-01").all()
        primary=m.risk.screen_metric(scores);non=m.risk.screen_metric(scores[scores.one_price.eq(0)])
        assert m.base.clean(primary)==report["primary"] and m.base.clean(non)==report["non_one_price"]
        assert report["pass_validation"]==bool(primary["hits"]>52 and non["hits"]>=48 and primary["decision_date_coverage"]>=.95)
        settlements={}
        for scope,non_price in [("primary",False),("non_one_price",True)]:
            a=scores[scores.research_selection_allowed]
            b=frozen[frozen.research_selection_allowed]
            if non_price:a,b=a[a.one_price.eq(0)],b[b.one_price.eq(0)]
            a=m.base.selected(a,"model_score",1).set_index("signal_date")
            b=m.base.selected(b,"model_score",1).set_index("signal_date")
            assert a.index.equals(b.index) and len(a)==117
            ah,bh=a[m.stage.TARGET].eq(1),b[m.stage.TARGET].eq(1)
            ak,bk=a[m.stage.TARGET].notna(),b[m.stage.TARGET].notna()
            settlements[scope]=dict(dates=117,challenger_hits=int(ah.sum()),baseline_hits=int(bh.sum()),challenger_known=int(ak.sum()),baseline_known=int(bk.sum()),
                common_wins=int((ah&bh).sum()),challenger_only_win=int((ah&~bh&bk).sum()),baseline_only_win=int((~ah&bh&ak).sum()),common_failures=int((~ah&~bh&ak&bk).sum()))
            for day in a.index:all_pairs.append(dict(arm=arm,scope=scope,signal_date=day,challenger_code=a.loc[day,"code"],baseline_code=b.loc[day,"code"],challenger_outcome=a.loc[day,m.stage.TARGET],baseline_outcome=b.loc[day,m.stage.TARGET]))
        compact=compact.merge(scores[m.KEYS+["model_score","calibrated_score"]].rename(columns={"model_score":arm+"_raw_score","calibrated_score":arm+"_calibrated_score"}),on=m.KEYS,validate="one_to_one")
        assert len(list((out/"models").glob("*_COMPLETED.json")))==2
        audits[arm]=dict(trainer=report["trainer"],contract_feature_count=len(head["features"]),all4724_scores_replayed=True,original_prefix_and_full_members_equal=True,core_fits=2,temporal_calibrator_fits=1,settlements=settlements,pass_validation=report["pass_validation"])
    compact.to_csv(m.OUTPUT/"COMPLETE4724_BOTH_ARMS_WITH_FROZEN_BASE.csv",index=False)
    pd.DataFrame(all_pairs).to_csv(m.OUTPUT/"COMPLETE117_DAILY_PAIRED_SELECTIONS_ALL_ARMS.csv",index=False)
    evaluation=json.loads((m.OUTPUT/"report.json").read_text("utf-8"))
    chosen=evaluation["selection"]["selected"]
    eval_audit=None
    if chosen is not None:
        directory=m.OUTPUT/"evaluation";parts=[]
        for path in sorted(directory.glob("????-??_ALL_PREDICTIONS_BEFORE_METRICS.csv")):
            month=path.name[:7]
            scored=pd.read_csv(path,dtype={"code":str},float_precision="round_trip")
            head=joblib.load(directory/"models"/(month+"_"+chosen+"_bundle.joblib"))
            assert head["train_max_label_end"]<scored.signal_date.min() and head["features"]==m.ARMS[chosen]
            replay=m.stage.predict(head,scored)
            for field in ("model_score","calibrated_score","train_prior"):np.testing.assert_allclose(replay[field],scored[field],rtol=1e-12,atol=1e-12,equal_nan=True)
            parts.append(scored)
        assert len(parts)==14
        scored=pd.concat(parts,ignore_index=True)
        assert len(scored)==13149 and scored.signal_date.nunique()==286
        baseline=m.risk.attach_frozen(rows,m.REFERENCE/"S1_ALL_PREDICTIONS.csv")
        metrics=m.history.period_metrics(scored,baseline)
        assert m.base.clean(metrics)==evaluation["evaluation"]
        metrics=metrics["all"]["new_screened"]
        success=bool(metrics["hit_rate"]>=.60 and metrics["known"]>=200 and metrics["outcome_coverage"]>=.95 and metrics["abstention_dates"]==0 and all(evaluation["evaluation"][p]["new_screened"]["hit_rate"]>=.55 for p in ("2025H2","2026")))
        assert success==evaluation["goal_60_achieved"]
        eval_audit=dict(months=14,all13149scores_replayed=True,complete286dates_preserved=True,all_period_metrics_recomputed=True,goal_60_achieved=success,core_fits=28,temporal_calibrator_fits=14)
    else:assert not (m.OUTPUT/"evaluation").exists() and not evaluation["goal_60_achieved"]
    result=dict(at=m.base.now(),status="PASS_NO_FIT_REPLAY",verifier_fits=0,all_bindings_and668raw_hashes_unchanged=True,all28503_original_fields_and_labels_preserved=True,
        validation_rows=4724,validation_unknown=10,first_unknown=126,arms=audits,evaluation=eval_audit,baseline_refits=0,second_stage_refits=0,production_changes=False,
        goal_60_achieved=evaluation["goal_60_achieved"],qualification_scope="Raw/provider/sourcebinding checks and saved-modelreplay;no new fitting,external exchange authentication or prospective-performance claim")
    m.base.write_json(m.OUTPUT/"JOINT_SAVED_RESULT_VERIFICATION.json",result)
    print(json.dumps(m.base.clean(result),ensure_ascii=True),flush=True)

if __name__=="__main__":verify()
