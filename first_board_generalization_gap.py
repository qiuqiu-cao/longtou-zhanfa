"""Read-only full-pool generalization audit of three frozen validation heads."""
from pathlib import Path
import argparse
import json
import shutil
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import first_board_leader_model as base
import first_second_stage_model as stage
import first_board_capital_risk_model as risk
import first_board_matured_event_model as prior

REF=risk.REFERENCE
SPECS={
 "BASE62":dict(root=REF,stem="validation_S1_STAGE_CYCLE_GENE",data="all_first_and_second_boards_WITH_SEPARATE_LABELS.csv",validation="validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv"),
 "MARKET70":dict(root=base.ROOT/"research/first_board_market_residual_20261020_run1",stem="validation_FIRST_CLOSE_MARKET_RESIDUAL",data="ALL_FIRST_BOARD_MARKET_CONTEXT_UNCHANGED_LABELS.csv",validation="validation_ALL_BEFORE_METRICS.csv"),
 "LEGACY80":dict(root=base.ROOT/"research/first_board_legacy_peer_20261020_run1",stem="validation_LEGACY_PEER_INFORMATION",data="ALL_FIRST_BOARD_LEGACY_PEER_FEATURES_UNCHANGED_LABELS.csv",validation="validation_LEGACY_PEER_INFORMATION_ALL_BEFORE_METRICS.csv")}
KEEP=["signal_date","code","name","stage","label_end",stage.TARGET,"one_price","research_selection_allowed","exclusion_reason"]

def score(model,fields,rows,members,kind):
    result=rows[KEEP].copy()
    result["model_score"]=model.predict_proba(rows[fields])[:,1]
    member_keys=set(zip(members.signal_date,members.code))
    result["is_model_training_member"]=list((d,c) in member_keys for d,c in zip(rows.signal_date,rows.code))
    result["evaluation_kind"]=kind
    return result

def metrics(pred):
    m=risk.screen_metric(pred)
    selected=base.selected(pred[pred.research_selection_allowed],"model_score",1)
    known=pred[stage.TARGET].notna()
    y=pred.loc[known,stage.TARGET]
    m.update(pool_rows=len(pred),pool_known=int(known.sum()),pool_unknown=int((~known).sum()),
        selected_training_members=int(selected.is_model_training_member.sum()),
        rank_auc=float(roc_auc_score(y,pred.loc[known,"model_score"])) if y.nunique()==2 else None,
        raw_score_is_not_qualified_probability=True)
    return m

def run(output):
    out=Path(output)
    if out.exists():raise FileExistsError("Preserve completed/stopped audit;do not overwrite")
    out.mkdir(parents=True)
    screen_file=prior.SCREEN/"COMPLETE_FIRST_BOARD_FACTS_FEATURES_LABELS_AND_SCREEN.csv"
    inputs=[screen_file,Path(base.__file__),Path(stage.__file__),Path(risk.__file__)]
    for spec in SPECS.values():
        directory=spec["root"]/"models";stem=spec["stem"]
        inputs.extend([spec["root"]/spec["data"],spec["root"]/spec["validation"],directory/(stem+"_bundle.joblib"),
            directory/(stem+"_calibration_receipt.json"),directory/(stem+"_calibration_rows.csv")])
        for suffix in ("_full","_calibration_prefix"):
            inputs.extend([directory/(stem+suffix+".joblib"),directory/(stem+suffix+"_members.csv"),directory/(stem+suffix+"_COMPLETED.json")])
    protocol=dict(formed_at=base.now(),batch_started_at="2026-10-20T14:28:10+00:00",budget_minutes=30,
        previous_goal_turn="NO_PROGRESS:interrupted after confirming common42datecutoff but before audit execution;now resume nextsafeaction. Earlier legacytrialmadeprogress andwasverified.",
        task="Replay exactly3saved heads to measure optimistic training versus temporal-heldout/full2025H1 gaps;no fits",
        hypotheses="Additional fields may improve training fit without transferable ranking;observed gaps diagnose optimism/drift but do not establish one causal explanation",
        heads=list(SPECS),windows=["TRAIN_FULL_OPTIMISTIC","CAL42_PREFIX_HELDOUT","CAL42_FULL_OPTIMISTIC","VALIDATION2025H1_FINAL"],
        pool_policy="Every candidate on the relevant signal dates,including unknown targets;never derive a forecast pool by joining only known calibration/training members",
        calibration_policy="Raw prefix scores on42heldout dates. Do not use a calibrator fitted on those same42outcomes to claim heldout probability accuracy",
        matching="Full/prefixtrainingmembers,labels and dates must match acrossheads;cal42optimistic and heldout pools identical",
        metrics="Fixed original risk-screen Top1hits/known/unknown/completepoolpositives;raw-score rankAUC;paired cal42full-versus-prefix outcomes;monthly2025H1stability",
        acceptance="In-sample orcal42rates cannot pass60%goal;primarycross-month criterion unchanged",
        stop="Complete fixedread-only audit andverification or30minutes. No newlearningalgorithm or thresholds thisbatch",
        model_fits=0,new_forward_results=0,actual_trades=0,goal_60_passed=False,production_changes=False,
        inputs={str(p.resolve()):base.sha(p) for p in inputs},source_sha256=base.sha(__file__))
    base.write_json(out/"protocol_BEFORE_REPLAY.json",protocol);shutil.copy2(__file__,out/"source_before_replay.py")
    screen=pd.read_csv(screen_file,dtype={"code":str},float_precision="round_trip")
    report=dict(formed_at=None,heads={},model_fits=0,goal_60_passed=False,new_forward_results=0,actual_trades=0)
    reference_members={};canonical=None
    for name,spec in SPECS.items():
        directory=spec["root"]/"models";stem=spec["stem"]
        rows=pd.read_csv(spec["root"]/spec["data"],dtype={"code":str},float_precision="round_trip")
        rows=rows[rows.stage.eq(1)].copy()
        if "research_selection_allowed" not in rows:
            rows=rows.merge(screen[["signal_date","code","research_selection_allowed","exclusion_reason"]],on=["signal_date","code"],validate="one_to_one")
        bundle=joblib.load(directory/(stem+"_bundle.joblib"));fields=bundle["features"]
        if canonical is None:canonical=rows.set_index(["signal_date","code"]).sort_index()
        else:
            indexed=rows.set_index(["signal_date","code"]).sort_index()
            assert indexed.index.tolist()==canonical.index.tolist()
            np.testing.assert_allclose(indexed[stage.VARIANTS["STAGE_CYCLE_GENE"]+[stage.TARGET]],canonical[stage.VARIANTS["STAGE_CYCLE_GENE"]+[stage.TARGET]],atol=0,rtol=0,equal_nan=True)
            assert indexed.research_selection_allowed.tolist()==canonical.research_selection_allowed.tolist()
        members={}
        for suffix in ("full","calibration_prefix"):
            receipt=json.loads((directory/(stem+"_"+suffix+"_COMPLETED.json")).read_text("utf-8"))
            assert base.sha(directory/(stem+"_"+suffix+".joblib"))==receipt["sha256"]
            members[suffix]=pd.read_csv(directory/(stem+"_"+suffix+"_members.csv"),dtype={"code":str},float_precision="round_trip")
            entries=set(map(tuple,members[suffix][["signal_date","code",stage.TARGET,"label_end"]].to_numpy()))
            if suffix not in reference_members:reference_members[suffix]=entries
            else:assert entries==reference_members[suffix]
        cal_info=json.loads((directory/(stem+"_calibration_receipt.json")).read_text("utf-8"))
        train_dates=sorted(members["full"].signal_date.unique());cal_dates=train_dates[-42:]
        assert cal_info["cutoff"]==cal_dates[0]=="2024-11-01"
        assert members["calibration_prefix"].label_end.max()<cal_dates[0]
        assert members["full"].label_end.max()<"2025-01-01"
        pools={"TRAIN_FULL_OPTIMISTIC":rows[rows.signal_date.isin(train_dates)],
            "CAL42_PREFIX_HELDOUT":rows[rows.signal_date.isin(cal_dates)],
            "VALIDATION2025H1_FINAL":rows[rows.signal_date.between("2025-01-01","2025-06-30")]}
        prefix=joblib.load(directory/(stem+"_calibration_prefix.joblib"))
        outputs={}
        for kind,pool in pools.items():
            model=prefix if kind=="CAL42_PREFIX_HELDOUT" else bundle["model"]
            fitted=members["calibration_prefix"] if kind=="CAL42_PREFIX_HELDOUT" else members["full"]
            pred=score(model,fields,pool,fitted,kind)
            pred.to_csv(out/(name+"_"+kind+"_FULL_POOL.csv"),index=False,encoding="utf-8-sig")
            outputs[kind]=pred
        outputs["CAL42_FULL_OPTIMISTIC"]=outputs["TRAIN_FULL_OPTIMISTIC"][outputs["TRAIN_FULL_OPTIMISTIC"].signal_date.isin(cal_dates)].copy()
        outputs["CAL42_FULL_OPTIMISTIC"]["evaluation_kind"]="CAL42_FULL_OPTIMISTIC"
        outputs["CAL42_FULL_OPTIMISTIC"].to_csv(out/(name+"_CAL42_FULL_OPTIMISTIC_FULL_POOL.csv"),index=False,encoding="utf-8-sig")
        cal_old=pd.read_csv(directory/(stem+"_calibration_rows.csv"),dtype={"code":str},float_precision="round_trip")
        cal_new=outputs["CAL42_PREFIX_HELDOUT"]
        overlap=cal_old.merge(cal_new[["signal_date","code","model_score"]],on=["signal_date","code"],validate="one_to_one")
        np.testing.assert_allclose(overlap.uncalibrated_oof,overlap.model_score,atol=1e-12,rtol=1e-12)
        assert len(cal_new)>=len(cal_old) and len(overlap)==len(cal_old)
        validation_old=pd.read_csv(spec["root"]/spec["validation"],dtype={"code":str},float_precision="round_trip")
        replay=outputs["VALIDATION2025H1_FINAL"].merge(validation_old[["signal_date","code","model_score"]],on=["signal_date","code"],validate="one_to_one",suffixes=("_replayed","_old"))
        assert len(replay)==len(validation_old)
        np.testing.assert_allclose(replay.model_score_replayed,replay.model_score_old,atol=1e-12,rtol=1e-12)
        summary={kind:metrics(pred) for kind,pred in outputs.items()}
        summary["monthly_validation"]={month:metrics(part) for month,part in outputs["VALIDATION2025H1_FINAL"].groupby(outputs["VALIDATION2025H1_FINAL"].signal_date.str[:7])}
        pair=[]
        for kind in ("CAL42_PREFIX_HELDOUT","CAL42_FULL_OPTIMISTIC"):
            chosen=base.selected(outputs[kind][outputs[kind].research_selection_allowed],"model_score",1)
            pair.append(chosen[["signal_date","code",stage.TARGET]].rename(columns={"code":"code_"+kind,stage.TARGET:"outcome_"+kind}))
        paired=pair[0].merge(pair[1],on="signal_date",validate="one_to_one")
        paired.to_csv(out/(name+"_CAL42_PAIRED_OPTIMISTIC_VS_HELDOUT_LABEL_ONLY.csv"),index=False,encoding="utf-8-sig")
        summary["head_members"]=dict(full=len(members["full"]),prefix=len(members["calibration_prefix"]),dates=len(train_dates),calibration_days=len(cal_dates),
            original_known_calibration_rows=len(cal_old),complete_calibration_rows=len(cal_new),
            added_unknown_calibration_candidates=int(cal_new[stage.TARGET].isna().sum()),fields=len(fields))
        summary["cal42_in_sample_uplift_pp"]=100*(summary["CAL42_FULL_OPTIMISTIC"]["hit_rate"]-summary["CAL42_PREFIX_HELDOUT"]["hit_rate"])
        report["heads"][name]=summary
        print(json.dumps({"head":name,"results":{kind:{k:summary[kind][k] for k in ("hits","known","unknown","hit_rate","pool_unknown")} for kind in outputs}},ensure_ascii=True),flush=True)
    for path,digest in protocol["inputs"].items():assert base.sha(path)==digest
    report.update(formed_at=base.now(),status="FIXED_READ_ONLY_REPLAY_FINISHED",matched_members_across_all_heads=True)
    base.write_json(out/"report.json",report)
    return report

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True)
    run(parser.parse_args().output)
