"""Fixed pre-2025 diversity diagnostic using complete saved prefix pools."""
from pathlib import Path
import argparse
import itertools
import json
import shutil
import joblib
import numpy as np
import pandas as pd
import first_board_leader_model as base
import first_second_stage_model as stage
import first_board_capital_risk_model as risk
import first_board_matured_event_model as previous
import first_board_regularized_ordered as learning

REF=risk.REFERENCE
OTHER=base.ROOT/"research/first_board_regularized_ordered_20261020_run1"
HEADS={"BASE":(REF,"validation_S1_STAGE_CYCLE_GENE"),"LINEAR":(OTHER,"validation_REGULARIZED_LINEAR"),"ORDERED":(OTHER,"validation_CHRONO_ORDERED")}


def select(frame,head):
    selected=frame[frame.research_selection_allowed].sort_values(["signal_date","score_"+head,"code"],ascending=[True,False,True]).groupby("signal_date",sort=False).head(1)
    return selected[["signal_date","code","name",stage.TARGET,"score_"+head]].rename(columns={"code":"code_"+head,"name":"name_"+head,stage.TARGET:"outcome_"+head})


def run(output):
    out=Path(output)
    if out.exists():raise FileExistsError("Read-only fixed comparison exists")
    out.mkdir(parents=True)
    source=REF/"all_first_and_second_boards_WITH_SEPARATE_LABELS.csv"
    screen_file=previous.SCREEN/"COMPLETE_FIRST_BOARD_FACTS_FEATURES_LABELS_AND_SCREEN.csv"
    inputs=[source,screen_file,Path(base.__file__),Path(stage.__file__),Path(risk.__file__),Path(learning.__file__)]
    for root,stem in HEADS.values():
        inputs.extend([root/"models"/(stem+suffix) for suffix in ("_calibration_prefix.joblib","_calibration_prefix_members.csv","_calibration_prefix_COMPLETED.json","_calibration_rows.csv","_calibration_receipt.json")])
    inputs.append(OTHER/"models/validation_CHRONO_ORDERED_calibration_prefix.cbm")
    inputs.extend([learning.DEP/"wheel_acquisition.json",learning.DEP/"pypi_metadata_request.json"])
    protocol=dict(formed_at=base.now(),batch_started_at="2026-10-20T15:12:34+00:00",budget_minutes=30,
        previous_goal_turn="PROGRESS:singleclockdrop actuallyfitted2cores/rejected46vs52,fullmember/scoreverificationpassed",
        task="Check whether fixed existing same-field learners have useful pre-2025heldoutdiversity before any ensemble hypothesis",
        heads=list(HEADS),pool="Exactcomplete42dates2024-11-01..12-30,2428candidates,including6unknown labels;originalrisk screen",
        source_label_policy="Read original2422knowncalibration labels only;join ontocompleteperiodfeatures. No2025outcome/score columns loaded",
        evaluation="Raw prefix scores only,no cal42-fitted calibrator;prefixmaturity ends2024-10-31;all3same8075trainingmembers",
        metrics="EachTop1hit/known/unknown,completepositivepool;dailycodeagreement,pairwisehitcontingencies,allTop1union oracle separatelylabelled,all3vote counts and sourceuncertainty",
        no_ensemble_weights_or_thresholds_selected=True,no_combined_ranker_scored=True,
        stop="Completefixeddiversityreadoutandverificationor30minutes;no newfitsor2025weighttuning",
        inputs={str(p.resolve()):base.sha(p) for p in inputs},source_sha256=base.sha(__file__),
        historical_research_not_pristine_blind=True,model_fits=0,new_forward_results=0,actual_trades=0,goal_60_passed=False,production_changes=False)
    base.write_json(out/"protocol_BEFORE_REPLAY.json",protocol);shutil.copy2(__file__,out/"source_before_replay.py")
    receipt=json.loads((REF/"models/validation_S1_STAGE_CYCLE_GENE_calibration_receipt.json").read_text("utf-8"))
    original_cal=pd.read_csv(REF/"models/validation_S1_STAGE_CYCLE_GENE_calibration_rows.csv",dtype={"code":str},float_precision="round_trip")
    dates=sorted(original_cal.signal_date.unique())
    assert len(dates)==42 and dates[0]=="2024-11-01" and dates[-1]=="2024-12-30"
    cols=["signal_date","code","name","stage","label_end"]+stage.VARIANTS["STAGE_CYCLE_GENE"]
    features=pd.read_csv(source,usecols=cols,dtype={"code":str},float_precision="round_trip")
    rows=features[features.stage.eq(1)&features.signal_date.isin(dates)].copy()
    assert stage.TARGET not in rows and len(rows)==2428
    labels=original_cal[["signal_date","code",stage.TARGET]]
    rows=rows.merge(labels,on=["signal_date","code"],how="left",validate="one_to_one")
    assert rows[stage.TARGET].notna().sum()==2422 and rows[stage.TARGET].isna().sum()==6
    screen=pd.read_csv(screen_file,usecols=["signal_date","code","research_selection_allowed","exclusion_reason"],dtype={"code":str})
    rows=rows.merge(screen,on=["signal_date","code"],validate="one_to_one")
    canonical_members=None;training_checks=[]
    for head,(root,stem) in HEADS.items():
        members=pd.read_csv(root/"models"/(stem+"_calibration_prefix_members.csv"),dtype={"code":str},float_precision="round_trip")
        keys=["signal_date","code","stage","label_end",stage.TARGET]
        sorted_members=members[keys].sort_values(["signal_date","code"]).reset_index(drop=True)
        if canonical_members is None:canonical_members=sorted_members
        else:pd.testing.assert_frame_equal(canonical_members,sorted_members)
        assert len(members)==8075 and members.label_end.max()<rows.signal_date.min()
        model=joblib.load(root/"models"/(stem+"_calibration_prefix.joblib"))
        complete=json.loads((root/"models"/(stem+"_calibration_prefix_COMPLETED.json")).read_text("utf-8"))
        assert base.sha(root/"models"/(stem+"_calibration_prefix.joblib"))==complete["sha256"]
        fields=complete.get("observed_fields",complete.get("features"))
        rows["score_"+head]=model.predict_proba(rows[fields])[:,1]
        saved=pd.read_csv(root/"models"/(stem+"_calibration_rows.csv"),dtype={"code":str},float_precision="round_trip")
        matched=saved.merge(rows[["signal_date","code","score_"+head]],on=["signal_date","code"],validate="one_to_one")
        assert len(matched)==2422
        np.testing.assert_allclose(matched.uncalibrated_oof,matched["score_"+head],atol=1e-12,rtol=1e-12)
        training_checks.append(dict(head=head,members=len(members),max_label_end=members.label_end.max()))
    rows.to_csv(out/"FULL2428_PREFIX_SCORES_AND_ONLY_PRE2025_LABELS.csv",index=False,encoding="utf-8-sig")
    picked=None;head_metrics={}
    for head in HEADS:
        selected=select(rows,head)
        picked=selected if picked is None else picked.merge(selected,on="signal_date",validate="one_to_one")
        score=rows.rename(columns={"score_"+head:"model_score"})
        head_metrics[head]=risk.screen_metric(score)
    picked["unique_top1_codes"]=picked[["code_"+h for h in HEADS]].nunique(axis=1)
    outcomes=picked[["outcome_"+h for h in HEADS]]
    picked["known_hits_across_heads"]=outcomes.eq(1).sum(axis=1)
    picked["known_failures_across_heads"]=outcomes.eq(0).sum(axis=1)
    picked["unknown_head_outcomes"]=outcomes.isna().sum(axis=1)
    picked["union_oracle_any_known_winner_LABEL_ONLY"]=outcomes.eq(1).any(axis=1)
    picked.to_csv(out/"ALL42_THREE_PREFIX_TOP1S_LABEL_ONLY.csv",index=False,encoding="utf-8-sig")
    pairs={}
    for a,b in itertools.combinations(HEADS,2):
        left=picked["outcome_"+a];right=picked["outcome_"+b]
        same=picked["code_"+a].eq(picked["code_"+b])
        pairs[a+"__"+b]=dict(same_code_days=int(same.sum()),different_code_days=int((~same).sum()),
            both_known_hit=int((left.eq(1)&right.eq(1)).sum()),both_known_failure=int((left.eq(0)&right.eq(0)).sum()),
            first_only_known_hit=int((left.eq(1)&right.eq(0)).sum()),second_only_known_hit=int((left.eq(0)&right.eq(1)).sum()),
            at_least_one_unknown=int((left.isna()|right.isna()).sum()))
    voting={}
    for count,group in picked.groupby("unique_top1_codes",sort=True):
        voting[str(count)]=dict(days=len(group),any_known_winner_days=int(group.union_oracle_any_known_winner_LABEL_ONLY.sum()),
            no_known_winner_with_unknown=int((group.known_hits_across_heads.eq(0)&group.unknown_head_outcomes.gt(0)).sum()),
            base_hit=int(group.outcome_BASE.eq(1).sum()),linear_hit=int(group.outcome_LINEAR.eq(1).sum()),ordered_hit=int(group.outcome_ORDERED.eq(1).sum()))
    result=dict(formed_at=base.now(),status="FIXED_PREFIX_DIVERSITY_FINISHED",complete_pool=len(rows),complete_dates=len(dates),
        pool_known=int(rows[stage.TARGET].notna().sum()),pool_unknown=int(rows[stage.TARGET].isna().sum()),
        risk_allowed=int(rows.research_selection_allowed.sum()),complete_positive_pool=int(rows[stage.TARGET].eq(1).sum()),
        training=training_checks,heads=head_metrics,pairwise=pairs,code_diversity=voting,
        union_oracle_any_known_winner_days_LABEL_ONLY=int(picked.union_oracle_any_known_winner_LABEL_ONLY.sum()),
        union_oracle_is_not_top1_success=True,combined_ranker_scored=False,model_fits=0,new_forward_results=0,actual_trades=0,goal_60_passed=False)
    for path,digest in protocol["inputs"].items():assert base.sha(path)==digest,path
    base.write_json(out/"report.json",result)
    print(json.dumps(base.clean(result),ensure_ascii=True,indent=2),flush=True)
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True)
    run(parser.parse_args().output)
