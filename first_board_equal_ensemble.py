"""Two preregistered equal-weight combinations of three fixed learner families."""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
import pandas as pd
import first_board_leader_model as base
import first_second_stage_model as stage
import first_board_capital_risk_model as risk
import first_board_matured_event_model as previous
import first_board_regularized_ordered as learning

REF=risk.REFERENCE
OTHER=base.ROOT/"research/first_board_regularized_ordered_20261020_run1"
DIVERSITY=base.ROOT/"research/first_board_prefix_diversity_20261020"
HEADS=("BASE","LINEAR","ORDERED")
ARMS=("EQUAL_PROBABILITY_MEAN","EQUAL_DAILY_RANK_MEAN")
VALIDATION={"BASE":REF/"validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv",
    "LINEAR":OTHER/"validation_REGULARIZED_LINEAR_ALL_BEFORE_METRICS.csv",
    "ORDERED":OTHER/"validation_CHRONO_ORDERED_ALL_BEFORE_METRICS.csv"}


def combine(rows,arm):
    if rows.duplicated(["signal_date","code"]).any():raise ValueError("Duplicate combination key")
    if arm not in ARMS:raise ValueError("Unexpected ensemble rule")
    result=rows.copy()
    if arm==ARMS[0]:
        fields=["component_"+head+"_calibrated" for head in HEADS]
        if not np.isfinite(result[fields].to_numpy(float)).all():raise ValueError("Complete three-head calibration required;no silent head dropping")
        if not result[fields].ge(0).all().all() or not result[fields].le(1).all().all():raise ValueError("Invalid calibrated component range")
        result["model_score"]=result[fields].mean(axis=1)
        # This aggregate is only an internal estimate until its own evaluation
        # calibration gate passes; no independent aggregate calibrator is fitted.
        result["calibrated_score"]=result.model_score
    else:
        fields=["component_"+head+"_raw" for head in HEADS]
        if not np.isfinite(result[fields].to_numpy(float)).all():raise ValueError("Complete three-head raw scores required")
        ranks=result[fields].groupby(result.signal_date).rank(method="average",pct=True)
        result["model_score"]=ranks.mean(axis=1)
        result["calibrated_score"]=np.nan
    result["ensemble_rule"]=arm
    return result


def attach_components(facts,components):
    result=facts.copy()
    for head in HEADS:
        component=components[head]
        keys=["signal_date","code"]
        if result.duplicated(keys).any() or component.duplicated(keys).any():raise ValueError("Duplicate component keys")
        if set(zip(result.signal_date,result.code))!=set(zip(component.signal_date,component.code)):raise ValueError("Incomplete component universe")
        indexed=component.set_index(keys).loc[pd.MultiIndex.from_frame(result[keys])]
        np.testing.assert_allclose(result[stage.VARIANTS["STAGE_CYCLE_GENE"]+[stage.TARGET]],indexed[stage.VARIANTS["STAGE_CYCLE_GENE"]+[stage.TARGET]],atol=0,rtol=0,equal_nan=True)
        result["component_"+head+"_raw"]=indexed.model_score.to_numpy()
        result["component_"+head+"_calibrated"]=indexed.calibrated_score.to_numpy()
        if "train_prior" in result:
            np.testing.assert_allclose(result.train_prior,indexed.train_prior,atol=1e-12,rtol=1e-12)
        else:result["train_prior"]=indexed.train_prior.to_numpy()
    return result


def run(output):
    out=Path(output)
    if out.exists():raise FileExistsError("One fixed ensemble comparison;refuse overwrite/retry")
    out.mkdir(parents=True)
    source=REF/"all_first_and_second_boards_WITH_SEPARATE_LABELS.csv"
    screen_file=previous.SCREEN/"COMPLETE_FIRST_BOARD_FACTS_FEATURES_LABELS_AND_SCREEN.csv"
    protected=[Path(base.__file__),Path(stage.__file__),Path(risk.__file__),Path(previous.__file__),Path(learning.__file__),
        REF/"report.json",REF/"S1_ALL_PREDICTIONS.csv",REF/"S2_ALL_PREDICTIONS.csv",OTHER/"report.json",OTHER/"combined_report_APPEND_ONLY.json"]
    learning.cat_class()
    import catboost._catboost as native_module
    sources=[source,screen_file,*VALIDATION.values(),DIVERSITY/"report.json",DIVERSITY/"verification.json",DIVERSITY/"completion.json",
        learning.DEP/"wheel_acquisition.json",learning.DEP/"runtime/catboost/core.py",Path(native_module.__file__)]
    protocol=dict(formed_at=base.now(),budget_minutes=30,
        previous_batch="Finishedread-onlypre2025diversity:BASE/LINEAR/ORDERED21/20/19of42,unionoracle28,BASEmissrescued5and4days;notanactualensemblegain",
        hypothesis="Fixed equal-weight averaging of already-qualified distinct learner families reduces individual ranking errors",
        heads=list(HEADS),arms={ARMS[0]:"Exactly1/3each savedtemporalPlatt score,no newaggregatecalibrator;all3required",ARMS[1]:"Exactly1/3each rawscore'ssame-datecomplete-pool percentileaverage-rank,averageties;scoreisnotprobability"},
        feature_policy="Original62fields/fullknownmaturemembers;componentvalidationcoresreusedexactly,no newfeaturesorfitweights;allmissing/unknown candidates retained",
        baseline="FrozenoriginalBASEscores plusunchangedrisk screen;noBASErefits;no passingbythree-pickany-hit",
        validation="Completeoriginal2025H1:4724rows/117days. Need>52Top1hits,>=48non-one-pricehits,>=95%datecoverage;baselinewinsties. Mosthitspassingarm,probabilitymeanwinsarmtie",
        evaluation="Onlyselectedarm14originalmonthly252officialpositionwindows. SavedmonthlyBASE reused;fit originalfixedLINEARandORDEREDheads oncepermonthwiththeir42datecalibrators. Thesearecomponentfitsforthefrozenensemble,notretryingtheirfailedstandalonevalidation",
        calibration="Probabilitymeanrequiresall3componentcalibrators,nofallbackorheadsubsets;aggregatequalificationonlyonfullfinalevaluation. Rankmean remainsuncalibrated",
        target60="FirstcloseonlydailyTop1next-official-close1to2hitrate>=60%,>=200known,>=95%outcomecoverage,noabstentiondates;2025H2and2026each>=55%;samecodesecondclose recheckedindependently",
        stop="Twofixedrulesonce;no weight/subset/threshold changesafterresults. No newfits ifbothfail. Stopfixedcomparisonor30minutes",
        already_seen_historical_validation_not_pristine_blind=True,weights_not_fitted_on_any_outcomes=True,
        inputs={str(p.resolve()):base.sha(p) for p in sources},protected={str(p.resolve()):base.sha(p) for p in protected},
        raw_source_hashes={str(p.resolve()):base.sha(p) for p in base.INPUTS.values()},source_sha256=base.sha(__file__),
        production_changes=False,new_forward_results=0,actual_trades=0)
    base.write_json(out/"protocol_BEFORE_COMPARISON.json",protocol);shutil.copy2(__file__,out/"source_before_comparison.py")
    rows=pd.read_csv(source,dtype={"code":str},float_precision="round_trip");rows=rows[rows.stage.eq(1)].copy()
    screen=pd.read_csv(screen_file,dtype={"code":str},float_precision="round_trip")
    rows=rows.merge(screen[["signal_date","code","research_selection_allowed","exclusion_reason","threeboard_evidence_status"]],on=["signal_date","code"],validate="one_to_one")
    rows.to_csv(out/"ALL_FIRST_BOARD_UNCHANGED_FACTS_LABELS_SCREEN.csv",index=False,encoding="utf-8-sig")
    val=rows[rows.signal_date.between("2025-01-01","2025-06-30")]
    components={head:pd.read_csv(file,dtype={"code":str},float_precision="round_trip") for head,file in VALIDATION.items()}
    merged=attach_components(val,components)
    merged.to_csv(out/"VALIDATION_THREE_FROZEN_COMPONENT_SCORES.csv",index=False,encoding="utf-8-sig")
    frozen=risk.attach_frozen(rows,VALIDATION["BASE"])
    baseline=risk.screen_metric(frozen);nonbase=risk.screen_metric(frozen[frozen.one_price.eq(0)])
    assert baseline["hits"]==52 and nonbase["hits"]==48
    decisions={}
    for arm in ARMS:
        if arm==ARMS[0] and not np.isfinite(merged[["component_"+h+"_calibrated" for h in HEADS]]).all().all():
            decisions[arm]=dict(status="COMPONENT_CALIBRATION_UNAVAILABLE",pass_validation=False);continue
        score=combine(merged,arm)
        score.to_csv(out/("validation_"+arm+"_ALL_BEFORE_METRICS.csv"),index=False,encoding="utf-8-sig")
        m=risk.screen_metric(score);non=risk.screen_metric(score[score.one_price.eq(0)])
        decisions[arm]=dict(status="SCORED",primary=m,non_one_price=non,unfiltered_diagnostic=stage.metric(score),
            pass_validation=bool(m["hits"]>52 and non["hits"]>=48 and m["decision_date_coverage"]>=.95))
        print(json.dumps(base.clean(dict(arm=arm,validation=decisions[arm])),ensure_ascii=True),flush=True)
    passing=[arm for arm in ARMS if decisions[arm]["pass_validation"]]
    selected=max(passing,key=lambda a:decisions[a]["primary"]["hits"]) if passing else None
    selection=dict(formed_at=base.now(),selected=selected,baseline=baseline,baseline_non_one_price=nonbase,arms=decisions)
    base.write_json(out/"selection_BEFORE_EVALUATION.json",selection)
    report=dict(formed_at=None,decision=selection,goal_60_passed=False,new_forward_results=0,actual_trades=0,
        validation_component_refits=0,weights_fitted=0)
    if selected:
        (out/"components").mkdir()
        known=rows[rows[stage.TARGET].notna()]
        sessions=json.loads(base.INPUTS["calendar"].read_text("utf-8"))["sessions"]
        frozen_all=risk.attach_frozen(rows,REF/"S1_ALL_PREDICTIONS.csv")
        parts=[]
        for month,part in rows[rows.signal_date.ge("2025-07-01")].groupby(rows.signal_date.str[:7],sort=True):
            boundary=month+"-01";p=int(np.searchsorted(sessions,boundary))
            train=known[known.p.ge(p-252)&known.label_end.lt(boundary)]
            baseline_members=pd.read_csv(REF/"models"/(month+"_S1_STAGE_CYCLE_GENE_full_members.csv"),dtype={"code":str})
            assert set(zip(train.signal_date,train.code))==set(zip(baseline_members.signal_date,baseline_members.code))
            comps={"BASE":frozen_all[frozen_all.signal_date.isin(part.signal_date)]}
            for head,arm in [("LINEAR","REGULARIZED_LINEAR"),("ORDERED","CHRONO_ORDERED")]:
                bundle=learning.fit_head(train,arm,out/"components",month+"_"+head,boundary)
                comps[head]=learning.predict(bundle,part)
            merged_month=attach_components(part,comps)
            merged_month.to_csv(out/(month+"_ALL_COMPONENTS_BEFORE_METRICS.csv"),index=False,encoding="utf-8-sig")
            scores=combine(merged_month,selected);scores["model_month"]=month
            scores.to_csv(out/(month+"_ALL_BEFORE_METRICS.csv"),index=False,encoding="utf-8-sig");parts.append(scores)
            print(month+" ensemble saved",flush=True)
        predictions=pd.concat(parts,ignore_index=True)
        predictions.to_csv(out/"EVALUATION_ALL_PREDICTIONS.csv",index=False,encoding="utf-8-sig")
        report["evaluation"]=previous.period_metrics(predictions,frozen_all)
        allowed=predictions[predictions.research_selection_allowed]
        second=pd.read_csv(REF/"S2_ALL_PREDICTIONS.csv",dtype={"code":str},float_precision="round_trip")
        chain,report["same_stock_second_recheck"]=stage.chain_recheck(allowed,second)
        chain.to_csv(out/"SAME_STOCK_SECOND_RECHECK_ALL.csv",index=False,encoding="utf-8-sig")
        report["aggregate_historical_calibration"]=stage.calibration_metrics(allowed)
        m=report["evaluation"]["all"]["new_screened"]
        report["goal_60_passed"]=bool(m["hit_rate"]>=.60 and m["known"]>=200 and m["outcome_coverage"]>=.95 and m["abstention_dates"]==0 and all(report["evaluation"][p]["new_screened"]["hit_rate"]>=.55 for p in ("2025H2","2026")))
    for path,digest in protocol["protected"].items():assert base.sha(path)==digest
    report.update(formed_at=base.now(),new_component_core_fits=len(list((out/"components/models").glob("*_COMPLETED.json"))),
        status="FIXED_MONTHLY_ENSEMBLE_COMPARISON_FINISHED" if selected else "BOTH_EQUAL_WEIGHT_RULES_REJECTED_NO_FITS")
    base.write_json(out/"report.json",report)
    print(json.dumps(base.clean(report),ensure_ascii=True,indent=2),flush=True)
    return report

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True)
    run(parser.parse_args().output)
