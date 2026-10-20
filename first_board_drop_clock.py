"""One fixed first-close ablation: remove only the archive-clock age field."""
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

REF=risk.REFERENCE
SCREEN=previous.SCREEN
EVIDENCE=base.ROOT/"research/first_board_input_shift_20261020"
VARIANT="DROP_ARCHIVE_CLOCK"
REMOVED="listing_age_sessions"
FIELDS=[field for field in stage.VARIANTS["STAGE_CYCLE_GENE"] if field!=REMOVED]


def train_members(rows,boundary,start_p=None):
    valid=rows.stage.eq(1)&rows[stage.TARGET].notna()&rows.signal_date.lt(boundary)&rows.label_end.lt(boundary)
    if start_p is not None:valid &= rows.p.ge(start_p)
    return rows[valid].copy()


def run(output):
    out=Path(output)
    if out.exists():raise FileExistsError("Fixed ablation exists;refuse retry or overwrite")
    out.mkdir(parents=True)
    source=REF/"all_first_and_second_boards_WITH_SEPARATE_LABELS.csv"
    screen_file=SCREEN/"COMPLETE_FIRST_BOARD_FACTS_FEATURES_LABELS_AND_SCREEN.csv"
    validation_file=REF/"validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv"
    protected=[Path(base.__file__),Path(stage.__file__),Path(risk.__file__),Path(previous.__file__),REF/"report.json",
        REF/"S1_ALL_PREDICTIONS.csv",REF/"S2_ALL_PREDICTIONS.csv",SCREEN/"report.json"]
    inputs=[source,screen_file,validation_file,EVIDENCE/"report.json",EVIDENCE/"verification.json",EVIDENCE/"saved_archive_clock_usage_CURRENT_DIAGNOSTIC.json"]
    assert len(FIELDS)==61 and set(stage.VARIANTS["STAGE_CYCLE_GENE"])-set(FIELDS)=={REMOVED}
    protocol=dict(formed_at=base.now(),budget_minutes=30,batch_started_at="2026-10-20T15:08:53+00:00",
        previous_goal_turn="PROGRESS:outcome-blind62fieldauditverifiedlisting_age_sessionsis archivecalendarposition for99%ofoldstocks and54BASEsplitsusedit;nowonesurgicalablation",
        hypothesis="Removing the left-censored archive-clock input improves first-close transfer beyondfrozen62fieldHGB",
        variant=VARIANT,fields=FIELDS,removed_only=REMOVED,
        source_semantics="Original field remains in complete facts as source-truncated tenure,not full listing age;no new age replacement or capping",
        baseline="Exact frozen original62-fieldHGBscores plusunchangedcapital/thinturnover screen;baseline andsecondheadneverrefitted",
        training="Exact samefullknownmaturemembers,label_end<boundary,originalequaldayweights/HGB160trees15leavesleaf60lr.06l2=8balancedseed42;last42datePlattC1 unchanged",
        validation="Completeoriginal2025H1:4724rows/117dates. Need>52Top1hits,>=48non-one-priceTop1hits,>=95%decisiondatecoverage;baselinewinsties",
        evaluation="Onlyifpass:oneheadperoriginal14monthlyprevious252officialsignalpositions,alllabel_end<monthstart;savefullpredictionsbeforemetrics,reportphase/non-one-price/samecodesecondclose",
        target60="DailyTop1next-official-close1to2knownhitrate>=60%,>=200known,>=95%outcomecoverage,noabstentiondates;2025H2and2026each>=55%;firstcloseonly,noauctiondata",
        stop="Singlearmonce;fixedcomparisonor30minutes. No agecap/replacement/otherfeature changes afterresult,no oldfailedschemereruns",
        inputs={str(p.resolve()):base.sha(p) for p in inputs},protected={str(p.resolve()):base.sha(p) for p in protected},
        raw_source_hashes={str(p.resolve()):base.sha(p) for p in base.INPUTS.values()},source_sha256=base.sha(__file__),
        historical_replay_not_pristine_blind=True,production_changes=False,actual_trades=0,new_forward_results=0)
    base.write_json(out/"protocol_BEFORE_ANY_FIT.json",protocol);shutil.copy2(__file__,out/"source_before_first_fit.py")
    rows=pd.read_csv(source,dtype={"code":str},float_precision="round_trip");rows=rows[rows.stage.eq(1)].copy()
    screen=pd.read_csv(screen_file,dtype={"code":str},float_precision="round_trip")
    rows=rows.merge(screen[["signal_date","code","research_selection_allowed","exclusion_reason","threeboard_evidence_status"]],on=["signal_date","code"],validate="one_to_one")
    rows.to_csv(out/"ALL_FIRST_BOARD_UNCHANGED_FACTS_LABELS_AND_SCREEN.csv",index=False,encoding="utf-8-sig")
    train=train_members(rows,"2025-01-01")
    baseline_members=pd.read_csv(REF/"models/validation_S1_STAGE_CYCLE_GENE_full_members.csv",dtype={"code":str})
    assert set(zip(train.signal_date,train.code))==set(zip(baseline_members.signal_date,baseline_members.code))
    val=rows[rows.signal_date.between("2025-01-01","2025-06-30")]
    frozen=risk.attach_frozen(rows,validation_file)
    baseline=risk.screen_metric(frozen);nonbase=risk.screen_metric(frozen[frozen.one_price.eq(0)])
    assert baseline["hits"]==52 and nonbase["hits"]==48
    stage.VARIANTS[VARIANT]=FIELDS
    bundle=stage.fit_head(train,VARIANT,out,"validation_"+VARIANT)
    predictions=stage.predict(bundle,val)
    predictions.to_csv(out/"validation_ALL_BEFORE_METRICS.csv",index=False,encoding="utf-8-sig")
    primary=risk.screen_metric(predictions);non=risk.screen_metric(predictions[predictions.one_price.eq(0)])
    passed=bool(primary["hits"]>52 and non["hits"]>=48 and primary["decision_date_coverage"]>=.95)
    decision=dict(formed_at=base.now(),selected=VARIANT if passed else None,baseline=baseline,baseline_non_one_price=nonbase,
        primary=primary,non_one_price=non,unfiltered_diagnostic=stage.metric(predictions),pass_validation=passed)
    base.write_json(out/"selection_BEFORE_EVALUATION.json",decision)
    print(json.dumps(base.clean(dict(validation=decision)),ensure_ascii=True),flush=True)
    report=dict(formed_at=None,decision=decision,goal_60_passed=False,new_forward_results=0,actual_trades=0)
    if passed:
        sessions=json.loads(base.INPUTS["calendar"].read_text("utf-8"))["sessions"];parts=[]
        for month,part in rows[rows.signal_date.ge("2025-07-01")].groupby(rows.signal_date.str[:7],sort=True):
            boundary=month+"-01";p=int(np.searchsorted(sessions,boundary))
            train=train_members(rows,boundary,p-252)
            bundle=stage.fit_head(train,VARIANT,out,month+"_"+VARIANT)
            scores=stage.predict(bundle,part);scores["model_month"]=month
            scores.to_csv(out/(month+"_ALL_BEFORE_METRICS.csv"),index=False);parts.append(scores)
            print(month+" saved",flush=True)
        scores=pd.concat(parts,ignore_index=True)
        scores.to_csv(out/"EVALUATION_ALL_PREDICTIONS.csv",index=False,encoding="utf-8-sig")
        frozen=risk.attach_frozen(rows,REF/"S1_ALL_PREDICTIONS.csv")
        report["evaluation"]=previous.period_metrics(scores,frozen)
        allowed=scores[scores.research_selection_allowed]
        second=pd.read_csv(REF/"S2_ALL_PREDICTIONS.csv",dtype={"code":str},float_precision="round_trip")
        chain,report["same_stock_second_recheck"]=stage.chain_recheck(allowed,second)
        chain.to_csv(out/"SAME_STOCK_SECOND_RECHECK_ALL.csv",index=False,encoding="utf-8-sig")
        report["historical_calibration_allowed_pool"]=stage.calibration_metrics(allowed)
        m=report["evaluation"]["all"]["new_screened"]
        report["goal_60_passed"]=bool(m["hit_rate"]>=.60 and m["known"]>=200 and m["outcome_coverage"]>=.95 and m["abstention_dates"]==0 and all(report["evaluation"][period]["new_screened"]["hit_rate"]>=.55 for period in ("2025H2","2026")))
    for path,digest in protocol["protected"].items():assert base.sha(path)==digest
    report.update(formed_at=base.now(),core_fit_count=len(list((out/"models").glob("*_COMPLETED.json"))),
        status="FIXED_MONTHLY_EVALUATION_FINISHED" if passed else "REJECTED_VALIDATION_NO_MONTHLY_FITS")
    base.write_json(out/"report.json",report)
    print(json.dumps(base.clean(report),ensure_ascii=True,indent=2),flush=True)
    return report

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True)
    run(parser.parse_args().output)
