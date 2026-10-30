"""Shared fixed first-close experiment: timestamp information versus timestamp plus seal quotes."""
from __future__ import annotations
import argparse
from datetime import datetime
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import shutil
import time
from zoneinfo import ZoneInfo
import joblib
import numpy as np
import pandas as pd
import first_board_leader_model as base
import first_second_stage_model as stage
import first_board_capital_risk_model as risk
import first_board_matured_event_model as history
import first_board_drop_clock as members

ROOT=base.ROOT
SOURCES=ROOT/"research/first_board_joint_design_20261020T185417Z"
QUAL=SOURCES/"qualification_resumed_20261021"
JOINT=ROOT/"research/gemini_codex_joint_training_20261020"
OUTPUT=ROOT/"research/first_board_joint_seal_20261021_run1"
REFERENCE=risk.REFERENCE
FACTS=REFERENCE/"all_first_and_second_boards_WITH_SEPARATE_LABELS.csv"
SCREEN=history.SCREEN/"COMPLETE_FIRST_BOARD_FACTS_FEATURES_LABELS_AND_SCREEN.csv"
KEYS=["signal_date","code"]
ORIGINAL=stage.VARIANTS["STAGE_CYCLE_GENE"].copy()
TIME=["reported_first_seal_trading_minutes","reported_first_seal_observed_pool_rank"]
FUND=["reported_close_seal_log_amount_ratio","reported_close_seal_log_floatcap_ratio"]
ARMS={"SEAL_TIME_64":ORIGINAL+TIME,"SEAL_TIME_FUND_66":ORIGINAL+TIME+FUND}
OWNERS={"SEAL_TIME_64":"Gemini","SEAL_TIME_FUND_66":"Codex"}

def exact_cents(value):
    try:
        x=Decimal(str(value))*100
        if not x.is_finite() or x<=0 or x!=x.to_integral_value():return None
        return int(x)
    except (InvalidOperation,TypeError,ValueError):return None

def first_minutes(stamp,signal_date):
    try:
        if isinstance(stamp,bool) or not np.isfinite(float(stamp)) or float(stamp)!=int(stamp):return np.nan
        dt=datetime.fromtimestamp(int(stamp),ZoneInfo("Asia/Shanghai"))
    except (ValueError,TypeError,OverflowError,OSError):return np.nan
    if dt.date().isoformat()!=signal_date:return np.nan
    seconds=dt.hour*3600+dt.minute*60+dt.second
    if not (33900<=seconds<=41400 or 46800<=seconds<=54000):return np.nan
    return (seconds-34200-(5400 if seconds>=46800 else 0))/60

def positive(value):
    try:x=float(value)
    except (ValueError,TypeError):return np.nan
    return x if np.isfinite(x) and x>0 else np.nan

def derive_fields(frame):
    if frame.duplicated(KEYS).any():raise ValueError("Duplicate full-pool key")
    result=frame.copy()
    groups=result.groupby("signal_date")[TIME[0]]
    observed=groups.transform("count");total=groups.transform("size")
    result[TIME[1]]=groups.rank(method="average",pct=True,ascending=True).where(observed.ge(3)&(observed/total).ge(.5))
    funds=pd.to_numeric(result.reported_close_seal_yuan,errors="coerce")
    for denominator,field in [("amount",FUND[0]),("float_cap",FUND[1])]:
        den=pd.to_numeric(result[denominator],errors="coerce")
        valid=funds.gt(0)&np.isfinite(funds)&den.gt(0)&np.isfinite(den)
        result[field]=np.log1p((funds/den).where(valid))
    return result

def qualify():
    if QUAL.exists():raise FileExistsError("Preserve original qualification;no overwrite")
    QUAL.mkdir()
    manifest_path=SOURCES/"ALL668_SOURCE_RECEIPTS.json"
    receipts=json.loads(manifest_path.read_text("utf-8"))
    assert len(receipts)==668 and len({r["date"] for r in receipts})==668
    usecols=KEYS+["stage","close","amount","float_cap","one_price"]
    facts=pd.read_csv(FACTS,usecols=usecols,dtype={"code":str},float_precision="round_trip")
    facts=facts[facts.stage.eq(1)].reset_index(drop=True)
    assert len(facts)==28503 and facts.signal_date.nunique()==668
    protocol=dict(at=base.now(),fits=0,labels_read=False,source_sha256=base.sha(__file__),facts_sha256=base.sha(FACTS),manifest_sha256=base.sha(manifest_path),
        fields=TIME+FUND,definition="Exact first-seal trading minutes,observed original-pool same-day mean percentile;log1p provider reported close-seal yuan divided by original signal-day amount/floatcap",
        field_scope="Only raw indices0/4/6/21 under exact dated daily envelope;quarantine reasons/concepts/group IDs/currentname/flow/maxseal/unmappedflags/finalseal/bomb count",
        gate="Each required field>=90% in development and2025H1,each signal date at least one timestamp observation;no target-based row removal. Missing/lunch/wrongdate/price mismatch remain NaN.",
        boundary="Late downloaded daily-provider quotes are historical proxies,not strict originalHTTPclock,verified exchange queue,capital-actor identity or eliminated universe-survivorship bias")
    base.write_json(QUAL/"PROTOCOL_BEFORE_FIELD_QUALIFICATION.json",protocol)
    records=[];date_summary=[]
    for receipt in receipts:
        day=receipt["date"]
        if not receipt.get("qualified_date_envelope"):
            date_summary.append(dict(signal_date=day,status="SOURCE_UNAVAILABLE",count=0));continue
        path=Path(receipt["source_path"])
        assert base.sha(path)==receipt["sha256"]
        payload=json.loads(path.read_bytes());info=payload["info"]
        assert str(payload["errcode"])=="0" and info[1]==day
        rows=info[0];codes=[str(r[0]).zfill(6) for r in rows if isinstance(r,list) and len(r)>=23]
        duplicated={code for code in codes if codes.count(code)>1}
        for index,row in enumerate(rows):
            if not isinstance(row,list) or len(row)<23:continue
            code=str(row[0]).zfill(6)
            if len(code)!=6 or not code.isdigit() or code in duplicated:continue
            records.append(dict(signal_date=day,code=code,provider_close=row[21],provider_first_stamp=row[4],provider_reported_seal=row[6],
                source_path=str(path.resolve()),source_sha256=receipt["sha256"],source_record_pointer=f"/info/0/{index}",
                source_received_at=receipt.get("received_at"),reused_probe_original_clock_in_source_receipt=bool(receipt.get("reused_original_probe")),
                source_time_minutes=first_minutes(row[4],day)))
        date_summary.append(dict(signal_date=day,status="DATED_ENVELOPE_OK",count=len(rows),duplicate_codes="|".join(sorted(duplicated))))
    provider=pd.DataFrame(records)
    assert not provider.duplicated(KEYS).any()
    joined=facts.merge(provider,on=KEYS,how="left",sort=False,validate="one_to_one")
    joined["provider_observation_present"]=joined.source_record_pointer.notna()
    joined["exact_cent_close_match"]=[exact_cents(a) is not None and exact_cents(a)==exact_cents(b) for a,b in zip(joined.close,joined.provider_close)]
    joined[TIME[0]]=joined.source_time_minutes.where(joined.exact_cent_close_match)
    joined["reported_close_seal_yuan"]=joined.provider_reported_seal.map(positive).where(joined.exact_cent_close_match&joined.source_time_minutes.notna())
    output=derive_fields(joined)
    checks=[]
    for cutoff in ("2025-06-30","2025-12-31"):
        prefix=derive_fields(joined[joined.signal_date.le(cutoff)])
        np.testing.assert_allclose(prefix[TIME+FUND],output.loc[output.signal_date.le(cutoff),TIME+FUND],atol=0,rtol=0,equal_nan=True)
        checks.append(dict(cutoff=cutoff,rows=len(prefix),all_four_fields_equal=True))
    periods={}
    for name,mask in [("development",output.signal_date.lt("2025-01-01")),("validation2025H1",output.signal_date.between("2025-01-01","2025-06-30")),("evaluation",output.signal_date.ge("2025-07-01"))]:
        part=output[mask]
        periods[name]=dict(rows=len(part),dates=part.signal_date.nunique(),observations=int(part.provider_observation_present.sum()),
            price_mismatch=int((part.provider_observation_present&~part.exact_cent_close_match).sum()),field_coverage=part[TIME+FUND].notna().mean().to_dict(),
            dates_without_time_observation=int(part.groupby("signal_date")[TIME[0]].count().eq(0).sum()))
    qualified={arm:all(all(periods[p]["field_coverage"][f]>=.90 for f in fields if f in TIME+FUND) and periods[p]["dates_without_time_observation"]==0 for p in ("development","validation2025H1")) for arm,fields in ARMS.items()}
    output.to_csv(QUAL/"COMPLETE28503_NEW_FIELDS_NO_LABELS.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(date_summary).to_csv(QUAL/"ALL668_DATE_SOURCE_STATUS.csv",index=False)
    provider.to_csv(QUAL/"PROVIDER_REPORTS_WITH_POINTERS_NO_LABELS.csv",index=False,encoding="utf-8-sig")
    report=dict(at=base.now(),fits=0,labels_read=False,full_rows=len(output),full_dates=668,periods=periods,qualified_arms=qualified,prefix_checks=checks,
        new_feature_fields=TIME+FUND,source_captured_later=True,strict_originalHTTPtime_certified=False,original_universe_survivorship_unchanged=True,
        fund_definition="Provider historical dated daily closingseal report in yuan,not exchange-verified executable queue or identity")
    base.write_json(QUAL/"qualification.json",report)
    print(json.dumps(base.clean(report),ensure_ascii=True),flush=True)
    return report

def load_rows():
    rows=pd.read_csv(FACTS,dtype={"code":str},float_precision="round_trip")
    rows=rows[rows.stage.eq(1)].reset_index(drop=True)
    new=pd.read_csv(QUAL/"COMPLETE28503_NEW_FIELDS_NO_LABELS.csv",dtype={"code":str},float_precision="round_trip")
    assert set(map(tuple,rows[KEYS].to_numpy()))==set(map(tuple,new[KEYS].to_numpy()))
    out=rows.merge(new[KEYS+TIME+FUND],on=KEYS,validate="one_to_one",sort=False)
    np.testing.assert_allclose(out[ORIGINAL+[stage.TARGET]],rows[ORIGINAL+[stage.TARGET]],atol=0,rtol=0,equal_nan=True)
    screen=pd.read_csv(SCREEN,dtype={"code":str},float_precision="round_trip")
    out=out.merge(screen[KEYS+["research_selection_allowed","exclusion_reason","threeboard_evidence_status"]],on=KEYS,validate="one_to_one",sort=False)
    return out

def train_arm(arm,trainer):
    freeze_path=JOINT/"JOINT_FROZEN_PROTOCOL.json"
    freeze=json.loads(freeze_path.read_text("utf-8"))
    assert freeze["status"]=="AGREED_BEFORE_NEW_FITS" and arm in freeze["arms"] and trainer==freeze["arms"][arm]["trainer"]
    assert freeze["source_sha256"]==base.sha(__file__) and freeze["qualification_sha256"]==base.sha(QUAL/"qualification.json")
    assert freeze["arms"][arm]["fields"]==ARMS[arm]
    for path,digest in freeze["bindings"].items():assert base.sha(path)==digest
    out=OUTPUT/arm
    if out.exists():raise FileExistsError("Arm already attempted;no retune/retry/overwrite")
    out.mkdir(parents=True)
    base.write_json(out/"protocol_BEFORE_ANY_FIT.json",dict(at=base.now(),trainer=trainer,arm=arm,freeze_sha256=base.sha(freeze_path),fields=ARMS[arm],fits_before=0))
    shutil.copy2(__file__,out/"source_before_fit.py")
    rows=load_rows()
    train=members.train_members(rows,"2025-01-01")
    old=pd.read_csv(REFERENCE/"models/validation_S1_STAGE_CYCLE_GENE_full_members.csv",dtype={"code":str},float_precision="round_trip")
    assert set(map(tuple,train[KEYS+["label_end",stage.TARGET]].to_numpy()))==set(map(tuple,old[KEYS+["label_end",stage.TARGET]].to_numpy()))
    val=rows[rows.signal_date.between("2025-01-01","2025-06-30")]
    assert len(val)==4724 and val.signal_date.nunique()==117
    stage.VARIANTS[arm]=ARMS[arm]
    head=stage.fit_head(train,arm,out,"validation_"+arm)
    scored=stage.predict(head,val)
    scored.to_csv(out/"VALIDATION_ALL_PREDICTIONS_BEFORE_METRICS.csv",index=False,encoding="utf-8-sig")
    primary=risk.screen_metric(scored);non=risk.screen_metric(scored[scored.one_price.eq(0)])
    passed=bool(primary["hits"]>52 and non["hits"]>=48 and primary["decision_date_coverage"]>=.95)
    result=dict(at=base.now(),trainer=trainer,arm=arm,primary=primary,non_one_price=non,unfiltered_diagnostic=stage.metric(scored),pass_validation=passed,
        core_fits=2,temporal_calibrator_fits=1,baseline_refits=0,second_stage_refits=0,new_forward_results=0,actual_trades=0,goal_60_achieved=False,
        monthly_fits=0,status="VALIDATION_PASS_AWAIT_JOINT_SELECTION" if passed else "REJECTED_VALIDATION",source_sha256=base.sha(__file__))
    base.write_json(out/"report.json",result)
    for path,digest in freeze["bindings"].items():assert base.sha(path)==digest
    print(json.dumps(base.clean(result),ensure_ascii=True),flush=True)
    return result

def evaluate():
    freeze=json.loads((JOINT/"JOINT_FROZEN_PROTOCOL.json").read_text("utf-8"))
    reports={arm:json.loads((OUTPUT/arm/"report.json").read_text("utf-8")) for arm in freeze["arms"]}
    passed=[arm for arm,report in reports.items() if report["pass_validation"]]
    winner=sorted(passed,key=lambda arm:(-reports[arm]["primary"]["hits"],len(ARMS[arm]),arm))[0] if passed else None
    selection=dict(at=base.now(),reports=reports,selected=winner,baseline_wins_ties=True,baseline_hits=52,baseline_non_one_price_hits=48)
    with (OUTPUT/"JOINT_SELECTION_BEFORE_EVALUATION.json").open("x",encoding="utf-8") as f:json.dump(base.clean(selection),f,ensure_ascii=False,indent=2)
    result=dict(at=base.now(),selection=selection,goal_60_achieved=False,production_deployment_allowed=False,baseline_refits=0,second_stage_refits=0,new_forward_results=0,actual_trades=0)
    if winner is None:
        result.update(status="BOTH_FIXED_ARMS_REJECTED",monthly_fits=0)
        base.write_json(OUTPUT/"report.json",result);print(json.dumps(base.clean(result),ensure_ascii=True));return result
    out=OUTPUT/"evaluation";out.mkdir()
    rows=load_rows();sessions=json.loads(base.INPUTS["calendar"].read_text("utf-8"))["sessions"]
    stage.VARIANTS[winner]=ARMS[winner];parts=[];began=time.monotonic()
    for month,part in rows[rows.signal_date.ge("2025-07-01")].groupby(rows.signal_date.str[:7],sort=True):
        if time.monotonic()-began>20*60:raise TimeoutError("Preserve checkpoint;no rerun fitted months")
        boundary=month+"-01";position=int(np.searchsorted(sessions,boundary));train=members.train_members(rows,boundary,position-252)
        old=pd.read_csv(REFERENCE/"models"/(month+"_S1_STAGE_CYCLE_GENE_full_members.csv"),dtype={"code":str},float_precision="round_trip")
        assert set(map(tuple,train[KEYS+["label_end",stage.TARGET]].to_numpy()))==set(map(tuple,old[KEYS+["label_end",stage.TARGET]].to_numpy()))
        head=stage.fit_head(train,winner,out,month+"_"+winner)
        assert head["train_max_label_end"]<part.signal_date.min()
        scored=stage.predict(head,part);scored["model_month"]=month
        scored.to_csv(out/(month+"_ALL_PREDICTIONS_BEFORE_METRICS.csv"),index=False,encoding="utf-8-sig");parts.append(scored)
        print(month+" saved",flush=True)
    assert len(parts)==14
    scored=pd.concat(parts,ignore_index=True);scored.to_csv(out/"ALL14_MONTHS_PREDICTIONS.csv",index=False,encoding="utf-8-sig")
    old=risk.attach_frozen(rows,REFERENCE/"S1_ALL_PREDICTIONS.csv")
    result["evaluation"]=history.period_metrics(scored,old)
    second=pd.read_csv(REFERENCE/"S2_ALL_PREDICTIONS.csv",dtype={"code":str},float_precision="round_trip")
    chain,result["same_stock_second_recheck"]=stage.chain_recheck(scored[scored.research_selection_allowed],second)
    chain.to_csv(out/"SAME_STOCK_SECOND_RECHECK.csv",index=False,encoding="utf-8-sig")
    result["calibration"]=stage.calibration_metrics(scored[scored.research_selection_allowed])
    m=result["evaluation"]["all"]["new_screened"]
    result["goal_60_achieved"]=bool(m["hit_rate"]>=.60 and m["known"]>=200 and m["outcome_coverage"]>=.95 and m["abstention_dates"]==0 and all(result["evaluation"][p]["new_screened"]["hit_rate"]>=.55 for p in ("2025H2","2026")))
    result.update(at=base.now(),status="FIXED_JOINT_MONTHLY_EVALUATION_COMPLETE",monthly_fits=14)
    base.write_json(OUTPUT/"report.json",result);print(json.dumps(base.clean(result),ensure_ascii=True),flush=True);return result

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("command",choices=["qualify","train","evaluate"]);parser.add_argument("--arm",choices=list(ARMS));parser.add_argument("--trainer",choices=["Codex","Gemini"])
    args=parser.parse_args()
    if args.command=="qualify":qualify()
    elif args.command=="evaluate":evaluate()
    elif not args.arm or not args.trainer:parser.error("train requires arm and actual trainer")
    else:train_arm(args.arm,args.trainer)
