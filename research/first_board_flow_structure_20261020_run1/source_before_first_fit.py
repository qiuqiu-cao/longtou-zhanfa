"""Observable first-close volume/price structure, not inferred fund ownership."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import first_board_leader_model as base
import first_second_stage_model as stage
import first_board_capital_risk_model as risk
import first_board_matured_event_model as previous
import joblib
import numpy as np
import pandas as pd

REF=risk.REFERENCE
SCREEN=previous.SCREEN
PAST=["structure_clv_volume20","structure_clv_observed_fraction20","structure_upday_volume_fraction20",
      "structure_volume_concentration20","structure_turnover_sum20","structure_direction_efficiency20",
      "structure_return_per_amount_median20","structure_gap_move_fraction20","structure_high_volume_narrow_range_fraction20"]
SIGNAL=["structure_signal_impact_vs_past20","structure_signal_range_vs_past20"]
VARIANTS={"PRIOR_PRICE_VOLUME_STRUCTURE":stage.VARIANTS["STAGE_CYCLE_GENE"]+PAST,
          "STRUCTURE_AND_SIGNAL_RESPONSE":stage.VARIANTS["STAGE_CYCLE_GENE"]+PAST+SIGNAL}
SOURCES=base.ROOT/"research/first_board_flow_structure_20261020_sources"
MINUTES=base.ROOT/"data/v289/full_august_five_minute_20261002_v1"
POLICY=dict(prior_window=20,prior_only_shift=1,high_volume_reference=20,
    clv="(2close-high-low)/(high-low); undefined one-price sessions excluded, observed fraction retained; at least15of20 defined pairs",
    volume_concentration="sum(volume squared)/(sum volume)squared over prior20, not shareholder concentration",
    direction_efficiency="abs(sum daily returns)/sum(abs daily returns) over prior20",
    impact="absolute daily return divided by amount in100million yuan; price-response proxy, not net inflow",
    high_volume_narrow_range="Past day's volume above its preceding20-day median and amplitude below its preceding20-day median; average over current prior20",
    continuity="Exact official prior20 positions;40 for reference-based high-volume/narrow-range. No bridging missing or invalid/nontrading bars.",
    identities="Absorption/control/emotion are competing hypotheses. Program/account identity UNKNOWN without appropriate evidence; no fabricated mechanism labels.",
    risk_screen="Exact previous fixed capital and thin-turnover screen, unchanged")


def structure_from_bars(bars):
    b=bars.sort_values(["code","p"]).copy()
    valid=b.eligible&b.valid_price&b.regular_price&b.amount.gt(0)&b.volume.gt(0)
    price_range=b.high-b.low
    volume=b.volume.where(valid)
    ret=(b.close/b.pre_close-1).where(valid)
    amp=((b.high-b.low)/b.pre_close).where(valid)
    clv=((2*b.close-b.high-b.low)/price_range.where(price_range.gt(0))).where(valid)
    impact=(ret.abs()/(b.amount/1e8).where(b.amount.gt(0))).where(valid)
    gap=(b.open/b.pre_close-1).abs().where(valid)
    intraday=(b.close/b.open-1).abs().where(valid)
    def roll(series,op,window=20,minimum=None,shift=1):
        return series.groupby(b.code,sort=False).transform(lambda s:getattr(s.shift(shift).rolling(window,min_periods=window if minimum is None else minimum),op)())
    span20=b.p.sub(b.groupby("code",sort=False).p.shift(20)).eq(20)
    span40=b.p.sub(b.groupby("code",sort=False).p.shift(40)).eq(40)
    sumvol=roll(volume,"sum")
    paired_volume=volume.where(clv.notna())
    clv_count=roll(clv,"count",minimum=1)
    clv_den=roll(paired_volume,"sum",minimum=15)
    out=b[["date","code"]].copy()
    out[PAST[0]]=roll(clv*volume,"sum",minimum=15)/clv_den.where(clv_den.gt(0))
    out[PAST[1]]=clv_count/20
    # Every observed trading day participates in both numerator/denominator;
    # a down/flat day contributes known zero up-day volume, not missing data.
    up_volume=volume.where(ret.gt(0),0.).where(ret.notna())
    out[PAST[2]]=roll(up_volume,"sum")/sumvol.where(sumvol.gt(0))
    out[PAST[3]]=roll(volume.pow(2),"sum")/sumvol.pow(2).where(sumvol.gt(0))
    out[PAST[4]]=roll(b.turnover.where(valid),"sum")
    movement=roll(ret.abs(),"sum")
    out[PAST[5]]=roll(ret,"sum").abs()/movement.where(movement.gt(0))
    out[PAST[6]]=roll(impact,"median")
    gap_sum=roll(gap,"sum");inside_sum=roll(intraday,"sum")
    out[PAST[7]]=gap_sum/(gap_sum+inside_sum).where((gap_sum+inside_sum).gt(0))
    volume_reference=roll(volume,"median")
    range_reference=roll(amp,"median")
    condition=(volume.gt(volume_reference)&amp.lt(range_reference)).astype(float).where(volume_reference.notna()&range_reference.notna()&valid)
    out[PAST[8]]=roll(condition,"mean").where(span40)
    out[SIGNAL[0]]=impact/out[PAST[6]].where(out[PAST[6]].gt(0))
    out[SIGNAL[1]]=amp/roll(amp,"median").where(lambda x:x.gt(0))
    out[PAST[:8]+SIGNAL]=out[PAST[:8]+SIGNAL].where(span20,axis=0)
    out[PAST+SIGNAL]=out[PAST+SIGNAL].replace([np.inf,-np.inf],np.nan)
    return out


def historical_cards(rows,out):
    """Complete historical61-stock illustration, never a new forward call."""
    day="2026-08-28"
    current=rows[rows.signal_date.eq(day)].copy()
    current=current[[c for c in current if "LABEL_ONLY" not in c and c not in ("label_end","baseline_score")]]
    raw=pd.read_csv(MINUTES/"five_minute_bars_UNRECONCILED.csv",dtype={"code":str},float_precision="round_trip")
    raw=raw[raw.minute.str[:10].eq(day)]
    old_cohort=pd.read_csv(MINUTES/"august_cohort_LABEL_ONLY.csv",dtype={"code":str},usecols=["date","code","boards"])
    first_enrol=old_cohort.groupby("code").date.min()
    facts=[]
    grid=[f"{h:02d}:{m:02d}" for h,minutes in [(9,range(35,60,5)),(10,range(0,60,5)),(11,range(0,35,5)),(13,range(5,60,5)),(14,range(0,60,5)),(15,[0])] for m in minutes]
    for row in current.itertuples():
        m=raw[raw.code.eq(row.code)].sort_values("minute")
        item=dict(signal_date=day,code=row.code,minute_rows=len(m),minute_OHLC_equal=False,minute_grid_complete=False,
                  minute_volume_reconciled=False,first_touch_bar_end=None,first_limit_close_bar_end=None,
                  observed_later_below_cap_bars=None,max_5minute_volume_share=None,
                  first_board_mechanism_truth="UNKNOWN",program_fund_identity="UNKNOWN",absorption_fund_identity="UNKNOWN",
                  source_selection="Original monthly relay cohort, not a complete first-board sample",source_first_relay_date=first_enrol.get(row.code),
                  later_relay_selection_for_this_day=bool(first_enrol.get(row.code,day)>day))
        if len(m):
            expected=np.rint(np.array([row.open,row.high,row.low,row.close])*100)
            observed=np.rint(np.array([m.open.iloc[0],m.high.max(),m.low.min(),m.close.iloc[-1]])*100)
            item["minute_OHLC_equal"]=bool(np.array_equal(expected,observed))
            item["minute_grid_complete"]=bool(len(m)==48 and sorted(m.minute.str[11:16])==sorted(grid) and not m.minute.duplicated().any())
            item["minute_volume_reconciled"]=bool(abs(m.volume_shares.sum()-row.volume)<=max(1,row.volume*.02))
            if item["minute_OHLC_equal"] and item["minute_grid_complete"]:
                cap=row.cap/100
                touch=m[m.high.ge(cap-.000001)]
                closing=m[m.close.ge(cap-.000001)]
                item["first_touch_bar_end"]=touch.minute.iloc[0] if len(touch) else None
                item["first_limit_close_bar_end"]=closing.minute.iloc[0] if len(closing) else None
                if len(touch):item["observed_later_below_cap_bars"]=int(m[m.minute.gt(touch.minute.iloc[0])].low.lt(cap-.000001).sum())
                if item["minute_volume_reconciled"]:item["max_5minute_volume_share"]=float(m.volume_shares.max()/m.volume_shares.sum())
        facts.append(item)
    current=current.merge(pd.DataFrame(facts),on=["signal_date","code"],validate="one_to_one")
    current.to_csv(out/"HISTORICAL_20260828_COMPLETE61_STRUCTURE_CARDS_NO_SCORES_NO_TARGETS.csv",index=False,encoding="utf-8-sig")
    return dict(date=day,complete_pool=len(current),minute_present=int(current.minute_rows.gt(0).sum()),
                OHLC_and_grid=int((current.minute_OHLC_equal&current.minute_grid_complete).sum()),
                price_grid_volume=int((current.minute_OHLC_equal&current.minute_grid_complete&current.minute_volume_reconciled).sum()),
                enrolled_only_after_signal=int(current.later_relay_selection_for_this_day.sum()),
                effect_training_uses_minutes=False,new_forecast=False,fund_identity_unknown=len(current))


def run(output):
    out=Path(output)
    if out.exists():raise FileExistsError("fixed experiment already exists")
    out.mkdir(parents=True)
    source=REF/"all_first_and_second_boards_WITH_SEPARATE_LABELS.csv"
    protected=[Path(base.__file__),Path(stage.__file__),Path(risk.__file__),Path(previous.__file__),REF/"report.json",REF/"S1_ALL_PREDICTIONS.csv",REF/"S2_ALL_PREDICTIONS.csv",SCREEN/"report.json"]
    sources=[source,SCREEN/"COMPLETE_FIRST_BOARD_FACTS_FEATURES_LABELS_AND_SCREEN.csv",SOURCES/"source_and_sampling_qualification.json",MINUTES/"five_minute_bars_UNRECONCILED.csv",MINUTES/"request_scope.json"]
    protocol=dict(formed_at=base.now(),budget_minutes=30,hypothesis="Observable prior liquidity,volume concentration,and volume-versus-price response distinguish first-close promotion better than62-field price/cycle/gene baseline.",
        user_new_request="Assess accumulation/control versus quant/emotion; preserve unknown or mixed mechanisms rather than invent actor labels",
        policy=POLICY,variants=VARIANTS,baseline="Frozen62-field scores with unchanged prior capital/risk screen; original unscreened metrics also retained",
        validation="2025H1 complete117date original pool; require strictly moreTop1hits than screened baseline,>=95% decision-date coverage and no fewer non-one-price Top1hits. Most hits wins, prior-only arm wins ties.",
        evaluation="Only selected arm14monthly original252official-position windows, originalHGB160/15/leaf60/lr.06/l2=8/balanced/seed42, equal-day weights,42date temporal calibration; originalBASE and second head reused.",
        target60="Known selectedTop1>=60%,>=200known,>=95% outcome coverage,no abstention dates,both2025H2/2026>=55%; same stock rescored at second close",
        fields_are_observed_proxies_not_mechanism_labels=True,minute_data_excluded_from_fits=True,
        stop="Fixed2arms once; no threshold/parameter changes after results; preserve all prior rejected experiments",
        protected={str(p.resolve()):base.sha(p) for p in protected},inputs={str(p.resolve()):base.sha(p) for p in sources},
        raw_sources={str(p.resolve()):base.sha(p) for p in base.INPUTS.values()},source_sha256=base.sha(__file__),actual_orders=0,production_changes=False,historical_research_not_pristine_blind=True)
    base.write_json(out/"protocol_BEFORE_ANY_FIT.json",protocol);shutil.copy2(__file__,out/"source_before_first_fit.py")
    original=pd.read_csv(source,dtype={"code":str},float_precision="round_trip")
    first=original[original.stage.eq(1)].copy()
    _,bars,sessions,audit=base.build_dataset()
    extra=structure_from_bars(bars).rename(columns={"date":"signal_date"})
    rows=first.merge(extra,on=["signal_date","code"],validate="one_to_one")
    screen=pd.read_csv(SCREEN/"COMPLETE_FIRST_BOARD_FACTS_FEATURES_LABELS_AND_SCREEN.csv",dtype={"code":str},float_precision="round_trip")
    rows=rows.merge(screen[["signal_date","code","research_selection_allowed","exclusion_reason","threeboard_evidence_status"]],on=["signal_date","code"],validate="one_to_one")
    rows.to_csv(out/"ALL_FIRST_BOARD_OBSERVABLE_STRUCTURE_AND_UNCHANGED_LABELS.csv",index=False,encoding="utf-8-sig")
    checks=[]
    for cutoff in ("2025-06-30","2025-12-31"):
        prefix=structure_from_bars(bars[bars.date.le(cutoff)]).rename(columns={"date":"signal_date"})
        new=first[first.signal_date.le(cutoff)][["signal_date","code"]].merge(prefix,on=["signal_date","code"],validate="one_to_one")
        old=rows[rows.signal_date.le(cutoff)]
        np.testing.assert_allclose(new[PAST+SIGNAL],old[PAST+SIGNAL],atol=1e-12,rtol=1e-12,equal_nan=True)
        checks.append(dict(cutoff=cutoff,rows=len(new),all11_equal=True,no_outcome_columns_in_calculation=True))
    case=historical_cards(rows,out)
    base.write_json(out/"feature_causality_and_case_coverage_BEFORE_FITS.json",dict(prefix_checks=checks,case=case,field_coverage=rows[PAST+SIGNAL].notna().mean().to_dict()))
    del bars,extra
    known=rows[rows[stage.TARGET].notna()]
    train=known[known.signal_date.lt("2025-01-01")&known.label_end.lt("2025-01-01")]
    val=rows[rows.signal_date.between("2025-01-01","2025-06-30")]
    frozen=risk.attach_frozen(rows,REF/"validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv")
    baseline=risk.screen_metric(frozen);nonbase=risk.screen_metric(frozen[frozen.one_price.eq(0)]);decisions={}
    for variant,fields in VARIANTS.items():
        stage.VARIANTS[variant]=fields
        bundle=stage.fit_head(train,variant,out,"validation_"+variant)
        scored=stage.predict(bundle,val)
        scored.to_csv(out/("validation_"+variant+"_ALL_BEFORE_METRICS.csv"),index=False)
        m=risk.screen_metric(scored);non=risk.screen_metric(scored[scored.one_price.eq(0)])
        decisions[variant]=dict(primary=m,non_one_price=non,unfiltered_diagnostic=stage.metric(scored),
            pass_validation=bool(m["hits"]>baseline["hits"] and m["decision_date_coverage"]>=.95 and non["hits"]>=nonbase["hits"]))
        print(json.dumps(base.clean(dict(variant=variant,metrics=decisions[variant])),ensure_ascii=True),flush=True)
    passing=[v for v in VARIANTS if decisions[v]["pass_validation"]]
    selected=max(passing,key=lambda v:decisions[v]["primary"]["hits"]) if passing else None
    decision=dict(formed_at=base.now(),selected=selected,baseline=baseline,baseline_non_one_price=nonbase,variants=decisions)
    base.write_json(out/"selection_BEFORE_EVALUATION.json",decision)
    report=dict(formed_at=None,decision=decision,goal_60_passed=False,mechanism_identity_validated=False,cases=case,actual_trades=0,new_forward_results=0)
    if selected:
        parts=[]
        for month,part in rows[rows.signal_date.ge("2025-07-01")].groupby(rows.signal_date.str[:7],sort=True):
            boundary=month+"-01";p=int(np.searchsorted(sessions,boundary))
            tr=known[known.p.ge(p-252)&known.label_end.lt(boundary)]
            head=stage.fit_head(tr,selected,out,month+"_"+selected)
            score=stage.predict(head,part);score["model_month"]=month
            score.to_csv(out/(month+"_ALL_BEFORE_METRICS.csv"),index=False);parts.append(score)
            print(month+" saved",flush=True)
        predictions=pd.concat(parts,ignore_index=True)
        predictions.to_csv(out/"EVALUATION_ALL_PREDICTIONS.csv",index=False,encoding="utf-8-sig")
        frozen=risk.attach_frozen(rows,REF/"S1_ALL_PREDICTIONS.csv")
        report["evaluation"]=previous.period_metrics(predictions,frozen)
        allowed=predictions[predictions.research_selection_allowed]
        second=pd.read_csv(REF/"S2_ALL_PREDICTIONS.csv",dtype={"code":str},float_precision="round_trip")
        chain,report["same_stock_second_recheck"]=stage.chain_recheck(allowed,second)
        chain.to_csv(out/"SAME_STOCK_SECOND_RECHECK.csv",index=False,encoding="utf-8-sig")
        report["historical_calibration_allowed_pool"]=stage.calibration_metrics(allowed)
        m=report["evaluation"]["all"]["new_screened"]
        report["goal_60_passed"]=bool(m["hit_rate"]>=.60 and m["known"]>=200 and m["outcome_coverage"]>=.95 and m["abstention_dates"]==0 and all(report["evaluation"][x]["new_screened"]["hit_rate"]>=.55 for x in ("2025H2","2026")))
    report.update(formed_at=base.now(),core_fit_count=len(list((out/"models").glob("*_COMPLETED.json"))),
        source_sha256=base.sha(__file__),status="FIXED_EVALUATION_FINISHED" if selected else "BOTH_ARMS_REJECTED_VALIDATION_NO_TEST_FITS")
    for path,digest in protocol["protected"].items():assert base.sha(path)==digest
    base.write_json(out/"report.json",report)
    print(json.dumps(base.clean(report),ensure_ascii=True,indent=2))
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True)
    run(parser.parse_args().output)
