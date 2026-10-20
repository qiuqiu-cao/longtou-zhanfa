"""Fixed legacy-cohort information experiment; no current industry/fund labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import first_board_leader_model as base
import first_second_stage_model as stage
import first_board_capital_risk_model as risk
import first_board_matured_event_model as previous
import legacy_industry_snapshot as legacy

REF=risk.REFERENCE
SCREEN=previous.SCREEN
CONTROL=["legacy_group_known"]
PEERS=["legacy_peer_observed_n","legacy_peer_catalog_fraction","legacy_peer_limit_rate",
    "legacy_peer_first_rate","legacy_peer_relay_rate","legacy_peer_failed_attempt_rate",
    "legacy_peer_mean_return","legacy_peer_positive_fraction","legacy_peer_max_height",
    "legacy_peer_previous_first_promotion","legacy_peer_previous_high_break_fraction",
    "legacy_first_competitor_observed_n","legacy_first_turnover_rank","legacy_first_amount_rank",
    "legacy_first_volume_rank","legacy_self_to_peer_return5","legacy_self_to_peer_amount5"]
VARIANTS={"LEGACY_COVERAGE_CONTROL":stage.VARIANTS["STAGE_CYCLE_GENE"]+CONTROL,
          "LEGACY_PEER_INFORMATION":stage.VARIANTS["STAGE_CYCLE_GENE"]+CONTROL+PEERS}
POLICY=dict(source="Qualified immutable2021static category proxy,not signal-date current industry or actual theme",
    reference_catalog="Raw2021mainboard-prefix category membership;denominator is old catalog,not current active/exchange-complete universe",
    observations="Eligible,valid,regular,positive amount/volume current bar and exact adjacent eligible/valid/regular previous bar;provider preclose agrees with previous unadjusted close within0.011yuan",
    peer_statistics="All peer counts/means/rates/maxima exclude own contribution;counts refer only to observed archived cohort",
    unknowns="Unmapped/identity-conflict stocks keep all17peerfieldsNaN. Zeroobserved count is not proof of zero actual peers;rates/returns undefined without observations or attempts remainNaN",
    first_rank="Own percentile among observed same-legacy-group first boards with the field observed;average ties;competitor count excludes self",
    previous_first="Verified adjacent previous first flag,not merely height1aftera missing predecessor",
    previous_high="Previous adjacent height>=3,current break observed at signal close;not a threeboardfund identity",
    relative5="Return5requires current/prior4qualified adjacent observations;amount-ratio5requires prior5qualified observations;exclude self from peer means",
    mapping_not_used_as_category_id=True,source_presence_control=True,old2018fallback=False)


def build_peer_features(bars,keys,mapping=None):
    if mapping is None:mapping,_=legacy.load_snapshot()
    columns=["date","code","p","close","pre_close","eligible","valid_price","regular_price","amount","volume",
             "limit","height","first","failed","turnover","volume_ratio_5","amount_ratio_5","return_5"]
    b=bars[columns].sort_values(["code","p"]).copy()
    if b.duplicated(["date","code"]).any() or keys.duplicated(["signal_date","code"]).any():raise ValueError("Duplicate price/signal key")
    lookup=mapping.set_index("code").legacy_group_PROXY_ALLOWED
    if lookup.index.duplicated().any():raise ValueError("Ambiguous legacy code mapping")
    b["legacy_group_PROXY"]=b.code.map(lookup)
    known=b.legacy_group_PROXY.notna()
    history=b.groupby("code",sort=False)
    adjacent=b.p.sub(history.p.shift()).eq(1)
    previous_ok=history.eligible.shift().fillna(False).astype(bool)&history.valid_price.shift().fillna(False).astype(bool)&history.regular_price.shift().fillna(False).astype(bool)
    agreement=b.pre_close.sub(history.close.shift()).abs().le(.011)
    observed=known&b.eligible&b.valid_price&b.regular_price&b.amount.gt(0)&b.volume.gt(0)&adjacent&previous_ok&agreement
    previous_first=history["first"].shift().fillna(False).astype(bool)&adjacent
    previous_high=history.height.shift().ge(3)&adjacent
    group=[b.date,b.legacy_group_PROXY]
    def total(series):return series.groupby(group,sort=False).transform("sum",min_count=1)
    def peer_mean(series):
        value=series.where(observed)
        count=value.groupby(group,sort=False).transform("count")-value.notna().astype(int)
        return (total(value)-value.fillna(0))/count.where(count.gt(0))
    def peer_ratio(numerator,denominator):
        top=total(numerator.astype(int))-numerator.astype(int)
        bottom=total(denominator.astype(int))-denominator.astype(int)
        return top/bottom.where(bottom.gt(0))
    peer_n=total(observed.astype(int))-observed.astype(int)
    b[CONTROL[0]]=known.astype(int)
    b[PEERS[0]]=peer_n
    catalog=mapping[mapping.code.str.startswith(("000","001","002","003","600","601","603","605"))&mapping.legacy_group_RAW.notna()].groupby("legacy_group_RAW").size()
    denominator=b.legacy_group_PROXY.map(catalog)-1
    b[PEERS[1]]=peer_n/denominator.where(denominator.gt(0))
    b[PEERS[2]]=peer_ratio(observed&b.limit,observed)
    b[PEERS[3]]=peer_ratio(observed&b["first"],observed)
    b[PEERS[4]]=peer_ratio(observed&b.height.ge(2),observed)
    b[PEERS[5]]=peer_ratio(observed&b.failed,observed&(b.failed|b.limit))
    daily_return=(b.close/history.close.shift()-1).where(observed)
    b[PEERS[6]]=peer_mean(daily_return)
    b[PEERS[7]]=peer_ratio(observed&daily_return.gt(0),observed)
    height=b.height.where(observed)
    top=height.groupby(group,sort=False).transform("max")
    top_count=total(observed&height.eq(top))
    second=height.where(height.lt(top)).groupby(group,sort=False).transform("max")
    own_unique_top=observed&height.eq(top)&top_count.eq(1)
    b[PEERS[8]]=top.where(~own_unique_top,second).where(peer_n.gt(0))
    b[PEERS[9]]=peer_ratio(observed&previous_first&b.limit,observed&previous_first)
    b[PEERS[10]]=peer_ratio(observed&previous_high&~b.limit,observed&previous_high)
    first_observed=observed&b["first"]
    b[PEERS[11]]=total(first_observed.astype(int))-first_observed.astype(int)
    for field,source in [(PEERS[12],"turnover"),(PEERS[13],"amount"),(PEERS[14],"volume_ratio_5")]:
        b[field]=b[source].where(first_observed).groupby(group,sort=False).rank(pct=True)
    valid5=observed.groupby(b.code,sort=False).transform(lambda s:s.astype(int).rolling(5,min_periods=5).sum()).eq(5)
    prior5=observed.groupby(b.code,sort=False).transform(lambda s:s.astype(int).shift(1).rolling(5,min_periods=5).sum()).eq(5)
    return5=b.return_5.where(observed&valid5)
    b[PEERS[15]]=return5-peer_mean(return5)
    amount5=b.amount_ratio_5.where(observed&prior5&b.amount_ratio_5.gt(0))
    peer_amount=peer_mean(amount5)
    b[PEERS[16]]=np.log(amount5/peer_amount.where(peer_amount.gt(0)))
    b[PEERS]=b[PEERS].where(known,axis=0).replace([np.inf,-np.inf],np.nan)
    produced=b[["date","code","legacy_group_PROXY"]+CONTROL+PEERS].rename(columns={"date":"signal_date"})
    out=keys[["signal_date","code"]].merge(produced,on=["signal_date","code"],validate="one_to_one",how="left")
    if len(out)!=len(keys) or out[CONTROL[0]].isna().any():raise ValueError("Signal missing exact source bar")
    return out


def run(output):
    out=Path(output)
    if out.exists():raise FileExistsError("Fixed experiment already exists")
    out.mkdir(parents=True)
    source=REF/"all_first_and_second_boards_WITH_SEPARATE_LABELS.csv"
    screen_file=SCREEN/"COMPLETE_FIRST_BOARD_FACTS_FEATURES_LABELS_AND_SCREEN.csv"
    mapping,qualification=legacy.load_snapshot()
    protected=[Path(base.__file__),Path(stage.__file__),Path(risk.__file__),Path(previous.__file__),Path(legacy.__file__),
        REF/"report.json",REF/"S1_ALL_PREDICTIONS.csv",REF/"S2_ALL_PREDICTIONS.csv",SCREEN/"report.json"]
    sources=[source,screen_file,legacy.BANK/"qualified_legacy_snapshot.json",legacy.BANK/qualification["mapping_file"],legacy.BANK/"verification.json"]
    protocol=dict(formed_at=base.now(),batch_started_at="2026-10-20T14:13:36+00:00",budget_minutes=30,
        previous_goal_turn="PROGRESS:immutable2021legacygroup proxyqualified,25808/28503mapped,2695unknown;3identityconflictsquarantined;strictcurrentindustryandoriginalpublicationnotcertified",
        hypothesis="Causally observed legacy-cohort breadth,ladder and relative competition add first-close Top1promotion information beyond the original62fields and source-presence control",
        variants=VARIANTS,policy=POLICY,baseline="Frozen62-field original scores plus unchanged capital/thin-turnover screen;no old models or stage2refit",
        validation="Complete original2025H1:4724rows/117dates. Require >52Top1hits,>=48non-one-price hits,>=95%decision-datecoverage;baselinewinsties. Passing arms mosthits,coveragecontrolwinsarmties",
        training="Exact original full known matured members before2025;only selected arm in14original252official-position monthly windows. Parameters/weights/42datecalibration unchanged;unmapped rows remainincluded",
        evaluation="Save fullscores beforemetrics,report samepool/phase/non-one-price and same-code savedsecond-head recheck;report control-versus-peer validation differences without choosing subset thresholds",
        target60="DailyTop1knownhitrate>=60%,>=200known,>=95%outcomecoverage,noabstentiondates;both2025H2and2026>=55%;firstcloseonly,no nextauctioninput",
        new_current_industry_truth=False,strict_original_publication_verified=False,old_proxy_source_retrieved_later=True,
        raw_source_hashes={str(p.resolve()):base.sha(p) for p in base.INPUTS.values()},inputs={str(p.resolve()):base.sha(p) for p in sources},
        protected={str(p.resolve()):base.sha(p) for p in protected},source_sha256=base.sha(__file__),
        historical_replay_not_pristine_blind=True,stop="Twoarms once;fixedcomparison or30minutes. No feature/window/weight/threshold changes afterresults",
        new_forward_results=0,actual_trades=0,production_changes=False)
    base.write_json(out/"protocol_BEFORE_ANY_FIT.json",protocol);shutil.copy2(__file__,out/"source_before_first_fit.py")
    original=pd.read_csv(source,dtype={"code":str},float_precision="round_trip")
    first=original[original.stage.eq(1)].copy()
    _,bars,sessions,audit=base.build_dataset()
    features=build_peer_features(bars,first,mapping)
    rows=first.merge(features,on=["signal_date","code"],validate="one_to_one")
    screen=pd.read_csv(screen_file,dtype={"code":str},float_precision="round_trip")
    rows=rows.merge(screen[["signal_date","code","research_selection_allowed","exclusion_reason","threeboard_evidence_status"]],on=["signal_date","code"],validate="one_to_one")
    rows.to_csv(out/"ALL_FIRST_BOARD_LEGACY_PEER_FEATURES_UNCHANGED_LABELS.csv",index=False,encoding="utf-8-sig")
    checks=[]
    for cutoff in ("2025-06-30","2025-12-31"):
        prefix=build_peer_features(bars[bars.date.le(cutoff)],first[first.signal_date.le(cutoff)],mapping)
        old=rows[rows.signal_date.le(cutoff)]
        np.testing.assert_allclose(prefix[CONTROL+PEERS],old[CONTROL+PEERS],atol=1e-11,rtol=1e-11,equal_nan=True)
        checks.append(dict(cutoff=cutoff,rows=len(prefix),all18_fields_equal=True))
    assert rows.legacy_group_known.eq(1).sum()==25808 and rows.loc[rows.legacy_group_known.eq(0),PEERS].isna().all().all()
    base.write_json(out/"causality_and_source_coverage_BEFORE_FITS.json",dict(prefix_checks=checks,field_coverage=rows[CONTROL+PEERS].notna().mean().to_dict(),
        groups_known=int(rows.legacy_group_known.sum()),groups_unknown=int(rows.legacy_group_known.eq(0).sum()),legacy_not_current_industry=True))
    del bars,features
    known=rows[rows[stage.TARGET].notna()]
    train=known[known.signal_date.lt("2025-01-01")&known.label_end.lt("2025-01-01")]
    val=rows[rows.signal_date.between("2025-01-01","2025-06-30")]
    frozen=risk.attach_frozen(rows,REF/"validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv")
    baseline=risk.screen_metric(frozen);nonbase=risk.screen_metric(frozen[frozen.one_price.eq(0)])
    assert baseline["hits"]==52 and nonbase["hits"]==48
    decisions={}
    for variant,fields in VARIANTS.items():
        stage.VARIANTS[variant]=fields
        bundle=stage.fit_head(train,variant,out,"validation_"+variant)
        score=stage.predict(bundle,val)
        score.to_csv(out/("validation_"+variant+"_ALL_BEFORE_METRICS.csv"),index=False)
        primary=risk.screen_metric(score);non=risk.screen_metric(score[score.one_price.eq(0)])
        decisions[variant]=dict(primary=primary,non_one_price=non,unfiltered_diagnostic=stage.metric(score),
            mapped_group_diagnostic=risk.screen_metric(score[score.legacy_group_known.eq(1)]),
            unknown_group_diagnostic=risk.screen_metric(score[score.legacy_group_known.eq(0)]),
            pass_validation=bool(primary["hits"]>52 and non["hits"]>=48 and primary["decision_date_coverage"]>=.95))
        print(json.dumps(base.clean(dict(variant=variant,validation=decisions[variant])),ensure_ascii=True),flush=True)
    passing=[v for v in VARIANTS if decisions[v]["pass_validation"]]
    selected=max(passing,key=lambda v:decisions[v]["primary"]["hits"]) if passing else None
    decision=dict(formed_at=base.now(),selected=selected,baseline=baseline,baseline_non_one_price=nonbase,variants=decisions)
    base.write_json(out/"selection_BEFORE_EVALUATION.json",decision)
    report=dict(formed_at=None,decision=decision,goal_60_passed=False,new_forward_results=0,actual_trades=0,independent_current_industry_validated=False)
    if selected:
        parts=[]
        for month,part in rows[rows.signal_date.ge("2025-07-01")].groupby(rows.signal_date.str[:7],sort=True):
            boundary=month+"-01";p=int(np.searchsorted(sessions,boundary))
            tr=known[known.p.ge(p-252)&known.label_end.lt(boundary)]
            bundle=stage.fit_head(tr,selected,out,month+"_"+selected)
            score=stage.predict(bundle,part);score["model_month"]=month
            score.to_csv(out/(month+"_ALL_BEFORE_METRICS.csv"),index=False);parts.append(score)
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
        report["goal_60_passed"]=bool(m["hit_rate"]>=.60 and m["known"]>=200 and m["outcome_coverage"]>=.95 and m["abstention_dates"]==0 and all(report["evaluation"][p]["new_screened"]["hit_rate"]>=.55 for p in ("2025H2","2026")))
    for path,digest in protocol["protected"].items():assert base.sha(path)==digest
    report.update(formed_at=base.now(),core_fit_count=len(list((out/"models").glob("*_COMPLETED.json"))),
        status="FIXED_MONTHLY_COMPARISON_FINISHED" if selected else "BOTH_ARMS_REJECTED_VALIDATION_NO_MONTHLY_FITS")
    base.write_json(out/"report.json",report)
    print(json.dumps(base.clean(report),ensure_ascii=True,indent=2),flush=True)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True)
    run(parser.parse_args().output)
