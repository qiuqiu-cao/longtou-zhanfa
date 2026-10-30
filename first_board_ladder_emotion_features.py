"""Causal EOD ladder structure and tier-specific crowd feedback, without targets.

Prices and ladders use the same archived nominal-price convention as the fixed
baseline. They describe the declared surviving universe, not every market stock
or verified sector/fund identities.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import first_board_leader_model as base

KEYS=["signal_date","code"]
LADDER=["ladder_n2","ladder_n3","ladder_n4","ladder_n5plus",
    "ladder_contiguous_from2","ladder_missing_tier_fraction"]
FEEDBACK=["emotion_prev2_survival","emotion_prev3_survival",
    "emotion_prev_relay_mean_return","emotion_prev_relay_loss7_share"]
PATH=["emotion_prev2_survival_change3","emotion_failed_rate_change3","emotion_relay_return_change3"]
FIELDS=LADDER+FEEDBACK+PATH
DIAGNOSTIC_HIGH_SURVIVAL="emotion_prev4plus_survival_DIAGNOSTIC"

def reject_targets(frame):
    if any("LABEL_ONLY" in c or c=="label_end" for c in frame):
        raise ValueError("Future target columns are not admitted to emotion feature construction")

def load_states(inputs=base.INPUTS,cutoff=None):
    """Reconstruct only causal price states; do not create any future labels."""
    sessions=json.loads(inputs["calendar"].read_text("utf-8"))["sessions"]
    assert sessions==sorted(set(sessions))
    positions={day:index for index,day in enumerate(sessions)}
    columns=["date","code","open","high","low","close","pre_close","st","new_listing","status_verified"]
    bars=pd.concat([pd.read_csv(inputs[key],usecols=columns,dtype={"code":str},float_precision="round_trip")
        for key in ["bars","recovered"]],ignore_index=True)
    bars["code"]=bars.code.str.zfill(6)
    bars["date"]=bars.date.astype(str).str[:10]
    if cutoff is not None:bars=bars[bars.date.le(cutoff)].copy()
    if bars.duplicated(["date","code"]).any():
        duplicate=bars[bars.duplicated(["date","code"],keep=False)]
        conflict=duplicate.groupby(["date","code"])[["open","high","low","close","pre_close"]].nunique().max(axis=1)
        if conflict.gt(1).any():raise ValueError("Conflicting original price sources")
        bars=bars.drop_duplicates(["date","code"],keep="last")
    prices=["open","high","low","close","pre_close"]
    bars[prices]=bars[prices].apply(pd.to_numeric,errors="coerce")
    bars["p"]=bars.date.map(positions)
    bars=bars[bars.p.notna()].sort_values(["code","p"]).reset_index(drop=True)
    bars["p"]=bars.p.astype(int)
    bars["eligible"]=base.true_values(bars.status_verified)&~base.true_values(bars.st)&~base.true_values(bars.new_listing)&bars.code.str.startswith(("000","001","002","003","600","601","603","605"))
    valid=bars[prices].gt(0).all(axis=1)&bars.high.ge(bars[["open","close","low"]].max(axis=1))&bars.low.le(bars[["open","close"]].min(axis=1))
    bars["valid_price"]=valid
    cap=base.limit_cents(bars.pre_close)
    floor=base.limit_cents(bars.pre_close,False)
    cents=(bars[["open","high","low","close"]]*100).round()
    bars["regular_price"]=valid&cents.high.le(cap)&cents.low.ge(floor)
    bars["limit"]=bars.eligible&bars.regular_price&cents.close.eq(cap)
    bars["failed"]=bars.eligible&bars.regular_price&cents.high.eq(cap)&~bars.limit
    group=bars.groupby("code",sort=False)
    adjacent=bars.p.sub(group.p.shift()).eq(1)
    previous_limit=group.limit.shift().fillna(False).astype(bool)
    link=bars.limit&previous_limit&adjacent
    run=(~link).groupby(bars.code).cumsum()
    bars["height"]=bars.limit.astype(int).groupby([bars.code,run]).cumsum()
    bars["day_return"]=bars.close/bars.pre_close-1
    return bars,sessions

def market_context(states,sessions):
    reject_targets(states)
    if states.duplicated(["p","code"]).any():raise ValueError("Duplicate stock/session")
    index=pd.Index(range(int(states.p.min()),int(states.p.max())+1),name="p")
    output=pd.DataFrame(index=index)
    observed=states[states.eligible&states.valid_price].groupby("p").size().reindex(index,fill_value=0).gt(0)
    eligible=states[states.eligible&states.valid_price]
    positive=eligible[eligible.limit]
    for field,mask in zip(LADDER[:4],[positive.height.eq(2),positive.height.eq(3),positive.height.eq(4),positive.height.ge(5)]):
        output[field]=positive[mask].groupby("p").size().reindex(index,fill_value=0).astype(float).where(observed)
    heights={p:set(group.height.astype(int)) for p,group in positive[positive.height.ge(2)].groupby("p")}
    depth=[];missing=[]
    for p in index:
        if not observed.loc[p]:depth.append(np.nan);missing.append(np.nan);continue
        present=heights.get(p,set())
        if not present:depth.append(0.);missing.append(0.);continue
        top=max(present)
        step=2
        while step in present:step+=1
        depth.append(float(step-2))
        missing.append(float(sum(h not in present for h in range(2,top+1))/(top-1)))
    output[LADDER[4]]=depth;output[LADDER[5]]=missing
    limit_count=positive.groupby("p").size().reindex(index,fill_value=0)
    failed_count=eligible[eligible.failed].groupby("p").size().reindex(index,fill_value=0)
    failed_rate=(failed_count/(failed_count+limit_count).replace(0,np.nan)).where(observed)
    # Membership is the entire previous OFFICIAL session's eligible relay cohort.
    # Its S-close observation is known at the feature date S, not a forecast of S+1.
    prior=states[states.eligible&states.regular_price&states.limit&states.height.ge(2)][["p","code","height"]].copy()
    prior["p"]=prior.p+1
    prior=prior[prior.p.isin(index)]
    current=states[["p","code","eligible","valid_price","regular_price","limit","day_return"]]
    joined=prior.merge(current,on=["p","code"],how="left",validate="one_to_one",indicator=True)
    joined["known"]=joined._merge.eq("both")&joined.eligible.eq(True)&joined.valid_price.eq(True)&joined.regular_price.eq(True)
    for mask,field in [(joined.height.eq(2),FEEDBACK[0]),(joined.height.eq(3),FEEDBACK[1]),
        (joined.height.ge(4),DIAGNOSTIC_HIGH_SURVIVAL)]:
        cohort=joined[mask]
        total=cohort.groupby("p").size().reindex(index,fill_value=0)
        known=cohort.groupby("p").known.sum().reindex(index,fill_value=0)
        hits=cohort[cohort.known&cohort.limit.eq(True)].groupby("p").size().reindex(index,fill_value=0)
        output[field]=(hits/total.replace(0,np.nan)).where(known.eq(total)&total.gt(0))
        output[field+"_members_DIAGNOSTIC"]=total
        output[field+"_unknown_DIAGNOSTIC"]=total-known
    total=joined.groupby("p").size().reindex(index,fill_value=0)
    known=joined.groupby("p").known.sum().reindex(index,fill_value=0)
    complete=known.eq(total)&total.gt(0)
    output[FEEDBACK[2]]=joined[joined.known].groupby("p").day_return.mean().reindex(index).where(complete)
    losses=joined[joined.known&joined.day_return.le(-.07)].groupby("p").size().reindex(index,fill_value=0)
    output[FEEDBACK[3]]=(losses/total.replace(0,np.nan)).where(complete)
    output["emotion_prev_relay_members_DIAGNOSTIC"]=total
    output["emotion_prev_relay_unknown_DIAGNOSTIC"]=total-known
    for field,series in zip(PATH,[output[FEEDBACK[0]],failed_rate,output[FEEDBACK[2]]]):
        output[field]=series-series.shift().rolling(3,min_periods=3).mean()
    output["emotion_failed_rate_DIAGNOSTIC"]=failed_rate
    output["signal_date"]=[sessions[p] for p in index]
    return output.reset_index()

def attach(candidates,context):
    reject_targets(candidates)
    if candidates.duplicated(KEYS).any() or context.signal_date.duplicated().any():
        raise ValueError("Duplicate candidates or market date")
    output=candidates[KEYS].merge(context.drop(columns="p"),on="signal_date",how="left",sort=False,validate="many_to_one")
    assert len(output)==len(candidates)
    return output
