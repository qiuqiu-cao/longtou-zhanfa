"""Corrected interpretation layer; never changes a frozen model's inputs.

Reported heights, observed 10% price paths, weak names, explicit narratives,
and independent role truth remain separate. All clocks belong to the caller's
bound source receipts; an official-date reconstruction is not original access.
"""
from collections import defaultdict
from functools import lru_cache
import numpy as np
import pandas as pd

import short_term_thinking_dataset_builder as b

VERSION="causal-review-scope-1"
POLICY=dict(active_anchor_sessions=5,anchor_reported_height_min=4,launch_lag_match=[2,6],
    cold_promotion_threshold=.25,divergence_touch_fail_threshold=.30,
    climax_prior_rank=.80,climax_one_price_share=.30,cross_state_window=5,
    cross_state_min_stress_sessions=2,volume_expansion_proxy=2,
    price_scope="EXACT_10PCT_PRICE_PATH_IN_BOUND_ARCHIVE_NOT_EXCHANGE_STATUS_CERTIFIED",
    roles="NONEXCLUSIVE_EVIDENCE_CANDIDATES_NOT_IDENTITY_GOLD",
    role_priority=["OBSERVED_SPACE_HEIGHT_CANDIDATE","CROSS_STATE_CONTINUATION_CANDIDATE",
        "BREAK_FROM_COLD_CANDIDATE","AUTHOR_SUPPORTED_SUPPLEMENT_CANDIDATE",
        "RELATIVE_POSITION_USURP_CANDIDATE","HIGH_NONWORD_PRICE_PATH_CANDIDATE",
        "RELAY_CONTENDER","UNKNOWN"],
    old_frozen_columns="RETAINED_UNMODIFIED_AND_LABELLED_LEGACY",
    model_training=False,automatic_author_judgement=False)


def finite(x):
    try:return float(x) if np.isfinite(float(x)) else None
    except (ValueError,TypeError):return None


def normalize_attempts(attempts,bars,sessions):
    """Fill absent source adapter fields from exact same/adjacent dated bars.

    Existing nonmissing facts always win, even when conflicting. No supplied
    source scope is created here and no classification is changed.
    """
    a=attempts.copy();a["date"]=a.get("date",pd.Series(index=a.index,dtype=object)).fillna(a.signal_date)
    relevant=set(a.date);indices={d:i for i,d in enumerate(sessions)}
    relevant|={sessions[indices[d]-1] for d in a.date if d in indices and indices[d]}
    fields=[x for x in ["date","code","high","close","pre_close"] if x in bars]
    data=bars[bars.date.isin(relevant)][fields]
    current=data.rename(columns={x:"daily_"+x for x in fields if x not in ["date","code"]})
    a=a.merge(current,on=["date","code"],how="left",validate="one_to_one")
    a["previous_date"]=[sessions[indices[d]-1] if d in indices and indices[d] else None for d in a.date]
    previous=data[["date","code","close"]].rename(columns={"date":"previous_date","close":"adjacent_daily_close"})
    a=a.merge(previous,on=["previous_date","code"],how="left",validate="many_to_one")
    for field in ["high","close","pre_close","limit_price_10"]:
        if field not in a:a[field]=np.nan
    for field in ["high","close"]:
        a[field]=a[field].combine_first(a.get("daily_"+field,pd.Series(np.nan,index=a.index)))
    a["pre_close"]=a.pre_close.combine_first(a.adjacent_daily_close)
    if "daily_pre_close" in a:a["pre_close"]=a.pre_close.combine_first(a.daily_pre_close)
    inferred=[b.cap_cents(x)/100 if finite(x) is not None and x>0 else np.nan for x in a.pre_close]
    a["limit_price_10"]=a.limit_price_10.combine_first(pd.Series(inferred,index=a.index))
    a["adapter_policy"]="ONLY_ABSENT_FIELDS_BOUND_SAME_AND_ADJACENT_DAILY_PRICES_NO_RECLASSIFICATION"
    a.attrs=attempts.attrs.copy()
    return a


def classify_node(current,past):
    """Independent flags with explicit overlap/missing states, no forced cycle."""
    def get(row,key):return finite(row.get(key))
    prior=past[-1] if past else {}
    height=get(current,"height");oldheight=get(prior,"height")
    promotion=get(current,"full_promotion");oldpromotion=get(prior,"full_promotion")
    breadth=get(current,"closing_count");oldbreadth=get(prior,"closing_count")
    premium=get(current,"premium");down=get(current,"down_count");olddown=get(prior,"down_count")
    low=height is not None and height<=3
    cold=len(past)>=2 and all(get(x,"full_promotion") is not None and x["full_promotion"]<.25 for x in past[-2:])
    prior_low=oldheight is not None and oldheight<=3
    height_up=height is not None and oldheight is not None and height>oldheight
    promotion_up=promotion is not None and oldpromotion is not None and promotion>oldpromotion
    breadth_up=breadth is not None and oldbreadth is not None and breadth>oldbreadth
    breaks=(cold or prior_low) and height is not None and height>=3 and sum([height_up,promotion_up,breadth_up])>=2 and premium is not None and premium>0
    failure=get(current,"touch_failure")
    divergence=(failure is not None and failure>.30) or (breadth is not None and oldbreadth is not None and breadth<oldbreadth and promotion is not None and oldpromotion is not None and promotion<oldpromotion)
    ebb=premium is not None and premium<0 and ((height is not None and oldheight is not None and height<oldheight) or (down is not None and olddown is not None and down>olddown))
    ranks={}
    for key,value in [("closing_count",breadth),("full_promotion",promotion)]:
        values=[get(x,key) for x in past[-20:]];values=[x for x in values if x is not None]
        ranks[key]=sum(x<=value for x in values)/len(values) if value is not None and len(values)==20 else None
    one=get(current,"one_price_share_proxy")
    climax=all(ranks[k] is not None and ranks[k]>=.8 for k in ranks) and one is not None and one>=.3
    flags=[label for label,on in [("ICE_BREAK_CANDIDATE",breaks),("DIVERGENCE_CONTEXT",divergence),("CLIMAX_CONTEXT",climax),("EBB_CONTEXT",ebb)] if on]
    complete=all(x is not None for x in [height,promotion,breadth,premium,down,failure])
    state="MIXED_CONTEXT" if len(flags)>1 else flags[0] if flags else "LOW_HEIGHT_WITHOUT_BREAK_CONFIRMATION" if low else "BALANCED_CONTEXT" if complete else "UNKNOWN_INCOMPLETE_CONTEXT"
    return dict(review_cycle_state=state,review_cycle_flags=flags,low_height_observation=low,
        break_from_cold_candidate=bool(breaks),cycle_context_complete=complete,
        cycle_missing_fields=[k for k in ["height","full_promotion","closing_count","premium","down_count","touch_failure"] if get(current,k) is None],
        prior20_breadth_rank=ranks["closing_count"],prior20_promotion_rank=ranks["full_promotion"])


def build_causal(candidates,market,attempts,bars,sessions,transitions,*,author_parent_edges=()):
    """One historical/live function, with explicit signal-field allowlists."""
    sessions=list(sessions);indices={d:i for i,d in enumerate(sessions)}
    if sessions!=sorted(set(sessions)) or candidates.duplicated(["date","code"]).any():raise ValueError("unique official dates/candidates required")
    columns=[k for k in ["date","code","name","boards","primary_role","cycle_regime","parent_candidates","parent_anchor_code","daily_one_word","same_height_amount_rank","same_height_turnover_rank"] if k in candidates]
    c=candidates[columns].copy();c["code"]=c.code.astype(str).str.zfill(6)
    if not set(c.date)<=set(indices):raise ValueError("candidate outside official calendar")
    last=c.date.max();first=attempts.signal_date.min()
    a=attempts[attempts.signal_date.le(last)].copy()
    scope=set(transitions["summary"]["complete_scope_dates"])
    pools=defaultdict(dict)
    for row in a[a.pool_assignment.eq("CLOSING_LIMIT_UP")].to_dict("records"):
        pools[row["signal_date"]][row["code"]]=row
    relevant_days={d for d in sessions if first<=d<=last}
    price_columns=[k for k in ["date","code","open","high","low","close","pre_close","volume","turnover"] if k in bars]
    # Compact tuples rather than retaining millions of mutable raw-row dictionaries.
    raw=bars[bars.date.isin(relevant_days)][price_columns]
    keys=price_columns[2:]
    prices={(row[0],row[1]):row[2:] for row in raw.itertuples(index=False,name=None)}
    @lru_cache(maxsize=None)
    def daily(day,code):
        values=prices.get((day,code))
        if values is None:return dict(status="UNKNOWN_MISSING_DAILY",closed=None,one_price=None)
        row=dict(zip(keys,values));pos=indices[day]
        prev=prices.get((sessions[pos-1],code)) if pos else None
        previous=dict(zip(keys,prev)) if prev else pools.get(sessions[pos-1],{}).get(code) if pos else None
        pc=finite(previous.get("close")) if previous else None
        shape=b.price_state(row)
        if pc is None or pc<=0 or not shape["observed"]:
            return dict(status="UNKNOWN_ADJACENT_REFERENCE_OR_SHAPE",closed=None,one_price=None)
        ref=finite(row.get("pre_close"))
        if ref is not None and abs(ref-pc)>.011:return dict(status="UNKNOWN_REFERENCE_CONFLICT",closed=None,one_price=None)
        vol=finite(row.get("volume"))
        if vol is None or vol<=0:return dict(status="UNKNOWN_NONTRADING_OR_VOLUME",closed=None,one_price=None)
        cap=b.cap_cents(pc)
        if b.cents(row["high"])>cap:return dict(status="UNKNOWN_10PCT_SCOPE_CONFLICT",closed=None,one_price=None)
        closed=b.cents(row["close"])==cap
        return dict(status="OBSERVED_10PCT_PRICE_PATH_PROXY",closed=closed,one_price=shape["one_word"] if closed else None,
            price_return=row["close"]/pc-1,volume_ratio=vol/previous["volume"] if previous and finite(previous.get("volume")) is not None and previous["volume"]>0 else None,
            previous_date=sessions[pos-1],relative_volume_reference="EXACT_PREVIOUS_OFFICIAL_SESSION_VOLUME_NOT_TURNOVER")
    @lru_cache(maxsize=None)
    def streak(day,code,height):
        pos=indices[day];start=pos-height+1
        if start<0:return dict(status="UNKNOWN_INSUFFICIENT_OFFICIAL_HISTORY",start=None)
        states=[daily(d,code) for d in sessions[start:pos+1]]
        if any(s["closed"] is False for s in states):return dict(status="REPORTED_HEIGHT_PRICE_PATH_CONFLICT",start=None)
        if any(s["closed"] is None for s in states):return dict(status="UNKNOWN_INCOMPLETE_CONTINUOUS_PATH",start=None)
        return dict(status="OBSERVED_CONTINUOUS_10PCT_PRICE_PATH_PROXY",start=sessions[start],nonprice_boards=sum(s["one_price"]==0 for s in states))
    markets={r["date"]:r for r in market[[k for k in ["date","down_limit_count","yesterday_premium","one_word_share"] if k in market]].to_dict("records")}
    groups=transitions["groups"];members=transitions["members"]
    perday={d:f for d,f in groups[groups.target_height.ge(2)].groupby("date")}
    days=[];daymap={};past=[];last_high={};anchors=[];rows=[]
    original={d:f.to_dict("records") for d,f in c.groupby("date")}
    known_edges={(x["date"],x["code"],x["parent_code"]):x for x in author_parent_edges}
    for day in [d for d in sessions if first<=d<=last]:
        pos=indices[day];previous=sessions[pos-1] if pos else None
        pool=pools.get(day,{});mr=markets.get(day,{})
        group=perday.get(day,groups.iloc[:0]);denom=group.full_cohort_previous_count.sum() if len(group) else 0
        all_known=bool(len(group) and group[group.full_cohort_observed_previous_count.gt(0)].full_cohort_status.eq("COMPLETE_PROXY_ELIGIBLE_COHORT").all())
        fullrate=group.full_cohort_promoted_count.sum()/denom if all_known and denom>0 else None
        touched=a[a.signal_date.eq(day)]
        touchratio=float(touched.pool_assignment.eq("FAILED_BOARD_OBSERVATION").mean()) if day in scope and len(touched) else None
        height=max([int(x["board_height"]) for x in pool.values()],default=None) if day in scope else None
        data=dict(date=day,height=height,closing_count=len(pool) if day in scope else None,full_promotion=fullrate,touch_failure=touchratio,
            premium=mr.get("yesterday_premium"),down_count=mr.get("down_limit_count"),one_price_share_proxy=mr.get("one_word_share"))
        data.update(classify_node(data,past));days.append(data);daymap[day]=data;past.append(data)
        active={code:r for code,r in last_high.items() if 0<pos-indices[r["signal_date"]]<=5}
        for row in original.get(day,[]):
            code=row["code"];n=int(row["boards"]);path=streak(day,code,n);shape=daily(day,code)
            eligible=path["status"]=="OBSERVED_CONTINUOUS_10PCT_PRICE_PATH_PROXY"
            all_parent_codes=sorted(set(active)-{code});legacy=set(row.get("parent_candidates") or [])
            accepted=[]
            for parent in all_parent_codes:
                anchor=active[parent];anchor_path=streak(anchor["signal_date"],parent,int(anchor["board_height"]))
                state=daily(day,parent);prevstate=daily(previous,parent) if previous else dict(closed=None)
                if state["closed"] is None:life="UNKNOWN_CURRENT_ANCHOR_PRICE"
                elif state["closed"]:
                    life="UPPER_PRICE_CONTINUATION" if prevstate["closed"] else "PRICE_RESTART_AFTER_BREAK" if prevstate["closed"] is False else "UNKNOWN_PREVIOUS_ANCHOR_STATE"
                    if prevstate["closed"] and finite(state.get("volume_ratio")) is not None and state["volume_ratio"]>2:life="HEAVY_VOLUME_CONTINUATION_PROXY"
                elif prevstate["closed"]:life="FIRST_PRICE_BREAK_OBSERVED"
                elif prevstate["closed"] is False and state.get("price_return",0)<=-.07:life="POST_BREAK_NEGATIVE_FEEDBACK_PROXY"
                else:life="NONCLOSING_ANCHOR_WITHOUT_NEW_BREAK"
                lag=indices[path["start"]]-indices[anchor_path["start"]] if path["start"] and anchor_path["start"] else None
                edge=known_edges.get((day,code,parent))
                author_supported=bool(edge and edge.get("accepted") is True and edge.get("hypothesis_public_date",day)<=day)
                if author_supported and lag is not None and 2<=lag<=6 and life in ["FIRST_PRICE_BREAK_OBSERVED","HEAVY_VOLUME_CONTINUATION_PROXY"]:accepted.append(parent)
                name=anchor.get("name","");shared,phonetic=b.name_relation(row.get("name",""),name)
                anchors.append(dict(date=day,code=code,parent_code=parent,parent_name=name,
                    active_reported_date=anchor["signal_date"],active_reported_height=anchor["board_height"],anchor_path_status=anchor_path["status"],
                    anchor_current_streak_start_observed=anchor_path["start"],child_current_streak_start_observed=path["start"],
                    observed_launch_lag=lag,in_legacy_lag_match_subset=parent in legacy,anchor_current_state=life,
                    shared_characters=shared,phonetic_families=phonetic,name_relation_status="WEAK_CANDIDATE_NOT_SUFFICIENT_FOR_PARENT_IDENTITY",
                    author_supported_hypothesis=author_supported,industry_relation="UNKNOWN_UNLESS_SEPARATELY_BOUND",funds_causality="UNKNOWN",feature_cutoff=day))
            trail=[daymap[d] for d in sessions[max(0,pos-min(n,5)+1):pos+1] if d in daymap]
            stressed=[d for d in trail if any(x in d["review_cycle_flags"] for x in ["DIVERGENCE_CONTEXT","EBB_CONTEXT"])]
            resolved=[d["review_cycle_state"] for d in trail if d["review_cycle_state"]!="UNKNOWN_INCOMPLETE_CONTEXT"]
            cross=eligible and n>=3 and len(stressed)>=2 and len(set(resolved))>=2
            weak_survivor=eligible and n>=3 and bool(stressed) and not cross
            roles=[]
            if eligible and height is not None and n==height:roles.append("OBSERVED_SPACE_HEIGHT_CANDIDATE")
            if cross:roles.append("CROSS_STATE_CONTINUATION_CANDIDATE")
            if eligible and n>=3 and data["break_from_cold_candidate"]:roles.append("BREAK_FROM_COLD_CANDIDATE")
            if eligible and accepted:roles.append("AUTHOR_SUPPORTED_SUPPLEMENT_CANDIDATE")
            # Position change is measured against the actual previous official cohort.
            prior_rows=original.get(previous,[]);oldself=next((x for x in prior_rows if x["code"]==code),None)
            prior_rank=finite(oldself.get("same_height_amount_rank")) if oldself else None
            rank=finite(row.get("same_height_amount_rank"))
            peer_exit=members[(members.date.eq(day))&members.target_height.eq(n)] if len(members) else members
            competitor_failed=bool(len(peer_exit) and peer_exit.transition_state.isin(["FAILED_TOUCHED_CAP","FAILED_NO_CAP_TOUCH"]).any())
            usurp=eligible and shape["one_price"]==0 and competitor_failed and rank is not None and prior_rank is not None and rank>prior_rank
            if usurp:roles.append("RELATIVE_POSITION_USURP_CANDIDATE")
            if eligible and n>=3 and shape["one_price"]==0 and path.get("nonprice_boards",0)>=2:roles.append("HIGH_NONWORD_PRICE_PATH_CANDIDATE")
            if not roles:roles=["RELAY_CONTENDER" if eligible else "UNKNOWN"]
            earlier=[d for d,p in pools.items() if d<day and code in p and int(p[code]["board_height"])==1 and (not path["start"] or d<path["start"])]
            rows.append(dict(date=day,code=code,name=row.get("name"),reported_height=n,legacy_primary_role=row.get("primary_role"),legacy_cycle_regime=row.get("cycle_regime"),
                **{k:data[k] for k in ["review_cycle_state","review_cycle_flags","low_height_observation","break_from_cold_candidate","cycle_context_complete"]},
                current_price_path_status=path["status"],current_streak_start_observed=path["start"],latest_earlier_reported_first_board=max(earlier) if earlier else None,
                current_price_shape=shape["one_price"],channel_review="A_ONE_PRICE_ANCHOR_PROXY" if shape["one_price"]==1 else "B_NONONE_PRICE_RESEARCH_PROXY" if shape["one_price"]==0 else "UNRESOLVED_IDENTITY",
                b_direction_allowed=shape["one_price"]==0,cross_state_continuation_candidate=cross,divergence_survivor_only=weak_survivor,
                complete_active_parent_count=len(all_parent_codes),complete_active_parent_codes=all_parent_codes,legacy_lag_parent_subset_count=len(legacy),
                independent_role_identity_truth=None,review_roles=roles,review_primary_role=next(x for x in POLICY["role_priority"] if x in roles),
                support="Scope-qualified price path, explicit node flags and complete/partial competing cohort are separate facts",
                counterevidence="No independent identity gold; price path and volume cannot prove true absorption, theme leadership or funds causality",
                cancel_condition="Invalidate this structural hypothesis if adjacent price/reference or eligibility conflicts, required peers/anchors are missing, or a separately bound narrative is contradicted",
                theme_core_status="UNKNOWN_NO_COMPLETE_TYPED_ACTIVE_THEME_EVIDENCE",feature_cutoff=day))
        for code,anchor in pool.items():
            if int(anchor["board_height"])>=4:last_high[code]=anchor
    return dict(rows=pd.DataFrame(rows),all_active_parent_relations=pd.DataFrame(anchors),cycle_days=pd.DataFrame(days),policy=POLICY,version=VERSION)
