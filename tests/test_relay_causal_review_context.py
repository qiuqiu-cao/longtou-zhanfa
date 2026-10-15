import numpy as np
import pandas as pd

import relay_causal_review_context as r
import relay_full_cohort_transition as t


def test_low_height_is_not_sufficient_to_call_a_break():
    node=r.classify_node(dict(height=2,closing_count=20),[])
    assert node["review_cycle_state"]=="LOW_HEIGHT_WITHOUT_BREAK_CONFIRMATION"
    assert node["break_from_cold_candidate"] is False


def test_break_needs_transition_and_multiple_confirming_changes():
    past=[dict(height=2,closing_count=20,full_promotion=.2)]*2
    weak=r.classify_node(dict(height=3,closing_count=20,full_promotion=.2,premium=.01),past)
    strong=r.classify_node(dict(height=3,closing_count=30,full_promotion=.4,premium=.01),past)
    assert not weak["break_from_cold_candidate"]
    assert strong["break_from_cold_candidate"]


def test_overlapping_ebb_and_divergence_are_mixed_and_ranks_need20_prior_dates():
    prior=dict(height=5,closing_count=50,full_promotion=.4,down_count=3)
    current=dict(height=4,closing_count=40,full_promotion=.2,down_count=6,premium=-.02,touch_failure=.5)
    out=r.classify_node(current,[prior])
    assert out["review_cycle_state"]=="MIXED_CONTEXT"
    assert set(out["review_cycle_flags"])=={"EBB_CONTEXT","DIVERGENCE_CONTEXT"}
    assert out["prior20_breadth_rank"] is None


def fixture():
    days=pd.bdate_range("2024-01-01",periods=8).strftime("%Y-%m-%d").tolist()
    c=pd.DataFrame([dict(date=days[4],code="000001",name="甲",boards=3,primary_role="CYCLE_CROSS_DRAGON",cycle_regime="ICE_BREAKING",parent_candidates=[],parent_anchor_code=None,daily_one_word=0),
        dict(date=days[5],code="000001",name="甲",boards=4,primary_role="SPACE_DRAGON",cycle_regime="DIVERGENCE_TEST",parent_candidates=[],parent_anchor_code=None,daily_one_word=0)])
    price=10.;bars=[];attempts=[]
    for i,day in enumerate(days):
        previous=price;price=round(price*1.1,2) if i>=2 else 10.
        bars.append(dict(date=day,code="000001",open=previous,high=price,low=previous,close=price,pre_close=previous,volume=100.))
        if i>=2:attempts.append(dict(signal_date=day,code="000001",name="甲",board_height=i-1,limit_price_10=price,high=price,close=price,pre_close=previous,pool_assignment="CLOSING_LIMIT_UP"))
    a=pd.DataFrame(attempts);a.attrs["failed_scope_complete_dates"]=days
    b=pd.DataFrame(bars);market=pd.DataFrame([dict(date=d,down_limit_count=0,yesterday_premium=.01,one_word_share=0) for d in days])
    return days,c,a,b,market


def test_future_and_outcome_values_do_not_affect_semantics():
    days,c,a,b,market=fixture();tr=t.build_transition_context(c,a,b,days)
    expected=r.build_causal(c,market,a,b,days,tr)
    altered=b.copy();altered.loc[altered.date.gt(days[5]),"close"]=999.
    c["target"]=999;c["score"]=-99
    actual=r.build_causal(c,market,a,altered,days,tr)
    pd.testing.assert_frame_equal(actual["rows"],expected["rows"])
    pd.testing.assert_frame_equal(actual["all_active_parent_relations"],expected["all_active_parent_relations"])


def test_daily_and_batch_semantic_outputs_match_and_unknown_shape_no_B_direction():
    days,c,a,b,market=fixture();tr=t.build_transition_context(c,a,b,days)
    full=r.build_causal(c,market,a,b,days,tr)["rows"]
    for day in c.date:
        one=r.build_causal(c[c.date.eq(day)],market,a,b,days,tr)["rows"]
        pd.testing.assert_frame_equal(one,full[full.date.eq(day)].reset_index(drop=True))
    b=b[~b.date.eq(days[5])]
    unknown=r.build_causal(c,market,a,b,days,tr)["rows"].iloc[-1]
    assert unknown.channel_review=="UNRESOLVED_IDENTITY"
    assert not unknown.b_direction_allowed
    assert unknown.review_primary_role=="UNKNOWN"


def test_adapter_uses_real_adjacent_reference_without_overwriting_conflicts():
    days,c,a,b,market=fixture()
    minimal=a.drop(columns=["limit_price_10","pre_close","high"])
    restored=r.normalize_attempts(minimal,b,days)
    assert restored.iloc[2].pre_close==b.iloc[3].close
    conflict=a.copy();conflict.loc[2,"pre_close"]=1.
    actual=r.normalize_attempts(conflict,b,days)
    assert actual.iloc[2].pre_close==1.
    assert actual.attrs==a.attrs


def test_one_stressed_session_is_only_survivor_but_two_across_states_can_continue():
    days,c,a,b,market=fixture()
    previous=c.iloc[:1].copy();previous["date"]=days[3];previous["boards"]=2
    c=pd.concat([previous,c],ignore_index=True)
    tr=t.build_transition_context(c,a,b,days)
    one=market.copy();one.loc[one.date.eq(days[5]),"yesterday_premium"]=-.01;one.loc[one.date.eq(days[5]),"down_limit_count"]=1
    single=r.build_causal(c,one,a,b,days,tr)["rows"].iloc[-1]
    assert single.divergence_survivor_only and not single.cross_state_continuation_candidate
    two=one.copy();two.loc[two.date.eq(days[4]),"yesterday_premium"]=-.01
    two.loc[two.date.eq(days[4]),"down_limit_count"]=1;two.loc[two.date.eq(days[5]),"down_limit_count"]=2
    repeated=r.build_causal(c,two,a,b,days,tr)["rows"].iloc[-1]
    assert repeated.cross_state_continuation_candidate


def test_explicit_suspension_overrides_stale_positive_volume_and_price():
    days,c,a,b,market=fixture();tr=t.build_transition_context(c,a,b,days)
    b["is_suspended"]=False;b.loc[b.date.eq(days[5]),"is_suspended"]=True
    row=r.build_causal(c,market,a,b,days,tr)["rows"].iloc[-1]
    assert row.channel_review=="UNRESOLVED_IDENTITY" and not row.b_direction_allowed
    assert row.review_primary_role=="UNKNOWN"
