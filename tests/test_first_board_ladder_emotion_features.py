import numpy as np
import pandas as pd
import pytest
import first_board_ladder_emotion_features as feature

SESSIONS=["2024-01-02","2024-01-03","2024-01-04","2024-01-05","2024-01-08"]

def states(items):
    return pd.DataFrame([dict(p=p,code=code,height=height,limit=is_limit,day_return=ret,
        eligible=True,regular_price=True,valid_price=True,failed=False)
        for p,code,height,is_limit,ret in items])

def test_equal_top_and_relay_total_can_have_different_ladder_depth():
    complete=states([(0,str(i),h,True,.1) for i,h in enumerate([2,3,4,5])])
    fragmented=states([(0,str(i),h,True,.1) for i,h in enumerate([2,2,5,5])])
    assert complete.height.max()==fragmented.height.max()==5
    assert len(complete)==len(fragmented)==4
    a=feature.market_context(complete,SESSIONS).iloc[0]
    b=feature.market_context(fragmented,SESSIONS).iloc[0]
    assert a.ladder_contiguous_from2==4 and b.ladder_contiguous_from2==1
    assert a.ladder_missing_tier_fraction==0 and b.ladder_missing_tier_fraction==.5

def test_missing_current_member_remains_unknown_in_full_previous_cohort():
    source=states([(0,"A",2,True,.1),(0,"B",2,True,.1),(1,"A",3,True,.1)])
    last=feature.market_context(source,SESSIONS).iloc[-1]
    assert last.emotion_prev2_survival_members_DIAGNOSTIC==2
    assert last.emotion_prev2_survival_unknown_DIAGNOSTIC==1
    assert pd.isna(last.emotion_prev2_survival)
    assert pd.isna(last.emotion_prev_relay_mean_return)
    assert pd.isna(last.emotion_prev3_survival)

def test_observed_loss_and_survival_use_today_feedback_not_future():
    source=states([(0,"A",2,True,.1),(0,"B",2,True,.1),
        (1,"A",3,True,.1),(1,"B",0,False,-.09)])
    current=feature.market_context(source,SESSIONS).iloc[-1]
    assert current.emotion_prev2_survival==.5
    assert current.emotion_prev_relay_mean_return==pytest.approx(.005)
    assert current.emotion_prev_relay_loss7_share==.5
    future=pd.concat([source,states([(2,"A",0,False,-.1),(2,"B",0,False,-.1)])],ignore_index=True)
    pd.testing.assert_frame_equal(feature.market_context(source,SESSIONS),
        feature.market_context(future,SESSIONS).iloc[:2].reset_index(drop=True))

def test_ineligible_current_observation_is_not_counted_as_a_loss():
    source=states([(0,"A",2,True,.1),(1,"A",0,False,-.1),(1,"B",1,True,.1)])
    source.loc[(source.p.eq(1)&source.code.eq("A")),"eligible"]=False
    last=feature.market_context(source,SESSIONS).iloc[-1]
    assert last.emotion_prev2_survival_unknown_DIAGNOSTIC==1
    assert pd.isna(last.emotion_prev_relay_loss7_share)

def test_targets_rejected_and_candidate_missing_date_never_dropped():
    source=states([(0,"A",2,True,.1)])
    with pytest.raises(ValueError,match="target"):
        feature.market_context(source.assign(promotion_next_LABEL_ONLY=1),SESSIONS)
    context=feature.market_context(source,SESSIONS)
    candidates=pd.DataFrame(dict(signal_date=SESSIONS[:2],code=["A","A"]))
    result=feature.attach(candidates,context)
    assert len(result)==2 and np.isnan(result.ladder_n2.iloc[1])
