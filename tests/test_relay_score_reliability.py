import numpy as np
import pandas as pd
import pytest
import relay_score_reliability as r


def data():
    return pd.DataFrame(dict(arm=["A"]*3,daily_one_word=[0]*3,score=[0.,1.,.8],
        target=[0.,1.,np.nan],label_valid=[True,True,False]))


def test_unknown_result_never_turns_into_false_and_endpoints_are_kept():
    summary,bins=r.reliability(data())
    assert summary.known_scored.iloc[0]==2 and summary.unknown_or_unscored.iloc[0]==1
    assert summary.Brier.iloc[0]==0 and bins.known_scored.sum()==2
    assert bins.iloc[0].known_scored==bins.iloc[9].known_scored==1


def test_pairwise_preference_never_gets_event_probability_metrics():
    f=data();f["arm"]="PAIRWISE_DAY_RANK"
    s,b=r.reliability(f)
    assert s.applicability.iloc[0]=="NO_ABSOLUTE_EVENT_HEAD" and s.Brier.iloc[0] is None and b.empty


def test_out_of_scale_score_is_rejected_instead_of_silently_clipped():
    f=data();f.loc[0,"score"]=7
    with pytest.raises(ValueError):r.reliability(f)


def test_reliability_does_not_change_inputs_or_certify_probabilities():
    f=data();before=f.copy(deep=True);s,_=r.reliability(f)
    pd.testing.assert_frame_equal(f,before)
    assert not s.event_probability_validated.iloc[0]
