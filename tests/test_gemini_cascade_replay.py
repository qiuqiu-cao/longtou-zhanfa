import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np

PATH=Path(__file__).resolve().parents[1]/"research/first_board_math_20261030/review_gemini_cascade.py"
SPEC=importlib.util.spec_from_file_location("cascade_replay",PATH)
M=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)

def fixture():
    d=pd.DataFrame({"signal_date":["D1"]*4+["D2"]*4,"code":["001","002","003","004"]*2,
        "s1_score":[.8,.7,.6,.5]*2,"height_score":[.1,.2,.3,.4]*2,
        "reported_close_seal_log_amount_ratio":[np.nan,.2,.3,.4]*2,
        "reported_first_seal_trading_minutes":[np.nan,10,20,30]*2,
        "breakout_20":[0.,.1,.2,.3]*2,"position_60":.5,"amount":1e8,"turnover":5.})
    w={c+"_pct":float(c=="s1_score") for c in M.FIELDS}
    return d,{"w1":w,"w2":w,"w3":w}

def test_outcomes_cannot_change_pool_ranks_or_cascade_picks():
    d,w=fixture()
    a=M.cascade_select(d.assign(**{M.P2:1.}),w)
    b=M.cascade_select(d.assign(**{M.P2:np.nan}),w)
    pd.testing.assert_frame_equal(a,b)
    assert len(a)==6 and a.groupby("signal_date").size().eq(3).all()

def test_missing_seal_remains_explicit_and_does_not_rank_as_observed_zero():
    d,_=fixture(); f=M.percentiles(d)
    assert f.loc[0,"seal_ratio_missing"]
    assert f.loc[0,"seal_ratio_pct"]==.5
    assert pd.isna(d.loc[0,"reported_close_seal_log_amount_ratio"])

def test_later_all_unknown_date_still_gets_three_candidates():
    d,w=fixture(); picks=M.cascade_select(d,w)
    outcomes=d[M.KEYS].copy(); outcomes[M.P2]=[1.,0.,0.,0.]+[np.nan]*4; outcomes[M.P4]=np.nan
    result,_=M.evaluate(outcomes,picks,["D1","D2"])
    assert result["selected"]==6 and result["unknown"]==3
    assert result["known"]==3 and result["hits"]==1
