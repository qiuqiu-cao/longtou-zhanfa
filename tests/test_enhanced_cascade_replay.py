import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

PATH=Path(__file__).resolve().parents[1]/"research/first_board_enhanced_cascade_review_20261030/replay_enhanced.py"
SPEC=importlib.util.spec_from_file_location("enhanced_replay",PATH)
M=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)

def fixture():
    d=pd.DataFrame({"signal_date":["D1"]*4+["D2"]*4,"code":["001","002","003","004"]*2,
        "s1_score":[.8,.7,.6,.5]*2,"height_score":[.1,.2,.3,.4]*2,
        "reported_close_seal_log_amount_ratio":[np.nan,.2,.3,.4]*2,
        "reported_close_seal_log_floatcap_ratio":[np.nan,.1,.2,.3]*2,
        "reported_first_seal_trading_minutes":[np.nan,10,20,30]*2,
        "reported_first_seal_observed_pool_rank":[np.nan,.2,.4,.6]*2,
        "breakout_20":[0.,.1,.2,.3]*2,"position_60":.5,"amount":1e8,"turnover":5.,
        "one_price":[1.,0.,0.,0.]*2})
    w={c:float(c=="s1_score_pct") for c in M.SCORES}
    return d,{"w1":w,"w2":w,"w3":w}

def test_nonlinear_feature_and_selection_are_label_blind():
    d,w=fixture()
    a=M.enhanced_features(d.assign(**{M.P2:1.}))
    b=M.enhanced_features(d.assign(**{M.P2:np.nan}))
    pd.testing.assert_frame_equal(a,b)
    chosen=M.select_features(a,w)
    assert len(chosen)==6 and not chosen.duplicated(M.KEYS).any()

def test_unknown_interaction_is_neutral_not_ranked_as_an_observed_product():
    d,_=fixture(); f=M.enhanced_features(d)
    assert f.loc[0,"seal_dominance_missing"]
    assert f.loc[0,"seal_dominance_pct"]==.5
    assert f.loc[0,"seal_rank_inv_pct"]==.5
    assert pd.isna(d.loc[0,"reported_first_seal_trading_minutes"])

def test_non_one_price_analysis_keeps_the_original_feature_ranks():
    d,w=fixture(); f=M.enhanced_features(d)
    take=M.select_features(f,w,d[d.one_price.eq(0)])
    assert len(take)==6 and set(take.code)=={"002","003","004"}
    assert take.loc[take.slot.eq(1),"score"].eq(.75).all()
