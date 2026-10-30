import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

P = Path(__file__).resolve().parents[1]/"research/first_board_theme_diffusion_review_20261030/industry_timing_proxy.py"
S = importlib.util.spec_from_file_location("industry_timing_review",P)
M = importlib.util.module_from_spec(S); S.loader.exec_module(M)


def sample(n=6):
    return pd.DataFrame(dict(signal_date=["2026-01-05"]*n,
        code=[f"{i:06d}" for i in range(1,n+1)],s1_score=np.linspace(.9,.1,n),
        legacy_industry_group_PROXY=[f"g{i}" for i in range(n)],
        reported_first_seal_trading_minutes=[10.]*n))


def test_future_labels_do_not_change_groups_or_rankings():
    x=sample(); x.loc[[3,4],M.GROUP]="shared"; x.loc[4,M.TIME]=15.
    a=M.timing_features(x.assign(promotion_next_LABEL_ONLY=1.))
    b=M.timing_features(x.assign(promotion_next_LABEL_ONLY=np.nan))
    pd.testing.assert_frame_equal(a,b)
    pd.testing.assert_frame_equal(M.choose_proxy_swap(a)[0],M.choose_proxy_swap(b)[0])
    assert len(a)==len(x)


def test_unknown_outcome_earliest_member_remains_in_the_feature_pool():
    x=sample(); x.loc[[3,4],M.GROUP]="shared"; x.loc[3,M.TIME]=1.;x.loc[4,M.TIME]=5.
    x["promotion_next_LABEL_ONLY"]=1.;x.loc[3,"promotion_next_LABEL_ONLY"]=np.nan
    f=M.timing_features(x)
    assert f.loc[3,"is_earliest_in_observed_group"]
    assert not f.loc[4,"is_earliest_in_observed_group"]
    take,swaps=M.choose_proxy_swap(f)
    assert swaps.iloc[0].promoted_code=="000004"
    assert len(take)==3


def test_unknown_group_is_not_negative_evidence_against_a_baseline_member():
    x=sample();x.loc[[0,1,2],M.GROUP]=pd.NA
    x.loc[[3,4],M.GROUP]="shared";x.loc[4,M.TIME]=15.
    f=M.timing_features(x); take,swaps=M.choose_proxy_swap(f)
    assert f.loc[[0,1,2],"is_earliest_in_observed_group"].isna().all()
    assert swaps.empty
    assert take.code.to_list()==["000001","000002","000003"]


def test_incomplete_group_timing_cannot_certify_the_observed_earliest():
    x=sample();x.loc[[3,4],M.GROUP]="shared";x.loc[4,M.TIME]=np.nan
    f=M.timing_features(x)
    assert f.loc[[3,4],"proxy_status"].eq("UNKNOWN_INCOMPLETE_GROUP_TIMES").all()
    assert f.loc[[3,4],"later_first_seals_within_window"].isna().all()
    assert M.choose_proxy_swap(f)[1].empty


def test_simultaneous_earliest_members_are_not_counted_as_later_response():
    x=sample();x.loc[[3,4,5],M.GROUP]="shared";x.loc[[3,4],M.TIME]=10.;x.loc[5,M.TIME]=40.
    f=M.timing_features(x)
    assert f.loc[[3,4],"earliest_tie_count"].eq(2).all()
    assert f.loc[[3,4],"later_first_seals_within_window"].eq(1).all()
    assert not f.dynamic_theme_membership_verified.any()
    assert not f.causal_leadership_verified.any()


def test_complete_challenger_pool_can_reach_rank_eleven():
    x=sample(12);x.loc[[10,11],M.GROUP]="shared";x.loc[11,M.TIME]=15.
    f=M.timing_features(x)
    full,swaps=M.choose_proxy_swap(f)
    restricted,old_swaps=M.choose_proxy_swap(f,8)
    assert swaps.iloc[0].promoted_original_rank==11
    assert "000011" in full.code.to_list()
    assert old_swaps.empty and "000011" not in restricted.code.to_list()
    assert len(full)==len(restricted)==3


def test_fixed_tie_rule_and_each_day_remain_stable_under_row_permutation():
    x=sample().assign(s1_score=.2)
    x=pd.concat([x,x.assign(signal_date="2026-01-06")],ignore_index=True)
    a=M.choose_proxy_swap(M.timing_features(x))[0]
    b=M.choose_proxy_swap(M.timing_features(x.sample(frac=1,random_state=2)))[0]
    pd.testing.assert_frame_equal(a,b)
    assert a.groupby("signal_date").size().eq(3).all()


def test_duplicate_and_invalid_times_do_not_silently_supply_evidence():
    x=sample();x.loc[3,M.TIME]=np.inf
    assert M.timing_features(x).loc[3,"proxy_status"]=="UNKNOWN_INCOMPLETE_GROUP_TIMES"
    with pytest.raises(ValueError):M.timing_features(pd.concat([x,x.iloc[[0]]]))
