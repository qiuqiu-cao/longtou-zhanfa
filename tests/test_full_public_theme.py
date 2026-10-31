import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
from scipy.special import logit

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'research/first_board_full_public_theme_20261030'
sys.path.insert(0,str(HERE))
import subset_features as f
import rank_offset as r
import experiment as e
spec=importlib.util.spec_from_file_location('old_fixture_for_full_source',ROOT/'tests/test_dynamic_theme_rank.py')
oldtest=importlib.util.module_from_spec(spec);spec.loader.exec_module(oldtest)


def test_no_observed_peers_and_no_anchor_are_unknown():
    x,p,s,days=oldtest.fixture()
    p=p[~p.code.isin(['000002','000004'])]
    row=f.build(x,p,s,days).iloc[0]
    assert pd.isna(row.ths_observed_subset_theme_peer_breadth)
    assert pd.isna(row.ths_observed_subset_prior_anchor_height_gap)
    assert pd.isna(row.ths_observed_subset_prior_anchor_relative_return)


def test_failed_prior_members_stay_and_missing_price_is_unknown():
    x,p,s,days=oldtest.fixture()
    a=f.build(x,p,s,days).iloc[0]
    assert a.ths_observed_subset_prior_anchor_height_gap==3
    assert a.ths_observed_subset_prior_anchor_relative_return==pytest.approx(-.1)
    s=s[~(s.code.eq('000003')&s.date.eq(days[5]))]
    b=f.build(x,p,s,days).iloc[0]
    assert pd.isna(b.ths_observed_subset_prior_cohort_relative_return)


def test_future_truncation_and_labels_rejected():
    x,p,s,days=oldtest.fixture()
    a=f.build(x,p,s,days)
    future=p.iloc[[0]].copy();future.signal_date=days[6];future['themes']=[['THS:FUTURE']]
    pd.testing.assert_frame_equal(a,f.build(x,pd.concat([p,future]),s,days))
    with pytest.raises(ValueError):f.build(x.assign(reach4=1),p,s,days)


def test_uniform_offset_preserves_oldA_fallback_and_fit_units():
    x=pd.DataFrame({'signal_date':['2025-07-01','2025-10-01'],'code':['000001','000001'],'s1_score':[.4,.6]})
    saved=x[['signal_date','code']].assign(arm='A_DYNAMIC_THEME',residual=[0.,.2],
        research_score=[.4,logit(.6)+.2],score_status=['S1_FALLBACK','FIT'])
    a=e.reconstruct_offsets(x,saved)
    np.testing.assert_allclose(a.old_a_offset,[logit(.4),logit(.6)+.2],rtol=0,atol=1e-15)
    assert a.old_a_offset.iloc[0]!=a.old_a_saved_score.iloc[0]


def test_all_missing_features_fall_back_to_each_base_exactly():
    x=pd.DataFrame({'signal_date':['2025-07-01']*4,'code':['000004','000001','000002','000003'],
                    'fixed_offset':[1.,1.,-.5,2.],'v':[np.nan]*4})
    y=x[r.KEYS].assign(**{r.TARGET:[0,1,0,np.nan]})
    m=r.fit(x,y,['v'])
    score,resid,status=r.score(x,m)
    np.testing.assert_array_equal(score,x.fixed_offset)
    assert (resid==0).all() and m['status']=='FALLBACK'
    selected=r.choose(x.assign(research_score=score),'research_score')
    assert selected.code.tolist()==['000003','000001','000004']


def test_pairwise_training_keeps_unknowns_in_scaling_only():
    x=pd.DataFrame({'signal_date':['2025-07-01']*3,'code':['000001','000002','000003'],
                   'fixed_offset':[0.,0.,0.],'v':[0.,1.,9.]})
    y=x[r.KEYS].assign(**{r.TARGET:[0,1,np.nan]})
    m=r.fit(x,y,['v'])
    assert m['pairs']==1 and m['means'][0]==pytest.approx(10/3)
    assert m['coefficients'][0]>0
    missing=x.copy();missing['v']=np.nan
    assert (r.score(missing,m)[1]==0).all()


def test_known_tied_anchors_and_missing_times_remain_distinct():
    x,p,s,days=oldtest.fixture();p['first_seal_minutes']=np.nan
    z=f.build(x,p,s,days)
    assert z.ths_observed_subset_relative_reported_first_seal.isna().all()
    assert z.ths_observed_subset_theme_peer_breadth.notna().all()


def test_raw_source_missing_and_date_mismatch_fail_closed():
    sys.path.insert(0,str(ROOT/'research/first_board_public_theme_sources_20261030'))
    from public_theme_sources import parse_ths,validate_detail_link
    rows,info=parse_ths(ROOT/'research/first_board_public_theme_sources_20261030/raw/ths_003029.json')
    assert rows==[] and info['status']=='SOURCE_UNAVAILABLE'
    row={'code':'000001','signal_date':'2025-07-01','source_ref':'r','detail_link_id':4}
    v={'source_ref':'r','detail_link_id':4,'result':{'status':'fulfilled','value':'https://www.iwencai.com/x?f=1&w=000001+20250702+涨停原因'}}
    assert not validate_detail_link(row,v)


def test_failed_target_fetch_still_exposes_original_link_date():
    from source_pipeline import validate_observed_target
    row={'code':'000008','signal_date':'2025-07-01','source_ref':'r','detail_link_id':4}
    v={'source_ref':'r','detail_link_id':4,'result':{'status':'fulfilled','value':
       'Failed to fetch https://www.iwencai.com/stockpick/search?f=1&w=000008+20250701+%E6%B6%A8%E5%81%9C%E5%8E%9F%E5%9B%A0: Cache miss'}}
    assert validate_observed_target(row,v)
    v['result']['value']=v['result']['value'].replace('20250701','20250702')
    assert not validate_observed_target(row,v)


def test_destination_content_errors_are_false_and_unverified_body_unknown():
    from source_pipeline import target_content_status
    for value in ['Internal Error: cannot be opened','Failed to fetch x: Cache miss','Unable to resolve link']:
        assert target_content_status({'result':{'status':'fulfilled','value':value}}) is False
    assert target_content_status({'result':{'status':'rejected'}}) is False
    assert target_content_status({'result':{'status':'fulfilled','value':'Some response without verified target body'}}) is None
