import copy
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'research/first_board_public_theme_sources_20261030'
spec=importlib.util.spec_from_file_location('public_theme_sources_under_test',HERE/'public_theme_sources.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_actual_timeline_has_dated_evolving_narrative():
    rows,status=m.parse_ths(HERE/'raw/ths_002696.json')
    assert status['event_count']==10
    early=next(x for x in rows if x['signal_date']=='2026-10-09')
    later=next(x for x in rows if x['signal_date']=='2026-10-14')
    assert not any('羊' in t for t in early['themes'])
    assert any('羊' in t for t in later['themes'])
    assert early['publication_time'] is None
    assert not early['actual_historical_available']


def test_detail_link_must_match_stock_date_and_source():
    row={'code':'002724','signal_date':'2026-10-16','source_ref':'r','detail_link_id':7}
    v={'source_ref':'r','detail_link_id':7,'result':{'status':'fulfilled','value':
       'https://www.iwencai.com/stockpick/search?f=1&w=002724+20261016+%E6%B6%A8%E5%81%9C%E5%8E%9F%E5%9B%A0'}}
    assert m.validate_detail_link(row,v)
    v['result']['value']=v['result']['value'].replace('20261016','20261019')
    assert not m.validate_detail_link(row,v)


def test_failed_page_is_unknown_not_empty_theme_evidence():
    rows,status=m.parse_ths(HERE/'raw/ths_003029.json')
    assert rows==[] and status['status']=='SOURCE_UNAVAILABLE'


def test_late_acquisition_cannot_become_original_signal():
    claim={'source_received_by':'2026-10-30 14:00:00 UTC','publication_time':'2026-10-12T17:28:51+08:00',
           'publication_timezone_verified':True}
    state=m.clock_gate(claim,'2026-10-16T15:00:00+08:00')
    assert state['published_by_cutoff'] is True
    assert state['acquired_by_cutoff'] is False
    assert state['actual_historical_available'] is False
    claim['publication_time']=None
    assert m.clock_gate(claim,'2026-11-02T15:00:00+08:00')['published_by_cutoff'] is None


def test_full_pool_join_preserves_scores_unknowns_and_dates():
    pool=pd.DataFrame({'signal_date':['2026-10-09','2026-10-16','2026-10-16'],
                       'code':['002696','002724','003029'],'s1_score':[0.2,0.8,0.6],'rank':[3,1,2]})
    row={'signal_date':'2026-10-16','code':'002724','themes':['AI照明'],'source_url':'u','source_sha256':'s',
         'event_date_binding':'DETAIL_LINK_DATE_VERIFIED'}
    group={'event_date':'2026-10-12','code':'002696','theme_id':'name','source_url':'g','source_sha256':'s'}
    out=m.attach_to_pool(pool,[row],[group])
    pd.testing.assert_frame_equal(pool,out[pool.columns],check_exact=True)
    assert out.public_theme_context_status.tolist()==['UNKNOWN','LATE_RESEARCH_CONTEXT','UNKNOWN']
    assert not out.public_context_actual_historical_available.any()
    assert out.loc[0,'public_reviewed_group_count']==0
    with pytest.raises(ValueError):
        m.attach_to_pool(pool.assign(promotion_next=[0,1,0]),[row],[])
    with pytest.raises(ValueError):
        m.attach_to_pool(pd.concat([pool,pool.iloc[[0]]]),[],[])


def test_unknown_date_link_cannot_reach_model_provider():
    rows,_=m.parse_ths(HERE/'raw/ths_002724.json')
    assert m.as_provider(rows).empty
    rows[0]['event_date_binding']='DETAIL_LINK_DATE_VERIFIED'
    p=m.as_provider(rows)
    assert len(p)==1 and pd.isna(p.first_seal_minutes.iloc[0])


def test_no_homophone_or_shared_character_auto_merge():
    assert m.split_themes('羊字辈+阳字辈+洋字辈')==['洋字辈','羊字辈','阳字辈']


def test_source_span_and_group_member_integrity():
    path=HERE/'raw/forum_1781454602.json'
    obj,text=m.read_archive(path)
    span=next(t for n,t in m.page_lines(text) if n==57)
    s={'source_sha256':m.sha(path),'source_span':span,'body_scope':'original_body',
       'members':[{'source_name':'九阳','name':'九阳股份','code':'002242'},
                  {'source_name':'三羊','name':None,'code':None}],
       'theme_id':'group','event_date':'2026-10-12','displayed_publication_time':'2026-10-12 17:28:51',
       'author':'author','relation':'unknown','reviewed_at':'2026-10-30T14:20:00+00:00',
       'complete_within_reviewed_group':False,'counterevidence':'limited claim'}
    rows=m.reviewed_memberships(s,path)
    assert len(rows)==2 and rows[1]['code'] is None
    bad=copy.deepcopy(s);bad['members'][0]['source_name']='海洋王'
    with pytest.raises(ValueError):m.reviewed_memberships(bad,path)
    bad=copy.deepcopy(s);bad['source_sha256']='wrong'
    with pytest.raises(ValueError):m.reviewed_memberships(bad,path)


def test_sparse_sample_zero_is_never_market_absence_input():
    frame=pd.DataFrame({'signal_date':['2025-10-21'],'code':['002748'],
        'closing_price_members':[63],'theme_bound_members':[2],'unbound_closing_members':[61],
        'theme_peer_breadth':[0.0],'prior_anchor_height_gap':[0.0]})
    out=m.guard_sparse_features(frame,['theme_peer_breadth','prior_anchor_height_gap'])
    assert out.theme_peer_breadth.isna().all() and out.prior_anchor_height_gap.isna().all()
    assert out.observed_subset_theme_peer_breadth.iloc[0]==0
    coverage=pd.DataFrame({'signal_date':['2025-10-21'],'current_source_universe_complete':[True],
                           'prior_source_universe_complete':[True]})
    # Even an acquisition claim does not override 61 unbound background stocks.
    out=m.guard_sparse_features(frame,['theme_peer_breadth','prior_anchor_height_gap'],coverage)
    assert out.theme_peer_breadth.isna().all()


def test_current_coverage_does_not_certify_previous_cohort():
    frame=pd.DataFrame({'signal_date':['2025-10-21'],'code':['002748'],
        'unbound_closing_members':[0],'theme_peer_breadth':[0.0],'prior_anchor_height_gap':[0.0]})
    coverage=pd.DataFrame({'signal_date':['2025-10-21'],'current_source_universe_complete':[True],
                           'prior_source_universe_complete':[False]})
    out=m.guard_sparse_features(frame,['theme_peer_breadth','prior_anchor_height_gap'],coverage)
    assert out.theme_peer_breadth.iloc[0]==0
    assert pd.isna(out.prior_anchor_height_gap.iloc[0])
