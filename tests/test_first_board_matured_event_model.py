import numpy as np
import pandas as pd

import first_board_matured_event_model as memory
import first_second_stage_model as stage


def sample():
    sessions = pd.bdate_range("2025-01-02", periods=15).strftime("%Y-%m-%d").tolist()
    first = pd.DataFrame([dict(signal_date=sessions[p], code=code, p=p, label_end=sessions[p+1], promotion_next_LABEL_ONLY=label)
        for p,code,label in [(0,"000001",1.),(2,"000002",0.),(4,"000001",0.),(7,"000001",1.),(9,"000001",1.)]])
    paths=first[["signal_date","code","p"]].copy()
    paths["label_end"]=[sessions[p+5] for p in paths.p]
    paths["reach2_LABEL_ONLY"]=[1.,0.,0.,1.,1.]
    paths["reach3_LABEL_ONLY"]=[1.,0.,0.,0.,1.]
    paths["reach4_LABEL_ONLY"]=[0.,0.,0.,0.,1.]
    paths["max_height_5_LABEL_ONLY"]=[3.,1.,1.,2.,4.]
    return first,paths,sessions


def test_event_labels_are_gated_by_exact_maturity():
    first,paths,sessions=sample()
    rows=memory.build_features(first,paths,sessions)
    at4=rows[rows.p.eq(4)].iloc[0]
    assert at4.event_first_known252==1
    assert at4.event_previous_first_outcome==1
    assert at4.event_complete_paths252==0
    at7=rows[rows.p.eq(7)].iloc[0]
    assert at7.event_first_known252==2
    assert at7.event_complete_paths252==1
    assert at7.event_stop_at3_count252==1
    assert at7.event_max_path_height252==3


def test_current_and_future_labels_do_not_change_current_features():
    first,paths,sessions=sample()
    original=memory.build_features(first,paths,sessions)
    first.loc[first.p.ge(7),stage.TARGET]=1-first.loc[first.p.ge(7),stage.TARGET]
    for col in ["reach2_LABEL_ONLY","reach3_LABEL_ONLY","reach4_LABEL_ONLY"]:
        paths.loc[paths.p.ge(4),col]=1-paths.loc[paths.p.ge(4),col]
    paths.loc[paths.p.ge(4),"max_height_5_LABEL_ONLY"]=99
    changed=memory.build_features(first,paths,sessions)
    np.testing.assert_allclose(original.loc[original.p.le(7),memory.FIRST+memory.PATH],changed.loc[changed.p.le(7),memory.FIRST+memory.PATH],equal_nan=True)


def test_unknown_old_outcome_never_counts_as_failure():
    first,paths,sessions=sample()
    first.loc[first.p.eq(0),stage.TARGET]=np.nan
    paths.loc[paths.p.eq(0),["reach2_LABEL_ONLY","reach3_LABEL_ONLY","reach4_LABEL_ONLY","max_height_5_LABEL_ONLY"]]=np.nan
    rows=memory.build_features(first,paths,sessions)
    row=rows[rows.p.eq(4)].iloc[0]
    assert row.event_first_known252==0
    assert np.isnan(row.event_previous_first_outcome)
    assert row.event_stop_at3_count252==0
    assert row.event_complete_paths252==0
