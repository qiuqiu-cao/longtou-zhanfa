from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

import first_board_leader_model as fb


def sample_inputs(tmp_path, missing=None, change_future=False):
    days=pd.bdate_range("2024-01-02",periods=30).strftime("%Y-%m-%d").tolist()
    rows=[]
    for code in ("000001","600001"):
        price=10.
        for i,day in enumerate(days):
            previous=price
            if i in (5,6,7,8):
                price=fb.limit_cents([previous])[0]/100
            elif i==15 and code=="600001":
                price=fb.limit_cents([previous])[0]/100-.01
            if change_future and i>=10:
                price=round(previous*.99,2)
            if (code,i)==missing:continue
            rows.append(dict(date=day,code=code,name=code,open=previous,high=max(price,previous),low=min(price,previous),close=price,pre_close=previous,volume=100000.,amount=1000000.,turnover=5.,float_cap=20000000.,st=False,new_listing=False,status_verified=True,source="synthetic_test_fixture"))
    frame=pd.DataFrame(rows)
    frame.to_csv(tmp_path/"bars.csv",index=False)
    frame.iloc[:0].to_csv(tmp_path/"recovered.csv",index=False)
    (tmp_path/"calendar.json").write_text(json.dumps({"sessions":days}))
    pd.DataFrame([dict(code=c,listing_date="2000-01-01") for c in ("000001","600001")]).to_csv(tmp_path/"security.csv",index=False)
    return dict(bars=tmp_path/"bars.csv",recovered=tmp_path/"recovered.csv",calendar=tmp_path/"calendar.json",security_intervals=tmp_path/"security.csv"),days


def test_exact_continuous_targets_and_first_identity(tmp_path):
    inputs,days=sample_inputs(tmp_path)
    rows,_,_,_=fb.build_dataset(inputs)
    assert len(rows)==2
    assert set(rows.signal_date)=={days[5]}
    assert rows.reach4_LABEL_ONLY.eq(1).all()
    assert rows.reach5_LABEL_ONLY.eq(0).all()
    assert rows.max_height_5_LABEL_ONLY.eq(4).all()
    assert rows.cohort_leader_LABEL_ONLY.eq(1).all()
    assert not any("LABEL" in f or "future" in f for f in fb.FEATURES)


def test_missing_official_session_is_unknown_and_never_skipped(tmp_path):
    inputs,days=sample_inputs(tmp_path,missing=("000001",6))
    rows,_,_,_=fb.build_dataset(inputs)
    candidate=rows[(rows.code=="000001") & (rows.signal_date==days[5])].iloc[0]
    assert pd.isna(candidate.reach4_LABEL_ONLY)
    assert pd.isna(candidate.cohort_leader_LABEL_ONLY)
    assert not ((rows.code=="000001") & (rows.signal_date==days[7])).any()


def test_future_prices_do_not_change_signal_features(tmp_path):
    a=tmp_path/"a";a.mkdir();b=tmp_path/"b";b.mkdir()
    first,days=sample_inputs(a)
    changed,_=sample_inputs(b,change_future=True)
    x=fb.build_dataset(first)[0]
    y=fb.build_dataset(changed)[0]
    x=x[x.signal_date==days[5]].set_index("code")[fb.FEATURES]
    y=y[y.signal_date==days[5]].set_index("code")[fb.FEATURES]
    pd.testing.assert_frame_equal(x,y)


def test_unknown_labels_remain_in_topk_denominator():
    rows=pd.DataFrame(dict(signal_date=["2025-01-02"]*3,code=["000001","000002","000003"],score=[.9,.8,.7],reach4_LABEL_ONLY=[1,np.nan,0]))
    result=fb.metric(rows,"score",3)
    assert result["selected"]==3
    assert result["known"]==2
    assert result["unknown"]==1
    assert result["precision"]==.5


def test_source_conflicts_refused(tmp_path):
    inputs,_=sample_inputs(tmp_path)
    frame=pd.read_csv(inputs["bars"],dtype={"code":str}).iloc[:1].copy()
    frame["close"]+=.01
    frame.to_csv(inputs["recovered"],index=False)
    with pytest.raises(ValueError,match="conflicting duplicate"):
        fb.build_dataset(inputs)
