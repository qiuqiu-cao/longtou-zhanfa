import copy
import pandas as pd
import pytest

import legacy_industry_snapshot as legacy


def test_host_future_push_rejected_even_with_old_git_author_date():
    metadata=dict(private=False,fork=False,full_name="a/b",created_at="2021-01-01T00:00:00Z",
        pushed_at="2021-03-05T00:00:00Z",updated_at="2021-06-06T00:00:00Z")
    head=dict(sha="abc",commit=dict(author=dict(date="2020-01-01T00:00:00Z")))
    identity=dict(commit_sha="abc",repo="a/b")
    legacy.validate_metadata(metadata,head,identity)
    altered=copy.deepcopy(metadata);altered["pushed_at"]="2026-10-20T00:00:00Z"
    with pytest.raises(ValueError,match="host metadata"):
        legacy.validate_metadata(altered,head,identity)


def test_listing_identity_conflict_and_absent_members_keep_unknown():
    source=pd.DataFrame(dict(ts_code=["000001.SZ","000004.SZ","000005.SZ"],symbol=["000001","000004","000005"],
        name=["旧甲","旧乙","旧丙"],industry=["银行","软件服务",None],list_date=["19910403","19910114","19901210"]))
    master=pd.DataFrame(dict(code=["000001","000004","000005"],listing_date=["1991-04-03","1990-12-01","1990-12-10"]))
    result=legacy.normalize_snapshot(source,master).set_index("code")
    assert result.loc["000001","legacy_group_PROXY_ALLOWED"]=="银行"
    assert pd.isna(result.loc["000004","legacy_group_PROXY_ALLOWED"])
    assert result.loc["000004","identity_status"]=="LISTING_DATE_CONFLICT_UNKNOWN"
    assert pd.isna(result.loc["000005","legacy_group_PROXY_ALLOWED"])
    source.loc[0,"list_date"]="20240101"
    with pytest.raises(ValueError,match="newer"):
        legacy.normalize_snapshot(source,master)


def test_full_rows_and_future_targets_unchanged_and_old_research_dates_rejected():
    rows=pd.DataFrame(dict(signal_date=["2025-01-02"]*3,code=["000001","000004","999999"],
        name=["当时甲","当时乙","当时丙"],promotion_next_LABEL_ONLY=[1.,0.,None]))
    result=legacy.attach_legacy_groups(rows)
    pd.testing.assert_frame_equal(result[rows.columns],rows)
    assert result.legacy_industry_group_PROXY.notna().tolist()==[True,False,False]
    changed=rows.copy();changed["promotion_next_LABEL_ONLY"]=[0.,1.,1.]
    actual=legacy.attach_legacy_groups(changed)
    pd.testing.assert_series_equal(result.legacy_industry_group_PROXY,actual.legacy_industry_group_PROXY)
    with pytest.raises(ValueError,match="domain"):
        legacy.attach_legacy_groups(rows.assign(signal_date="2020-01-02"))
