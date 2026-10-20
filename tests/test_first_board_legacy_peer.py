import numpy as np
import pandas as pd
import pytest
import first_board_legacy_peer as peer


def fixture():
    dates=pd.bdate_range("2024-01-02",periods=10).strftime("%Y-%m-%d").tolist()
    rows=[]
    for j,code in enumerate(["000001","000002","000003","000004"]):
        limit_days={0:{7},1:{6,7},2:{4,5,6},3:{7}}[j]
        prices=[];height=0;was_limit=False
        for i,date in enumerate(dates):
            is_limit=i in limit_days
            rate=.1 if is_limit else (-.03 if j==2 and i==7 else .01)
            previous=prices[-1] if prices else 10.
            close=previous*(1+rate);prices.append(close)
            first=is_limit and not was_limit
            height=height+1 if is_limit and was_limit else (1 if is_limit else 0)
            rows.append(dict(date=date,code=code,p=i,close=close,pre_close=previous,eligible=True,valid_price=True,regular_price=True,
                amount=(j+1)*1e8,volume=(j+1)*1e7,limit=is_limit,height=height,first=first,failed=bool(j==2 and i==7),
                turnover=(j+1)*10.,volume_ratio_5=[2.,1.5,1.2,1.][j],amount_ratio_5=1.,
                return_5=close/prices[i-5]-1 if i>=5 else np.nan))
            was_limit=is_limit
    mapping=pd.DataFrame(dict(code=["000001","000002","000003","000004"],legacy_group_RAW="旧组G",legacy_group_PROXY_ALLOWED="旧组G"))
    keys=pd.DataFrame(dict(signal_date=[dates[7]],code=["000001"]))
    return dates,pd.DataFrame(rows),mapping,keys


def test_manual_peer_counts_returns_ladder_and_rank():
    dates,bars,mapping,keys=fixture()
    row=peer.build_peer_features(bars,keys,mapping).iloc[0]
    expected=dict(legacy_group_known=1,legacy_peer_observed_n=3,legacy_peer_catalog_fraction=1.,
        legacy_peer_limit_rate=2/3,legacy_peer_first_rate=1/3,legacy_peer_relay_rate=1/3,
        legacy_peer_failed_attempt_rate=1/3,legacy_peer_mean_return=(.1-.03+.1)/3,
        legacy_peer_positive_fraction=2/3,legacy_peer_max_height=2,
        legacy_peer_previous_first_promotion=1.,legacy_peer_previous_high_break_fraction=1.,
        legacy_first_competitor_observed_n=1,legacy_first_turnover_rank=.5,legacy_first_amount_rank=.5,legacy_first_volume_rank=1.)
    for field,value in expected.items():assert row[field]==pytest.approx(value,abs=1e-12)
    signal=bars[bars.date.eq(dates[7])].set_index("code")
    assert row.legacy_self_to_peer_return5==pytest.approx(signal.loc["000001","return_5"]-signal.drop("000001").return_5.mean())
    assert row.legacy_self_to_peer_amount5==pytest.approx(0)


def test_self_exclusion_future_isolation_and_prefix_identity():
    dates,bars,mapping,keys=fixture()
    expected=peer.build_peer_features(bars,keys,mapping)
    changed=bars.copy()
    own=changed.code.eq("000001")&changed.date.eq(dates[7])
    changed.loc[own,"height"]=9
    changed.loc[own,["amount","turnover","volume_ratio_5"]]*=30
    actual=peer.build_peer_features(changed,keys,mapping)
    peer_only=peer.PEERS[:12]
    np.testing.assert_allclose(expected[peer_only],actual[peer_only],atol=1e-12)
    changed=bars.copy();changed.loc[changed.date.gt(dates[7]),["close","pre_close","amount"]]*=50
    pd.testing.assert_frame_equal(expected,peer.build_peer_features(changed,keys.assign(future_LABEL_ONLY=1),mapping))
    pd.testing.assert_frame_equal(expected,peer.build_peer_features(bars[bars.date.le(dates[7])],keys,mapping))


def test_unmapped_keeps_all_fields_unknown_and_missing_official_predecessor_not_bridged():
    dates,bars,mapping,keys=fixture()
    unknown=mapping.copy();unknown.loc[unknown.code.eq("000004"),"legacy_group_PROXY_ALLOWED"]=None
    pair=pd.DataFrame(dict(signal_date=[dates[7]]*2,code=["000001","000004"]))
    result=peer.build_peer_features(bars,pair,unknown)
    assert result.legacy_group_known.tolist()==[1,0]
    assert result.loc[1,peer.PEERS].isna().all()
    missing=bars[~(bars.code.eq("000003")&bars.date.eq(dates[6]))]
    row=peer.build_peer_features(missing,keys,mapping).iloc[0]
    assert row.legacy_peer_observed_n==2
    assert row.legacy_peer_catalog_fraction==pytest.approx(2/3)
    assert pd.isna(row.legacy_peer_previous_high_break_fraction)


def test_empty_observed_peer_set_has_undefined_rates_not_fake_zero():
    dates,bars,mapping,keys=fixture()
    mapping.loc[mapping.code.ne("000001"),["legacy_group_RAW","legacy_group_PROXY_ALLOWED"]]="其他组"
    row=peer.build_peer_features(bars,keys,mapping).iloc[0]
    assert row.legacy_peer_observed_n==0
    assert pd.isna(row.legacy_peer_limit_rate) and pd.isna(row.legacy_peer_mean_return)
    assert pd.isna(row.legacy_peer_max_height) and pd.isna(row.legacy_peer_failed_attempt_rate)
