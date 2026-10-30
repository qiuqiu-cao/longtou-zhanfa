import numpy as np
import pandas as pd
import pytest
import first_board_long_high_features as features


def source():
    return pd.DataFrame({"date":pd.date_range("2024-01-01", periods=6).strftime("%Y-%m-%d"),
        "code":["000001"]*6,"p":np.arange(6),"open":[10.,10.,10.,10.,10.,10.],
        "high":[11.,10.5,10.5,10.,11.,11.],"low":[10.]*6,"close":[10.]*6,
        "pre_close":[10.]*6,"volume":[100.]*6,"amount":[1000.]*6})


def test_current_high_is_excluded_and_equal_high_uses_latest_session():
    data=source()
    actual=features.from_bars(data,horizons=(120,))
    assert actual[features.DISTANCE].isna().all()
    # The same algorithm on a small exact window exposes off-by-one errors directly.
    small=features.from_bars(data,horizons=(3,))
    assert small.loc[3,features.DIAGNOSTIC250] == pytest.approx(np.log(10/11))
    assert small.loc[4,features.DIAGNOSTIC250] == pytest.approx(np.log(10/10.5))


def test_reference_link_cancels_mechanical_split_and_future_data_cannot_rewrite_prefix():
    old=source()
    split=old.copy()
    split.loc[3:,["open","high","low","close","pre_close"]]/=2
    normal=features.from_bars(old,horizons=(3,))
    actual=features.from_bars(split,horizons=(3,))
    np.testing.assert_allclose(normal[features.DIAGNOSTIC250],actual[features.DIAGNOSTIC250],atol=1e-14,rtol=0,equal_nan=True)
    prefix=features.from_bars(split.iloc[:5],horizons=(3,))
    np.testing.assert_allclose(prefix[features.DIAGNOSTIC250],actual.iloc[:5][features.DIAGNOSTIC250],atol=0,rtol=0,equal_nan=True)


def test_missing_official_session_and_zero_volume_do_not_bridge_window():
    data=source()
    missing=data.drop(index=2)
    out=features.from_bars(missing,horizons=(3,))
    assert out[features.DIAGNOSTIC250].isna().all()
    data.loc[2,"volume"]=0
    out=features.from_bars(data,horizons=(3,))
    assert out[features.DIAGNOSTIC250].isna().all()


def test_duplicate_official_key_is_rejected():
    with pytest.raises(ValueError,match="Duplicate"):
        features.from_bars(pd.concat([source(),source().iloc[:1]]),horizons=(3,))


def test_full120_age_selects_last_equal_high_and_excludes_signal_high():
    data=pd.DataFrame({"date":pd.date_range("2024-01-01",periods=125).strftime("%Y-%m-%d"),
        "code":["000001"]*125,"p":np.arange(125),"open":[10.]*125,"high":[10.]*125,
        "low":[10.]*125,"close":[10.]*125,"pre_close":[10.]*125,"volume":[100.]*125,"amount":[1000.]*125})
    data.loc[[6,10,122],"high"]=11.
    actual=features.from_bars(data,horizons=(120,))
    assert actual.loc[122,features.DISTANCE] == pytest.approx(np.log(10/11))
    assert actual.loc[122,features.AGE] == 112
    assert actual.loc[123,features.AGE] == 1
