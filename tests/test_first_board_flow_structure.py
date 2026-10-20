import numpy as np
import pandas as pd

import first_board_flow_structure as flow


def sample():
    rng=np.random.default_rng(42)
    dates=pd.bdate_range("2024-01-02",periods=85).strftime("%Y-%m-%d").tolist()
    rows=[]
    for code in ("000001","600001"):
        close=10.
        for p,day in enumerate(dates):
            previous=close;opening=previous*(1+rng.normal(0,.004));close=previous*(1+rng.normal(0,.015))
            high=max(opening,close)*1.004;low=min(opening,close)*.996
            rows.append(dict(date=day,code=code,p=p,open=opening,high=high,low=low,close=close,pre_close=previous,
                volume=float(rng.integers(10000,100000)),amount=2e7,turnover=5.,eligible=True,valid_price=True,regular_price=True))
    return pd.DataFrame(rows)


def test_future_bars_cannot_change_any_current_structure():
    bars=sample()
    old=flow.structure_from_bars(bars)
    cutoff=bars.loc[bars.p.eq(65),"date"].iloc[0]
    prefix=flow.structure_from_bars(bars[bars.p.le(65)])
    expected=old[old.date.le(cutoff)]
    np.testing.assert_allclose(prefix[flow.PAST+flow.SIGNAL],expected[flow.PAST+flow.SIGNAL],atol=1e-12,rtol=1e-12,equal_nan=True)
    changed=bars.copy();changed.loc[changed.p.gt(65),"volume"]*=1000
    new=flow.structure_from_bars(changed)
    np.testing.assert_allclose(old.loc[old.date.le(cutoff),flow.PAST+flow.SIGNAL],new.loc[new.date.le(cutoff),flow.PAST+flow.SIGNAL],equal_nan=True)


def test_prior_structure_excludes_current_first_board():
    bars=sample();old=flow.structure_from_bars(bars)
    day=bars.loc[bars.p.eq(65),"date"].iloc[0]
    bars.loc[bars.p.eq(65),"volume"]*=1000
    bars.loc[bars.p.eq(65),"turnover"]*=100
    new=flow.structure_from_bars(bars)
    np.testing.assert_allclose(old.loc[old.date.eq(day),flow.PAST],new.loc[new.date.eq(day),flow.PAST],equal_nan=True)


def test_undefined_close_location_and_missing_official_session_are_visible():
    bars=sample()
    one=bars.code.eq("000001")&bars.p.eq(64)
    price=bars.loc[one,"close"].iloc[0]
    bars.loc[one,["open","high","low","close"]]=price
    result=flow.structure_from_bars(bars)
    day=bars.loc[bars.p.eq(65),"date"].iloc[0]
    r=result[result.code.eq("000001")&result.date.eq(day)].iloc[0]
    assert r.structure_clv_observed_fraction20==19/20
    missing=bars[~(bars.code.eq("000001")&bars.p.eq(60))]
    r=flow.structure_from_bars(missing)
    r=r[r.code.eq("000001")&r.date.eq(day)].iloc[0]
    assert r[flow.PAST+flow.SIGNAL].isna().all()
