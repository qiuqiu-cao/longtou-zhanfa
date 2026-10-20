import numpy as np
import pandas as pd
import pytest

import first_board_market_residual as m


def fixture():
    sessions = pd.bdate_range("2024-01-01", periods=95).strftime("%Y-%m-%d").tolist()
    rng = np.random.default_rng(29)
    market = rng.normal(0, .009, 95)
    returns = market[:, None] * np.array([1.2, .7, 1., 1.4, .6]) + rng.normal(0, .013, (95, 5))
    prices = 10 * np.cumprod(1 + returns, axis=0)
    records = []
    for i, day in enumerate(sessions):
        for j in range(5):
            records.append(dict(date=day, code=f"{j+1:06}", p=i, close=prices[i, j],
                pre_close=prices[i-1, j] if i else 10, eligible=True, valid_price=True,
                regular_price=True, amount=1e8, volume=1e7))
    keys = pd.DataFrame([dict(signal_date=sessions[i], code="000001") for i in (75, 80)])
    return sessions, pd.DataFrame(records), keys


def test_direct_leave_self_out_regression_agrees_and_excludes_signal():
    sessions, bars, keys = fixture()
    result = m.build_features(bars, sessions, keys)
    y, x = m.return_grid(bars, sessions)
    for row in result.itertuples():
        i = sessions.index(row.signal_date)
        for w in (20, 60):
            yy = y[row.code].iloc[i-w:i].to_numpy()
            xx = x[row.code].iloc[i-w:i].to_numpy()
            alpha, beta = np.linalg.lstsq(np.column_stack([np.ones(w), xx]), yy, rcond=None)[0]
            residual_vol = np.sqrt(np.mean((yy-alpha-beta*xx)**2))
            expected = dict(beta=beta, corr=np.corrcoef(xx, yy)[0, 1], residual_vol=residual_vol,
                signal_residual_z=(y[row.code].iloc[i]-alpha-beta*x[row.code].iloc[i])/residual_vol)
            for name, value in expected.items():
                assert getattr(row, f"market_{name}{w}") == pytest.approx(value, abs=1e-10)
    # Altering this stock on the signal day cannot change its prior regression
    # or its own leave-self-out contemporaneous market reference.
    day = keys.signal_date.iloc[0]
    changed = bars.copy()
    changed.loc[changed.date.eq(day) & changed.code.eq("000001"), "close"] *= 1.04
    changed_result = m.build_features(changed, sessions, keys.iloc[:1])
    prior_fields = [f for f in m.FIELDS if "signal_residual_z" not in f]
    np.testing.assert_allclose(result[prior_fields].iloc[:1], changed_result[prior_fields], atol=1e-12)
    _, changed_x = m.return_grid(changed, sessions)
    assert x.loc[day, "000001"] == pytest.approx(changed_x.loc[day, "000001"])


def test_future_and_targets_do_not_change_prior_predictions():
    sessions, bars, keys = fixture()
    expected = m.build_features(bars, sessions, keys.iloc[:1])
    altered = bars.copy()
    altered.loc[altered.date.gt(keys.signal_date.iloc[0]), ["close", "pre_close"]] *= 50
    targets = keys.iloc[:1].assign(future_promotion_LABEL_ONLY=1)
    actual = m.build_features(altered, sessions, targets)
    pd.testing.assert_frame_equal(expected, actual)


def test_missing_official_session_or_reference_mismatch_remains_unknown():
    sessions, bars, keys = fixture()
    missing = bars[~(bars.date.eq(sessions[70]) & bars.code.eq("000001"))]
    result = m.build_features(missing, sessions, keys.iloc[:1])
    assert result[m.FIELDS].isna().all().all()
    mismatch = bars.copy()
    mismatch.loc[mismatch.date.eq(sessions[70]) & mismatch.code.eq("000001"), "pre_close"] += .02
    assert m.build_features(mismatch, sessions, keys.iloc[:1])[m.FIELDS].isna().all().all()
