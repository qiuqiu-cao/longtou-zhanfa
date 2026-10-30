import numpy as np
import pandas as pd
import pytest
import first_board_event_path_features as features


def fixture():
    n = 26
    source = pd.DataFrame(dict(date=pd.date_range("2024-01-01", periods=n).strftime("%Y-%m-%d"),
        code="000001", p=np.arange(n), open=10., high=10.1, low=9.9, close=10., pre_close=10., volume=100., amount=1000.))
    source.loc[10, ["open", "high", "low", "close", "volume"]] = [10., 11., 10., 11., 400.]
    source.loc[11:19, "pre_close"] = 10.
    source.loc[11, "pre_close"] = 11.
    source.loc[15, ["open", "high", "low", "close", "volume"]] = [10., 11., 9.9, 10.5, 200.]
    source.loc[16, "pre_close"] = 10.5
    source.loc[20, ["open", "high", "low", "close", "volume"]] = [10., 11., 10., 11., 500.]
    candidates = pd.DataFrame(dict(signal_date=[source.date.iloc[20]], code=["000001"]))
    return source, candidates


def test_event_alignment_and_excluding_current_event():
    b, k = fixture()
    r = features.from_bars(b, k).iloc[0]
    assert r.event_path_full_window_observed
    assert r[features.FIELDS["sealed"][1]] == 10
    assert r[features.FIELDS["probe"][1]] == 5
    assert r[features.FIELDS["sealed"][4]] == .25
    assert r[features.FIELDS["sealed"][5]] == 5
    assert r[features.FIELDS["sealed"][6]] == 0
    assert r[features.FIELDS["probe"][3]] == pytest.approx(10/10.5-1)


@pytest.mark.parametrize("change", ["missing", "zero", "off_cent"])
def test_invalid_or_nonadjacent_window_never_becomes_zero(change):
    b, k = fixture()
    if change == "missing": b = b.drop(index=14)
    elif change == "zero": b.loc[14, "volume"] = 0
    else: b.loc[14, "high"] = 10.101
    r = features.from_bars(b, k).iloc[0]
    assert not r.event_path_full_window_observed
    assert r[features.FIELDS["sealed"] + features.FIELDS["probe"]].isna().all()


def test_future_prefix_isolation_and_target_rejection():
    b, k = fixture()
    all_rows = features.from_bars(b, k)
    b.loc[21:, ["high", "low", "close", "volume"]] = [200., 1., 100., 1e8]
    pd.testing.assert_frame_equal(all_rows, features.from_bars(b, k))
    pd.testing.assert_frame_equal(all_rows, features.from_bars(b.iloc[:21], k))
    with pytest.raises(ValueError, match="target"):
        features.from_bars(b, k.assign(promotion_next_LABEL_ONLY=1))


def test_observed_no_event_and_no_correction_days_are_distinct():
    b, k = fixture()
    b.loc[[10, 15], ["open", "high", "low", "close"]] = [10., 10.1, 9.9, 10.]
    b.loc[[11, 16], "pre_close"] = 10.
    no_event = features.from_bars(b, k).iloc[0]
    assert no_event[features.FIELDS["sealed"][0]] == 0
    assert np.isnan(no_event[features.FIELDS["sealed"][1]])
    b.loc[19, ["open", "high", "low", "close"]] = [10., 11., 10., 11.]
    b.loc[20, "pre_close"] = 11.
    recent = features.from_bars(b, k).iloc[0]
    assert recent[features.FIELDS["sealed"][1]] == 1
    assert recent[features.FIELDS["sealed"][2:6]].isna().all()
