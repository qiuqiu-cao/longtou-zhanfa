"""Synthetic causality/boundary checks; no real labels or model fitting."""
import json

import numpy as np
import pandas as pd
import pytest

from research.wyckoff_features import FEATURE, extract_features


def fixture():
    # Includes the actual holiday-sized gap; official adjacency is not weekday adjacency.
    sessions = ["2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24",
                "2026-09-28", "2026-09-29", "2026-09-30", "2026-10-08",
                "2026-10-09", "2026-10-12", "2026-10-13"]
    rows = []
    for i, date in enumerate(sessions):
        close = 10+.2*i
        rows.append(dict(date=date, code="000001", open=close, high=close+.5,
                         low=close-.5, close=close, volume=100., source="same-unadjusted-provider"))
    # Event=T=index6: higher effort, smaller range, weak close after rising background.
    rows[6].update(open=10.9, high=11.2, low=10.7, close=10.8, volume=200.)
    candidates = pd.DataFrame([dict(date=sessions[8], code="000001", boards=2)])
    return candidates, pd.DataFrame(rows), sessions


def get(candidates, bars, sessions):
    return extract_features(candidates, bars, sessions).iloc[0]


def test_positive_exact_window_age_and_holiday_adjacency():
    c, b, s = fixture()
    r = get(c, b, s)
    assert r[FEATURE] == 1 and r.wyckoff_input_missing == 0
    assert r.feature_date == "2026-09-30" and r.age_sessions == 2
    assert json.loads(r.window_dates) == s[:7]
    assert r.volume_ratio == 2
    assert r.close_location == pytest.approx(.2)


@pytest.mark.parametrize("field,value", [("volume", 100.), ("close", 11.05), ("high", 12.0)])
def test_valid_negative_conjunction_is_zero(field, value):
    c, b, s = fixture()
    b.loc[6, field] = value
    r = get(c, b, s)
    assert r[FEATURE] == 0 and r.wyckoff_input_missing == 0


def test_background_false_is_zero():
    c, b, s = fixture()
    b.loc[0, ["open", "high", "low", "close"]] = [11.1, 11.5, 10.9, 11.1]
    r = get(c, b, s)
    assert r[FEATURE] == 0 and r.wyckoff_input_missing == 0


@pytest.mark.parametrize("kind", ["gap", "volume_zero", "nonfinite", "bounds", "duplicate", "mixed_source", "missing_source"])
def test_missing_or_invalid_input_stays_unknown(kind):
    c, b, s = fixture()
    if kind == "gap": b = b.drop(index=3)
    elif kind == "volume_zero": b.loc[3, "volume"] = 0
    elif kind == "nonfinite": b.loc[3, "volume"] = np.inf
    elif kind == "bounds": b.loc[3, "close"] = 99
    elif kind == "duplicate": b = pd.concat([b, b.iloc[[3]]], ignore_index=True)
    elif kind == "mixed_source": b.loc[3, "source"] = "other-provider"
    elif kind == "missing_source": b.loc[3, "source"] = None
    r = get(c, b, s)
    assert pd.isna(r[FEATURE]) and r.wyckoff_input_missing == 1


@pytest.mark.parametrize("side", ["high", "low"])
def test_exact_one_cent_endpoint_and_float_noise_are_excluded(side):
    c, b, s = fixture()
    b.loc[6, side] = 10.81+1e-12 if side == "high" else 10.79-1e-12
    if side == "high": b.loc[6, "open"] = 10.8
    r = get(c, b, s)
    assert pd.isna(r[FEATURE]) and r.feature_status == "EVENT_ENDPOINT_EXCLUDED"


def test_future_and_board_period_mutations_do_not_change_features():
    c, b, s = fixture()
    expected = extract_features(c, b, s)
    b.loc[7:, ["open", "high", "low", "close", "volume"]] = [1000., 1001., 999., 1000., 1e9]
    b.loc[7:, "source"] = "future-other-source"
    # Even future duplicate and invalid OHLC do not invalidate the prior window.
    b.loc[10, "close"] = np.inf
    b = pd.concat([b, b.iloc[[9]]], ignore_index=True)
    pd.testing.assert_frame_equal(expected, extract_features(c, b, s))


def test_same_first_board_origin_across_two_board_ages():
    c, b, s = fixture()
    c = pd.concat([c, pd.DataFrame([dict(date=s[9], code="000001", boards=3)])], ignore_index=True)
    r = extract_features(c, b, s)
    assert r.feature_date.tolist() == [s[6], s[6]]
    assert r.age_sessions.tolist() == [2, 3]
    assert r[FEATURE].tolist() == [1, 1]


def test_calendar_insufficient_or_nonofficial_or_bad_boards_unknown():
    c, b, s = fixture()
    for field, value, reason in [("date", s[4], "INSUFFICIENT_OFFICIAL_WINDOW"),
                                 ("date", "2026-10-07", "SIGNAL_NOT_OFFICIAL_SESSION"),
                                 ("boards", 2.5, "INVALID_BOARDS")]:
        x = c.copy()
        if field == "boards":
            x["boards"] = x.boards.astype(float)
        x.loc[0, field] = value
        r = get(x, b, s)
        assert pd.isna(r[FEATURE]) and r.feature_status == reason


def test_duplicate_candidate_keys_are_preserved_as_unknown():
    c, b, s = fixture()
    r = extract_features(pd.concat([c, c], ignore_index=True), b, s)
    assert len(r) == 2 and r[FEATURE].isna().all()
    assert r.feature_status.eq("DUPLICATE_CANDIDATE_KEY").all()


def test_invalid_official_calendar_is_rejected():
    c, b, s = fixture()
    with pytest.raises(ValueError, match="unique and increasing"):
        extract_features(c, b, s+[s[-1]])
