import pandas as pd
import pytest

import relay_full_cohort_transition as m


S = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]


def attempt(day, code, height=1, close=11, high=11, prior=10):
    return dict(signal_date=day, code=code, name=code, board_height=height,
        close=close, high=high, pre_close=prior, limit_price_10=11 if prior == 10 else 12.1,
        pool_assignment="CLOSING_LIMIT_UP" if close == (11 if prior == 10 else 12.1) else "FAILED_BOARD_OBSERVATION")


def bar(day, code, close, high=None, prior=11, volume=100):
    hi = close if high is None else high
    return dict(date=day, code=code, open=min(close, hi), high=hi, low=min(close, hi), close=close,
                pre_close=float(prior), volume=volume)


def fixture():
    a = pd.DataFrame([attempt(S[0], "1"), attempt(S[0], "2"), attempt(S[0], "3"),
                      attempt(S[1], "1", 2, 12.1, 12.1, 11), attempt(S[1], "2", 2, 11.5, 12.1, 11)])
    a.attrs["failed_scope_complete_dates"] = S[:2]
    b = pd.DataFrame([bar(S[1], "1", 12.1), bar(S[1], "2", 11.5, 12.1), bar(S[1], "3", 10.5, 11.2)])
    c = pd.DataFrame([dict(date=S[1], code="1", boards=2)])
    return c, a, b


def test_full_cohort_keeps_non_touchers_and_separates_denominators():
    c, a, b = fixture()
    out = m.build_transition_context(c, a, b, S)
    row = out["rows"].iloc[0]
    assert row.full_cohort_previous_count == 3
    assert row.full_cohort_promotion_rate == pytest.approx(1 / 3)
    assert row.touch_failure_rate == .5
    assert row.full_cohort_failed_no_touch_count == 1
    assert set(out["members"].transition_state) == m.KNOWN_STATES
    assert out["summary"]["exchange_status_verified"] is False


def test_missing_or_suspended_members_never_count_as_failure():
    c, a, b = fixture()
    b = b[b.code != "3"].copy()
    b.loc[b.code == "2", "trade_status"] = "SUSPENDED"
    out = m.build_transition_context(c, a, b, S)
    row = out["rows"].iloc[0]
    assert pd.isna(row.full_cohort_promotion_rate)
    assert row.full_cohort_unknown_count == 2
    assert row.full_cohort_promotion_lower_bound == pytest.approx(1 / 3)
    assert row.full_cohort_promotion_upper_bound == 1
    assert set(out["members"].transition_state) == {"PROMOTED_EXACT_CAP", "UNKNOWN_SUSPENDED_EXPLICIT", "UNKNOWN_MISSING_DAILY"}


def test_presence_of_attempts_does_not_certify_complete_scope():
    c, a, b = fixture()
    a.attrs.clear()
    row = m.build_transition_context(c, a, b, S)["rows"].iloc[0]
    assert row.full_cohort_status == "UNKNOWN_PREVIOUS_POOL_SCOPE"
    assert pd.isna(row.full_cohort_previous_count)
    assert pd.isna(row.full_cohort_promotion_rate)
    assert pd.isna(row.touch_failure_rate)


def test_explicit_empty_touch_scope_is_not_zero_rate():
    c, a, b = fixture()
    a = a[a.signal_date == S[0]].copy()
    a.attrs["failed_scope_complete_dates"] = S[:2]
    row = m.build_transition_context(c, a, b, S)["rows"].iloc[0]
    assert row.touch_failure_attempt_count == 0
    assert row.touch_failure_scope_status == "NOT_APPLICABLE_NO_TOUCHED_ATTEMPTS"
    assert pd.isna(row.touch_failure_rate)
    assert row.full_cohort_promotion_rate == pytest.approx(1 / 3)


def test_previous_official_day_cannot_skip_missing_session():
    c, a, b = fixture()
    c["date"] = S[2]
    a = a[a.signal_date == S[0]].copy()
    a.attrs["failed_scope_complete_dates"] = [S[0]]
    b["date"] = S[2]
    row = m.build_transition_context(c, a, b, S)["rows"].iloc[0]
    assert row.full_cohort_previous_date == S[1]
    assert row.full_cohort_status == "UNKNOWN_PREVIOUS_POOL_SCOPE"
    assert pd.isna(row.full_cohort_promotion_rate)


def test_conflicting_preclose_or_previous_bar_remains_unknown():
    c, a, b = fixture()
    b.loc[b.code == "1", "pre_close"] = 10.99
    b = pd.concat([b, pd.DataFrame([bar(S[0], "2", 10.99, prior=10)])], ignore_index=True)
    out = m.build_transition_context(c, a, b, S)
    assert out["rows"].iloc[0].full_cohort_unknown_count == 2
    assert set(out["members"].qualification_reason) >= {
        "CURRENT_PRECLOSE_DIFFERS_FROM_BOUND_PREVIOUS_CLOSE", "PREVIOUS_DAILY_CLOSE_DIFFERS_FROM_BOUND_POOL"}


def test_zero_volume_is_unknown_and_optional_preclose_can_be_absent():
    c, a, b = fixture()
    b.loc[b.code == "2", "volume"] = 0
    b = b.drop(columns="pre_close")
    out = m.build_transition_context(c, a, b, S)
    states = out["members"].set_index("code").transition_state
    assert states["000001"] == "PROMOTED_EXACT_CAP"
    assert states["000002"] == "UNKNOWN_NONTRADING"


def test_invalid_exact_attempt_quarantines_entire_declared_scope():
    c, a, b = fixture()
    a.loc[a.signal_date.eq(S[0]) & a.code.eq("3"), "high"] = 11.01
    out = m.build_transition_context(c, a, b, S)
    assert len(out["attempt_quarantine"]) == 1
    assert out["rows"].iloc[0].full_cohort_status == "UNKNOWN_PREVIOUS_POOL_SCOPE"


def test_all_success_all_failure_and_empty_groups_are_preserved():
    c, a, b = fixture()
    b.loc[:, ["open", "high", "low", "close"]] = 12.1
    out = m.build_transition_context(c, a, b, S)
    assert out["rows"].iloc[0].full_cohort_promotion_rate == 1
    b.loc[:, ["open", "high", "low", "close"]] = 10
    out = m.build_transition_context(c, a, b, S)
    assert out["rows"].iloc[0].full_cohort_promotion_rate == 0
    c["boards"] = 4
    out = m.build_transition_context(c, a, b, S)
    assert out["rows"].iloc[0].full_cohort_status == "NOT_APPLICABLE_EMPTY_COHORT"
    assert 2 in set(out["groups"].target_height)  # no surviving candidate in this group


def test_day_and_batch_use_same_engine_and_ignore_future_and_labels():
    c, a, b = fixture()
    future = c.copy()
    future["date"] = S[2]
    batch = pd.concat([c, future], ignore_index=True)
    c["target_next_up"] = 999
    c["score"] = -999
    one = m.build_for_day(S[1], c, a, b, S)
    full = m.build_transition_context(batch, a, b, S)
    pd.testing.assert_frame_equal(one["rows"].reset_index(drop=True), full["rows"][full["rows"].date.eq(S[1])].reset_index(drop=True))
    bad_future = pd.concat([b, pd.DataFrame([bar(S[3], "1", 999), bar(S[3], "1", -1)])], ignore_index=True)
    again = m.build_for_day(S[1], c, a, bad_future, S)
    pd.testing.assert_frame_equal(one["rows"], again["rows"])


def test_exact_half_cent_rounding_and_candidate_membership_conflict():
    assert m._cap(1005) == 1106
    c, a, b = fixture()
    c["code"] = "999999"
    row = m.build_transition_context(c, a, b, S)["rows"].iloc[0]
    assert row.full_cohort_candidate_membership == "NOT_IN_PREVIOUS_K_MINUS_1_COHORT"
    assert row.full_cohort_candidate_transition_state == "UNKNOWN_CANDIDATE_HEIGHT_OR_POOL_CONFLICT"
    assert row.full_cohort_promotion_rate == pytest.approx(1 / 3)


def test_duplicate_relevant_source_rejected_and_inputs_not_mutated():
    c, a, b = fixture()
    saved = a.copy(deep=True)
    with pytest.raises(ValueError, match="duplicate dated"):
        m.build_transition_context(c, a, pd.concat([b, b.iloc[:1]], ignore_index=True), S)
    m.build_transition_context(c, a, b, S)
    pd.testing.assert_frame_equal(a, saved)
