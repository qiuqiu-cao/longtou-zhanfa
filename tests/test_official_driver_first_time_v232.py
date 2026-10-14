import pytest

from official_driver_first_time_v232 import evaluate_row, _assert_terms


def _row(**updates):
    row = {
        "signal_date": "2026-09-21",
        "official_published_at": "2026-08-21",
        "first_public_time_basis": "AT_LATEST_OFFICIAL_DISCLOSURE",
        "lifecycle_status": "ONGOING_ACTION_VERIFIED",
    }
    row.update(updates)
    return row


def test_ongoing_action_is_not_mislabeled_as_new_signal_date_catalyst():
    result = evaluate_row(_row())
    assert result["official_driver_age_lower_bound_days"] == 31
    assert result["signal_date_novelty_status"] == "PREEXISTING_AT_LEAST_31_CALENDAR_DAYS"
    assert result["driver_freshness_readthrough"] == "ONGOING_ACTION_BUT_NOT_NEW_ON_SIGNAL_DATE"
    assert not result["driver_is_new_on_signal_date"]
    assert not result["driver_age_is_ranking_score"]


def test_exact_historical_policy_time_is_kept_distinct_from_upper_bound():
    result = evaluate_row(_row(
        official_published_at="2022-12-10 19:56:04",
        first_public_time_basis="EXACT_OFFICIAL_PAGE_TIME",
        lifecycle_status="HISTORICAL_POLICY_INCLUSION",
    ))
    assert result["first_public_time_status"] == "VERIFIED_EXACT"
    assert result["driver_freshness_readthrough"] == "LEGACY_OR_RECYCLED_DRIVER_NOT_NEW"


def test_source_term_mismatch_fails_closed():
    with pytest.raises(ValueError, match="term mismatch"):
        _assert_terms("only one phrase", ["one phrase", "missing"], "x")
