import pandas as pd

from theme_driver_history_v207 import build


def test_same_height_missing_driver_is_prioritized_and_not_zero():
    archive = pd.DataFrame([
        {"signal_date": "2026-01-05", "code": "1", "name": "A", "boards": 2,
         "theme_group_id": "x", "theme_status": "verified", "point_in_time_available_for_next_session": True,
         "next_session_preopen_cutoff": "2026-01-06T09:15:00+08:00", "full_market_completeness": "verified_full_market"},
        {"signal_date": "2026-01-05", "code": "2", "name": "B", "boards": 2,
         "theme_group_id": None, "theme_status": "unavailable", "point_in_time_available_for_next_session": False,
         "next_session_preopen_cutoff": "2026-01-06T09:15:00+08:00", "full_market_completeness": "verified_full_market"},
    ])
    evidence = pd.DataFrame([{"signal_date": "2026-01-05", "theme_group_id": "x",
        "available_at": "2026-01-05T16:00:00+08:00", "evidence_status": "verified",
        "theme_granularity": "specific", "driver_type": "POLICY", "driver_summary": "x",
        "driver_first_available_at": "2026-01-05T10:00:00+08:00",
        "driver_claim_status": "verified_primary", "expectation_status": "not_fully_priced"}])
    rows, queue, report = build(archive, evidence)
    assert rows.active_driver_verified.sum() == 1
    assert queue.iloc[0].same_height_race
    assert not rows.missing_driver_is_zero.any()
    assert report["formal_theme_factor_allowed"] is False
