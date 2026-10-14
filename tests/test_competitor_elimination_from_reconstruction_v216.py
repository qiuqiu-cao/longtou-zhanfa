import pandas as pd

from competitor_elimination_from_reconstruction_v216 import STATUS, build


def test_same_date_and_height_define_elimination_pressure():
    attempts = pd.DataFrame([
        {"signal_date": "2026-01-05", "code": "1", "pool_assignment": "CLOSING_LIMIT_UP", "board_height": 1},
        {"signal_date": "2026-01-05", "code": "2", "pool_assignment": "CLOSING_LIMIT_UP", "board_height": 1},
        {"signal_date": "2026-01-05", "code": "3", "pool_assignment": "FAILED_BOARD_OBSERVATION", "board_height": 1},
        {"signal_date": "2026-01-05", "code": "4", "pool_assignment": "FAILED_BOARD_OBSERVATION", "board_height": 2},
    ])
    candidates = pd.DataFrame([
        {"signal_date": "2026-01-05", "code": "1", "name": "A", "boards": 1},
        {"signal_date": "2026-01-05", "code": "9", "name": "B", "boards": 2},
    ])
    rows, report = build(attempts, candidates)
    first, second = rows.iloc[0], rows.iloc[1]
    assert first.competitor_elimination == 1 / 3
    assert first.same_height_survivor_count == 2
    assert second.competitor_elimination == 1
    assert second.competitor_elimination_evidence_status == STATUS
    assert report["formal_factor_allowed"] is False


def test_missing_group_stays_unavailable_not_zero():
    attempts = pd.DataFrame([
        {"signal_date": "2026-01-05", "code": "1", "pool_assignment": "CLOSING_LIMIT_UP", "board_height": 1},
    ])
    candidates = pd.DataFrame([
        {"signal_date": "2026-01-06", "code": "2", "name": "B", "boards": 1},
    ])
    rows, _ = build(attempts, candidates)
    assert pd.isna(rows.iloc[0].competitor_elimination)
    assert rows.iloc[0].competitor_elimination_evidence_status == "UNAVAILABLE"
