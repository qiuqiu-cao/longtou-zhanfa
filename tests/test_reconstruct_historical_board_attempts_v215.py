import pandas as pd

from reconstruct_historical_board_attempts_v215 import reconstruct


def bar(date, code, close, high=None, pre=10, st=False, new=False, verified=True):
    return {"date": date, "code": code, "name": code, "high": close if high is None else high,
            "close": close, "pre_close": pre, "st": st, "new_listing": new,
            "status_verified": verified}


def test_failed_attempt_uses_immediately_previous_session_streak():
    bars = pd.DataFrame([
        bar("2026-01-05", "1", 11, pre=10),
        bar("2026-01-06", "1", 12.10, pre=11),
        bar("2026-01-07", "1", 12.00, high=13.31, pre=12.10),
        bar("2026-01-05", "2", 9, pre=10),
        bar("2026-01-06", "2", 9, pre=9),
        bar("2026-01-07", "2", 9, high=9.90, pre=9),
    ])
    rows, _ = reconstruct(bars)
    failed = rows[rows.pool_assignment.eq("FAILED_BOARD_OBSERVATION")].set_index("code")
    assert failed.loc["000001", "board_height"] == 3
    assert failed.loc["000002", "board_height"] == 1


def test_missing_market_session_breaks_streak_instead_of_skipping():
    bars = pd.DataFrame([
        bar("2026-01-05", "1", 11, pre=10),
        bar("2026-01-06", "2", 9, pre=10),
        bar("2026-01-07", "1", 12.10, pre=11),
        bar("2026-01-07", "2", 9, pre=9),
    ])
    rows, _ = reconstruct(bars)
    closing = rows[(rows.code.eq("000001")) & rows.pool_assignment.eq("CLOSING_LIMIT_UP")]
    assert closing.board_height.tolist() == [1, 1]


def test_unverified_status_is_kept_only_as_diagnostic_reconstruction():
    bars = pd.DataFrame([bar("2026-01-05", "1", 11, pre=10, verified=False)])
    rows, report = reconstruct(bars)
    assert len(rows) == 1
    assert rows.iloc[0].scope_evidence_status == "CURRENT_UNIVERSE_STATUS_RECONSTRUCTION_UNVERIFIED"
    assert report["status_unverified_candidate_rows"] == 1
    assert report["formal_factor_allowed"] is False
