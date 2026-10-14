import numpy as np
import pandas as pd
import pytest

from relay_turnover_phase_research import EXTRA, add_phase


SESSIONS = ["2026-09-30", "2026-10-08", "2026-10-09", "2026-10-12"]


def cohort(rows):
    return pd.DataFrame(rows, columns=["date", "code", "signal_one_word", "boards"])


def test_exact_official_predecessor_connects_long_holiday_and_weekend():
    data = cohort([
        ("2026-09-30", "600241", 1, 3),
        ("2026-10-08", "600241", 0, 4),
        ("2026-10-09", "600241", 1, 5),
        ("2026-10-12", "600241", 0, 6),
    ])
    result = add_phase(data, SESSIONS)
    assert result[EXTRA[0]].iloc[1:].tolist() == [1, 0, 1]
    assert result[EXTRA[1]].iloc[1:].tolist() == [1, 0, 1]
    assert result[EXTRA].iloc[0].isna().all()


def test_missing_exact_previous_session_does_not_use_last_observed_appearance():
    data = cohort([
        ("2026-09-30", "002058", 1, 2),
        ("2026-10-09", "002058", 0, 3),
    ])
    assert add_phase(data, SESSIONS)[EXTRA].isna().all().all()


@pytest.mark.parametrize("prior,current,prior_boards,current_boards,known_prior", [
    (np.nan, 0, 3, 4, False), (2, 0, 3, 4, False),
    (1, 0, 3, 5, False), (1, 0, np.nan, 4, False),
    (1, np.nan, 3, 4, True), (1, 2, 3, 4, True),
])
def test_unknown_identity_and_broken_board_continuity_remain_unknown(
        prior, current, prior_boards, current_boards, known_prior):
    data = cohort([("2026-09-30", "600241", prior, prior_boards),
                   ("2026-10-08", "600241", current, current_boards)])
    row = add_phase(data, SESSIONS).iloc[1]
    assert pd.isna(row[EXTRA[1]])
    assert (row[EXTRA[0]] == 1) if known_prior else pd.isna(row[EXTRA[0]])


def test_full_candidates_and_unknown_first_board_history_are_preserved():
    data = cohort([("2026-10-08", "600241", 0, 4),
                   ("2026-09-30", "600241", 1, 3),
                   ("2026-10-08", "603928", 0, 2)])
    result = add_phase(data, SESSIONS)
    pd.testing.assert_frame_equal(result[data.columns], data)
    assert len(result) == 3
    assert result[EXTRA].iloc[2].isna().all()
    assert result[EXTRA[1]].iloc[0] == 1


def test_future_labels_and_future_identity_do_not_change_past_phase():
    data = cohort([("2026-09-30", "600241", 1, 3),
                   ("2026-10-08", "600241", 0, 4),
                   ("2026-10-09", "600241", 0, 5)])
    data["target"] = [1, 0, 0]
    expected = add_phase(data, SESSIONS)[EXTRA].iloc[:2]
    data["target"] = [0, 1, 1]
    data.loc[2, "signal_one_word"] = 1
    pd.testing.assert_frame_equal(add_phase(data, SESSIONS)[EXTRA].iloc[:2], expected)


def test_duplicate_or_nonofficial_cohort_and_invalid_session_order_are_refused():
    data = cohort([("2026-10-08", "600241", 0, 4)])
    with pytest.raises(ValueError, match="unique"):
        add_phase(pd.concat([data, data]), SESSIONS)
    with pytest.raises(ValueError, match="unique"):
        add_phase(data, SESSIONS[::-1])
    data.loc[0, "date"] = "2026-10-01"
    with pytest.raises(ValueError, match="official"):
        add_phase(data, SESSIONS)
