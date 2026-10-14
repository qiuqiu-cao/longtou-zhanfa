import pandas as pd

from relay_promotion_walkforward_v257 import FOLDS, _random_top, split_fold


def test_fold_purges_rows_whose_label_is_after_training_freeze():
    rows = pd.DataFrame({
        "date": ["2024-06-28", "2024-06-30", "2024-07-01", "2024-07-01"],
        "trade_date": ["2024-07-01", "2024-07-02", "2024-07-02", "2024-07-02"],
        "promoted_next_day": [1, 0, 1, 0],
        "promotion_label_status": [
            "verified_from_next_daily_limit_close",
            "verified_from_next_daily_limit_close",
            "verified_from_next_daily_limit_close",
            "verified_from_next_daily_limit_close",
        ],
    })
    train, test, complete = split_fold(rows, FOLDS[0])
    assert train.empty  # both pre-cutoff rows label after the cutoff
    assert len(test) == 2
    assert complete.loc["2024-07-01"]


def test_incomplete_label_date_is_excluded_only_from_evaluation():
    rows = pd.DataFrame({
        "date": ["2024-07-01", "2024-07-01", "2024-07-02"],
        "trade_date": ["2024-07-02"] * 3,
        "promoted_next_day": [1, None, 0],
        "promotion_label_status": [
            "verified_from_next_daily_limit_close", "unavailable",
            "verified_from_next_daily_limit_close",
        ],
    })
    _, test, complete = split_fold(rows, FOLDS[0])
    assert len(test) == 3
    assert not complete.loc["2024-07-01"]
    assert complete.loc["2024-07-02"]


def test_random_control_is_stable_and_selects_one_row_per_date():
    rows = pd.DataFrame({
        "date": ["2024-07-01"] * 3,
        "code": ["000003", "000001", "000002"],
        "promoted_next_day": [0, 1, 0],
    })
    first = _random_top(rows)
    second = _random_top(rows)
    assert len(first) == 1
    assert first.iloc[0].code == second.iloc[0].code
