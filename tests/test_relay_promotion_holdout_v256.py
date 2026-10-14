import numpy as np
import pandas as pd

from relay_promotion_holdout_v256 import (
    SPECS, FORBIDDEN_MODEL_FIELDS, _fit, _same_height_score, _top,
    heuristic_top, split_rows,
)


def test_purge_future_training_label_and_keep_unknown_predictions():
    rows = pd.DataFrame({
        "date": ["2025-12-30", "2025-12-31", "2026-01-05", "2026-01-05"],
        "trade_date": ["2025-12-31", "2026-01-05", "2026-01-06", "2026-01-06"],
        "promoted_next_day": [1, 1, 0, np.nan],
        "promotion_label_status": ["verified_from_next_daily_limit_close"]*4,
    })
    train, test, complete = split_rows(rows)
    assert list(train.index) == [0]
    assert len(test) == 2  # missing label stays in the prediction universe
    assert not complete.loc["2026-01-05"]


def test_bad_label_status_invalidates_entire_evaluation_date():
    rows = pd.DataFrame({
        "date": ["2026-02-02"]*2,
        "trade_date": ["2026-02-03"]*2,
        "promoted_next_day": [1, 0],
        "promotion_label_status": ["verified_from_next_daily_limit_close", "unavailable"],
    })
    _, test, complete = split_rows(rows)
    assert len(test) == 2
    assert not complete.any()


def test_shared_identity_baseline_is_not_individual_strength():
    rows = pd.DataFrame({
        "date": ["2026-01-05"]*2, "code": ["000001", "000002"],
        "boards": [2, 2], "height_gap": [3, 3], "unique_space": [0, 0],
        "same_height_rivals": [1, 1], "is_space_leader": [0, 0],
        "turnover": [np.nan, 0.05], "amplitude": [3, 1], "one_word_proxy": [0, 0],
    })
    assert _same_height_score(rows).nunique() == 1
    assert _top(rows).iloc[0].code == "000001"
    assert heuristic_top(rows, "low_turnover").iloc[0].code == "000002"


def test_future_columns_cannot_change_prediction_or_selection():
    assert not set(sum(SPECS.values(), [])) & FORBIDDEN_MODEL_FIELDS
    train = pd.DataFrame({"date": ["2025-01-01"]*6,
                          "x": [1, 2, 3, 4, 5, 6], "promoted_next_day": [0, 0, 1, 0, 1, 1]})
    test = pd.DataFrame({"x": [1, 6], "promoted_next_day": [0, 1], "net_return": [-1, 1]})
    before = _fit(train, test, ["x"])
    test["promoted_next_day"] = [1, 0]
    test["net_return"] = [99, -99]
    np.testing.assert_allclose(before, _fit(train, test, ["x"]))
