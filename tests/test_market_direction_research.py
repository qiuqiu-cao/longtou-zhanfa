"""Synthetic direction contracts; no historical observations or outcome files."""
from unittest.mock import patch
import json
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

from market_direction_research import direction_forecasts, label_only, metrics


CHANNELS = [
    "sh_a_return_pct", "sz_a_return_pct", "limit_count", "down_limit_count",
    "failed_rate", "max_board", "relay_promotion_rate", "yesterday_premium",
]
ARMS = ["LR120", "KNN120", "LR120_NEGATIVE2", "LR252"]
SCHEMA = [
    "signal_date", "target_date", "arm", "status", "positive_probability",
    "predicted_positive", "train_n", "max_training_target_date",
    "neighbor_end_dates_json",
]


def synthetic(n=360, constant=False):
    """An explicitly synthetic exchange grid, with a holiday-sized date gap."""
    dates = pd.bdate_range("2023-01-02", periods=n + 5).strftime("%Y-%m-%d").tolist()
    sessions = dates[:45] + dates[50:]
    t = np.arange(n, dtype=float)
    if constant:
        values = np.tile([0.2, 0.1, 45.0, 4.0, 0.2, 5.0, 0.35, 0.01], (n, 1))
    else:
        values = np.column_stack([
            np.sin(t / 7), np.cos(t / 9), 45 + (t % 17),
            1 + (t % 7), 0.1 + (t % 6) / 40,
            3 + (t % 5), 0.2 + (t % 8) / 20,
            np.where(t % 3 == 0, 0.0, np.where(t % 3 == 1, 0.012, -0.009)),
        ])
    return pd.DataFrame(values, index=sessions, columns=CHANNELS), sessions


def one(obs, sessions, pos, arm):
    result = direction_forecasts(obs, sessions, [pos], [arm])
    assert len(result) == 1
    assert set(SCHEMA).issubset(result.columns)
    return result.iloc[0]


@pytest.mark.parametrize("arm", ARMS)
def test_future_observations_cannot_change_prior_prediction(arm):
    obs, sessions = synthetic()
    pos = 130
    before = direction_forecasts(obs, sessions, [pos], [arm])
    altered = obs.copy()
    altered.iloc[pos + 1:] = np.tile(
        [-900.0, 900.0, 1.0, 99.0, 0.99, 30.0, 0.99, -0.9],
        (len(obs) - pos - 1, 1),
    )
    after = direction_forecasts(altered, sessions, [pos], [arm])
    pd.testing.assert_frame_equal(before[SCHEMA], after[SCHEMA])


@pytest.mark.parametrize("arm", ARMS)
def test_training_labels_end_at_or_before_signal_and_target_is_official_successor(arm):
    obs, sessions = synthetic()
    result = direction_forecasts(obs, sessions, [100, 130, 300], [arm])
    assert len(result) == 3
    for pos, row in zip([100, 130, 300], result.itertuples()):
        assert row.signal_date == sessions[pos]
        assert row.target_date == sessions[pos + 1]
        assert row.max_training_target_date == sessions[pos]
        assert row.max_training_target_date <= row.signal_date
        if arm == "KNN120":
            neighbors = json.loads(row.neighbor_end_dates_json)
            assert len(neighbors) == 21
            assert len(set(neighbors)) == 21
            assert all(d <= row.signal_date for d in neighbors)


@pytest.mark.parametrize("arm,window", [("LR120", 120), ("KNN120", 120),
                                         ("LR120_NEGATIVE2", 120), ("LR252", 252)])
def test_window_counts_use_exact_target_date_window_without_borrowing_future(arm, window):
    obs, sessions = synthetic()
    early = one(obs, sessions, 59, arm)
    assert early.train_n == 59
    assert early.status == "INSUFFICIENT_HISTORY"
    assert pd.isna(early.positive_probability)
    assert pd.isna(early.predicted_positive)
    sufficient = one(obs, sessions, 60, arm)
    assert sufficient.train_n == 60
    assert sufficient.status == "KNOWN_DIRECTION"
    mature = one(obs, sessions, 300, arm)
    assert mature.train_n == window
    if arm == "KNN120":
        neighbors = json.loads(mature.neighbor_end_dates_json)
        assert all(sessions[300 + 1 - window] <= d <= sessions[300] for d in neighbors)


@pytest.mark.parametrize("arm", ARMS)
def test_current_all_missing_stays_unknown_with_sufficient_past(arm):
    obs, sessions = synthetic()
    pos = 130
    obs.iloc[pos] = np.nan
    row = one(obs, sessions, pos, arm)
    assert row.status == "UNKNOWN_CURRENT_LEVELS"
    assert pd.isna(row.positive_probability)
    assert pd.isna(row.predicted_positive)
    assert json.loads(row.neighbor_end_dates_json) == []


@pytest.mark.parametrize("arm", ["LR120", "LR120_NEGATIVE2", "LR252"])
@pytest.mark.parametrize("value,positive", [(0.0, False), (-0.01, False), (0.01, True)])
def test_lr_one_class_uses_laplace_training_prior_without_fitting(arm, value, positive):
    obs, sessions = synthetic()
    pos = 100
    obs["yesterday_premium"] = value
    # Unknown labels are excluded; literal zero is a valid nonpositive label.
    missing_ends = [12, 26, 47, 70]
    obs.loc[[sessions[i] for i in missing_ends], "yesterday_premium"] = np.nan
    with patch.object(LogisticRegression, "fit", side_effect=AssertionError("single-class LR fit forbidden")):
        row = one(obs, sessions, pos, arm)
    n = pos - len(missing_ends)
    assert row.train_n == n
    assert row.status == "ONE_CLASS_TRAIN_PRIOR"
    expected = (n + 1) / (n + 2) if positive else 1 / (n + 2)
    assert row.positive_probability == pytest.approx(expected)
    assert bool(row.predicted_positive) is positive
    assert row.max_training_target_date == sessions[pos]
    assert json.loads(row.neighbor_end_dates_json) == []


@pytest.mark.parametrize("arm", ARMS)
def test_missing_labels_never_count_as_nonpositive_or_get_imputed(arm):
    obs, sessions = synthetic()
    pos = 100
    # 41 unavailable targets leave only 59 finite labels despite a 100-day history.
    obs.loc[sessions[1:42], "yesterday_premium"] = np.nan
    row = one(obs, sessions, pos, arm)
    assert row.train_n == 59
    assert row.status == "INSUFFICIENT_HISTORY"
    assert pd.isna(row.positive_probability)


@pytest.mark.parametrize("arm", ARMS)
def test_unknown_source_row_and_unknown_label_are_separate_exclusions(arm):
    obs, sessions = synthetic()
    pos = 100
    # End 20 has no label; end 21 has a known label but its source (day 20) is all unknown.
    obs.iloc[20] = np.nan
    row = one(obs, sessions, pos, arm)
    assert row.train_n == 98
    if arm == "KNN120":
        neighbors = json.loads(row.neighbor_end_dates_json)
        assert sessions[20] not in neighbors
        assert sessions[21] not in neighbors


def test_knn_equal_distance_ties_choose_earliest_training_target_dates():
    obs, sessions = synthetic(constant=True)
    pos = 130
    row = one(obs, sessions, pos, "KNN120")
    assert row.status == "KNOWN_DIRECTION"
    assert row.train_n == 120
    assert json.loads(row.neighbor_end_dates_json) == sessions[11:32]
    assert row.positive_probability == pytest.approx(1.0)
    assert bool(row.predicted_positive)


def test_holiday_gap_does_not_change_exact_official_successor():
    obs, sessions = synthetic()
    # The first synthetic holiday is too early for a forecast, so shift an artificial
    # holiday gap into a mature grid while keeping one row for every listed session.
    revised = sessions[:131] + [(pd.Timestamp(d) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
                               for d in sessions[131:]]
    obs.index = revised
    row = one(obs, revised, 130, "LR120")
    assert row.target_date == revised[131]
    assert row.target_date != (pd.Timestamp(row.signal_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


@pytest.mark.parametrize("kind", ["missing_grid_row", "duplicate_session", "reversed_grid", "extra_channel"])
def test_invalid_official_grid_or_channel_contract_is_rejected(kind):
    obs, sessions = synthetic()
    if kind == "missing_grid_row":
        obs = obs.drop(index=sessions[50])
    elif kind == "duplicate_session":
        sessions = sessions[:51] + [sessions[50]] + sessions[51:]
    elif kind == "reversed_grid":
        obs = obs.iloc[::-1]
        sessions = sessions[::-1]
    elif kind == "extra_channel":
        obs["unregistered_future_feature"] = 1.0
    with pytest.raises(ValueError):
        direction_forecasts(obs, sessions, [100], ["LR120"])


def manual_scoring_case():
    """Six registrations: TP, FN, FP at zero, TN, unknown call, unknown truth."""
    dates = pd.bdate_range("2000-01-03", periods=7).strftime("%Y-%m-%d").tolist()
    obs = pd.DataFrame(0.0, index=dates, columns=CHANNELS)
    obs.loc[dates[1:], "yesterday_premium"] = [0.02, 0.01, 0.0, -0.03, 0.04, np.nan]
    predictions = pd.DataFrame({
        "signal_date": dates[:-1], "target_date": dates[1:], "arm": "LR120",
        "status": ["KNOWN_DIRECTION"] * 4 + ["UNKNOWN_CURRENT_LEVELS", "KNOWN_DIRECTION"],
        "positive_probability": [0.8, 0.3, 0.7, 0.2, np.nan, 0.9],
        "predicted_positive": [1.0, 0.0, 1.0, 0.0, np.nan, 1.0],
    })
    return label_only(predictions, obs)


def test_scoring_contract_zero_is_nonpositive_and_unknown_truth_stays_unknown():
    scored = manual_scoring_case()
    assert scored.actual_positive.iloc[2] == 0.0
    assert bool(scored.evaluated.iloc[2])
    assert scored.false_positive.iloc[2] == 1.0
    assert pd.isna(scored.actual_positive.iloc[5])
    assert not bool(scored.evaluated.iloc[5])
    assert pd.isna(scored.correct.iloc[5])
    assert pd.isna(scored.false_positive.iloc[5])


def test_scoring_contract_unknown_prediction_keeps_full_registered_coverage_denominator():
    scored = manual_scoring_case()
    assert len(scored) == 6
    result = metrics(scored)
    assert result["registered_dates"] == 6
    assert result["known_target_dates"] == 5
    assert result["predicted_dates"] == 5
    assert result["scored_dates"] == 4
    assert result["unknown_prediction_dates"] == 1
    assert result["unknown_target_dates"] == 1
    assert result["scoring_coverage"] == pytest.approx(4 / 6)
    assert result["label_coverage"] == pytest.approx(5 / 6)
    assert result["valid_label_scoring_coverage"] == pytest.approx(4 / 5)


def test_scoring_contract_fpr_uses_nonpositive_dates_and_complete_recall_keeps_uncalled_positive():
    result = metrics(manual_scoring_case())
    assert (result["tp"], result["tn"], result["fp"], result["fn"]) == (1, 1, 1, 1)
    assert result["nonpositive_count"] == 2
    assert result["FPR"] == pytest.approx(1 / 2)
    assert result["nonpositive_recall"] == pytest.approx(1 / 2)
    assert result["precision"] == pytest.approx(1 / 2)
    assert result["scored_positive_recall"] == pytest.approx(1 / 2)
    # The fifth row is a known positive with an unknown call, still in this denominator.
    assert result["complete_positive_recall"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("problem,expected", [("warning", "FIT_NONCONVERGENCE"),
                                              ("failure", "FIT_FAILED")])
def test_fit_problem_returns_unknown_instead_of_score(problem, expected):
    obs, sessions = synthetic()

    def fit_problem(*_args, **_kwargs):
        if problem == "warning":
            warnings.warn("synthetic convergence failure", ConvergenceWarning)
            return None
        raise ValueError("synthetic fitting failure")

    with patch.object(LogisticRegression, "fit", side_effect=fit_problem):
        row = one(obs, sessions, 100, "LR120")
    assert row.train_n == 100
    assert row.status == expected
    assert pd.isna(row.positive_probability)
    assert pd.isna(row.predicted_positive)
    assert row.fit_failure and "synthetic" in row.fit_failure
