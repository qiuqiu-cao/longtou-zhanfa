import numpy as np
import pandas as pd
import pytest
import first_board_emotion_forecast as m


def test_daily_target_keeps_unknown_cohort_denominator_and_uses_no_stock_labels_as_features():
    rows = pd.DataFrame({"signal_date": ["2024-01-02"] * 20 + ["2024-01-03"] * 3,
        "code": [str(x) for x in range(23)], "p": [1] * 20 + [2] * 3,
        "label_end": ["2024-01-03"] * 20 + ["2024-01-04"] * 3,
        m.stage.TARGET: [1.] * 5 + [0.] * 14 + [np.nan] + [1., 0., np.nan],
        "previous_first_promotion": [.2] * 20 + [.25] * 3})
    context = pd.DataFrame({"signal_date": ["2024-01-02", "2024-01-03"],
                            **{f: [1., 2.] for f in m.features.FIELDS}})
    daily = m.macro_table(rows, context)
    assert daily.members.tolist() == [20, 3]
    assert daily.known.tolist() == [19, 2]
    assert daily.loc[0, m.TARGET] == 5/19
    assert pd.isna(daily.loc[1, m.TARGET])
    assert daily[m.features.FIELDS].equals(context[m.features.FIELDS])


def test_macro_window_is_official_positions_not_252_known_dates_and_end_is_strict():
    sessions = pd.bdate_range("2023-01-02", periods=320).strftime("%Y-%m-%d").tolist()
    daily = pd.DataFrame(dict(signal_date=sessions[:-1], p=np.arange(319), label_end=sessions[1:],
                             **{m.TARGET: np.ones(319)}))
    daily.loc[100:125, m.TARGET] = np.nan
    cut = sessions[300]
    keep = m.mature_members(daily, cut, sessions)
    assert keep.p.min() == 48 and keep.p.max() == 298
    assert keep.label_end.lt(cut).all() and len(keep) == 225
    changed = daily.copy()
    changed.loc[changed.label_end.ge(cut), m.TARGET] = 999
    pd.testing.assert_frame_equal(keep, m.mature_members(changed, cut, sessions))


def test_cold_start_is_nan_with_every_requested_date_preserved(tmp_path):
    sessions = pd.bdate_range("2024-01-01", periods=10).strftime("%Y-%m-%d").tolist()
    daily = pd.DataFrame(dict(signal_date=sessions[:-1], p=np.arange(9), label_end=sessions[1:],
        members=1, known=1, **{m.TARGET: np.ones(9)}, **{f: np.zeros(9) for f in m.features.FIELDS}))
    cut = sessions[5]
    plan = [m.cut_description(daily, cut, sessions)]
    emotion = m.ForwardEmotion(daily, sessions, tmp_path, plan)
    result = emotion.predict_at(cut, sessions[5:8])
    assert result.signal_date.tolist() == sessions[5:8]
    assert result[m.FIELD].isna().all()
    assert not list(tmp_path.glob("*/ridge.joblib"))
    with pytest.raises(ValueError, match="precede"):
        emotion.predict_at(cut, [sessions[4]])
    # A fully receipted cold start may be loaded without fitting anything.
    pd.testing.assert_frame_equal(result, m.ForwardEmotion(daily, sessions, tmp_path, plan).predict_at(cut, sessions[5:8]))


def test_unfinished_macro_cut_cannot_restart(tmp_path):
    sessions = ["2024-01-01", "2024-01-02"]
    daily = pd.DataFrame(dict(signal_date=[sessions[0]], p=[0], label_end=[sessions[1]], members=[1], known=[1],
        **{m.TARGET: [0.]}, **{f: [1.] for f in m.features.FIELDS}))
    cut = sessions[1]
    (tmp_path / cut).mkdir()
    manager = m.ForwardEmotion(daily, sessions, tmp_path, [m.cut_description(daily, cut, sessions)])
    with pytest.raises(FileExistsError, match="Unfinished"):
        manager.model_at(cut)


class FakeClassifier:
    def predict_proba(self, rows):
        p = np.linspace(.3, .7, len(rows))
        return np.column_stack([1-p, p])


class FakePlatt:
    coef_ = np.array([[1.]])

    def __init__(self, **kwargs):
        pass

    def fit(self, x, y):
        return self


class FakeEmotion:
    def __init__(self):
        self.calls = []
        self.allowed = {}

    def oof(self, rows):
        return pd.DataFrame({"signal_date": sorted(rows.signal_date.unique()), m.FIELD: .15})

    def predict_at(self, cut, dates):
        self.calls.append((cut, dates))
        self.allowed[cut] = {"max_label_end": "2022-12-31"}
        return pd.DataFrame({"signal_date": dates, m.FIELD: .42, "macro_model_cut": cut})


def test_stock_calibration_uses_its_own_frozen_macro_prefix(monkeypatch, tmp_path):
    dates = pd.bdate_range("2023-01-02", periods=151).strftime("%Y-%m-%d").tolist()
    train = pd.DataFrame(dict(signal_date=dates[:-1], code="000001", stage=1, label_end=dates[1:],
                              **{m.stage.TARGET: np.arange(150) % 2}))
    score = pd.DataFrame(dict(signal_date=["2024-01-02", "2024-06-28"], code="000001", stage=1))
    captured = []
    monkeypatch.setattr(m.stage, "fit_core", lambda rows, fields, out, name: (
        captured.append((name, rows.copy())) or FakeClassifier()))
    monkeypatch.setattr(m, "LogisticRegression", FakePlatt)
    monkeypatch.setattr(m.joblib, "dump", lambda *args: None)
    emotion = FakeEmotion()
    bundle, scored = m.fit_stock_head(train, score, [m.FIELD], tmp_path, "nested", emotion)
    cut = dates[108]
    assert emotion.calls[0][0] == cut
    assert emotion.calls[0][1] == dates[108:150]
    assert emotion.calls[1][0] == "2024-01-01"  # all H1 uses this one head
    assert captured[0][1].label_end.lt(cut).all()
    assert captured[0][1][m.FIELD].eq(.15).all()  # forward OOF, not cal-head output
    assert captured[1][1][m.FIELD].eq(.15).all()
    cal = pd.read_csv(tmp_path / "models/nested_calibration_rows.csv")
    assert cal[m.FIELD].eq(.42).all() and len(cal) == 42
    assert scored[m.FIELD].eq(.42).all() and bundle["train_max_label_end"] < "2024-01-01"


def test_macro_accuracy_requires_same_dates_and_preserves_missing_day():
    dates = pd.bdate_range("2024-01-01", periods=20).strftime("%Y-%m-%d").tolist()
    daily = pd.DataFrame(dict(signal_date=dates, label_end=dates, members=10, known=10, outcome_coverage=1.,
        **{m.TARGET: np.zeros(20)}, naive_last_mature_promotion=.4))
    predictions = pd.DataFrame({"signal_date": dates, m.FIELD: [0.] * 19 + [np.nan]})
    paired, report = m.score_forecast(predictions, daily)
    assert len(paired) == 20 and report["paired_days"] == 19 and report["unknown_or_unpaired"] == 1
    assert report["pass_emotion_validation"]
    predictions.loc[18, m.FIELD] = np.nan
    _, report = m.score_forecast(predictions, daily)
    assert not report["pass_emotion_validation"]
