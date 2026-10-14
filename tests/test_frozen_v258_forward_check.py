import pandas as pd
import pytest
import json

import frozen_v258_forward_check as replay
from frozen_v258_forward_check import TRAIN_END_EXCLUSIVE, audit_target_calendar


def test_forward_check_training_cutoff_is_before_target_date():
    assert TRAIN_END_EXCLUSIVE == pd.Timestamp("2026-01-01")


def calendar():
    return pd.DataFrame({"date": ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04"],
        "official_query_success": [True]*4, "provider_cross_section_complete": [True]*4})


def candidates():
    return pd.DataFrame({"date": ["2026-09-01"]*2, "trade_date": ["2026-09-04"]*2,
        "code": ["000001", "000002"], "boards": [2, 2], "one_word_proxy": [0, 0],
        "state_divergence": [1, 1], "promoted_next_day": [0., 1.],
        "promotion_label_status": ["verified_from_next_daily_limit_close"]*2})


def test_skipped_session_labels_are_unknown_not_failures():
    result = audit_target_calendar(candidates(), calendar())
    assert result.expected_trade_date.eq("2026-09-02").all()
    assert result.calendar_label_status.eq("WRONG_SESSION_LABEL").all()
    assert result.promoted_next_day.isna().all()
    assert not result.evaluation_label_valid.any()


def test_matching_session_keeps_labels_and_boundary_is_unknown():
    rows = candidates()
    rows["trade_date"] = "2026-09-02"
    assert audit_target_calendar(rows, calendar()).evaluation_label_valid.all()
    rows["date"] = "2026-09-04"
    assert audit_target_calendar(rows, calendar()).promoted_next_day.isna().all()


def test_missing_calendar_evidence_must_not_be_skipped():
    cal = calendar()
    cal.loc[1, "provider_cross_section_complete"] = False
    with pytest.raises(ValueError, match="incomplete calendar"):
        audit_target_calendar(candidates(), cal)


def test_run_writes_invalid_labels_and_does_not_replace_model_on_divergence(tmp_path, monkeypatch):
    rows = candidates()
    train = rows.copy()
    train["date"], train["trade_date"] = "2025-12-29", "2025-12-30"
    source, cal_path = tmp_path / "input.csv", tmp_path / "calendar.csv"
    pd.concat([train, rows]).to_csv(source, index=False)
    calendar().to_csv(cal_path, index=False)
    monkeypatch.setattr(replay, "load_rows", lambda p: (pd.read_csv(p, dtype={"code":str}), {}))
    monkeypatch.setattr(replay, "_fit", lambda tr, te, fields: [0.1, 0.9])
    out = tmp_path / "out"
    result = replay.run(str(source), "2026-09-01", str(out), str(cal_path))
    assert result["status"] == "LABEL_INVALID_DIAGNOSTIC_ONLY"
    assert result["interpretation"]["is_forward_test"] is False
    selected = result["selections"]["two_board:non_one_price"]["identity_model"][0]
    assert selected["code"] == "000002"  # not the code-order fallback
    assert selected["promoted_next_day"] is None
    assert json.loads((out / "report.json").read_text(encoding="utf-8"))["target_date_complete"] is False
    with pytest.raises(FileExistsError):
        replay.run(str(source), "2026-09-01", str(out), str(cal_path))
