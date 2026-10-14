import json

import pandas as pd
import pytest

from first_board_theme_forward_v220 import freeze, snapshot, verify


def test_forward_snapshot_freezes_one_candidate_and_forbids_replacement(tmp_path):
    artifact = tmp_path / "model.json"
    artifact.write_text("{}", encoding="utf-8")
    registry = tmp_path / "registry"
    freeze(registry, "2026-09-18", [artifact])
    watchlist = tmp_path / "watch.csv"
    pd.DataFrame([
        {"signal_date": "2026-09-21", "code": "600001", "name": "甲",
         "theme_research_gate": True, "close_quality_gate": True,
         "best_theme_lane": "OBSERVED_THEME_ROSTER", "best_theme_name": "医药",
         "community_probability_not_calibrated": .3, "entry_authorized": False},
        {"signal_date": "2026-09-21", "code": "600002", "name": "乙",
         "theme_research_gate": True, "close_quality_gate": True,
         "best_theme_lane": "OBSERVED_THEME_ROSTER", "best_theme_name": "地产",
         "community_probability_not_calibrated": .2, "entry_authorized": False},
    ]).to_csv(watchlist, index=False)
    result = snapshot(registry, watchlist, "2026-09-21", "2026-09-21T23:40:00+08:00")
    assert result["candidate"]["code"] == "600001"
    assert result["candidate"]["replacement_after_cancel"] is False
    assert verify(registry)["status"] == "PASS"
    with pytest.raises(ValueError, match="duplicate"):
        snapshot(registry, watchlist, "2026-09-21", "2026-09-21T23:50:00+08:00")


def test_frozen_artifact_change_is_detected(tmp_path):
    artifact = tmp_path / "model.json"
    artifact.write_text(json.dumps({"v": 1}), encoding="utf-8")
    registry = tmp_path / "registry"
    freeze(registry, "2026-09-18", [artifact])
    artifact.write_text(json.dumps({"v": 2}), encoding="utf-8")
    watchlist = tmp_path / "watch.csv"
    pd.DataFrame(columns=[
        "signal_date", "code", "name", "theme_research_gate", "close_quality_gate",
        "best_theme_lane", "best_theme_name", "community_probability_not_calibrated",
        "entry_authorized",
    ]).to_csv(watchlist, index=False)
    with pytest.raises(ValueError, match="changed or missing"):
        snapshot(registry, watchlist, "2026-09-21", "2026-09-21T23:40:00+08:00")
