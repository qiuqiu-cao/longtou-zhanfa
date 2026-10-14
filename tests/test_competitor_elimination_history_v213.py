import json
from pathlib import Path

import pandas as pd

from competitor_elimination_history_v213 import build


def write_pool(root: Path, date: str, rows):
    folder = root / "results" / f"current_full_pool_{date.replace('-', '')}_v1"
    folder.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(folder / "all_closing_and_failed_candidates.csv", index=False)
    (folder / "manifest.json").write_text(json.dumps({
        "complete_within_archived_roster": True,
        "full_market_completeness": "verified_full_market"}), encoding="utf-8")


def test_attempted_height_uses_immediate_previous_session_only(tmp_path):
    write_pool(tmp_path, "2026-01-05", [
        {"signal_date": "2026-01-05", "code": "1", "boards": 1,
         "pool_assignment": "CLOSING_LIMIT_UP"}])
    write_pool(tmp_path, "2026-01-06", [
        {"signal_date": "2026-01-06", "code": "2", "boards": 2,
         "pool_assignment": "CLOSING_LIMIT_UP"},
        {"signal_date": "2026-01-06", "code": "1", "boards": 0,
         "pool_assignment": "FAILED_BOARD_OBSERVATION"}])
    candidates = pd.DataFrame([{"signal_date": "2026-01-06", "code": "2",
                                "name": "B", "boards": 2}])
    rows, report, _ = build(tmp_path, candidates, ["2026-01-05", "2026-01-06"])
    assert rows.loc[0, "competitor_elimination"] == .5
    assert rows.loc[0, "competitor_elimination_evidence_status"] == "POINT_IN_TIME_VERIFIED"
    assert report["formal_point_in_time_rows"] == 1
