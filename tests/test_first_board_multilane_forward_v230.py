import json

import pandas as pd

from first_board_multilane_forward_v230 import freeze, snapshot, verify


def test_freeze_snapshot_and_verify(tmp_path):
    script = tmp_path / "model.py"
    script.write_text("MODEL = 1\n", encoding="utf-8")
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "version": "v229-first-board-multilane-identity-1",
        "signal_date": "2026-09-21",
        "narrative_lane_candidate": {"code": "000001", "name": "甲"},
        "symbolic_pure_buyable_pioneer": {"code": "000002", "name": "乙"},
        "symbolic_dual_identity_candidate": None,
    }, ensure_ascii=False), encoding="utf-8")
    candidates = tmp_path / "candidates.csv"
    pd.DataFrame([
        {"code": "000001", "name": "甲", "primary_business_theme": "医药",
         "symbolic_theme_name": "华字辈", "multilane_role": "DUAL",
         "first_seal_minute": 20},
        {"code": "000002", "name": "乙", "primary_business_theme": "无",
         "symbolic_theme_name": "华字辈", "multilane_role": "PURE",
         "first_seal_minute": 18},
    ]).to_csv(candidates, index=False)
    registry = tmp_path / "registry"
    frozen = freeze(str(registry), str(script), str(report))
    snap = snapshot(str(registry), str(candidates), str(report))
    assert frozen["freeze_id"] == snap["freeze_id"]
    assert len(snap["role_candidates"]) == 3
    assert snap["role_candidates"][2]["candidate"] is None
    assert verify(str(registry))["status"] == "PASS"


def test_verify_detects_snapshot_tampering(tmp_path):
    script = tmp_path / "model.py"; script.write_text("x=1", encoding="utf-8")
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "version": "v229-first-board-multilane-identity-1",
        "signal_date": "2026-09-21",
        "narrative_lane_candidate": None,
        "symbolic_pure_buyable_pioneer": None,
        "symbolic_dual_identity_candidate": None,
    }), encoding="utf-8")
    candidates = tmp_path / "candidates.csv"
    pd.DataFrame(columns=["code", "name"]).to_csv(candidates, index=False)
    registry = tmp_path / "registry"
    freeze(str(registry), str(script), str(report))
    snap = snapshot(str(registry), str(candidates), str(report))
    path = registry / "snapshots" / snap["freeze_id"] / "2026-09-21.json"
    path.write_text("{}", encoding="utf-8")
    assert verify(str(registry))["status"] == "FAIL"
