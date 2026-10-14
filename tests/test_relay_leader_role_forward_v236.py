import json

import pandas as pd

from relay_leader_role_forward_v236 import freeze, snapshot, verify


def _files(tmp_path):
    script = tmp_path / "gate.py"
    script.write_text("RULE = 1\n", encoding="utf-8")
    candidates = tmp_path / "relay_role_candidates.csv"
    pd.DataFrame([
        {"code": "001216", "name": "华瓷股份", "boards": 5,
         "relay_role": "HEIGHT_EMOTION_ANCHOR_NOT_CLOSE_ENTRY",
         "board_path_status": "ONE_PRICE_BUYABILITY_UNPROVEN",
         "security_eligibility_status": "UNAVAILABLE",
         "theme_evidence_status": "INFERRED_OR_LIMITED",
         "close_relay_challenger_eligible": False},
        {"code": "603230", "name": "内蒙新华", "boards": 4,
         "relay_role": "DIVERGENCE_ABSORBED_RELAY_CHALLENGER",
         "board_path_status": "QUICK_REPAIR",
         "security_eligibility_status": "UNAVAILABLE",
         "theme_evidence_status": "INFERRED_OR_LIMITED",
         "close_relay_challenger_eligible": True},
    ]).to_csv(candidates, index=False)
    import hashlib
    digest = hashlib.sha256(candidates.read_bytes()).hexdigest()
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "version": "v235-relay-leader-role-gate-1", "signal_date": "2026-09-21",
        "height_emotion_anchor": {"code": "001216", "name": "华瓷股份"},
        "close_relay_challenger": {"code": "603230", "name": "内蒙新华"},
        "artifact_hashes": {"relay_role_candidates.csv": digest},
    }, ensure_ascii=False), encoding="utf-8")
    return script, candidates, report


def test_freeze_snapshot_and_verify(tmp_path):
    script, candidates, report = _files(tmp_path)
    registry = tmp_path / "registry"
    frozen = freeze(str(registry), str(script), str(report))
    snap = snapshot(str(registry), str(candidates), str(report))
    assert frozen["freeze_id"] == snap["freeze_id"]
    assert snap["role_candidates"][0]["candidate"]["boards"] == 5
    assert snap["role_candidates"][1]["candidate"]["code"] == "603230"
    assert verify(str(registry))["status"] == "PASS"


def test_snapshot_tampering_is_detected(tmp_path):
    script, candidates, report = _files(tmp_path)
    registry = tmp_path / "registry"
    freeze(str(registry), str(script), str(report))
    snap = snapshot(str(registry), str(candidates), str(report))
    path = registry / "snapshots" / snap["freeze_id"] / "2026-09-21.json"
    path.write_text("{}", encoding="utf-8")
    assert verify(str(registry))["status"] == "FAIL"
