import json

import pandas as pd

from same_height_pk_forward_v226 import freeze, snapshot, verify


def test_v226_snapshot_is_strict_json_and_chain_verifies(tmp_path):
    script = tmp_path / "model.py"
    script.write_text("# frozen\n", encoding="utf-8")
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "version": "v225-same-height-leader-pk-1", "signal_date": "2026-09-21",
        "group_descriptive_leaders": [{
            "height_group_id": "2026-09-21:H4", "boards": 4,
            "descriptive_leader_code": "603230", "descriptive_leader_name": "内蒙新华",
            "score_margin": 2.0, "official_champion": None, "winner_claimed": False,
        }],
    }, ensure_ascii=False), encoding="utf-8")
    scores = tmp_path / "scores.csv"
    pd.DataFrame([{
        "height_group_id": "2026-09-21:H4", "code": "603230", "name": "内蒙新华",
        "boards": 4, "research_score_NOT_PROBABILITY": 16,
        "board_path_status": "QUICK_REPAIR", "announcement_risk_tier": "TITLE_ONLY",
        "symbolic_tags": "华|新", "missing_evidence": float("nan"),
    }]).to_csv(scores, index=False)
    registry = tmp_path / "registry"
    freeze(str(registry), str(script), str(report))
    payload = snapshot(str(registry), str(scores), str(report))
    target = registry / "snapshots" / payload["freeze_id"] / "2026-09-21.json"
    text = target.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert payload["groups"][0]["members"][0]["missing_evidence"] is None
    assert verify(str(registry))["status"] == "PASS"
