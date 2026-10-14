import json

import pandas as pd

from relay_path_forward_v244 import _sha, freeze, snapshot, record, verify


def test_v244_freeze_snapshot_record_and_verify(tmp_path):
    expectation = tmp_path / "v242.py"; expectation.write_text("RULE=1\n", encoding="utf-8")
    competition = tmp_path / "v243.py"; competition.write_text("RULE=1\n", encoding="utf-8")
    matrix = tmp_path / "path_expectation_matrix.csv"
    pd.DataFrame([{"code": "603230", "name": "内蒙新华", "boards": 4,
                   "expectation_type": "DIVERGENCE_ABSORPTION_CONTINUATION",
                   "normal_gap_low": 0, "normal_gap_high": 5,
                   "predeclared_role": "PRIMARY_EXPECTED_WINNER"}]).to_csv(matrix, index=False)
    scenarios = tmp_path / "scenario_tree.csv"; pd.DataFrame([{"scenario": "x"}]).to_csv(scenarios, index=False)
    baseline = tmp_path / "report.json"
    baseline.write_text(json.dumps({"version": "v242-relay-path-expectation-1",
        "signal_date": "2026-09-21", "primary_close_expectation": {"code": "603230"},
        "artifact_hashes": {matrix.name: _sha(matrix), scenarios.name: _sha(scenarios)}}), encoding="utf-8")
    registry = tmp_path / "registry"
    frozen = freeze(str(registry), str(expectation), str(competition), str(baseline))
    snap = snapshot(str(registry), str(matrix), str(scenarios), str(baseline))
    stage = tmp_path / "stage.json"
    stage.write_text(json.dumps({"version": "v243-relay-stage-competition-1",
        "signal_date": "2026-09-21", "decision_date": "2026-09-22",
        "decision": "DATA_UNAVAILABLE"}), encoding="utf-8")
    record(str(registry), str(stage))
    assert snap["full_denominator_count"] == 1
    assert frozen["policy"]["community_sources_are_hypotheses_only"]
    assert verify(str(registry))["status"] == "PASS"
