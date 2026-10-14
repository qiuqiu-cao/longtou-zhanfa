import json

import pandas as pd

from relay_cross_theme_forward_v241 import freeze, snapshot, record_confirmation, settle, verify


def _artifacts(tmp_path):
    expectation = tmp_path / "expectation.py"; expectation.write_text("RULE=1\n", encoding="utf-8")
    confirmation = tmp_path / "confirmation.py"; confirmation.write_text("RULE=1\n", encoding="utf-8")
    pool = tmp_path / "pool.csv"
    pd.DataFrame([
        {"code": "603230", "name": "内蒙新华", "boards": 4, "theme_group_id": "media",
         "theme_name": "传媒", "predeclared_role": "PRIMARY_EXPECTED_WINNER",
         "winner_hierarchy_rank": 1, "tradable_expectation_eligible": True},
        {"code": "000504", "name": "南华生物", "boards": 3, "theme_group_id": "medicine",
         "theme_name": "医药", "predeclared_role": "FULL_DENOMINATOR_WATCH_ONLY",
         "winner_hierarchy_rank": float("nan"), "tradable_expectation_eligible": False},
    ]).to_csv(pool, index=False)
    tree = tmp_path / "tree.csv"
    pd.DataFrame([{"code": "603230", "hierarchy_rank": 1}]).to_csv(tree, index=False)
    from relay_cross_theme_forward_v241 import _sha
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "version": "v239-relay-cross-theme-expectation-1", "signal_date": "2026-09-21",
        "artifact_hashes": {pool.name: _sha(pool), tree.name: _sha(tree)},
        "primary_expected_winner": {"code": "603230", "name": "内蒙新华"},
        "conditional_fallback_winners": [],
    }, ensure_ascii=False), encoding="utf-8")
    return expectation, confirmation, pool, tree, report


def test_freeze_snapshot_confirmation_and_settlement_keep_full_denominator(tmp_path):
    expectation, confirmation, pool, tree, report = _artifacts(tmp_path)
    registry = tmp_path / "registry"
    frozen = freeze(str(registry), str(expectation), str(confirmation), str(report))
    snap = snapshot(str(registry), str(pool), str(tree), str(report))
    snap_path = registry / "snapshots" / frozen["freeze_id"] / "2026-09-21.json"
    open_report = tmp_path / "open.json"
    open_report.write_text(json.dumps({
        "version": "v240-relay-cross-theme-confirmation-1", "signal_date": "2026-09-21",
        "decision_date": "2026-09-22", "decision": "PRIMARY_EXPECTATION_CONFIRMED",
        "conditional_winner": {"code": "603230", "name": "内蒙新华"},
    }, ensure_ascii=False), encoding="utf-8")
    record_confirmation(str(registry), str(open_report))
    next_pool = tmp_path / "next.csv"
    pd.DataFrame([{"code": "000504", "boards": 4,
                   "pool_assignment": "CLOSING_LIMIT_UP"}]).to_csv(next_pool, index=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"full_market_completeness": "verified"}), encoding="utf-8")
    result = settle(str(registry), str(snap_path), str(next_pool), str(manifest), str(open_report))
    assert snap["full_denominator_count"] == 2
    assert result["unexpected_promoted_codes"] == ["000504"]
    assert result["primary_expected_winner_promoted"] is False
    assert verify(str(registry))["status"] == "PASS"


def test_frozen_artifact_tampering_is_detected(tmp_path):
    expectation, confirmation, _, _, report = _artifacts(tmp_path)
    registry = tmp_path / "registry"
    freeze(str(registry), str(expectation), str(confirmation), str(report))
    expectation.write_text("RULE=2\n", encoding="utf-8")
    assert verify(str(registry))["status"] == "FAIL"
