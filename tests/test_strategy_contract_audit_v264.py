from strategy_contract_audit_v264 import audit


def test_contract_audit_is_fail_closed_when_execution_and_theme_are_missing():
    report = {
        "status": "DIAGNOSTIC_ONLY",
        "version": "v263",
        "aggregate": {},
        "data_audit": {"theme_status": "UNAVAILABLE_IN_FEATURE_SCHEMA"},
        "interpretation": {
            "actual_fill_count": 0,
            "is_pristine_blind_test": False,
            "survivorship_risk": True,
        },
    }
    result = audit(report)
    assert result["live_ready"] is False
    assert result["decision"] == "CASH"
    statuses = {row["requirement"]: row["status"] for row in result["requirements"]}
    assert statuses["verified topic-leader label"] == "UNAVAILABLE"
    assert statuses["actual fills and executable win rate"] == "UNAVAILABLE"
