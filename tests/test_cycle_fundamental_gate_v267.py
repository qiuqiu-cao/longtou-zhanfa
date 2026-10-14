import pandas as pd

from cycle_fundamental_gate_v267 import audit, classify_cleanliness, classify_cycle


def _row(**updates):
    row = {
        "code": "000001",
        "signal_date": "2026-09-21",
        "boards": 3,
        "old_cycle_anchor": True,
        "old_leader_break_date": "2026-09-18",
        "middle_board_clearance_status": "CLEARED",
        "old_cycle_negative_feedback": True,
        "supplement_window": False,
        "new_theme_independent_status": "INDEPENDENT",
        "new_ladder_first_board": True,
        "new_ladder_middle_support": True,
        "new_cycle_breakout_status": "BROKEN",
        "cycle_evidence_asof": "2026-09-21T15:00:00+08:00",
        "fundamental_clean_status": "CLEAN",
        "hard_risk_flags": "NONE",
        "business_relevance": "DIRECT",
        "theme_business_share": 0.35,
        "earnings_quality": "STABLE",
        "cashflow_quality": "STABLE",
        "capital_structure_status": "NORMAL",
        "audit_litigation_status": "NORMAL",
        "fundamental_asof": "2026-09-21T15:00:00+08:00",
    }
    row.update(updates)
    return row


def test_new_cycle_and_clean_context():
    frame = pd.DataFrame([_row()])
    result = audit(frame, asof="2026-09-21T15:00:00+08:00")
    assert result["cycle_fundamental_context_review"]["allowed"] is True
    assert result["row_state_counts"]["cycle"]["NEW_CYCLE_CONFIRMED"] == 1
    assert result["row_state_counts"]["fundamental"]["CLEAN"] == 1
    assert result["trade_authorization"]["allowed"] is False


def test_dirty_overrides_explicit_clean_status():
    row = pd.Series(_row(hard_risk_flags="MAJOR_REDUCTION"))
    assert classify_cleanliness(row) == "DIRTY"


def test_empty_columns_fail_coverage_gate():
    row = _row()
    frame = pd.DataFrame([row]).drop(columns=["fundamental_asof"])
    result = audit(frame)
    assert result["fundamental_cleanliness_review"]["allowed"] is False
    assert result["cycle_fundamental_context_review"]["allowed"] is False


def test_old_cycle_repair_is_not_new_cycle():
    row = pd.Series(_row(
        old_cycle_anchor=True,
        old_cycle_negative_feedback=False,
        middle_board_clearance_status="NOT_CLEARED",
        new_theme_independent_status="DEPENDENT",
        new_ladder_first_board=False,
        new_ladder_middle_support=False,
        new_cycle_breakout_status="NOT_BROKEN",
        supplement_window=False,
    ))
    assert classify_cycle(row) == "OLD_CYCLE_REPAIR"
