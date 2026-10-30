"""Unit tests for first_board_execution_upgrade.py."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import first_board_execution_upgrade as up

ROOT = Path(__file__).resolve().parent.parent


def test_original_screen_rules():
    """Verify original frozen risk screen: cap upper limit and thin turnover proxy."""
    df = pd.DataFrame([
        # Case 1: Normal eligible stock
        {"code": "000001", "name": "合格标的", "float_cap": 45e8, "turnover": 8.5, "one_price": 0, "volume_ratio_5": 1.5},
        # Case 2: Oversize (>93.27亿)
        {"code": "000002", "name": "大盘蓝筹", "float_cap": 120e8, "turnover": 6.0, "one_price": 0, "volume_ratio_5": 1.2},
        # Case 3: Thin turnover microcap (<=30亿, <5% turnover, one_price=1) - Lianming archetype
        {"code": "000003", "name": "薄换手一字", "float_cap": 25e8, "turnover": 1.2, "one_price": 1, "volume_ratio_5": 0.8},
        # Case 4: Thin turnover microcap (<=30亿, <5% turnover, volume_ratio_5 < 1.0)
        {"code": "000004", "name": "薄换手缩量", "float_cap": 28e8, "turnover": 3.5, "one_price": 0, "volume_ratio_5": 0.7},
        # Case 5: Microcap but with healthy turnover (>=5%)
        {"code": "000005", "name": "小微盘充分换手", "float_cap": 26e8, "turnover": 12.0, "one_price": 0, "volume_ratio_5": 1.8},
    ])

    screened = up.apply_original_screen(df)
    
    assert screened.loc[0, "original_screen_allowed"] == True
    assert screened.loc[0, "exclusion_reason"] == "QUALIFIED"

    assert screened.loc[1, "original_screen_allowed"] == False
    assert "ABOVE_DEVELOPMENT_CAP_NO_VERIFIED_EXCEPTION" in screened.loc[1, "exclusion_reason"]

    assert screened.loc[2, "original_screen_allowed"] == False
    assert "CONSERVATIVE_MICROCAP_THIN_TURNOVER_PROXY" in screened.loc[2, "exclusion_reason"]

    assert screened.loc[3, "original_screen_allowed"] == False
    assert "CONSERVATIVE_MICROCAP_THIN_TURNOVER_PROXY" in screened.loc[3, "exclusion_reason"]

    assert screened.loc[4, "original_screen_allowed"] == True
    assert screened.loc[4, "exclusion_reason"] == "QUALIFIED"


def test_capacity_dragon_exemption_evidence_pending():
    """Verify that capacity dragon exemption requires positive evidence and marks missing data as PENDING."""
    # 1. Missing seal delta must be PENDING_VERIFICATION, not granted!
    row_missing = {
        "float_cap": 50e8,
        "amount": 9e8,
        "turnover": 16.0,
        "seal_first": "09:40:00",
        "seal_delta": None,
    }
    status, reason = up.audit_capacity_dragon_exemption(row_missing)
    assert status == "PENDING_VERIFICATION"
    assert "待核" in reason

    # 2. Late seal (> 10:30) must be INELIGIBLE
    row_late = {
        "float_cap": 50e8,
        "amount": 9e8,
        "turnover": 16.0,
        "seal_first": 133000, # 13:30:00 integer
        "seal_delta": 1.3,
    }
    status, reason = up.audit_capacity_dragon_exemption(row_late)
    assert status == "INELIGIBLE"
    assert "偏晚" in reason

    # 3. Valid reinforced seal (Seal_Delta >= 1.2) must be SATISFIED
    row_good = {
        "float_cap": 60e8,
        "amount": 10e8,
        "turnover": 18.0,
        "seal_first": "09:35:00",
        "seal_delta": 1.25,
    }
    status, reason = up.audit_capacity_dragon_exemption(row_good)
    assert status == "SATISFIED"

    # 4. Net withdrawal (Seal_Delta < 1.0) must be VETOED
    row_bad = {
        "float_cap": 60e8,
        "amount": 10e8,
        "turnover": 18.0,
        "seal_first": "09:35:00",
        "seal_delta": 0.85,
    }
    status, reason = up.audit_capacity_dragon_exemption(row_bad)
    assert status == "VETOED"


def test_time_parsing():
    """Verify time parsing for integers, floats, and strings."""
    assert up.parse_seal_time_to_minutes("09:30:00") == 0.0
    assert up.parse_seal_time_to_minutes(93500) == 5.0
    assert up.parse_seal_time_to_minutes(103000) == 60.0
    assert up.parse_seal_time_to_minutes("13:00:00") == 120.0
    assert up.parse_seal_time_to_minutes("13:30:00") == 150.0
    assert up.parse_seal_time_to_minutes(145000) == 230.0


def test_theme_chronology_full_pool():
    """Verify theme chronology analysis across the full pool."""
    df = pd.DataFrame([
        {"signal_date": "2026-10-28", "code": "000001", "industry": "智能驾驶", "seal_first": "09:32:00"},
        {"signal_date": "2026-10-28", "code": "000002", "industry": "智能驾驶", "seal_first": "09:45:00"},
        {"signal_date": "2026-10-28", "code": "000003", "industry": "智能驾驶", "seal_first": "10:15:00"},
        {"signal_date": "2026-10-28", "code": "000004", "industry": "消费电子", "seal_first": "09:31:00"},
    ])

    audited = up.audit_theme_chronology_full_pool(df)
    
    # 000001 is pioneer with 2 followers
    pioneer = audited[audited["code"] == "000001"].iloc[0]
    assert "题材先锋身位" in pioneer["theme_pioneer_tag"]
    assert pioneer["industry_peers_count"] == 3

    # 000002 is follower
    follower = audited[audited["code"] == "000002"].iloc[0]
    assert "题材同位跟风" in follower["theme_pioneer_tag"]

    # 000004 is single isolated
    solo = audited[audited["code"] == "000004"].iloc[0]
    assert "独立无助攻" in solo["theme_pioneer_tag"]


def test_auction_execution_card():
    """Verify 09:25 execution card enforcing review before authorization."""
    cand = {"code": "000678", "name": "襄阳轴承", "close": 7.50}
    card = up.generate_auction_execution_card(cand)

    assert "09:20-09:25交易所不接受撤单" in card["exchange_rule_warning"]
    assert "审核先行" in card["execution_protocol"]
    assert len(card["visual_checklist"]) >= 4


def test_score_legacy_first_boards():
    """Verify loading and scoring using the frozen 44-feature bundle."""
    bundle_path = ROOT / "research/first_board_leader_20261020T114600Z/first_board_leader_bundle.joblib"
    if not bundle_path.exists():
        pytest.skip("Legacy bundle not present")

    # Load 5 sample rows from historical dataset that have all 44 features
    data_path = ROOT / "research/first_board_leader_20261020T114600Z/all_first_boards_features_and_LABEL_ONLY.csv"
    if not data_path.exists():
        pytest.skip("Data path not present")

    sample_df = pd.read_csv(data_path, nrows=5)
    scored = up.score_legacy_first_boards(sample_df, bundle_path=bundle_path)

    assert "legacy_model_score" in scored.columns
    assert "legacy_rank" in scored.columns
    assert scored["legacy_model_score"].notna().all()
    assert (scored["legacy_model_score"] >= 0.0).all() and (scored["legacy_model_score"] <= 1.0).all()

    # Verify missing feature raises error
    corrupted = sample_df.drop(columns=["log_price"])
    with pytest.raises(ValueError, match="Missing 1 fields"):
        up.score_legacy_first_boards(corrupted, bundle_path=bundle_path)


def test_dual_accounting_ledger():
    """Verify dual accounting ledger: counting both avoided losses and killed profits."""
    ledger = up.DualAccountingLedger()

    # Case 1: Excluded an unpromoted stock that actually failed (avoided loss)
    ledger.record_decision(
        signal_date="2026-10-28", code="000001", name="失败杂毛",
        legacy_rank=2, legacy_score=0.45, final_selected=False,
        intervention_type="EXCLUDE_BY_SCREEN", reason="薄换手风险",
        actual_reach4=0, actual_trade_profitable=False, trade_return=-0.08
    )

    # Case 2: Excluded a stock that actually produced profit (killed profit)
    ledger.record_decision(
        signal_date="2026-10-28", code="000002", name="错杀盈利股",
        legacy_rank=1, legacy_score=0.55, final_selected=False,
        intervention_type="EXCLUDE_BY_AUDIT", reason="题材跟风",
        actual_reach4=0, actual_trade_profitable=True, trade_return=0.06
    )

    # Case 3: Promoted a true dragon from below (rank 12 -> selected)
    ledger.record_decision(
        signal_date="2026-10-28", code="000003", name="真龙襄阳",
        legacy_rank=12, legacy_score=0.25, final_selected=True,
        intervention_type="PROMOTE_FROM_BELOW", reason="大容量换手特赦",
        actual_reach4=1, actual_trade_profitable=True, trade_return=0.35
    )

    summary = ledger.summarize()
    assert summary["total_decisions"] == 3
    assert summary["excluded_count"] == 2
    assert summary["avoided_losses_count"] == 2
    assert summary["avoided_negative_trades"] == 1
    assert summary["killed_profitable_trades"] == 1
    assert summary["net_loss_prevention"] == 0  # 1 avoided - 1 killed
    assert summary["promoted_dragons"] == 1
