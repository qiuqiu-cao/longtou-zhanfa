"""Unit tests for the Four Levers Comprehensive First-Board Engine."""
import numpy as np
import pandas as pd
import pytest
from first_board_comprehensive_engine import (
    EngineConfig,
    ComprehensiveFirstBoardEngine,
    DualAccountingLedger,
    parse_seal_time_to_minutes,
    FROZEN_UPPER_CAP_CNY,
)


def test_parse_seal_time():
    assert parse_seal_time_to_minutes("09:30:00") == 0.0
    assert parse_seal_time_to_minutes("09:35:00") == 5.0
    assert parse_seal_time_to_minutes(93500) == 5.0
    assert parse_seal_time_to_minutes("10:30:00") == 60.0
    assert parse_seal_time_to_minutes("11:30:00") == 120.0
    assert parse_seal_time_to_minutes("13:00:00") == 120.0
    assert parse_seal_time_to_minutes("13:30:00") == 150.0
    assert parse_seal_time_to_minutes("14:00:00") == 180.0
    assert parse_seal_time_to_minutes(145000) == 230.0
    assert parse_seal_time_to_minutes(None) is None
    assert parse_seal_time_to_minutes("invalid") is None
    assert parse_seal_time_to_minutes("08:00") is None


def test_emotion_defense_trigger():
    # 0 flags -> no defense
    row0 = {"market_down_limits": 2, "market_failed_rate": 0.20, "previous_first_promotion": 0.25, "market_positive_fraction": 0.55}
    is_def, flags = ComprehensiveFirstBoardEngine.check_emotion_defense(row0)
    assert not is_def
    assert len(flags) == 0

    # 1 flag -> no defense
    row1 = {"market_down_limits": 9, "market_failed_rate": 0.20, "previous_first_promotion": 0.25, "market_positive_fraction": 0.55}
    is_def, flags = ComprehensiveFirstBoardEngine.check_emotion_defense(row1)
    assert not is_def
    assert len(flags) == 1

    # 2 flags -> defense triggered
    row2 = {"market_down_limits": 9, "market_failed_rate": 0.45, "previous_first_promotion": 0.25, "market_positive_fraction": 0.55}
    is_def, flags = ComprehensiveFirstBoardEngine.check_emotion_defense(row2)
    assert is_def
    assert len(flags) == 2

    # All 4 flags -> defense triggered
    row4 = {"market_down_limits": 12, "market_failed_rate": 0.50, "previous_first_promotion": 0.05, "market_positive_fraction": 0.15}
    is_def, flags = ComprehensiveFirstBoardEngine.check_emotion_defense(row4)
    assert is_def
    assert len(flags) == 4


def test_sneak_attack_trap_veto():
    # Morning seal with low turnover -> not a sneak attack
    r1 = {"seal_first": "10:15:00", "turnover": 2.1}
    veto, reason = ComprehensiveFirstBoardEngine.audit_sneak_attack_trap(r1)
    assert not veto

    # Afternoon 14:00+ seal with low turnover (< 3%) -> sneak attack veto
    r2 = {"seal_first": "14:15:00", "turnover": 2.1}
    veto, reason = ComprehensiveFirstBoardEngine.audit_sneak_attack_trap(r2)
    assert veto
    assert "偷袭" in reason

    # Afternoon 14:00+ seal with high turnover (>= 3%) -> not sneak attack veto
    r3 = {"seal_first": "14:15:00", "turnover": 5.5}
    veto, reason = ComprehensiveFirstBoardEngine.audit_sneak_attack_trap(r3)
    assert not veto

    # Verified tail seal reduction > 30% (delta < 0.7) -> veto
    r4 = {"seal_delta": 0.55, "seal_delta_verified": True}
    veto, reason = ComprehensiveFirstBoardEngine.audit_sneak_attack_trap(r4)
    assert veto
    assert "撤单" in reason

    # Unverified tail seal reduction -> not vetoed (data pending)
    r5 = {"seal_delta": 0.55, "seal_delta_verified": False}
    veto, reason = ComprehensiveFirstBoardEngine.audit_sneak_attack_trap(r5)
    assert not veto


def test_capacity_dragon_exemption():
    # Base qualified capacity candidate
    base = {
        "float_cap": 50e8,
        "amount": 9e8,
        "turnover": 16.0,
        "seal_first": "09:45:00",
        "seal_delta": 1.35,
        "seal_delta_verified": True,
        "has_bad_history": False,
    }
    exempt, reason = ComprehensiveFirstBoardEngine.audit_capacity_dragon_exemption(base)
    assert exempt
    assert "特赦豁免" in reason

    # Market cap > 93.27亿 -> no exemption
    r_oversize = {**base, "float_cap": FROZEN_UPPER_CAP_CNY + 1e6}
    exempt, reason = ComprehensiveFirstBoardEngine.audit_capacity_dragon_exemption(r_oversize)
    assert not exempt
    assert "市值超限" in reason

    # Late seal (after 10:30) -> no exemption
    r_late = {**base, "seal_first": "11:00:00"}
    exempt, reason = ComprehensiveFirstBoardEngine.audit_capacity_dragon_exemption(r_late)
    assert not exempt
    assert "非10:30前早盘首封" in reason

    # Unverified delta -> no exemption
    r_unver = {**base, "seal_delta_verified": False}
    exempt, reason = ComprehensiveFirstBoardEngine.audit_capacity_dragon_exemption(r_unver)
    assert not exempt


def test_dual_accounting_ledger():
    ledger = DualAccountingLedger()
    # Record a normal trade
    ledger.record_decision(
        signal_date="2025-07-01",
        code="000001",
        name="平安银行",
        legacy_rank=1,
        legacy_score=0.85,
        final_rank=1,
        final_score=0.88,
        selected=True,
        intervention="KEEP",
        reason="双目标优选",
        reach4_outcome=1,
        promotion_outcome=1,
        trade_status="SCENARIO_CLOSED",
        trade_return=0.10,
        trade_net_pnl=20000.0,
    )
    # Record an avoided loss via defense
    ledger.record_decision(
        signal_date="2025-07-04",
        code="000002",
        name="万科A",
        legacy_rank=1,
        legacy_score=0.80,
        final_rank=1,
        final_score=0.80,
        selected=False,
        intervention="DEFENSE_EMPTY_CASH",
        reason="情绪防守留空",
        reach4_outcome=0,
        promotion_outcome=0,
        trade_status="SCENARIO_CLOSED",
        trade_return=-0.08,
        trade_net_pnl=-16000.0,
    )
    # Record a killed profit via defense
    ledger.record_decision(
        signal_date="2025-07-15",
        code="000004",
        name="国华网安",
        legacy_rank=1,
        legacy_score=0.75,
        final_rank=1,
        final_score=0.75,
        selected=False,
        intervention="DEFENSE_EMPTY_CASH",
        reason="情绪防守留空",
        reach4_outcome=1,
        promotion_outcome=1,
        trade_status="SCENARIO_CLOSED",
        trade_return=0.12,
        trade_net_pnl=24000.0,
    )

    summary = ledger.summarize()
    assert summary["total_decisions"] == 3
    audit = summary["four_dimensional_defense_audit"]
    assert audit["avoided_negative_trades_count"] == 1
    assert audit["avoided_loss_cny"] == 16000.0
    assert audit["killed_profitable_trades_count"] == 1
    assert audit["killed_profit_cny"] == 24000.0
    assert audit["missed_four_board_dragons"] == 1
    assert audit["net_pnl_effect_cny"] == -8000.0


def test_process_daily_candidates_pool_dynamics():
    engine = ComprehensiveFirstBoardEngine(EngineConfig(fusion_weight_s1=0.5))
    candidates = pd.DataFrame([
        {
            "signal_date": "2025-07-01",
            "code": "000001",
            "name": "个股1",
            "height_score": 0.80,
            "s1_score": 0.60,
            "legacy_industry_group_PROXY": "汽车",
            "amount": 5e8,
            "turnover": 8.0,
            "seal_first": "09:35:00",
        },
        {
            "signal_date": "2025-07-01",
            "code": "000002",
            "name": "个股2",
            "height_score": 0.60,
            "s1_score": 0.80,
            "legacy_industry_group_PROXY": "汽车",
            "amount": 4e8,
            "turnover": 7.0,
            "seal_first": "09:40:00",
        },
        {
            "signal_date": "2025-07-01",
            "code": "000003",
            "name": "个股3-孤勇",
            "height_score": 0.70,
            "s1_score": 0.70,
            "legacy_industry_group_PROXY": "医药",
            "amount": 3e8,
            "turnover": 6.0,
            "seal_first": "10:00:00",
        },
        {
            "signal_date": "2025-07-01",
            "code": "000004",
            "name": "个股4-偷袭",
            "height_score": 0.90,
            "s1_score": 0.90,
            "legacy_industry_group_PROXY": "汽车",
            "amount": 1e8,
            "turnover": 2.0,  # Turnover < 3%
            "seal_first": "14:30:00",  # Sneak after 14:00
        },
    ])
    emotion = {"market_down_limits": 1, "market_failed_rate": 0.15, "previous_first_promotion": 0.30, "market_positive_fraction": 0.60}
    
    processed = engine.process_daily_candidates("2025-07-01", candidates, emotion)
    
    # Check that candidate 4 is vetoed
    c4 = processed[processed["code"] == "000004"].iloc[0]
    assert c4["sneak_veto"] is True or c4["sneak_veto"] == 1
    assert c4["final_score"] == -999.0
    
    # Check sector peer count: 汽车 has 3 members, 医药 has 1
    c1 = processed[processed["code"] == "000001"].iloc[0]
    c3 = processed[processed["code"] == "000003"].iloc[0]
    assert c1["peer_count"] == 3
    assert c3["peer_count"] == 1
    assert "协同共振" in c1["sector_tag"]
    assert "单兵孤勇" in c3["sector_tag"]
    
    # Check final ranking: c4 is last because of veto
    assert processed.iloc[-1]["code"] == "000004"
    assert bool(processed.iloc[0]["selected"]) is True
