"""Regression tests for factual gates, score clocks and realized outcome joins."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
import first_board_execution_upgrade as up
from audit_execution_upgrade import build_trade_lookup

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope="module")
def complete_features():
    d = pd.read_csv(ROOT/"research/first_board_leader_20261020T114600Z/all_first_boards_features_and_LABEL_ONLY.csv",
                    dtype={"code":str}, float_precision="round_trip", low_memory=False)
    d = d.dropna(subset=up.LEGACY_44_FEATURES).head(2).copy()
    assert len(d)==2
    d["signal_date"] = "2026-10-28"
    d[["first","eligible","scope_known"]] = True
    return d

def test_exact_screen_boundary_and_pending():
    rows = pd.DataFrame([
        dict(float_cap=up.FROZEN_UPPER_CAP_CNY,turnover=8,one_price=0,volume_ratio_5=1.2),
        dict(float_cap=up.FROZEN_UPPER_CAP_CNY+1,turnover=8,one_price=0,volume_ratio_5=1.2),
        dict(float_cap=25e8,turnover=1.2,one_price=1,volume_ratio_5=.8),
        dict(float_cap=25e8,turnover=1.2,one_price=0,volume_ratio_5=np.nan)])
    got = up.apply_original_screen(rows)
    assert got.original_screen_allowed.tolist()==[True,False,False,True]
    assert got.loc[3,"screen_review_status"]=="PENDING_VERIFICATION"
    with pytest.raises(ValueError,match="Missing screen"):
        up.apply_original_screen(rows.drop(columns="one_price"))

@pytest.mark.parametrize("value,minutes",[
    ("09:30:00",0),(93500,5),(93500.,5),("103000",60),("11:30",120),
    ("13:00:00",120),("13:30:00",150),("14:00",180),(145000,230),("09:25",-5)])
def test_time(value,minutes):
    assert up.parse_seal_time_to_minutes(value)==minutes

@pytest.mark.parametrize("value",[None,np.nan,"not a time","11:99","12:15","08:00",250000,-1,93500.5])
def test_invalid_time_unknown(value):
    assert up.parse_seal_time_to_minutes(value) is None

def capacity(**kw):
    row=dict(float_cap=50e8,amount=9e8,turnover=16,first_seal_time="09:40",seal_delta=1.3,
             seal_delta_verified=True,seal_delta_source="reviewed_queue_pair",
             has_bad_history=False,history_verified=True,history_source="reviewed_past_year")
    row.update(kw)
    return row

@pytest.mark.parametrize("change,expected",[
    ({}, "SATISFIED"),
    ({"first_seal_time":None},"PENDING_VERIFICATION"),
    ({"first_seal_time":133000},"INELIGIBLE"),
    ({"seal_delta":None},"PENDING_VERIFICATION"),
    ({"seal_delta_verified":False},"PENDING_VERIFICATION"),
    ({"seal_delta_source":None},"PENDING_VERIFICATION"),
    ({"seal_delta":1.1},"INELIGIBLE"),
    ({"seal_delta":.85},"INELIGIBLE"),
    ({"seal_delta":.6},"VETOED"),
    ({"seal_delta":0},"VETOED"),
    ({"has_bad_history":True},"VETOED"),
    ({"has_bad_history":np.nan},"PENDING_VERIFICATION"),
    ({"history_verified":False},"PENDING_VERIFICATION"),
    ({"float_cap":120e8},"INELIGIBLE"),
    ({"amount":np.nan,"turnover":np.nan},"PENDING_VERIFICATION")])
def test_capacity_requires_all_evidence(change,expected):
    assert up.audit_capacity_dragon_exemption(capacity(**change))[0]==expected

def test_trap_clock_and_unknown_queue():
    assert up.audit_sneak_attack_trap_veto(dict(first_seal_time="13:45",turnover=2.5))[0] is False
    assert up.audit_sneak_attack_trap_veto(dict(first_seal_time="14:00",turnover=2.5))[0] is True
    assert up.audit_sneak_attack_trap_veto(dict(seal_delta=.5))[0] is False
    assert up.audit_sneak_attack_trap_veto(capacity(seal_delta=0))[0] is True
    assert up.audit_sneak_attack_trap_veto(dict(has_bad_history=True))[0] is True

def test_emotion_zero_and_missing():
    row=dict(market_down_limits=8,market_failed_rate=.1,previous_first_promotion=0,market_positive_fraction=.6)
    assert up.audit_market_emotion_defense(row)["is_defense"] is True
    assert up.audit_market_emotion_defense({})["is_defense"] is None
    assert up.audit_market_emotion_defense({})["regime"]=="UNKNOWN"

def test_industry_order_does_not_claim_leadership():
    d=pd.DataFrame([dict(signal_date="2026-10-28",code="000001",industry="机械",seal_first=93500),
                    dict(signal_date="2026-10-28",code="000002",industry="机械",seal_first=145900)])
    got=up.audit_theme_chronology_full_pool(d)
    assert not got.theme_pioneer_tag.str.contains("先锋|助攻|跟风").any()
    assert got.loc[0,"same_group_later_seals_5m"]==0
    assert got.group_basis.eq("INDUSTRY_PROXY").all()
    d.loc[1,"seal_first"]=None
    partial=up.audit_theme_chronology_full_pool(d)
    assert partial.chronology_status.eq("INCOMPLETE_GROUP_TIMES").all()
    assert pd.isna(partial.loc[1,"industry_seal_rank"])
    empty=up.audit_theme_chronology_full_pool(d.drop(columns="seal_first"))
    assert empty.industry_seal_rank.isna().all()

def test_chronology_windows_do_not_collapse_lunch():
    d=pd.DataFrame([dict(signal_date="2026-10-28",industry="机械",seal_first="11:30"),
                    dict(signal_date="2026-10-28",industry="机械",seal_first="13:00")])
    assert up.audit_theme_chronology_full_pool(d).loc[0,"same_group_later_seals_5m"]==0

def test_new_date_score_is_frozen_and_historical_rejected(complete_features):
    d=complete_features
    b=joblib.load(up.LEGACY_BUNDLE_PATH)
    got=up.score_legacy_first_boards(d)
    assert np.array_equal(got.legacy_model_score.to_numpy(),b["model"].predict_proba(d[b["features"]])[:,1])
    assert not any("LABEL_ONLY" in c for c in got)
    with pytest.raises(ValueError,match="Historical"):
        up.score_legacy_first_boards(d.assign(signal_date="2026-08-31"))
    with pytest.raises(ValueError,match="entirely missing"):
        bad=d.copy()
        bad[up.LEGACY_44_FEATURES]=np.nan
        up.score_legacy_first_boards(bad)
    with pytest.raises(ValueError,match="Unverified"):
        up.score_legacy_first_boards(d.assign(first=False))
    with pytest.raises(ValueError,match="Missing"):
        up.score_legacy_first_boards(d.drop(columns="log_price"))

def test_saved_oof_cohort_and_deterministic_rank():
    d=pd.DataFrame([dict(signal_date="2025-07-01",code="000002"),dict(signal_date="2025-07-01",code="000001")])
    p=d.assign(model_score=[.5,.5])
    got=up.load_historical_legacy_scores(d,p)
    assert got.legacy_rank.tolist()==[2,1]
    assert got.legacy_model_score.tolist()==[.5,.5]
    with pytest.raises(ValueError,match="cohort mismatch"):
        up.load_historical_legacy_scores(d,p.head(1))
    with pytest.raises(ValueError,match="Unknown historical"):
        up.load_historical_legacy_scores(d,p.assign(model_score=np.nan))

def test_trade_reader_real_column_and_status():
    d=pd.DataFrame([
        dict(signal_date="2025-07-01",code="000001",status="SCENARIO_CLOSED",net_return=.06,net_pnl=600),
        dict(signal_date="2025-07-01",code="000002",status="OPEN_AT_UPPER_LIMIT_NO_QUEUE_ASSUMPTION",net_return=np.nan,net_pnl=np.nan),
        dict(signal_date="2025-07-01",code="000003",status="OPEN_UNRESOLVED",net_return=np.nan,net_pnl=np.nan)])
    got=build_trade_lookup(d)
    assert got[("2025-07-01","000001")]["trade_return"]==.06
    assert got[("2025-07-01","000002")]["trade_return"] is None
    assert got[("2025-07-01","000003")]["trade_return"] is None
    with pytest.raises(ValueError,match="Missing scenario"):
        build_trade_lookup(d.drop(columns="net_return"))

def test_ledger_separates_failure_profit_unknown():
    ledger=up.DualAccountingLedger()
    base=dict(signal_date="2025-07-01",name="test",legacy_rank=1,legacy_score=.5,final_selected=False,
              intervention_type="EXCLUDE_BY_SCREEN",reason="risk")
    ledger.record_decision(**base,code="000001",actual_reach4=0,trade_status="SCENARIO_CLOSED",trade_return=.06)
    ledger.record_decision(**base,code="000002",actual_reach4=0,trade_status="SCENARIO_CLOSED",trade_return=-.08)
    ledger.record_decision(**base,code="000003",trade_status="OPEN_AT_UPPER_LIMIT_NO_QUEUE_ASSUMPTION")
    ledger.record_decision(**base,code="000004",trade_status="OPEN_UNRESOLVED")
    s=ledger.summarize()
    assert s["excluded_non_reach4"]==2
    assert s["avoided_negative_trades"]==1
    assert s["killed_profitable_trades"]==1
    assert s["excluded_profit_unknown"]==2
    assert s["new_strategy_net_return"] is None
    assert "avoided_losses_count" not in s
    with pytest.raises(ValueError,match="Duplicate"):
        ledger.record_decision(**base,code="000001")
    with pytest.raises(ValueError,match="Profit flag"):
        ledger.record_decision(**base,code="000005",actual_trade_profitable=False)

def test_card_pending_and_decimal_reference():
    got=up.generate_auction_execution_card(dict(code="000678",close=7.55,target_date="2026-10-29"))
    assert got["limit_ref"]==8.31
    assert got["authorization_status"]=="PENDING_HUMAN_VISUAL_REVIEW"
    assert "本卡不自动授权" in got["authorized_action_if_passed"]
    assert "09:20-09:25交易所不接受撤单" in got["exchange_rule_warning"]
    assert up.generate_auction_execution_card(dict(code="000678",close=7.55))["authorization_status"]=="PENDING_TARGET_DATE"

