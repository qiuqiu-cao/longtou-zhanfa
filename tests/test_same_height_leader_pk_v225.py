import json

import pandas as pd

from same_height_leader_pk_v225 import (
    build, classify_board_path, classify_notice_risk,
)


def test_quick_repair_uses_duration_not_raw_reopen_count():
    result = classify_board_path("092502", "093711", 3)
    assert result["status"] == "QUICK_REPAIR"
    assert result["span_minutes"] == 12.15
    assert result["score"] == 2


def test_one_price_is_buyability_risk():
    result = classify_board_path("092500", "092500", 0)
    assert result["status"] == "ONE_PRICE_BUYABILITY_UNPROVEN"
    assert result["score"] < 0


def test_title_only_risk_notice_is_not_hard_veto():
    rows = pd.DataFrame([{
        "risk_window_status": "active_within_5d", "risk_hard": 1,
        "risk_type": "风险提示_风险公告", "title": "股票交易风险提示公告",
    }])
    result = classify_notice_risk(rows)
    assert result["tier"] == "TITLE_ONLY_RISK_NOTICE_REQUIRES_FULLTEXT"
    assert result["hard_veto"] is False


def test_active_within_15d_is_still_an_active_notice():
    rows = pd.DataFrame([{
        "risk_window_status": "active_within_15d", "risk_hard": 1,
        "risk_type": "风险提示_监管", "title": "交易异常波动问询函复函",
    }])
    result = classify_notice_risk(rows)
    assert result["tier"] == "TITLE_ONLY_RISK_NOTICE_REQUIRES_FULLTEXT"
    assert result["score"] == -4


def test_late_fulltext_is_not_backdated_into_score():
    rows = pd.DataFrame([{
        "risk_window_status": "active_within_5d", "risk_hard": 1,
        "risk_type": "风险提示_风险公告", "title": "股票交易风险提示公告",
        "fulltext_review_status": "RETROSPECTIVE_REVIEW_NOT_POINT_IN_TIME",
        "point_in_time_eligible": False,
        "content_risk_class": "ROUTINE_MARKET_AND_PERFORMANCE_RISK",
        "fulltext_risk_score": 0, "fulltext_hard_veto": False,
    }])
    result = classify_notice_risk(rows)
    assert result["tier"] == "RETROSPECTIVE_FULLTEXT_NOT_APPLIED_TITLE_ONLY"
    assert result["score"] == -4
    assert result["hard_veto"] is False


def test_point_in_time_routine_fulltext_is_soft_not_veto():
    rows = pd.DataFrame([{
        "risk_window_status": "active_within_5d", "risk_hard": 1,
        "risk_type": "风险提示_风险公告", "title": "股票交易风险提示公告",
        "fulltext_review_status": "POINT_IN_TIME_REVIEWED",
        "point_in_time_eligible": True,
        "content_risk_class": "ROUTINE_MARKET_AND_PERFORMANCE_RISK",
        "fulltext_risk_score": -4, "fulltext_hard_veto": False,
    }])
    result = classify_notice_risk(rows)
    assert result["tier"] == "FULLTEXT_ROUTINE_MARKET_AND_PERFORMANCE_RISK"
    assert result["score"] == -4
    assert result["hard_veto"] is False


def test_fulltext_substantive_risk_can_veto():
    rows = pd.DataFrame([{
        "risk_window_status": "active_within_5d", "risk_hard": 1,
        "risk_type": "风险提示", "title": "风险提示公告",
        "full_text_status": "POINT_IN_TIME_VERIFIED",
        "full_text": "公司因涉嫌违法违规被立案调查。",
    }])
    result = classify_notice_risk(rows)
    assert result["tier"] == "FULLTEXT_VERIFIED_SUBSTANTIVE_HARD_RISK"
    assert result["hard_veto"] is True


def test_symbolic_identity_remains_non_tradeable_and_pairwise():
    review = {
        "signal_date": "2026-09-21",
        "relay_candidates": [
            {"code": "002285", "name": "世联行", "boards": 4,
             "turnover": 14, "first_seal_time": 131706, "failed_seal_count": 0,
             "theme_role": "THEME_HEIGHT_LEADER", "symbolic_name_clusters": [],
             "theme_evidence": {"status": "INFERRED_OR_LIMITED", "ladder_verified": True,
                                "support_ex_self": 7}},
            {"code": "603230", "name": "内蒙新华", "boards": 4,
             "turnover": 10, "first_seal_time": 92502, "failed_seal_count": 3,
             "theme_role": "THEME_HEIGHT_LEADER",
             "symbolic_name_clusters": [
                 {"token": "华", "classification": "SYMBOLIC_LADDER_CANDIDATE"},
                 {"token": "新", "classification": "SYMBOLIC_DIFFUSION_CANDIDATE"}],
             "theme_evidence": {"status": "INFERRED_OR_LIMITED", "ladder_verified": True,
                                "support_ex_self": 2}},
        ],
    }
    raw = pd.DataFrame([
        {"代码": "002285", "首次封板时间": "131706", "最后封板时间": "131706", "炸板次数": 0},
        {"代码": "603230", "首次封板时间": "092502", "最后封板时间": "093711", "炸板次数": 3},
    ])
    notices = pd.DataFrame([
        {"code": "002285", "risk_window_status": "not_hard_or_closed", "risk_hard": 0,
         "risk_type": "其他", "title": "普通公告"},
        {"code": "603230", "risk_window_status": "active_within_5d", "risk_hard": 1,
         "risk_type": "风险提示_风险公告", "title": "风险提示公告"},
    ])
    manifest = {"signal_date": "2026-09-21", "candidate_count": 2,
                "announcements": {"candidate_codes": 2, "failed_codes": 0,
                                  "status": "verified_index_date_only"}}
    rows, pairs, report = build(review, raw, notices, manifest)
    inner = rows[rows.code.eq("603230")].iloc[0]
    assert not bool(inner.symbolic_is_real_theme)
    assert not bool(inner.entry_authorized)
    assert len(pairs) == 1
    assert not bool(pairs.iloc[0].official_winner_claimed)
    assert report["policy"]["official_champion_claimed"] is False
