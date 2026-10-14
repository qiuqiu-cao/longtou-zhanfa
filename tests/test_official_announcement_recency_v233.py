import pandas as pd

from official_announcement_recency_v233 import (
    build_overlays, classify_title, is_mainboard, parse_cninfo_rows,
)


def test_scope_and_title_classification():
    assert is_mainboard("600785")
    assert is_mainboard("002084")
    assert not is_mainboard("300110")
    assert classify_title("关于以集中竞价方式回购股份的方案") == "CONTROL_OR_CAPITAL_ACTION"
    assert classify_title("2026年半年度业绩预告") == "PERFORMANCE_DISCLOSURE"
    assert classify_title("股票交易异常波动暨风险提示公告") == "NEGATIVE_OR_CLARIFICATION"


def test_cninfo_parser_keeps_date_only_and_official_url():
    payload = {"announcements": [{
        "secCode": "002084", "secName": "海鸥住工",
        "announcementTime": 1789056000000,
        "announcementTitle": "<em>控制权</em>变更进展公告",
        "adjunctUrl": "finalpage/x.pdf", "announcementId": "a1",
    }]}
    rows = parse_cninfo_rows(payload, "002084", "2026-09-22T00:00:00Z")
    assert rows[0]["announcement_date"] == "2026-09-11"
    assert rows[0]["published_at_status"] == "DATE_ONLY"
    assert rows[0]["url"] == "https://static.cninfo.com.cn/finalpage/x.pdf"


def test_same_day_date_only_notice_is_not_available_at_close():
    claims = pd.DataFrame([{
        "signal_date": "2026-09-21", "code": "600001", "name": "甲",
        "driver_claim": "拟回购", "driver_claim_category": "CONTROL_OR_CAPITAL_ACTION",
        "recency_language_status": "RECENCY_LANGUAGE_PRESENT_NOT_FIRST_TIME_VERIFIED",
    }])
    announcements = pd.DataFrame([
        {"code": "600001", "announcement_date": "2026-09-21", "title": "回购方案",
         "title_category": "CONTROL_OR_CAPITAL_ACTION", "url": "same", "announcement_id": "2"},
        {"code": "600001", "announcement_date": "2026-09-20", "title": "回购进展",
         "title_category": "CONTROL_OR_CAPITAL_ACTION", "url": "prior", "announcement_id": "1"},
    ])
    overlay, queue = build_overlays(
        claims, announcements, "2026-09-21", "2026-09-21T15:00:00+08:00",
        {"600001": {"status": "SUCCESS", "row_count": 2}}, {"600001"})
    assert overlay.iloc[0].compatible_title_count == 1
    assert overlay.iloc[0].latest_compatible_url == "prior"
    assert not bool(overlay.iloc[0].title_match_proves_claim)
    assert queue.iloc[0].required_next_step == "DOWNLOAD_AND_MATCH_OFFICIAL_FULLTEXT"


def test_no_title_match_does_not_prove_no_catalyst():
    claims = pd.DataFrame([{
        "signal_date": "2026-09-21", "code": "600001", "name": "甲",
        "driver_claim": "政策纳入", "driver_claim_category": "POLICY_OR_LIST_INCLUSION",
        "recency_language_status": "RECENCY_LANGUAGE_PRESENT_NOT_FIRST_TIME_VERIFIED",
    }])
    announcements = pd.DataFrame(columns=[
        "code", "announcement_date", "title", "title_category", "url", "announcement_id"])
    overlay, _ = build_overlays(
        claims, announcements, "2026-09-21", "2026-09-21T15:00:00+08:00",
        {"600001": {"status": "SUCCESS", "row_count": 0}})
    assert overlay.iloc[0].official_recency_status == "NO_CATEGORY_MATCHED_TITLE_IN_BOUNDED_WINDOW"
    assert not bool(overlay.iloc[0].bounded_absence_proves_no_catalyst)
