import unittest

import pandas as pd

from current_reset_chronology_v192 import (
    build_candidate_evidence,
    next_session,
    parse_cninfo_payload,
    parse_sse_daily_payload,
)


class CurrentResetChronologyV192Tests(unittest.TestCase):
    def test_sse_parser_keeps_only_z3_to_z8_and_deduplicates(self):
        payload = {"pageHelp": {"data": [
            {"secCode": "605058", "secAbbr": "澳弘电子", "tradeDate": "20260916",
             "abnormalStart": "20260905", "abnormalEnd": "20260916", "refType": "Z5",
             "secValue": "101.2", "bsType": "B"},
            {"secCode": "605058", "secAbbr": "澳弘电子", "tradeDate": "20260916",
             "abnormalStart": "20260905", "abnormalEnd": "20260916", "refType": "Z5",
             "secValue": "101.2", "bsType": "S"},
            {"secCode": "605058", "tradeDate": "20260916", "refType": "1"},
        ]}}
        result = parse_sse_daily_payload(payload, "2026-09-16")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["event_type"], "10D_POSITIVE_100")

    def test_sse_parser_rejects_wrong_date(self):
        payload = {"pageHelp": {"data": [{
            "secCode": "605058", "tradeDate": "20260915", "abnormalStart": "20260901",
            "abnormalEnd": "20260915", "refType": "Z5",
        }]}}
        with self.assertRaises(ValueError):
            parse_sse_daily_payload(payload, "2026-09-16")

    def test_sse_parser_accepts_official_severe_field_variant(self):
        payload = {"pageHelp": {"data": [{
            "secCode": "603137", "secAbbr": "恒尚节能", "tradeDate": "20260710",
            "unnormalStart": "20260701", "unnormalEnd": "20260710",
            "refType": "Z5", "secIndct": "109.13",
        }]}}
        result = parse_sse_daily_payload(payload, "2026-07-10")
        self.assertEqual(result[0]["abnormal_start"], "2026-07-01")
        self.assertEqual(result[0]["abnormal_end"], "2026-07-10")
        self.assertEqual(result[0]["sec_value"], "109.13")

    def test_cninfo_requires_explicit_severe_phrase(self):
        payload = {"announcements": [
            {"secCode": "000993", "secName": "闽东电力", "announcementId": "1",
             "announcementTitle": "关于<em>股票交易异常波动</em>的公告",
             "announcementTime": 1789401600000, "adjunctUrl": "ordinary.pdf"},
            {"secCode": "000993", "secName": "闽东电力", "announcementId": "2",
             "announcementTitle": "股票交易<em>严重异常波动</em>公告",
             "announcementTime": 1789401600000, "adjunctUrl": "severe.pdf"},
        ]}
        result = parse_cninfo_payload(payload, "000993")
        self.assertEqual([row["announcement_id"] for row in result], ["2"])

    def test_next_session_does_not_invent_future_calendar(self):
        calendar = ["2026-09-14", "2026-09-15", "2026-09-16"]
        self.assertEqual(next_session(calendar, "2026-09-15"), "2026-09-16")
        self.assertIsNone(next_session(calendar, "2026-09-16"))

    def test_sse_bounded_absence_verified_but_szse_absence_is_partial(self):
        candidates = pd.DataFrame([
            {"code": "605058", "name": "澳弘电子"},
            {"code": "000993", "name": "闽东电力"},
        ])
        calendar = ["2026-09-15", "2026-09-16"]
        evidence = build_candidate_evidence(
            candidates, calendar, {date: True for date in calendar}, [],
            {"000993": {"status": "SUCCESS", "query_coverage": "2026-07-06~2026-09-17"}},
            [],
        ).set_index("code")
        self.assertEqual(
            evidence.loc["605058", "reset_chronology_status"],
            "verified_no_exchange_event_since_rule_effective",
        )
        self.assertTrue(evidence.loc["605058", "official_exchange_chronology_complete"])
        self.assertEqual(
            evidence.loc["000993", "reset_chronology_status"],
            "company_index_no_event_found_not_exchange_exhaustive",
        )
        self.assertFalse(evidence.loc["000993", "official_exchange_chronology_complete"])


if __name__ == "__main__":
    unittest.main()
