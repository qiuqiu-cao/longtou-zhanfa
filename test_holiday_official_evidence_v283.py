import unittest

from holiday_official_evidence_v283 import validate_page, choose_documents


def payload(code="000678", day="2026-09-29", title="股票交易异常波动公告"):
    import pandas as pd
    return dict(hasMore=False, totalAnnouncement=1, announcements=[dict(secCode=code, secName="TEST_ONLY",
        announcementTime=int(pd.Timestamp(day, tz="Asia/Shanghai").timestamp()*1000),
        announcementTitle=title, announcementId="123", adjunctUrl="finalpage/"+day+"/123.PDF")])


class OfficialEvidenceTests(unittest.TestCase):
    def test_wrong_code_and_date_fail_instead_of_becoming_no_risk(self):
        with self.assertRaises(ValueError):
            validate_page(payload(code="600825"), "000678", "2026-09-15", "2026-10-02")
        with self.assertRaises(ValueError):
            validate_page(payload(day="2026-10-03"), "000678", "2026-09-15", "2026-10-02")

    def test_empty_needs_explicit_success_schema(self):
        self.assertEqual(validate_page(dict(hasMore=False, totalAnnouncement=0, announcements=None),
                                       "000678", "2026-09-15", "2026-10-02"), [])
        for bad in ({}, dict(hasMore=False), dict(hasMore="false", totalAnnouncement=0, announcements=[])):
            with self.assertRaises(ValueError):
                validate_page(bad, "000678", "2026-09-15", "2026-10-02")

    def test_publication_date_does_not_become_a_clock_time(self):
        row = validate_page(payload(), "000678", "2026-09-15", "2026-10-02")[0]
        self.assertEqual(row["announcement_date"], "2026-09-29")
        self.assertEqual(row["published_at_status"], "DATE_ONLY")
        self.assertIsNone(row["published_at"])
        self.assertIn("+08:00", row["available_at"])

    def test_nonofficial_document_link_rejected(self):
        p = payload();p["announcements"][0]["adjunctUrl"] = "https://example.com/123.PDF"
        with self.assertRaises(ValueError):
            validate_page(p, "000678", "2026-09-15", "2026-10-02")

    def test_title_selection_is_queue_not_risk_verdict(self):
        rows = validate_page(payload(), "000678", "2026-09-15", "2026-10-02")
        chosen = choose_documents(rows)
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0]["selection_reason"], "RISK_NOTICE_TITLE")
        self.assertNotIn("risk_class", chosen[0])
        self.assertNotIn("hard_veto", chosen[0])

    def test_capital_plan_and_summary_retained_without_boilerplate(self):
        base = dict(code="605303", announcement_date="2026-09-29")
        rows = [{**base, "announcement_id":str(i), "title":t} for i,t in enumerate([
            "发行股份购买资产预案", "发行股份购买资产预案（摘要）", "董事会关于本次购买资产预案的说明"])]
        chosen = choose_documents(rows)
        self.assertEqual({r["announcement_id"] for r in chosen}, {"0", "1"})


if __name__ == "__main__":
    unittest.main()
