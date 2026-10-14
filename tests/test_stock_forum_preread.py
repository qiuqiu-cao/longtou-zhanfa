import unittest
from copy import deepcopy
from stock_forum_preread import project


class PreReadTests(unittest.TestCase):
    def fixture(self):
        stocks=[dict(code="000001",name="甲",boards=2),dict(code="000002",name="乙",boards=3)]
        sources={"a":dict(kind="ORIGINAL_FORUM",published_at="2026-10-13T18:00:00+08:00",obtained_at="2026-10-14T12:00:00+08:00",url="https://example.com/post",author="作者")}
        claims=[dict(scope="STOCK",code="000001",kind="NARRATIVE",source_id="a",excerpt="甲的补涨线索",offset=1,source_section="COMMENT",comment_author="评论者",comment_published_at="2026-10-13T19:00:00+08:00")]
        return stocks,claims,sources

    def test_old_post_cannot_be_signal_day_coverage(self):
        rows=project(*self.fixture(),"2026-10-14")
        self.assertEqual(rows[0]["coverage"],"BACKGROUND_ONLY")
        self.assertEqual(rows[0]["forum_evidence"][0]["source_section"],"COMMENT")

    def test_current_post_and_uncovered_candidates_retained(self):
        values=self.fixture();before=deepcopy(values);rows=project(*values,"2026-10-13")
        self.assertEqual(len(rows),2);self.assertEqual(rows[0]["coverage"],"SIGNAL_DAY_LIMITED_SOURCE")
        self.assertEqual(rows[1]["coverage"],"NOT_COVERED_IN_CHECKED_SOURCES")
        self.assertIsNone(rows[1]["narrative_is_absent"])
        self.assertFalse(rows[1]["unknown_is_negative_evidence"])
        self.assertFalse(rows[1]["unknown_is_positive_evidence"])
        self.assertEqual(values,before)

    def test_market_summary_not_forum_narrative(self):
        stocks,claims,sources=self.fixture();claims[0]["kind"]="BUSINESS_LINK"
        self.assertFalse(project(stocks,claims,sources,"2026-10-13")[0]["forum_evidence"])

    def test_comment_time_and_author_do_not_inherit_main_post(self):
        stocks,claims,sources=self.fixture();claims[0].pop("comment_published_at")
        e=project(stocks,claims,sources,"2026-10-13")[0]["forum_evidence"][0]
        self.assertFalse(e["published_on_signal_day"])
        self.assertEqual(e["use"],"COMMENT_DATE_UNKNOWN")
        self.assertEqual(e["author"],"评论者")
        claims[0]["comment_published_at"]="2026-10-15T19:00:00+08:00"
        with self.assertRaises(ValueError):project(stocks,claims,sources,"2026-10-13")

    def test_date_only_publication_retains_explicit_precision(self):
        stocks,claims,sources=self.fixture();claims[0]["source_section"]="AUTHOR_BODY"
        sources["a"]["publication_precision"]="DATE";sources["a"]["published_at"]="2026-10-13"
        e=project(stocks,claims,sources,"2026-10-13")[0]["forum_evidence"][0]
        self.assertTrue(e["published_on_signal_day"])
        self.assertEqual(e["discussion_published_at"],"2026-10-13")

    def test_duplicate_outside_pool_and_bad_section_rejected(self):
        stocks,claims,sources=self.fixture()
        with self.assertRaises(ValueError):project(stocks+stocks,claims,sources,"2026-10-13")
        claims[0]["code"]="000003"
        with self.assertRaises(ValueError):project(stocks,claims,sources,"2026-10-13")
        claims[0]["code"]="000001";claims[0]["source_section"]="AUTHOR_GUARANTEED_FACT"
        with self.assertRaises(ValueError):project(stocks,claims,sources,"2026-10-13")


if __name__ == "__main__":unittest.main()
