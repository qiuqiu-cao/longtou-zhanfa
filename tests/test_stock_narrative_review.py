import copy
from pathlib import Path
import tempfile
import unittest

import leader_role_engine as e
import stock_narrative_review as n


class NarrativeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/"author.txt"
        self.text = "九阳股份六板。慈文传媒在作者讨论的影视传媒方向晋级。"
        self.path.write_text(self.text, encoding="utf-8")
        self.sources = {"author": {"text_path": str(self.path), "text_sha256": e.sha(self.path),
            "kind": "ORIGINAL_FINANCIAL_REPORT", "published_at": "2026-10-12T17:00:00+08:00",
            "obtained_at": "2026-10-12T17:01:00+08:00", "url": "https://fixture.example/author"}}
        self.candidates = {"002242": {"name": "九阳股份"}, "002343": {"name": "慈文传媒"}}
        self.bank = {"formed_at": "2026-10-12T17:02:00+08:00", "claims": [
            {"kind": "NARRATIVE", "code": "002242", "source_id": "author", "entity_anchor": "九阳股份"},
            {"kind": "NARRATIVE", "code": "002343", "source_id": "author", "entity_anchor": "慈文传媒"}]}
        self.reviews = {"version": n.VERSION, "formed_at": "2026-10-13T00:40:00+08:00",
            "reviewer": "synthetic_fixture_only", "reviews": [
                {"claim_index": 0, "claim_sha256": n.digest(self.bank["claims"][0]), "code": "002242",
                 "content_type": "PRICE_BOARD_SUMMARY", "reason": "只报板数，不给题材归因",
                 "reviewed_at": "2026-10-13T00:39:00+08:00", "source_span": {"offset": 0, "text": "九阳股份六板。"}},
                {"claim_index": 1, "claim_sha256": n.digest(self.bank["claims"][1]), "code": "002343",
                 "content_type": "EXPLICIT_TRADE_NARRATIVE", "reason": "原作者说明该股票与题材的联系",
                 "reviewed_at": "2026-10-13T00:39:00+08:00", "source_span": {"offset": 7, "text": self.text[7:]},
                 "topic_id": "media", "topic_label": "影视传媒", "topic_anchor": "影视传媒", "topic_level": "PARENT",
                 "relationship_review": "作者明确讨论关系，仅限此来源"}]}
        # Python character offsets are deliberately bound to the raw span.
        start = self.text.index("慈文传媒")
        self.reviews["reviews"][1]["source_span"] = {"offset": start, "text": self.text[start:]}

    def checked(self):
        return n.validate_annotations(self.bank, self.reviews, self.sources, self.candidates,
                                      e.stamp("2026-10-13T00:45:00+08:00"))[0]

    def test_price_summary_and_explicit_relation_remain_separate(self):
        rows = self.checked()
        self.assertEqual(rows[0]["content_type"], "PRICE_BOARD_SUMMARY")
        self.assertEqual(rows[1]["content_type"], "EXPLICIT_TRADE_NARRATIVE")
        self.assertFalse(rows[1]["fund_causality_confirmed"])
        self.assertFalse(rows[1]["business_link_confirmed_by_narrative"])

    def test_summary_cannot_silently_acquire_a_topic(self):
        self.reviews["reviews"][0]["topic_id"] = "media"
        with self.assertRaises(ValueError): self.checked()

    def test_neighboring_stock_relation_is_rejected(self):
        self.reviews["reviews"][0].update(content_type="EXPLICIT_TRADE_NARRATIVE", topic_id="media",
            topic_label="影视传媒", topic_level="PARENT", topic_anchor="影视传媒", relationship_review="错误跨股推断",
            source_span=self.reviews["reviews"][1]["source_span"])
        with self.assertRaises(ValueError): self.checked()

    def test_company_source_is_not_a_trading_report(self):
        self.sources["author"]["kind"] = "COMPANY_DISCLOSURE"
        with self.assertRaises(ValueError): self.checked()

    def test_missing_and_duplicate_original_claims_are_rejected(self):
        original = copy.deepcopy(self.reviews)
        self.reviews["reviews"].pop()
        with self.assertRaises(ValueError): self.checked()
        self.reviews = original
        self.reviews["reviews"].append(copy.deepcopy(self.reviews["reviews"][0]))
        with self.assertRaises(ValueError): self.checked()

    def test_span_offset_and_topic_anchor_must_be_exact(self):
        self.reviews["reviews"][1]["source_span"]["offset"] += 1
        with self.assertRaises(ValueError): self.checked()
        self.setUp()
        self.reviews["reviews"][1]["topic_anchor"] = "机器人"
        with self.assertRaises(ValueError): self.checked()

    def test_actual_review_cannot_be_backdated_or_future(self):
        self.reviews["reviews"][1]["reviewed_at"] = "2026-10-12T17:01:00+08:00"
        with self.assertRaises(ValueError): self.checked()
        self.reviews["reviews"][1]["reviewed_at"] = "2026-10-13T01:00:00+08:00"
        with self.assertRaises(ValueError): self.checked()

    def test_claim_identity_and_stock_must_not_change(self):
        self.reviews["reviews"][0]["claim_sha256"] = "0"*64
        with self.assertRaises(ValueError): self.checked()
        self.reviews["reviews"][0]["claim_sha256"] = n.digest(self.bank["claims"][0])
        self.reviews["reviews"][0]["code"] = "002343"
        with self.assertRaises(ValueError): self.checked()

    def test_unknown_is_allowed_without_guessed_topic(self):
        self.reviews["reviews"][0].update(content_type="UNKNOWN", reason="缺少明确归属")
        self.assertEqual(self.checked()[0]["content_type"], "UNKNOWN")
        self.reviews["reviews"][0]["topic_anchor"] = "机器人"
        with self.assertRaises(ValueError): self.checked()

    def test_old_facts_without_typed_sidecar_remain_compatible(self):
        self.assertIsNone(n.checked({}))

    def test_current_unknown_is_not_negative_or_present(self):
        context = {"rows": [{"code": "002242", "current_trade_narrative_presence": "UNKNOWN"}]}
        draft = {"rows": [{"code": "002242", "narrative_type_assessment": {
            "current_trade_narrative_presence": "UNKNOWN", "content_types_reviewed": True,
            "unknown_not_negative": True, "reason": "摘要不解释当前题材", "fund_causality_confirmed": False}}]}
        n.validate_assessments(context, draft)
        draft["rows"][0]["narrative_type_assessment"]["current_trade_narrative_presence"] = "PRESENT_IN_LIMITED_REVIEWED_SOURCE"
        with self.assertRaises(ValueError): n.validate_assessments(context, draft)


if __name__ == "__main__": unittest.main()
