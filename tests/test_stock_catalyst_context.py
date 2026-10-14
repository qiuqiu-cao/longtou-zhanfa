"""Synthetic contract tests; fixtures are not market observations or accuracy evidence."""

import copy
import unittest

import stock_catalyst_context as context_review


class CatalystContextIntegrityTests(unittest.TestCase):
    def setUp(self):
        # Codes, names, excerpts and timestamps below are deliberately synthetic.
        self.original_cutoff = "2026-10-11T16:00:00+00:00"
        self.signal_cutoff = "2026-10-11T18:00:00+00:00"
        self.candidates = [
            {"code": "000003", "name": "合成丙", "boards": 2,
             "actual_one_word_daily": 1.0,
             "features": {"signal_one_word": 0.0, "turnover_vs_prev": 8.0}},
            {"code": "000001", "name": "合成甲", "boards": 2,
             "actual_one_word_daily": 0.0,
             "features": {"signal_one_word": 1.0, "turnover_vs_prev": 5.0}},
            {"code": "000002", "name": "合成乙", "boards": 2,
             "actual_one_word_daily": 0.0,
             "features": {"signal_one_word": 0.0, "turnover_vs_prev": None}},
        ]
        self.sources = {
            "forum": {"kind": "ORIGINAL_FORUM", "url": "https://example.invalid/forum"},
            "company": {"kind": "COMPANY_DISCLOSURE", "url": "https://example.invalid/company"},
            "official": {"kind": "OFFICIAL_EVENT_RELEASE", "url": "https://example.invalid/event"},
            "report": {"kind": "ORIGINAL_FINANCIAL_REPORT", "url": "https://example.invalid/report"},
        }
        self.claims = [
            {"source_id": "forum", "scope": "STOCK", "code": "000001",
             "event_id": "synthetic-event-a", "kind": "NARRATIVE", "state": "PRESENT",
             "available_at": "2026-10-11T15:00:00+00:00", "excerpt": "合成叙事关联甲"},
            {"source_id": "company", "scope": "STOCK", "code": "000001",
             "event_id": "synthetic-event-a", "kind": "BUSINESS_LINK",
             "state": "PRODUCT_LABEL_RELATION", "available_at": "2026-10-11T15:10:00+00:00",
             "excerpt": "合成公司明确产品用途关联"},
            {"source_id": "company", "scope": "STOCK", "code": "000001",
             "event_id": "synthetic-event-a", "kind": "EARNINGS_IMPACT",
             "state": "LIMITED_NO_MAJOR_EFFECT", "available_at": "2026-10-11T15:10:00+00:00",
             "excerpt": "合成披露业绩影响有限"},
            {"source_id": "official", "scope": "EVENT", "event_id": "synthetic-event-a",
             "kind": "EVENT_STATUS", "state": "UNCONFIRMED_CAUSE_NO_ONGOING_TRANSMISSION",
             "available_at": "2026-10-11T15:20:00+00:00", "excerpt": "合成事件病原尚未确认"},
            {"source_id": "report", "scope": "STOCK", "code": "000003",
             "event_id": "synthetic-event-b", "kind": "NARRATIVE", "state": "PRESENT",
             "available_at": "2026-10-11T15:30:00+00:00", "excerpt": "合成丙属于另一个叙事"},
        ]

    def rows(self, claims=None, cutoff=None):
        return context_review.build_rows(
            self.candidates,
            self.claims if claims is None else claims,
            self.sources,
            self.signal_cutoff if cutoff is None else cutoff,
            self.original_cutoff,
        )

    @staticmethod
    def indexed(rows):
        return {row["code"]: row for row in rows}

    def valid_draft(self, rows):
        return {"rows": [
            {"code": row["code"], "promotion_judgement": "UNKNOWN",
             "catalyst_assessment": {
                 "narrative_presence": row["narrative_presence"],
                 "event_and_business_reviewed": True,
                 "earnings_impact": row["earnings_impact"],
                 "negative_reason_codes": [],
             }} for row in rows
        ]}

    def test_complete_sorted_pool_preserves_unchecked_and_unknown_candidates(self):
        rows = self.rows()
        self.assertEqual([row["code"] for row in rows], ["000001", "000002", "000003"])
        unknown = self.indexed(rows)["000002"]
        self.assertEqual(unknown["narrative_presence"], "UNKNOWN")
        self.assertEqual(unknown["business_link"], "UNKNOWN")
        self.assertEqual(unknown["earnings_impact"], "UNKNOWN")
        self.assertEqual(unknown["coverage"], "NOT_CHECKED")
        self.assertFalse(unknown["unknown_is_negative_evidence"])
        self.assertEqual(unknown["evidence"], [])

    def test_event_status_reaches_only_stock_with_matching_checked_event_link(self):
        rows = self.indexed(self.rows())
        expected = "UNCONFIRMED_CAUSE_NO_ONGOING_TRANSMISSION"
        self.assertIn(expected, rows["000001"]["event_states"])
        self.assertNotIn(expected, rows["000002"]["event_states"])
        self.assertNotIn(expected, rows["000003"]["event_states"])

    def test_industry_similarity_does_not_create_narrative_or_event_membership(self):
        candidates = copy.deepcopy(self.candidates)
        for candidate in candidates:
            candidate["industry"] = "合成同一行业"
        rows = self.indexed(context_review.build_rows(
            candidates, self.claims, self.sources, self.signal_cutoff, self.original_cutoff))
        self.assertEqual(rows["000002"]["narrative_presence"], "UNKNOWN")
        self.assertEqual(rows["000002"]["business_link"], "UNKNOWN")
        self.assertEqual(rows["000002"]["event_states"], [])

    def test_later_retrieved_evidence_cannot_be_backfilled_to_original_cutoff(self):
        late = [{"source_id": "forum", "scope": "STOCK", "code": "000002",
                 "event_id": "synthetic-event-a", "kind": "NARRATIVE", "state": "PRESENT",
                 "available_at": "2026-10-11T17:00:00+00:00", "excerpt": "原截止之后取得的合成正文"}]
        old = self.indexed(self.rows(late, self.original_cutoff))["000002"]
        new = self.indexed(self.rows(late, self.signal_cutoff))["000002"]
        self.assertEqual(old["narrative_presence"], "UNKNOWN")
        self.assertEqual(old["evidence"], [])
        self.assertEqual(new["narrative_presence"], "PRESENT_IN_CHECKED_SOURCE")
        self.assertTrue(new["evidence"])
        self.assertIn("2026-10-11T17:00:00+00:00", str(new["evidence"]))
        self.assertTrue(all(item["usable_at_original_cutoff"] is False for item in new["evidence"]))
        before = self.indexed(self.rows())["000001"]
        self.assertTrue(all(item["usable_at_original_cutoff"] is True for item in before["evidence"]))

    def test_future_available_at_does_not_leak_into_current_context(self):
        future = copy.deepcopy(self.claims)
        for claim in future:
            claim["available_at"] = "2026-10-11T19:00:00+00:00"
        rows = self.rows(future)
        self.assertTrue(all(row["coverage"] == "NOT_CHECKED" for row in rows))
        self.assertTrue(all(row["evidence"] == [] for row in rows))

    def test_forum_business_claim_does_not_certify_company_business(self):
        claim = copy.deepcopy(self.claims[1])
        claim["source_id"] = "forum"
        with self.assertRaises(ValueError):
            self.rows([claim])
        primary = self.indexed(self.rows())["000001"]
        self.assertEqual(primary["business_link"], "PRIMARY_SOURCE_CONFIRMED")
        self.assertEqual(primary["coverage"], "LIMITED_CHECKED")

    def test_proxy_and_last_observed_pool_semantics_remain_separate_from_actual_identity(self):
        rows = self.indexed(self.rows())
        for row in rows.values():
            semantics = row["feature_semantics"]
            self.assertEqual(semantics["turnover_vs_prev"], "LAST_OBSERVED_RELAY_POOL_NOT_PREVIOUS_SESSION")
            self.assertEqual(semantics["signal_one_word"], "LOW_TURNOVER_OHLC_PROXY_NOT_ACTUAL_ONE_WORD")
        self.assertEqual(rows["000003"]["actual_one_word_daily"], 1.0)
        self.assertEqual(rows["000001"]["actual_one_word_daily"], 0.0)

    def test_catalyst_review_requires_same_complete_pool(self):
        rows = self.rows()
        draft = self.valid_draft(rows)
        context_review.validate_catalyst_assessments({"rows": rows}, draft)
        draft["rows"].pop()
        with self.assertRaises(ValueError):
            context_review.validate_catalyst_assessments({"rows": rows}, draft)
        draft = self.valid_draft(rows)
        draft["rows"][1]["code"] = draft["rows"][0]["code"]
        with self.assertRaises(ValueError):
            context_review.validate_catalyst_assessments({"rows": rows}, draft)

    def test_missing_coverage_no_catalyst_and_model_negative_cannot_be_negative_reasons(self):
        rows = self.rows()
        for invalid in ("SOURCE_COVERAGE_INCOMPLETE", "NO_CATALYST", "MODEL_NEGATIVE"):
            with self.subTest(reason=invalid):
                draft = self.valid_draft(rows)
                draft["rows"][1]["promotion_judgement"] = "FAIL"
                draft["rows"][1]["catalyst_assessment"]["negative_reason_codes"] = [invalid]
                with self.assertRaises(ValueError):
                    context_review.validate_catalyst_assessments({"rows": rows}, draft)

    def test_fail_requires_actual_counterevidence_and_allows_checked_price_reason(self):
        rows = self.rows()
        draft = self.valid_draft(rows)
        draft["rows"][1]["promotion_judgement"] = "FAIL"
        with self.assertRaises(ValueError):
            context_review.validate_catalyst_assessments({"rows": rows}, draft)
        draft["rows"][1]["catalyst_assessment"]["negative_reason_codes"] = ["PRICE_PATH_WEAKNESS"]
        context_review.validate_catalyst_assessments({"rows": rows}, draft)

    def test_limited_earnings_cannot_be_rewritten_as_material_and_review_cannot_be_skipped(self):
        rows = self.rows()
        self.assertEqual(self.indexed(rows)["000001"]["earnings_impact"], "LIMITED")
        for change in ({"earnings_impact": "MATERIAL"}, {"event_and_business_reviewed": False},
                       {"narrative_presence": "UNKNOWN"}):
            with self.subTest(change=change):
                draft = self.valid_draft(rows)
                draft["rows"][0]["catalyst_assessment"].update(change)
                with self.assertRaises(ValueError):
                    context_review.validate_catalyst_assessments({"rows": rows}, draft)


if __name__ == "__main__":
    unittest.main()
