import unittest

import pandas as pd

from build_observed_theme_roles_v168 import build, validate
from capture_theme_roster_v168 import parse_page, select_primary_theme


class ThemeSourceConfidenceV168Tests(unittest.TestCase):
    def pool(self):
        return pd.DataFrame([
            {"code": "600001", "name": "甲", "boards": 2, "amount": 5e8,
             "first_seal_time": 93500, "pool_assignment": "CLOSING_LIMIT_UP", "is_mainboard": True},
            {"code": "000002", "name": "乙", "boards": 1, "amount": 8e8,
             "first_seal_time": 93000, "pool_assignment": "CLOSING_LIMIT_UP", "is_mainboard": True},
        ])

    def evidence(self, claim="secondary_market_narrative"):
        return {
            "signal_date": "2026-09-16",
            "sources": [{
                "source_id": "s1", "url": "https://example.test/review",
                "available_at": "2026-09-17T04:00:00+08:00", "published_at": None,
                "timestamp_basis": "retrieval_time_only_publication_time_unavailable",
                "source_tier": "C_PUBLIC_REVIEW_EXPLICIT_ROSTER",
                "claim_status": claim, "page_sha256": "a" * 64,
            }],
            "groups": [{
                "theme_group_id": "pcb", "theme": "PCB板",
                "codes": ["600001", "000002"], "source_id": "s1",
                "claim_status": claim, "source_tier": "C_PUBLIC_REVIEW_EXPLICIT_ROSTER",
            }],
        }

    def test_secondary_roster_is_not_promoted_to_verified(self):
        result = build(self.pool(), self.evidence(), "2026-09-17T04:10:00+08:00")
        self.assertTrue(result.theme_status.eq("observed_explicit_secondary_roster").all())
        self.assertTrue(result.evidence_published_at.eq("").all())
        self.assertTrue(result.evidence_available_at.ne("").all())

    def test_future_retrieval_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "after asof"):
            validate(self.evidence(), "2026-09-17T03:59:00+08:00")

    def test_capture_keeps_multi_tag_as_ambiguous_instead_of_forcing_first(self):
        html = '''<div data-stock-card><a href="https://quote.eastmoney.com/sh600001.html">600001</a>
        <div class="text-sm line-clamp-2"><span title="PCB板|国产芯片">[PCB板]</span></div></div>
        <div data-stock-card><a href="https://quote.eastmoney.com/sz000002.html">000002</a>
        <div class="text-sm line-clamp-2"><span title="PCB板">[PCB板]</span></div></div>'''
        observations, groups = parse_page(html, self.pool())
        self.assertIsNone(observations[0]["primary_theme"])
        self.assertTrue(observations[0]["primary_driver_ambiguous"])
        self.assertEqual(observations[0]["all_source_tags"], ["PCB板", "国产芯片"])
        self.assertEqual(groups, [])

    def test_generic_tags_are_removed_only_when_one_specific_tag_remains(self):
        self.assertEqual(
            select_primary_theme(["其他", "业绩增长", "国产芯片"]),
            ("国产芯片", "unique_specific_tag_after_generic_filter"),
        )
        self.assertEqual(
            select_primary_theme(["智能电网", "福建自贸-海西概念"]),
            (None, "multiple_specific_tags_primary_driver_ambiguous"),
        )


if __name__ == "__main__":
    unittest.main()
