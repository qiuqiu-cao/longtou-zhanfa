import unittest

import pandas as pd

from build_source_bound_theme_roles_v75 import build, validate_evidence


class SourceBoundThemeRolesTests(unittest.TestCase):
    def setUp(self):
        self.pool = pd.DataFrame([
            {"code": "600001", "boards": 2, "amount": 5e8, "first_seal_time": 100000,
             "pool_assignment": "CLOSING_LIMIT_UP"},
            {"code": "600002", "boards": 1, "amount": 8e8, "first_seal_time": 93000,
             "pool_assignment": "CLOSING_LIMIT_UP"},
            {"code": "600003", "boards": 1, "amount": 9e8, "first_seal_time": 92500,
             "pool_assignment": "FAILED_BOARD_OBSERVATION"},
        ])
        self.evidence = {"signal_date": "2026-09-09", "groups": [{
            "theme_group_id": "coal", "theme": "煤炭开采", "codes": ["600001", "600002", "600003"],
            "url": "https://example.test/coal", "published_at": "2026-09-09T11:04:00+08:00",
            "source_tier": "B_FINANCIAL_NEWS_EXPLICIT_ROSTER",
        }]}

    def test_only_closing_members_and_structure_roles(self):
        result = build(self.pool, self.evidence, "2026-09-09T16:00:00+08:00")
        self.assertEqual(set(result.code), {"600001", "600002"})
        leader = result.set_index("code").loc["600001"]
        support = result.set_index("code").loc["600002"]
        self.assertEqual(leader.theme_role, "THEME_HEIGHT_LEADER")
        self.assertIn("CAPACITY_CORE_BY_CLOSE_AMOUNT", support.identity_tags)
        self.assertIn("FIRST_SEAL_PIONEER", support.identity_tags)
        self.assertTrue(bool(leader.theme_ladder_verified))

    def test_future_publication_rejected(self):
        with self.assertRaisesRegex(ValueError, "future evidence"):
            validate_evidence(self.evidence, "2026-09-09T10:00:00+08:00")

    def test_duplicate_primary_theme_rejected(self):
        duplicate = dict(self.evidence)
        duplicate["groups"] = self.evidence["groups"] * 2
        with self.assertRaisesRegex(ValueError, "multiple primary themes"):
            validate_evidence(duplicate, "2026-09-09T16:00:00+08:00")


if __name__ == "__main__":
    unittest.main()
