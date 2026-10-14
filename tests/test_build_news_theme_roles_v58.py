import unittest

import pandas as pd

from build_news_theme_roles_v58 import build


class NewsThemeRolesTests(unittest.TestCase):
    def test_only_pool_members_are_emitted_and_leader_is_height_based(self):
        pool = pd.DataFrame([
            {"code": "600108", "boards": 4, "pool_assignment": "CLOSING_LIMIT_UP"},
            {"code": "600354", "boards": 3, "pool_assignment": "CLOSING_LIMIT_UP"},
            {"code": "601952", "boards": 1, "pool_assignment": "FAILED_BOARD_OBSERVATION"},
        ])
        result = build(pool)
        self.assertEqual(set(result.code), {"600108", "600354"})
        leader = result[result.code.eq("600108")].iloc[0]
        self.assertEqual(leader.theme_role, "THEME_HEIGHT_LEADER")
        self.assertEqual(leader.theme_support_ex_self, 1)

    def test_absence_is_not_emitted_as_negative_evidence(self):
        pool = pd.DataFrame([{
            "code": "002403", "boards": 4, "pool_assignment": "CLOSING_LIMIT_UP"
        }])
        self.assertTrue(build(pool).empty)


if __name__ == "__main__":
    unittest.main()
