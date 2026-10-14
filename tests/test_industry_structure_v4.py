import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import industry_structure_v4 as structure


class IndustryStructureV4Tests(unittest.TestCase):
    def _pool(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"date": "2026-09-01", "code": "600001", "name": "甲", "industry": "设备",
             "boards": 1, "first_seal": "092500", "last_seal": "092500", "bomb_count": 0, "turnover": 2},
            {"date": "2026-09-01", "code": "600002", "name": "乙", "industry": "设备",
             "boards": 2, "first_seal": "093000", "last_seal": "100000", "bomb_count": 1, "turnover": 8},
            {"date": "2026-09-01", "code": "600003", "name": "丙", "industry": "设备",
             "boards": 3, "first_seal": "094000", "last_seal": "100000", "bomb_count": 1, "turnover": 12},
            {"date": "2026-09-01", "code": "000001", "name": "丁", "industry": "零售",
             "boards": 4, "first_seal": "100000", "last_seal": "110000", "bomb_count": 2, "turnover": 20},
        ])

    def test_industry_ladder_is_proxy_not_verified_theme(self):
        structures = structure.build_industry_structure(
            "2026-09-01", self._pool(), pd.DataFrame(columns=self._pool().columns),
            pd.DataFrame(columns=self._pool().columns),
        )
        equipment = structures.set_index("industry").loc["设备"]
        self.assertTrue(equipment["ladder_reinforced_proxy"])
        self.assertEqual(equipment["structure_label"], "PROXY_LADDER_REINFORCEMENT")
        self.assertFalse(equipment["theme_verified"])
        self.assertEqual(equipment["evidence_level"], "INDUSTRY_PROXY_NOT_THEME")

    def test_failed_pool_changes_divergence_without_erasing_ladder(self):
        failed = pd.DataFrame([
            {"industry": "设备", "name": "炸一"}, {"industry": "设备", "name": "炸二"},
            {"industry": "设备", "name": "炸三"},
        ])
        structures = structure.build_industry_structure(
            "2026-09-01", self._pool(), failed,
            pd.DataFrame(columns=self._pool().columns),
        )
        equipment = structures.set_index("industry").loc["设备"]
        self.assertTrue(equipment["ladder_reinforced_proxy"])
        self.assertTrue(equipment["divergence_pressure_proxy"])
        self.assertEqual(equipment["structure_label"], "PROXY_LADDER_WITH_DIVERGENCE")

    def test_candidate_support_excludes_itself(self):
        pool = self._pool()
        structures = structure.build_industry_structure(
            "2026-09-01", pool, pd.DataFrame(columns=pool.columns),
            pd.DataFrame(columns=pool.columns),
        )
        candidates = pd.DataFrame([{"date": "2026-09-01", "code": "600003", "name": "丙", "boards": 3}])
        evidence = structure.candidate_industry_evidence("2026-09-01", candidates, pool, structures).iloc[0]
        self.assertEqual(evidence["industry_limit_support_ex_self"], 2)
        self.assertEqual(evidence["industry_first_support_ex_self"], 1)
        self.assertEqual(evidence["industry_middle_support_ex_self"], 1)

    def test_missing_industry_stays_unavailable_not_zero_support(self):
        pool = self._pool()
        candidates = pd.DataFrame([{"date": "2026-09-01", "code": "600999", "name": "未知", "boards": 2}])
        evidence = structure.candidate_industry_evidence(
            "2026-09-01", candidates, pool,
            structure.build_industry_structure(
                "2026-09-01", pool, pd.DataFrame(columns=pool.columns),
                pd.DataFrame(columns=pool.columns),
            ),
        ).iloc[0]
        self.assertEqual(evidence["industry_evidence_status"], "unavailable")
        self.assertTrue(math.isnan(evidence["industry_limit_support_ex_self"]))
        self.assertFalse(evidence["theme_verified"])

    def test_temporal_context_uses_only_prior_archived_day(self):
        current = pd.DataFrame([{
            "date": "2026-09-02", "industry": "设备", "limit_up_count": 4,
            "consecutive_count": 2, "failed_rate": 0.5, "max_board": 3,
        }])
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "archive.csv"
            pd.DataFrame([
                {"date": "2026-09-01", "industry": "设备", "limit_up_count": 2,
                 "consecutive_count": 1, "failed_rate": 0.1, "max_board": 2},
                {"date": "2026-09-03", "industry": "设备", "limit_up_count": 99,
                 "consecutive_count": 99, "failed_rate": 0.0, "max_board": 9},
            ]).to_csv(archive, index=False)
            enriched = structure.attach_temporal_context(current, archive, "2026-09-02").iloc[0]
        self.assertEqual(enriched["previous_proxy_date"], "2026-09-01")
        self.assertEqual(enriched["limit_up_change"], 2)
        self.assertEqual(enriched["consecutive_change"], 1)
        self.assertEqual(enriched["industry_proxy_trend"], "PROXY_DIVERGENCE_EXPANDING")


if __name__ == "__main__":
    unittest.main()
