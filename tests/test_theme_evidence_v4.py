import tempfile
import unittest
from pathlib import Path

import pandas as pd

import theme_evidence_v4 as themes


class ThemeEvidenceV4Tests(unittest.TestCase):
    def _evidence_row(self, code: str = "600001", name: str = "甲") -> dict:
        return {
            "signal_date": "2026-09-01",
            "code": code,
            "name": name,
            "theme_group_id": "robotics_20260901",
            "theme": "人形机器人",
            "theme_granularity": "细分题材",
            "driver": "公司公告确认机器人业务",
            "evidence_kind": "MEMBERSHIP",
            "source_type": "company_announcement",
            "source_ref": f"https://example.test/{code}",
            "source_published_at": "2026-09-01 10:00:00",
            "timestamp_precision": "exact",
            "membership_basis": "explicit_company_disclosure",
            "evidence_status": "verified",
            "collected_at": "2026-09-01 15:10:00",
        }

    def _registry(self) -> pd.DataFrame:
        frame = pd.DataFrame([
            self._evidence_row("600001", "甲"),
            self._evidence_row("600002", "乙"),
            self._evidence_row("600003", "丙"),
        ])
        return themes.validate_evidence(frame)

    def _pool(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"signal_date": "2026-09-01", "code": "600001", "name": "甲", "boards": 3},
            {"signal_date": "2026-09-01", "code": "600002", "name": "乙", "boards": 2},
            {"signal_date": "2026-09-01", "code": "600003", "name": "丙", "boards": 1},
        ])

    def _candidates(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"date": "2026-09-01", "code": "600001", "name": "甲", "boards": 3,
             "valid_identity": 1, "buyable_eod": 1, "emotion_state": "升温"},
            {"date": "2026-09-01", "code": "600009", "name": "未知", "boards": 2,
             "valid_identity": 1, "buyable_eod": 1, "emotion_state": "升温"},
        ])

    def test_date_only_same_day_is_not_known_by_close(self):
        row = self._evidence_row()
        row["timestamp_precision"] = "date_only"
        row["source_published_at"] = "2026-09-01"
        checked = themes.validate_evidence(pd.DataFrame([row])).iloc[0]
        self.assertFalse(checked["usable_at_signal"])
        self.assertIn("not_known_by_signal_cutoff", checked["usable_reason"])
        self.assertEqual(checked["effective_published_at"], "2026-09-01 23:59:59")

    def test_community_or_unverified_evidence_cannot_become_usable(self):
        row = self._evidence_row()
        row["source_type"] = "stock_forum"
        checked = themes.validate_evidence(pd.DataFrame([row])).iloc[0]
        self.assertFalse(checked["usable_at_signal"])
        self.assertIn("untrusted_source_type", checked["usable_reason"])

    def test_generic_style_concept_is_rejected(self):
        row = self._evidence_row()
        row["theme"] = "融资融券"
        with self.assertRaisesRegex(ValueError, "generic"):
            themes.validate_evidence(pd.DataFrame([row]))

    def test_registry_is_immutable_by_source_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            registry = root / "registry.csv"
            pd.DataFrame([self._evidence_row()]).to_csv(source, index=False)
            themes.import_evidence(source, registry)
            changed = self._evidence_row()
            changed["driver"] = "事后改写"
            pd.DataFrame([changed]).to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, "source reference already registered"):
                themes.import_evidence(source, registry)

    def test_theme_role_uses_verified_same_day_peers_and_excludes_self(self):
        roster, roles = themes.build_theme_roster(
            "2026-09-01", self._registry(), self._pool(), self._candidates()
        )
        self.assertEqual(len(roster), 3)
        leader = roles.set_index("code").loc["600001"]
        self.assertEqual(leader["theme_status"], "verified_same_day_group")
        self.assertEqual(leader["theme_role"], "HEIGHT_CORE")
        self.assertEqual(leader["theme_support_ex_self"], 2)
        self.assertEqual(leader["theme_first_support_ex_self"], 1)
        self.assertEqual(leader["theme_middle_support_ex_self"], 1)
        unknown = roles.set_index("code").loc["600009"]
        self.assertEqual(unknown["theme_status"], "unavailable")

    def test_same_height_tie_is_competing_cluster_not_unique_core(self):
        registry = self._registry()
        pool = self._pool().copy()
        pool.loc[pool["code"].isin(["600001", "600002"]), "boards"] = 3
        candidates = self._candidates().iloc[[0]].copy()
        _, roles = themes.build_theme_roster("2026-09-01", registry, pool, candidates)
        leader = roles.iloc[0]
        self.assertEqual(leader["same_height_rivals"], 1)
        self.assertEqual(leader["theme_role"], "CO_HEIGHT_CLUSTER")

    def test_broad_theme_cannot_supply_ladder_or_open_sync(self):
        broad = pd.DataFrame([
            {**self._evidence_row("600001", "甲"), "theme_granularity": "板块题材"},
            {**self._evidence_row("600002", "乙"), "theme_granularity": "板块题材"},
            {**self._evidence_row("600003", "丙"), "theme_granularity": "板块题材"},
        ])
        registry = themes.validate_evidence(broad)
        roster, roles = themes.build_theme_roster(
            "2026-09-01", registry, self._pool(), self._candidates().iloc[[0]]
        )
        leader = roles.iloc[0]
        self.assertEqual(leader["theme_status"], "verified_same_day_broad_group")
        self.assertEqual(leader["theme_role"], "BROAD_THEME_REFERENCE")
        self.assertFalse(leader["theme_ladder_verified"])

        open_members = pd.DataFrame([
            {"signal_date": "2026-09-01", "trade_date": "2026-09-02", "asof": "09:35",
             "theme_group_id": "robotics_20260901", "member_code": code, "member_name": name,
             "supportive": 1, "source": "broker", "source_timestamp": "2026-09-02 09:35:00",
             "data_status": "verified_broker_export"}
            for code, name in [("600002", "乙"), ("600003", "丙")]
        ])
        sync = themes.compute_open_sync(open_members, roster, roles).iloc[0]
        self.assertTrue(pd.isna(sync["theme_sync_open"]))
        self.assertEqual(sync["theme_sync_basis"], "insufficient_verified_theme_members")
        self.assertIn("broad", sync["warning"])

    def test_open_sync_is_recomputed_from_two_observed_peers(self):
        roster, roles = themes.build_theme_roster(
            "2026-09-01", self._registry(), self._pool(), self._candidates()
        )
        open_members = pd.DataFrame([
            {"signal_date": "2026-09-01", "trade_date": "2026-09-02", "asof": "09:35",
             "theme_group_id": "robotics_20260901", "member_code": "600002", "member_name": "乙",
             "supportive": 1, "source": "broker", "source_timestamp": "2026-09-02 09:35:00",
             "data_status": "verified_broker_export"},
            {"signal_date": "2026-09-01", "trade_date": "2026-09-02", "asof": "09:35",
             "theme_group_id": "robotics_20260901", "member_code": "600003", "member_name": "丙",
             "supportive": 0, "source": "broker", "source_timestamp": "2026-09-02 09:35:00",
             "data_status": "verified_broker_export"},
        ])
        sync = themes.compute_open_sync(open_members, roster, roles)
        leader = sync.set_index("code").loc["600001"]
        self.assertEqual(leader["theme_sync_basis"], "verified_theme_registry")
        self.assertAlmostEqual(leader["theme_sync_open"], 0.5)
        self.assertEqual(leader["theme_observed_peers"], 2)

    def test_open_sync_rejects_member_outside_frozen_roster(self):
        roster, roles = themes.build_theme_roster(
            "2026-09-01", self._registry(), self._pool(), self._candidates()
        )
        open_members = pd.DataFrame([{
            "signal_date": "2026-09-01", "trade_date": "2026-09-02", "asof": "09:35",
            "theme_group_id": "robotics_20260901", "member_code": "600999", "member_name": "伪成员",
            "supportive": 1, "source": "broker", "source_timestamp": "2026-09-02 09:35:00",
            "data_status": "verified_broker_export",
        }])
        with self.assertRaisesRegex(ValueError, "outside verified"):
            themes.compute_open_sync(open_members, roster, roles)


if __name__ == "__main__":
    unittest.main()
