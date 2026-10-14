import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import symbolic_name_theme_v217 as symbolic


class SymbolicNameThemeV217Tests(unittest.TestCase):
    def snapshot(self):
        rows = [
            ("600001", "华甲公司", 4, "行业甲"),
            ("600002", "乙华公司", 3, "行业乙"),
            ("600003", "南华公司", 2, "行业丙"),
            ("600004", "华丁公司", 1, "行业丁"),
            ("600005", "普通一", 1, "行业甲"),
            ("600006", "普通二", 1, "行业乙"),
            ("600007", "普通三", 1, "行业丙"),
            ("600008", "普通四", 1, "行业丁"),
        ]
        records = []
        for code, name, boards, industry in rows:
            records.append({
                "signal_date": "2026-09-18", "code": code, "name": name,
                "boards": boards, "board_group": (
                    "FIRST_BOARD" if boards == 1 else "CONSECUTIVE_BOARD"),
                "pool_assignment": "CLOSING_LIMIT_UP", "is_mainboard": True,
                "industry_provider": industry,
                "observed_at": "2026-09-18T15:00:00+08:00",
                "available_at": "2026-09-18T18:00:00+08:00",
            })
        records.append({
            "signal_date": "2026-09-18", "code": "600009", "name": "华戊公司",
            "boards": 0, "board_group": "FAILED_BOARD",
            "pool_assignment": "FAILED_BOARD_OBSERVATION", "is_mainboard": True,
            "industry_provider": "行业戊",
            "observed_at": "2026-09-18T15:00:00+08:00",
            "available_at": "2026-09-18T18:00:00+08:00",
        })
        return pd.DataFrame(records)

    def manifest(self):
        return {
            "signal_date": "2026-09-18",
            "available_at": "2026-09-18T18:00:00+08:00",
            "mainboard_closing_limit_rows": 8,
            "mainboard_failed_board_rows": 1,
            "provider_pool_status": "verified",
            "full_market_completeness": "not_independently_verified",
            "point_in_time_security_scope_complete": False,
        }

    def test_detects_name_ladder_but_never_promotes_it_to_real_theme(self):
        report = symbolic.build(
            self.snapshot(), self.manifest(), "2026-09-18T20:00:00+08:00")
        top = report["top_cluster"]
        self.assertEqual(top["token"], "华")
        self.assertEqual(top["member_count"], 4)
        self.assertEqual(top["relay_member_count"], 3)
        self.assertEqual(top["failed_board_member_count"], 1)
        self.assertEqual(top["distinct_board_levels"], [1, 2, 3, 4])
        self.assertEqual(top["classification"], "SYMBOLIC_LADDER_CANDIDATE")
        self.assertFalse(top["real_theme_eligible"])
        self.assertFalse(top["candidate_reranking_allowed"])
        self.assertFalse(report["governance"]["direct_entry_allowed"])

    def test_corporate_descriptor_tokens_are_excluded(self):
        report = symbolic.build(
            self.snapshot(), self.manifest(), "2026-09-18T20:00:00+08:00")
        tokens = {row["token"] for row in report["clusters"]}
        self.assertNotIn("股", tokens)
        self.assertNotIn("份", tokens)
        self.assertNotIn("科技", tokens)

    def test_rejects_evidence_that_was_not_available_by_asof(self):
        with self.assertRaisesRegex(ValueError, "not available by asof"):
            symbolic.build(
                self.snapshot(), self.manifest(), "2026-09-18T17:00:00+08:00")

    def test_previous_session_comparison_requires_frozen_rules(self):
        previous = symbolic.build(
            self.snapshot(), self.manifest(), "2026-09-18T20:00:00+08:00")
        current_snapshot = self.snapshot().copy()
        current_snapshot["signal_date"] = "2026-09-21"
        current_snapshot["observed_at"] = "2026-09-21T15:00:00+08:00"
        current_snapshot["available_at"] = "2026-09-21T18:00:00+08:00"
        current_manifest = dict(self.manifest())
        current_manifest.update({
            "signal_date": "2026-09-21",
            "available_at": "2026-09-21T18:00:00+08:00",
        })
        report = symbolic.build(
            current_snapshot, current_manifest, "2026-09-21T20:00:00+08:00", previous)
        comparison = report["top_cluster"]["previous_session_comparison"]
        self.assertEqual(comparison["status"], "COMPARABLE_FROZEN_RULES")
        self.assertEqual(comparison["member_count_change"], 0)

    def test_cli_artifact_is_immutable_and_hashes_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot_path = root / "snapshot.csv"
            manifest_path = root / "manifest.json"
            output = root / "result"
            self.snapshot().to_csv(snapshot_path, index=False)
            manifest_path.write_text(
                json.dumps(self.manifest(), ensure_ascii=False), encoding="utf-8")
            report = symbolic.run(
                str(snapshot_path), str(manifest_path), str(output),
                "2026-09-18T20:00:00+08:00")
            self.assertTrue((output / "report.json").exists())
            self.assertIn(str(snapshot_path.resolve()), report["source_hashes"])
            with self.assertRaises(FileExistsError):
                symbolic.run(
                    str(snapshot_path), str(manifest_path), str(output),
                    "2026-09-18T20:00:00+08:00")


if __name__ == "__main__":
    unittest.main()
