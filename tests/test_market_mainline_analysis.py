"""Synthetic analysis checks; no fitting or real prediction changes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import market_mainline_analysis as analysis


class MainlineAnalysis(unittest.TestCase):
    def fixture(self, name="机器人"):
        rows = [{"code": f"{i:06d}", "name": f"synthetic{i}", "boards": b,
                 "pool_kind": "limit_up", "category_status": "PROVIDER_CURRENT_CATEGORY_ONLY",
                 "original_base": {"predicted_positive": False, "original_raw_score": .2}}
                for i, b in enumerate([1, 2, 3, 1], 1)]
        rows.append({"code": "000005", "name": "unknown failed", "boards": None,
                     "pool_kind": "failed_board", "category_status": "UNKNOWN"})
        theme = {"signal_date": "2026-10-08", "knowledge_cutoff": "2026-10-08T10:00:00Z",
                 "generated_at": "2026-10-08T10:00:01Z", "official_full_market": False,
                 "scope": "PROVIDER_NOT_EXCHANGE_CERTIFIED", "all_candidates": rows,
                 "groups": [{"group_name": name+"概念", "evidence_lane": "PROVIDER_CURRENT_CATEGORY",
                             "group_id": "t", "member_codes": ["000001", "000002", "000003"]},
                            {"group_name": "融资融券", "evidence_lane": "PROVIDER_CURRENT_CATEGORY",
                             "group_id": "generic", "member_codes": ["000001", "000002", "000003", "000004"]}]}
        inputs = {"signal_date": "2026-10-08", "knowledge_cutoff": "2026-10-08T12:00:00Z",
                  "defined_at": "2026-10-08T11:59:59Z",
                  "sources": [{"source_id": "synthetic", "url": "https://example.test/original",
                               "published_at": "2026-10-08T09:00:00Z", "obtained_at": "2026-10-08T09:01:00Z",
                               "text_path": "unused.txt", "text_sha256": "synthetic",
                               "claims": [{"theme_name": name, "stance": "ACTIVE", "excerpt": name+"板块走强"},
                                          {"theme_name": "融资融券", "stance": "ACTIVE", "excerpt": "融资融券活跃"}]}]}
        policy = {"version": "FIXED_SYNTHETIC", "defined_at": "2026-10-08T11:00:00Z",
                  "min_relay": 2, "min_first": 1, "min_ladder_prefix": 3,
                  "excluded_names": ["融资融券"], "excluded_patterns": [], "aliases": {}, "families": {},
                  "positive_tokens": ["走强", "活跃", "涨幅居前"], "negative_tokens": ["走弱", "下挫", "下跌"]}
        return theme, inputs, policy

    def test_generic_theme_negative_model_and_immutable_inputs(self):
        args = self.fixture()
        before = copy.deepcopy(args)
        result = analysis.analyse(*args)
        self.assertEqual(result["mainline"], "机器人")
        self.assertEqual(result["status"], "INFERRED_SHORT_TERM_MAINLINE")
        self.assertEqual(result["candidates"][0]["original_negative_relay_codes"], ["000002", "000003"])
        self.assertEqual(result["unresolved"]["failed_category_unknown_codes"], ["000005"])
        self.assertFalse(result["candidates"][0]["failed_membership_coverage_complete"])
        self.assertFalse(result["model_or_prediction_changed"])
        self.assertEqual(args, before)

    def test_battery_is_not_special_winner(self):
        args = self.fixture("电池")
        self.assertEqual(analysis.analyse(*args)["mainline"], "电池")

    def test_isolated_eight_board_is_unknown(self):
        args = self.fixture()
        args[0]["groups"][0]["member_codes"] = ["000002"]
        args[0]["all_candidates"][1]["boards"] = 8
        self.assertEqual(analysis.analyse(*args)["status"], "UNKNOWN")

    def test_competing_equal_structures(self):
        theme, inputs, policy = self.fixture()
        extra = copy.deepcopy(theme["all_candidates"][:3])
        for row in extra:
            row["code"] = str(int(row["code"])+10).zfill(6)
        theme["all_candidates"] += extra
        theme["groups"].append({"group_name": "航运", "evidence_lane": "PROVIDER_INDUSTRY_PROXY",
                                "group_id": "shipping", "member_codes": [r["code"] for r in extra]})
        inputs["sources"][0]["claims"].append({"theme_name": "航运", "stance": "ACTIVE", "excerpt": "航运涨幅居前"})
        self.assertEqual(analysis.analyse(theme, inputs, policy)["status"], "COMPETING")

    def test_parent_child_not_competitors(self):
        theme, inputs, policy = self.fixture("固态电池")
        theme["groups"].append({"group_name": "电池", "evidence_lane": "PROVIDER_SEMANTIC_FAMILY",
                                "group_id": "battery", "member_codes": ["000001", "000002", "000003", "000004"]})
        policy["families"] = {"battery": {"canonical": "电池", "match_active_keyword": "电池",
            "source_lane": "PROVIDER_SEMANTIC_FAMILY", "group_id": "battery", "children": ["固态电池"]}}
        result = analysis.analyse(theme, inputs, policy)
        self.assertEqual(result["mainline"], "电池")
        self.assertNotIn("固态电池", result["competitive_lines"])

    def test_closing_unknown_blocks_confident_mainline(self):
        args = self.fixture()
        args[0]["all_candidates"][3]["category_status"] = "UNKNOWN"
        self.assertEqual(analysis.analyse(*args)["status"], "UNKNOWN")

    def test_weak_and_background_do_not_activate(self):
        for stance in ["WEAK", "BACKGROUND", "CAUTION"]:
            args = self.fixture()
            args[1]["sources"][0]["claims"][0]["stance"] = stance
            self.assertEqual(analysis.analyse(*args)["status"], "UNKNOWN")

    def test_hand_written_active_without_positive_or_negated_rejected(self):
        for excerpt in ["机器人政策发布", "机器人并未走强", "机器人走强后下跌"]:
            args = self.fixture()
            args[1]["sources"][0]["claims"][0]["excerpt"] = excerpt
            with self.assertRaises(ValueError):
                analysis.analyse(*args)

    def test_future_news_and_policy_rejected(self):
        for location in ["news", "policy", "theme"]:
            args = self.fixture()
            if location == "news":
                args[1]["sources"][0]["obtained_at"] = "2026-10-09T10:00:00Z"
            elif location == "policy":
                args[2]["defined_at"] = "2026-10-09T10:00:00Z"
            else:
                args[0]["knowledge_cutoff"] = "2026-10-09T10:00:00Z"
            with self.assertRaises(ValueError):
                analysis.analyse(*args)

    def test_file_hash_and_excerpts_and_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            theme, inputs, policy = self.fixture()
            text = root/"original.txt"
            text.write_text("机器人板块走强 融资融券活跃", encoding="utf-8")
            inputs["sources"][0].update(text_path=str(text), text_sha256=analysis.sha(text))
            original = root/"frozen_prediction.json"
            original.write_text('{"prediction":"unchanged"}', encoding="utf-8")
            frozen_bytes = original.read_bytes()
            theme["source_hashes"] = {str(original): analysis.sha(original)}
            paths = [root/"theme.json", root/"inputs.json", root/"policy.json"]
            for p, obj in zip(paths, [theme, inputs, policy]):
                p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
            result = analysis.run(*paths, root/"output")
            self.assertEqual(result["mainline"], "机器人")
            self.assertEqual(original.read_bytes(), frozen_bytes)
            with self.assertRaises(FileExistsError):
                analysis.run(*paths, root/"output")
            text.write_text("tampered", encoding="utf-8")
            with self.assertRaises(ValueError):
                analysis.run(*paths, root/"second_output")


if __name__ == "__main__":
    unittest.main()
