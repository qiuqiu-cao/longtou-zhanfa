"""Regression checks for the native-output/G1 rank boundary; no model gain."""
import copy
import unittest

from model_prediction_view import relay_view
import theme_structure_g1 as g1


class OriginalProjectionTests(unittest.TestCase):
    def fixture(self):
        roster = [{"code": c, "name": c, "boards": 2, "actual_one_word_daily": 0}
                  for c in ("000003", "000001", "000002")]
        scores = {"000001": .7, "000002": .456, "000003": .4}
        rows = [{**r, "raw_score": scores[r["code"]], "predicted_positive": scores[r["code"]] > .5}
                for r in roster]
        native = {"status": "KNOWN", "candidates": rows, "ranking": ["000001", "000002", "000003"],
                  "ranking_rule": "raw_score descending; code ascending; no replacement", "reason": None}
        return native, roster

    def test_native_code_order_g1_rank_and_calls_roundtrip_without_mutation(self):
        native, roster = self.fixture()
        before = copy.deepcopy((native, roster))
        view = relay_view(native, roster)
        self.assertEqual((native, roster), before)
        self.assertEqual([r["code"] for r in view["candidates"]], [r["code"] for r in native["candidates"]])
        theme = {"signal_date": "2026-10-13", "knowledge_cutoff": "2026-10-13T20:00:00+08:00",
                 "all_candidates": [{**r, "pool_kind": "limit_up", "category_status": "PROVIDER_CURRENT_CATEGORY_ONLY"}
                                    for r in roster], "groups": [], "pool_counts": {"closing": 3}}
        base = {"signal_date": "2026-10-13", "target_date": "2026-10-14",
                "knowledge_cutoff": "2026-10-13T19:00:00+08:00", "relay": view}
        overlay = g1.analyse(theme, base, {"aliases": {}, "families": {}})
        expected_ranks = {code: i for i, code in enumerate(native["ranking"], 1)}
        self.assertEqual([r["rank"] for r in overlay["all_candidates"]], [1, 2, 3])
        for row in overlay["all_candidates"]:
            original = next(r for r in native["candidates"] if r["code"] == row["code"])
            self.assertEqual({k: row[k] for k in ("boards", "raw_score", "predicted_positive")},
                             {k: original[k] for k in ("boards", "raw_score", "predicted_positive")})
            self.assertEqual(row["rank"], expected_ranks[row["code"]])

    def test_tie_uses_fixed_code_order_and_retains_negative_top_candidate(self):
        native, roster = self.fixture()
        for r in native["candidates"]:
            r.update(raw_score=.5, predicted_positive=False)
        native["ranking"] = sorted(r["code"] for r in roster)
        view = relay_view(native, roster)
        self.assertEqual({r["code"]: r["rank"] for r in view["candidates"]},
                         {"000001": 1, "000002": 2, "000003": 3})
        self.assertTrue(all(r["predicted_positive"] is False for r in view["candidates"]))

    def test_incomplete_duplicate_or_extra_native_roster_rejected(self):
        for operation in ("missing", "duplicate", "extra"):
            native, roster = self.fixture()
            if operation == "missing": native["candidates"].pop()
            if operation == "duplicate": native["candidates"][0] = copy.deepcopy(native["candidates"][1])
            if operation == "extra": native["candidates"][0]["code"] = "000009"
            with self.subTest(operation=operation), self.assertRaisesRegex(ValueError, "complete original"):
                relay_view(native, roster)

    def test_supplied_rank_or_native_order_conflict_rejected(self):
        native, roster = self.fixture()
        native["ranking"].reverse()
        with self.assertRaisesRegex(ValueError, "fixed ranking"):
            relay_view(native, roster)
        native, roster = self.fixture()
        native["candidates"][0]["rank"] = 1
        with self.assertRaisesRegex(ValueError, "candidate rank"):
            relay_view(native, roster)

    def test_score_call_and_fact_conflicts_rejected(self):
        for key, value in (("raw_score", float("nan")), ("raw_score", None),
                           ("predicted_positive", True), ("actual_one_word_daily", 1)):
            native, roster = self.fixture()
            native["candidates"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                relay_view(native, roster)

    def test_unknown_retains_all_candidates_and_does_not_invent_rank(self):
        native, roster = self.fixture()
        native.update(status="UNKNOWN", ranking=[])
        for row in native["candidates"]:
            row.update(raw_score=None, predicted_positive=None)
        view = relay_view(native, roster)
        self.assertEqual(len(view["candidates"]), 3)
        self.assertTrue(all(r["rank"] is None for r in view["candidates"]))
        native["ranking"] = ["000001"]
        with self.assertRaisesRegex(ValueError, "unknown scores"):
            relay_view(native, roster)

    def test_no_topfive_truncation(self):
        native, roster = self.fixture()
        for i in range(4, 9):
            row = {"code": f"{i:06}", "name": f"{i:06}", "boards": 2, "actual_one_word_daily": 0}
            roster.append(row)
            native["candidates"].append({**row, "raw_score": .2, "predicted_positive": False})
            native["ranking"].append(row["code"])
        view = relay_view(native, roster)
        self.assertEqual(len(view["candidates"]), 8)
        self.assertEqual(sorted(r["rank"] for r in view["candidates"]), list(range(1, 9)))


if __name__ == "__main__":
    unittest.main()
