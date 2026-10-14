import copy
import unittest

from full_pool_rank_dataset import TARGET_DEFINITION, SELECTION_SCOPE, build


def observation(code="000001", boards=1, label=True, touched=True, one_price=False):
    group = "FIRST_BOARD" if boards == 1 else "CONSECUTIVE_BOARD"
    return {
        "candidate_id": "2026-01-02:" + code,
        "code": code,
        "signal_date": "2026-01-02",
        "boards": boards,
        "board_group": group,
        "pool_assignment": "CLOSING_LIMIT_UP",
        "is_mainboard": True,
        "is_st": False,
        "security_status": "verified",
        "limit_up_status": "verified",
        "identity_status": "verified",
        "observed_at": "2026-01-02T15:00:00+08:00",
        "available_at": "2026-01-02T15:01:00+08:00",
        "decision_at": "2026-01-02T15:05:00+08:00",
        "source": "archive",
        "features": {"boards": boards, "future_extra": 9},
        "feature_evidence": {"boards": {"observed_at": "2026-01-02T15:00:00+08:00",
                                           "available_at": "2026-01-02T15:01:00+08:00",
                                           "source": "archive", "definition_id": "boards-v1",
                                           "data_status": "verified"}},
        "opportunity_outcome": {"status": "VERIFIED", "available_at": "2026-01-06T15:05:00+08:00",
                                "next_session_touched_limit": touched,
                                "next_session_one_price_limit": one_price,
                                "b2_close_mark_net_return_proxy": .03 if label else -.02,
                                "target_label": label, "source": "daily-bars",
                                "execution_verified": False},
    }


def policy():
    return {"policy_id": "rank-v1", "frozen_at": "2026-01-01T12:00:00+08:00",
            "available_at": "2026-01-01T12:01:00+08:00", "feature_allowlist": ["boards"],
            "target_definition": TARGET_DEFINITION, "selection_scope": SELECTION_SCOPE}


def manifest(rows, **changes):
    counts = {}
    for row in rows:
        counts[str(row["boards"])] = counts.get(str(row["boards"]), 0) + 1
    item = {"session_date": "2026-01-02", "candidate_ids": [row["candidate_id"] for row in rows],
            "board_level_counts": counts, "closing_limit_count": len(rows), "universe_complete": True,
            "point_in_time_status_complete": True, "all_observed_board_levels_reported": True,
            "failed_board_pool_mixed_with_main": False, "source": "closing-pool"}
    item.update(changes)
    return item


class FullPoolRankDatasetTests(unittest.TestCase):
    def test_valid_rows_are_proxy_only_and_include_both_groups(self):
        rows = [observation(), observation("600001", 2)]
        result = build(rows, [manifest(rows)], policy(), "2026-01-10T12:00:00+08:00")
        self.assertEqual(result["labeled_proxy_count"], 2)
        self.assertFalse(result["actual_trade_target"])
        self.assertFalse(result["can_claim_complete_trade_win_probability"])
        self.assertTrue(result["nonselected_candidates_are_included"])
        self.assertEqual(result["group_labeled_counts"]["FIRST_BOARD"], 1)
        self.assertEqual(result["group_labeled_counts"]["CONSECUTIVE_BOARD"], 1)

    def test_one_price_case_cannot_be_positive_opportunity(self):
        row = observation(one_price=True)
        result = build([row], [manifest([row])], policy(), "2026-01-10T12:00:00+08:00")
        self.assertIn("INVALID_OR_INCONSISTENT_OPPORTUNITY_TARGET", result["rows"][0]["issues"])

    def test_unavailable_is_preserved_not_negative(self):
        row = observation()
        row["opportunity_outcome"] = {"status": "UNAVAILABLE"}
        result = build([row], [manifest([row])], policy(), "2026-01-10T12:00:00+08:00")
        self.assertIsNone(result["rows"][0]["opportunity_proxy_label"])
        self.assertEqual(result["unresolved_count"], 1)
        self.assertFalse(result["unresolved_candidates_are_negative_labels"])

    def test_verified_no_touch_can_be_resolved_false_without_mark(self):
        row = observation(label=False, touched=False)
        row["opportunity_outcome"]["b2_close_mark_net_return_proxy"] = None
        result = build([row], [manifest([row])], policy(), "2026-01-10T12:00:00+08:00")
        self.assertFalse(result["rows"][0]["opportunity_proxy_label"])
        self.assertTrue(result["rows"][0]["label_available"])

    def test_future_feature_is_rejected(self):
        row = observation()
        row["feature_evidence"]["boards"]["available_at"] = "2026-01-03T09:25:00+08:00"
        result = build([row], [manifest([row])], policy(), "2026-01-10T12:00:00+08:00")
        self.assertIn("FEATURE_UNAVAILABLE_OR_FUTURE_boards", result["rows"][0]["issues"])

    def test_manifest_must_match_every_candidate(self):
        rows = [observation(), observation("600001", 2)]
        bad = manifest(rows)
        bad["candidate_ids"] = bad["candidate_ids"][:1]
        result = build(rows, [bad], policy(), "2026-01-10T12:00:00+08:00")
        self.assertIn("INCOMPLETE_FULL_POOL_SESSION_MANIFEST", result["global_issues"])

    def test_security_scope_is_not_inferred_from_code(self):
        row = observation()
        row["security_status"] = "inferred"
        result = build([row], [manifest([row])], policy(), "2026-01-10T12:00:00+08:00")
        self.assertIn("INVALID_OBSERVATION_SCOPE_OR_SOURCE", result["rows"][0]["issues"])

    def test_duplicate_candidate_date_rejected(self):
        row = observation()
        with self.assertRaises(ValueError):
            build([row, copy.deepcopy(row)], [manifest([row])], policy(), "2026-01-10T12:00:00+08:00")

    def test_unregistered_feature_is_ignored(self):
        row = observation()
        result = build([row], [manifest([row])], policy(), "2026-01-10T12:00:00+08:00")
        self.assertEqual(result["rows"][0]["ignored_unregistered_features"], ["future_extra"])

    def test_policy_target_is_exact(self):
        bad = policy()
        bad["target_definition"] = "profit"
        row = observation()
        with self.assertRaises(ValueError):
            build([row], [manifest([row])], bad, "2026-01-10T12:00:00+08:00")
