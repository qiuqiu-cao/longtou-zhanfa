import copy
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

import leader_role_engine as e
import relay_pairwise_comparison as p

DAY = "2026-09-11"


def path(code, closes, complete=True):
    return {"signal_date": DAY, "code": code, "previous_session": "2026-09-10",
            "previous_close": 100., "price_reconciled": True, "grid_complete": complete,
            "bars": [{"minute": DAY+" "+minute, "close": value}
                     for minute, value in zip(("09:35:00", "09:40:00"), closes)]}


def fixture():
    row = {"candidate_id": DAY+":000001", "signal_date": DAY, "code": "000001"}
    data = {"peer_codes": ["000002", "000003"], "scope_complete": True,
            "complete_price_comparison": True, "features": {"relative_min_pp": 1.}}
    value = {"current": data, "prior": {"peer_codes": [], "scope_complete": False,
             "complete_price_comparison": False, "features": {"relative_min_pp": None}}}
    paths = {DAY+":000001": path("000001", [105., 105.]),
             DAY+":000002": path("000002", [108., 108.]),
             DAY+":000003": path("000003", [100., 100.])}
    return row, value, paths


class PairwiseTests(unittest.TestCase):
    def review(self):
        row, value, paths = fixture()
        facts = p.describe(row, value, paths)
        scopes = {}
        for scope, data in facts["scopes"].items():
            scopes[scope] = {"state": "REVIEWED" if data["complete_scope_comparison"] else "UNKNOWN",
                "reason": "均值领先仍落后某个对手；原对手缺口保留",
                "comparison_peer_codes": data["declared_peer_codes"],
                "incomplete_peer_codes": data["missing_or_incomplete_peer_codes"],
                "stronger_observed_peer_codes": sorted(x["peer_code"] for x in data["pairs"] if x["peer_ahead_at_some_sample"] is True),
                "next_session_paths": {key: {"state": "UNKNOWN", "reason": "次日结果待发生，需实际环境与竞争核对"}
                    for key in ("joint_promotion", "joint_failure", "divergence")}}
        context = {"rows": [{"code": row["code"], "pairwise_price_competition": facts}]}
        draft = {"rows": [{"code": row["code"], "pairwise_competition_assessment": {
            "scopes": scopes, "price_rank_is_not_core_identity": True,
            "role_causality_or_entry_confirmed": False}}]}
        return context, draft

    def test_review_requires_stronger_observed_peer_and_all_three_paths(self):
        context, draft = self.review()
        p.validate_assessments(context, draft)
        draft["rows"][0]["pairwise_competition_assessment"]["scopes"]["current"]["stronger_observed_peer_codes"] = []
        with self.assertRaises(ValueError): p.validate_assessments(context, draft)

    def test_review_cannot_replace_full_peer_list_with_mean_or_chosen_winner(self):
        context, draft = self.review()
        draft["rows"][0]["pairwise_competition_assessment"]["scopes"]["current"]["comparison_peer_codes"] = ["000003"]
        with self.assertRaises(ValueError): p.validate_assessments(context, draft)

    def test_review_cannot_make_unknown_original_rival_scope_complete(self):
        context, draft = self.review()
        draft["rows"][0]["pairwise_competition_assessment"]["scopes"]["prior"]["state"] = "REVIEWED"
        with self.assertRaises(ValueError): p.validate_assessments(context, draft)

    def test_review_cannot_delete_joint_promotion_or_certify_role_identity(self):
        context, draft = self.review()
        del draft["rows"][0]["pairwise_competition_assessment"]["scopes"]["current"]["next_session_paths"]["joint_promotion"]
        with self.assertRaises(ValueError): p.validate_assessments(context, draft)
        context, draft = self.review()
        draft["rows"][0]["pairwise_competition_assessment"]["role_causality_or_entry_confirmed"] = True
        with self.assertRaises(ValueError): p.validate_assessments(context, draft)

    def test_mean_ahead_can_hide_a_stronger_individual_peer(self):
        row, value, paths = fixture()
        result = p.describe(row, value, paths)["scopes"]["current"]
        self.assertTrue(result["complete_scope_comparison"])
        self.assertTrue(result["mean_advantage_with_peer_strictly_ahead"])
        self.assertFalse(result["own_ahead_of_every_peer_at_every_sample"])
        self.assertEqual(result["pairs"][0]["state"], "PEER_PRICE_AHEAD_AT_EVERY_SAMPLE")
        self.assertEqual(result["price_change_rank_last"], 2)

    def test_tied_peer_is_distinct_from_strictly_stronger_peer(self):
        row, value, paths = fixture()
        paths[DAY+":000002"]["bars"][0]["close"] = 105.
        paths[DAY+":000002"]["bars"][1]["close"] = 105.
        result = p.describe(row, value, paths)["scopes"]["current"]
        self.assertTrue(result["mean_advantage_does_not_hold_against_every_peer"])
        self.assertFalse(result["mean_advantage_with_peer_strictly_ahead"])
        self.assertEqual(result["pairs"][0]["equal_samples"], 2)

    def test_missing_peer_keeps_full_declared_denominator_and_unknown_rank(self):
        row, value, paths = fixture()
        paths.pop(DAY+":000002")
        result = p.describe(row, value, paths)["scopes"]["current"]
        self.assertEqual(result["declared_peer_codes"], ["000002", "000003"])
        self.assertFalse(result["complete_scope_comparison"])
        self.assertIsNone(result["own_ahead_of_every_peer_at_every_sample"])
        self.assertIsNone(result["price_change_rank_last"])
        self.assertEqual(result["missing_or_incomplete_peer_codes"], ["000002"])

    def test_incomplete_grid_retains_observed_difference_without_full_claim(self):
        row, value, paths = fixture()
        paths[DAY+":000002"]["grid_complete"] = False
        pair = p.describe(row, value, paths)["scopes"]["current"]["pairs"][0]
        self.assertEqual(pair["relative_min_pp"], -3.)
        self.assertEqual(pair["state"], "UNKNOWN")
        self.assertIsNone(pair["peer_ahead_at_some_sample"])

    def test_different_reference_sessions_cannot_establish_complete_pair(self):
        row, value, paths = fixture()
        paths[DAY+":000002"]["previous_session"] = "2026-09-09"
        result = p.describe(row, value, paths)["scopes"]["current"]
        self.assertFalse(result["pairs"][0]["complete_pair"])

    def test_actual_grid_mismatch_is_unknown_despite_declared_complete_flag(self):
        row, value, paths = fixture()
        paths[DAY+":000002"]["bars"][0]["minute"] = DAY+" 09:36:00"
        pair = p.describe(row, value, paths)["scopes"]["current"]["pairs"][0]
        self.assertFalse(pair["complete_pair"])
        self.assertEqual(pair["matched_sample_count"], 1)

    def test_rank_change_is_price_only_and_not_active_displacement(self):
        row, value, paths = fixture()
        paths[DAY+":000002"]["bars"][1]["close"] = 101.
        result = p.describe(row, value, paths)
        self.assertEqual(result["scopes"]["current"]["price_change_rank_first"], 2)
        self.assertEqual(result["scopes"]["current"]["price_change_rank_last"], 1)
        self.assertFalse(result["automatic_role_identity"])
        self.assertFalse(result["scopes"]["current"]["pairs"][0]["role_displacement_confirmed"])

    def test_current_survivors_and_original_rivals_are_separate(self):
        row, value, paths = fixture()
        value["prior"] = {**value["current"], "peer_codes": ["000002"]}
        result = p.describe(row, value, paths)["scopes"]
        self.assertEqual(len(result["current"]["pairs"]), 2)
        self.assertEqual(len(result["prior"]["pairs"]), 1)
        self.assertIn("SURVIVORS", result["current"]["comparison_cohort"])
        self.assertIn("PRIOR", result["prior"]["comparison_cohort"])

    def test_decimal_ratios_do_not_turn_exact_equality_into_leadership(self):
        row, value, paths = fixture()
        for code, previous, close in [("000001", 4.3, 4.73), ("000002", 10., 11.)]:
            paths[DAY+":"+code]["previous_close"] = previous
            for bar in paths[DAY+":"+code]["bars"]: bar["close"] = close
        pair = p.describe(row, value, paths)["scopes"]["current"]["pairs"][0]
        self.assertEqual(pair["relative_last_pp"], 0.)
        self.assertEqual(pair["state"], "EQUAL_PRICE_CHANGE_AT_EVERY_SAMPLE")

    def test_raw_flag_without_qualified_feature_is_not_a_source(self):
        row, value, paths = fixture()
        row["source_groups"] = {"sampled_peers": {"value": value, "verified": True}}
        result = p.source_projection(row)
        self.assertFalse(result["source_qualified"])
        self.assertFalse(result["scopes"]["current"]["complete_scope_comparison"])

    def test_missing_price_paths_keep_known_qualified_same_height_peers(self):
        row, value, paths = fixture()
        row.update(roles={"HEIGHT": {"supporting_groups": ["pool"]}}, peer_codes=["000002", "000003"])
        result = p.source_projection(row)
        self.assertFalse(result["source_qualified"])
        self.assertEqual(result["scopes"]["current"]["declared_peer_codes"], ["000002", "000003"])
        self.assertEqual(result["scopes"]["current"]["missing_or_incomplete_peer_codes"], ["000002", "000003"])
        self.assertIsNone(result["scopes"]["current"]["price_change_rank_last"])

    def test_unreconciled_price_and_duplicate_timestamp_are_not_complete(self):
        row, value, paths = fixture()
        paths[DAY+":000002"]["price_reconciled"] = False
        paths[DAY+":000003"]["bars"][1]["minute"] = DAY+" 09:35:00"
        result = p.describe(row, value, paths)["scopes"]["current"]
        self.assertEqual(result["missing_or_incomplete_peer_codes"], ["000002", "000003"])


if __name__ == "__main__": unittest.main()
