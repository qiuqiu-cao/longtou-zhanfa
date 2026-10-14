import copy
import unittest

from leader_role_mapper_v146 import map_roles


ASOF = "2026-09-17T15:20:00+08:00"


def row(candidate_id, code, boards, **updates):
    base = {
        "candidate_id": candidate_id,
        "code": code,
        "name": candidate_id,
        "boards": boards,
        "pool_assignment": "CLOSING_LIMIT_UP",
        "is_mainboard": True,
        "is_st": False,
        "is_delisting_period": False,
        "security_status": "verified",
        "limit_up_status": "verified",
        "identity_status": "verified",
        "theme_status": "verified_same_day_group",
        "theme_granularity": "specific",
        "theme_group_id": "theme-a",
        "theme": "细分A",
        "theme_state": "STABLE",
        "theme_support_ex_self": 2,
        "same_height_peer_count": 0,
        "tradability": "buyable",
        "one_price_board": False,
        "competition_identity": "capacity_core",
        "is_emotion_anchor": False,
        "abnormal_rule_status": "verified",
        "indicator_status": "verified",
        "scenario_status": "verified_inputs_counterfactual",
        "target_status": "TARGET_RESEARCH_CANDIDATE",
        "distance_100_pp": 20,
        "scenario_limit_boards_to_100": 2,
    }
    base.update(updates)
    return base


def fixture():
    rows = [
        row("high", "600001", 5, is_emotion_anchor=True,
            competition_identity="height_anchor", scenario_limit_boards_to_100=2),
        row("low", "600002", 1, competition_identity="pioneer",
            scenario_limit_boards_to_100=None),
        row("other", "000001", 3, theme_group_id="theme-b", theme="细分B",
            competition_identity="unique_survivor", theme_support_ex_self=1,
            scenario_limit_boards_to_100=1),
    ]
    return {
        "universe": {
            "status": "VERIFIED_FULL_CLOSING_LIMIT_POOL",
            "universe_complete": True,
            "signal_date": "2026-09-17",
            "closing_limit_count": len(rows),
            "candidates": rows,
        },
        "next_session_emotion_risk": {
            "signal_date": "2026-09-17",
            "observed_at": "2026-09-17T15:00:00+08:00",
            "available_at": "2026-09-17T15:10:00+08:00",
            "model_status": "DIAGNOSTIC_ONLY",
            "risk_band": "MIXED",
            "policy_effect": "RISK_ONLY_NO_POSITIVE_RANK_BONUS",
            "data_status": "inferred",
            "source": "emotion_actionable_risk_v144",
        },
    }


class LeaderRoleMapperTests(unittest.TestCase):
    def test_roles_are_separate_and_never_buy_signals(self):
        result = map_roles(fixture(), ASOF)
        self.assertEqual(result["emotion_anchor"]["candidate_id"], "high")
        # Better verified identity outranks mere closeness to the threshold.
        self.assertEqual(result["hundred_percent_path_candidate"]["candidate_id"], "high")
        self.assertEqual(result["high_low_supplement_candidate"]["candidate_id"], "low")
        self.assertFalse(result["buy_signal"])
        self.assertIsNone(result["calibrated_probability"])
        self.assertEqual(result["high_low_mapping_coverage"]["identified_high_cores"], 2)
        self.assertEqual(len(result["high_low_supplement_mappings"]), 2)

    def test_distance_alone_cannot_win(self):
        item = fixture()
        item["universe"]["candidates"][2]["distance_100_pp"] = 1
        item["universe"]["candidates"][2]["competition_identity"] = "supplement"
        result = map_roles(item, ASOF)
        self.assertEqual(result["hundred_percent_path_candidate"]["candidate_id"], "high")

    def test_broad_or_collapsed_theme_blocks_mapping(self):
        for field, value in (("theme_granularity", "broad"), ("theme_state", "COLLAPSED")):
            item = fixture()
            item["universe"]["candidates"][0][field] = value
            result = map_roles(item, ASOF)
            self.assertIsNone(result["high_low_supplement_candidate"])

    def test_random_locked_or_one_price_low_is_rejected(self):
        for updates in ({"theme_group_id": "theme-z"}, {"tradability": "locked_limit_up"},
                        {"one_price_board": True}, {"competition_identity": "supplement"}):
            item = fixture()
            item["universe"]["candidates"][1].update(updates)
            result = map_roles(item, ASOF)
            self.assertIsNone(result["high_low_supplement_candidate"])

    def test_unverified_rule_removes_only_path_role(self):
        item = fixture()
        for candidate in item["universe"]["candidates"]:
            candidate["abnormal_rule_status"] = "inferred"
        result = map_roles(item, ASOF)
        self.assertIsNone(result["hundred_percent_path_candidate"])
        self.assertIsNotNone(result["emotion_anchor"])

    def test_full_pool_and_emotion_evidence_fail_closed(self):
        for mutation in ("pool", "emotion"):
            item = fixture()
            if mutation == "pool":
                item["universe"]["closing_limit_count"] += 1
            else:
                item["next_session_emotion_risk"]["available_at"] = "2026-09-17T16:00:00+08:00"
            result = map_roles(item, ASOF)
            self.assertEqual(result["status"], "DATA_LIMITED")
            self.assertIsNone(result["emotion_anchor"])

    def test_high_risk_never_turns_mapping_into_buy(self):
        item = fixture()
        item["next_session_emotion_risk"]["risk_band"] = "HIGH"
        result = map_roles(item, ASOF)
        self.assertEqual(result["selection_effect"], "HARDEN_CONFIRMATION_OR_CASH")
        self.assertFalse(result["high_low_supplement_candidate"]["buy_signal"])

    def test_every_high_core_gets_mapped_or_explicitly_unmapped(self):
        item = fixture()
        item["universe"]["candidates"].append(
            row("other-low", "000002", 1, theme_group_id="theme-b", theme="细分B",
                competition_identity="pioneer", scenario_limit_boards_to_100=None)
        )
        item["universe"]["closing_limit_count"] += 1
        result = map_roles(item, ASOF)
        by_high = {
            mapping["high_core"]["candidate_id"]: mapping
            for mapping in result["high_low_supplement_mappings"]
        }
        self.assertEqual(by_high["high"]["supplement_candidate"]["candidate_id"], "low")
        self.assertEqual(by_high["other"]["supplement_candidate"]["candidate_id"], "other-low")
        self.assertTrue(all(not mapping["buy_signal"] for mapping in by_high.values()))

    def test_high_core_with_unverified_theme_is_not_silently_omitted(self):
        item = fixture()
        item["universe"]["candidates"][2]["theme_status"] = "unavailable"
        result = map_roles(item, ASOF)
        mapping = next(
            value for value in result["high_low_supplement_mappings"]
            if value["high_core"]["candidate_id"] == "other"
        )
        self.assertEqual(mapping["mapping_status"], "NO_ELIGIBLE_MAPPING")
        self.assertIn("HIGH_CORE_SPECIFIC_THEME_NOT_VERIFIED", mapping["blocking_reasons"])


if __name__ == "__main__":
    unittest.main()
