import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import leader_role_engine as e
import stock_short_term_context as s


DAYS = ["2026-10-08", "2026-10-09", "2026-10-12", "2026-10-13"]


def row(day, boards, theme_id="topic", theme_scope="fixture"):
    competition = {"complete_within_scope": True, "peer_codes": ["600241"],
        "prior_at": day+"T09:35:00+08:00", "current_at": day+"T15:15:00+08:00",
        "prior": {"002242": "CORE", "600241": "COMPETING"},
        "current": {"002242": "STRONG", "600241": "FAILED"}}
    pressure = {"event_id": "fixture-pressure", "pre_event_role": "CORE",
        "started_at": day+"T10:00:00+08:00", "ended_at": day+"T15:00:00+08:00",
        "pre_cycle_id": "fixture-cycle", "post_cycle_id": "fixture-cycle",
        "pre_event_observation": {"code": "002242", "observed_at": competition["prior_at"],
            "state": "CORE", "relative_rank": 1, "price": 10.},
        "post_event_observation": {"code": "002242", "observed_at": competition["current_at"],
            "state": "CORE", "relative_rank": 1, "price": 11.},
        "post_event_competition_verified": True, "post_competition": competition}
    return {"candidate_id": day+":002242", "code": "002242", "boards": boards,
        "signal_date": day, "scope_id": "fixture", "available_at": day+"T16:00:00+08:00",
        "decision_at": day+"T16:01:00+08:00", "actual_one_word_daily": 0,
        "roles": {"TEST": {"supporting_groups": ["theme", "dependency", "path", "pressure", "competition"]}},
        "peer_codes": ["600241"], "features": {"theme_return_ex_self_pp": 11.,
            "pressure_pre_event_core": 1., "pressure_survived": 1., "pressure_role_retained": 1., "actual_path_resealed": 1.},
        "source_groups": {
            "theme": {"value": {"theme_id": theme_id, "comparison_scope_id": theme_scope,
                "membership_verified": True, "complete_within_scope": True,
                "granularity": "specific", "codes": ["002242", "600241"],
                "scope": "DECLARED_FIXTURE_COHORT", "full_theme_complete": False,
                "members": {"002242": {"return_pct": 10.}, "600241": {"return_pct": -1.}}}},
            "dependency": {"value": {"own_start": "2026-10-08", "anchor_code": "600241"}},
            "path": {"value": {"events": [{"state": "OPEN"}, {"state": "SEALED"}]}},
            "pressure": {"value": pressure},
            "competition": {"value": competition}}}


def cat(boards=6, topic=False):
    claim = {"kind": "NARRATIVE", "excerpt": "九阳股份晋级六连板", "source_id": "fixture"}
    if topic:
        claim.update(theme_id="topic", excerpt="九阳股份与同题材成员被作者讨论")
    return {"code": "002242", "name": "九阳股份", "boards": boards, "evidence": [claim]}


class ShortTermTests(unittest.TestCase):
    def test_missing_new_loader_rejects_own_kind_without_breaking_legacy_raw_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)/"legacy.json"
            new = Path(tmp)/"new.json"
            e.write_new(raw, {"legacy_fixture": True})
            e.write_new(new, {"derivation_kind": "HISTORICAL_CONFIDENCE_RELATIONS"})
            with patch.object(e, "_replay_confidence_bundle", None, create=True):
                self.assertEqual(e._rich_source_json(str(raw), e.sha(raw)), {"legacy_fixture": True})
                with self.assertRaisesRegex(ValueError, "source replay loader unavailable"):
                    e._rich_source_json(str(new), e.sha(new))

    def projected(self, rows=None, catalyst=None):
        return s.project_rows(rows or [row(DAYS[1], 5), row(DAYS[2], 6)],
            [catalyst or cat()], DAYS, DAYS[2], "DIAGNOSTIC")[0]

    def test_stock_mention_is_not_launch_narrative(self):
        check = self.projected()["checks"]["launch_and_continuation"]
        self.assertFalse(check["supported_allowed"])
        self.assertFalse(check["gates"]["checked_specific_topic_narrative_present"])
        self.assertFalse(check["facts"]["stock_mention_or_board_count_is_launch_cause"])

    def test_topic_plus_observed_continuation_still_does_not_certify_cause(self):
        check = self.projected(catalyst=cat(topic=True))["checks"]["launch_and_continuation"]
        self.assertTrue(check["supported_allowed"])
        self.assertFalse(check["facts"]["launch_or_continuation_cause_confirmed"])

    def test_single_day_or_missing_official_day_is_not_a_timeline(self):
        for rows in ([row(DAYS[2], 6)], [row(DAYS[0], 4), row(DAYS[2], 6)]):
            self.assertFalse(self.projected(rows)["multi_day_connection_complete"])
            self.assertFalse(self.projected(rows)["checks"]["independent_strength"]["supported_allowed"])

    def test_membership_change_is_not_market_or_narrative_switch(self):
        p = self.projected([row(DAYS[1], 5, "old"), row(DAYS[2], 6, "new")])
        check = p["checks"]["theme_transition"]
        self.assertTrue(check["supported_allowed"])
        self.assertEqual(check["facts"]["market_theme_switch"]["state"], "UNKNOWN")
        self.assertEqual(check["facts"]["stock_narrative_switch"]["state"], "UNKNOWN")
        self.assertFalse(check["facts"]["stock_theme_membership_changes"][0]["exact_intraday_switch_known"])

    def test_different_theme_comparison_scopes_cannot_establish_switch(self):
        p = self.projected([row(DAYS[1], 5, "old", "one"), row(DAYS[2], 6, "new", "two")])
        self.assertFalse(p["checks"]["theme_transition"]["supported_allowed"])

    def test_independent_candidate_needs_actual_pressure_and_weak_peers(self):
        self.assertTrue(self.projected()["checks"]["independent_strength"]["supported_allowed"])
        current = row(DAYS[2], 6)
        current["source_groups"].pop("pressure")
        self.assertFalse(self.projected([row(DAYS[1], 5), current])["checks"]["independent_strength"]["supported_allowed"])
        current = row(DAYS[2], 6)
        current["source_groups"]["theme"]["value"]["members"]["600241"]["return_pct"] = 1.
        self.assertFalse(self.projected([row(DAYS[1], 5), current])["checks"]["independent_strength"]["supported_allowed"])

    def test_volume_or_board_height_does_not_prove_absorption(self):
        current = row(DAYS[2], 6)
        current["source_groups"].pop("path")
        p = self.projected([row(DAYS[1], 5), current])
        self.assertFalse(p["checks"]["absorption_and_competition"]["supported_allowed"])
        self.assertFalse(p["role_identity_confirmed"])

    def test_competing_to_core_is_not_retained_pre_event_core(self):
        current = row(DAYS[2], 6)
        current["features"]["pressure_pre_event_core"] = 0.
        self.assertFalse(self.projected([row(DAYS[1], 5), current])["checks"]["independent_strength"]["supported_allowed"])

    def test_valid_old_pressure_does_not_support_current_window(self):
        current = row(DAYS[2], 6)
        old = row(DAYS[0], 4)["source_groups"]["pressure"]["value"]
        self.assertTrue(e._valid_group("pressure", old, current))
        current["source_groups"]["pressure"]["value"] = old
        check = self.projected([row(DAYS[1], 5), current])["checks"]["independent_strength"]
        self.assertFalse(check["supported_allowed"])
        self.assertFalse(check["gates"]["pressure_matches_current_comparison_window"])

    def test_prior_day_after_observation_is_not_current_retention(self):
        current = row(DAYS[2], 6)
        current["source_groups"]["pressure"]["value"] = row(DAYS[1], 5)["source_groups"]["pressure"]["value"]
        check = self.projected([row(DAYS[1], 5), current])["checks"]["independent_strength"]
        self.assertFalse(check["gates"]["pressure_matches_current_comparison_window"])

    def test_limited_cohort_candidate_does_not_confirm_theme_wide_weakness(self):
        check = self.projected()["checks"]["independent_strength"]
        self.assertTrue(check["supported_allowed"])
        self.assertFalse(check["facts"]["full_theme_weakness_confirmed"])
        self.assertFalse(check["facts"]["identity_confirmed"])
        self.assertIn("exact declared cohort", check["facts"]["support_scope"])

    def test_explicit_full_theme_flag_remains_separate_from_identity(self):
        current = row(DAYS[2], 6)
        current["source_groups"]["theme"]["value"]["full_theme_complete"] = True
        check = self.projected([row(DAYS[1], 5), current])["checks"]["independent_strength"]
        self.assertTrue(check["facts"]["full_declared_theme_mean_nonpositive_with_own_strength"])
        self.assertFalse(check["facts"]["full_theme_weakness_confirmed"])
        self.assertFalse(check["facts"]["identity_confirmed"])

    def test_negative_peer_mean_keeps_rising_peers_visible(self):
        current = row(DAYS[2], 6)
        members = current["source_groups"]["theme"]["value"]["members"]
        members["600241"]["return_pct"] = 1.
        members["600242"] = {"return_pct": -3.}
        observation = s._theme_observation(current)
        self.assertEqual(observation["declared_peer_mean_return_pct_ex_self"], -1.)
        self.assertEqual(observation["peer_positive_count"], 1)
        self.assertEqual(observation["peer_nonpositive_count"], 1)
        self.assertTrue(observation["peer_mean_nonpositive_does_not_mean_all_peers_fell"])

    def test_price_not_rising_is_distinct_from_relative_defense(self):
        current = row(DAYS[2], 6)
        members = current["source_groups"]["theme"]["value"]["members"]
        members["002242"]["return_pct"] = -1.
        members["600241"]["return_pct"] = -3.
        check = self.projected([row(DAYS[1], 5), current])["checks"]["independent_strength"]
        self.assertEqual(check["facts"]["declared_cohort_observation"]["observation_state"],
                         "OWN_PRICE_NOT_RISING_DESPITE_RELATIVE_ADVANTAGE")
        self.assertFalse(check["supported_allowed"])

    def test_rising_peers_are_not_weakness_pressure(self):
        current = row(DAYS[2], 6)
        current["source_groups"]["theme"]["value"]["members"]["600241"]["return_pct"] = 2.
        check = self.projected([row(DAYS[1], 5), current])["checks"]["independent_strength"]
        self.assertEqual(check["facts"]["declared_cohort_observation"]["observation_state"],
                         "OUTPERFORMANCE_WITH_DECLARED_PEERS_RISING")
        self.assertFalse(check["supported_allowed"])

    def test_member_churn_cannot_reuse_comparison_label_to_claim_persistence(self):
        current = row(DAYS[2], 6)
        group = current["source_groups"]["theme"]["value"]
        group["members"]["600242"] = group["members"].pop("600241")
        group["codes"] = ["002242", "600242"]
        check = self.projected([row(DAYS[1], 5), current])["checks"]["independent_strength"]
        self.assertFalse(check["gates"]["consecutive_comparable_theme_relative_strength"])
        self.assertEqual(check["facts"]["declared_cohort_changes"][0]["removed_member_codes"], ["600241"])

    def test_missing_qualified_theme_is_not_recorded_as_member_exit(self):
        current = row(DAYS[2], 6)
        current["source_groups"].pop("theme")
        facts = self.projected([row(DAYS[1], 5), current])["checks"]["independent_strength"]["facts"]
        self.assertEqual(facts["declared_cohort_changes"], [])
        self.assertEqual(facts["declared_cohort_qualification_gap_dates"], [DAYS[2]])
        self.assertFalse(facts["missing_qualification_is_member_exit"])

    def test_post_pressure_complete_subset_cannot_omit_original_theme_peer(self):
        current = row(DAYS[2], 6)
        competition = current["source_groups"]["pressure"]["value"]["post_competition"]
        competition["peer_codes"] = ["600242"]
        for key in ("prior", "current"):
            competition[key]["600242"] = competition[key].pop("600241")
        self.assertTrue(e._valid_group("pressure", current["source_groups"]["pressure"]["value"], current))
        check = self.projected([row(DAYS[1], 5), current])["checks"]["independent_strength"]
        self.assertFalse(check["supported_allowed"])
        self.assertEqual(check["facts"]["pressure_window_alignment"]["missing_original_peer_codes"], ["600241"])

    def test_missing_peer_returns_are_not_restored_from_existing_factor(self):
        current = row(DAYS[2], 6)
        current["source_groups"]["theme"]["value"]["members"]["600241"]["return_pct"] = None
        check = self.projected([row(DAYS[1], 5), current])["checks"]["independent_strength"]
        self.assertIsNone(check["facts"]["relative_return_ex_self_pp"])
        self.assertEqual(check["facts"]["declared_cohort_observation"]["missing_peer_return_codes"], ["600241"])
        self.assertFalse(check["supported_allowed"])

    def test_exact_declared_source_return_diff_keeps_original_factor_separate(self):
        current = row(DAYS[2], 6)
        current["features"]["theme_return_ex_self_pp"] = 123.
        observation = self.projected([row(DAYS[1], 5), current])["checks"]["independent_strength"]["facts"]["declared_cohort_observation"]
        self.assertEqual(observation["relative_return_ex_self_pp"], 11.)
        self.assertEqual(observation["original_role_relative_return_ex_self_pp"], 123.)

    def test_stale_cycle_is_not_current_market_phase(self):
        current = row(DAYS[2], 6)
        current["roles"]["TEST"]["supporting_groups"].append("cycle")
        current["source_groups"]["cycle"] = {"value": {"records": [{"date": DAYS[0], "state": "CONSENSUS"}]}}
        check = self.projected([row(DAYS[1], 5), current])["checks"]["theme_transition"]
        self.assertEqual(check["facts"]["market_stage_history"], [])

    def test_same_membership_has_bound_counterevidence_without_denial_of_narrative(self):
        context, draft = self.assessment()
        check = draft["rows"][0]["short_term_assessment"]["checks"]["theme_transition"]
        check.update(state="CONTRADICTED", reason="此观察窗口主题成员身份未变，叙事退出仍未知",
                     evidence_refs=["same_specific_theme_membership_in_observed_window"],
                     counter_claims=["SPECIFIC_THEME_MEMBERSHIP_CHANGED_IN_OBSERVED_WINDOW"])
        s.validate_assessments(context, draft)
        check["counter_claims"] = ["STOCK_NARRATIVE_CHANGE"]
        with self.assertRaises(ValueError): s.validate_assessments(context, draft)

    def test_full_original_pool_and_board_identity_required(self):
        with self.assertRaises(ValueError):
            s.project_rows([row(DAYS[2], 6)], [dict(cat(), code="000001")], DAYS, DAYS[2], "STRICT")
        with self.assertRaises(ValueError):
            self.projected(catalyst=cat(boards=5))

    def assessment(self):
        p = self.projected()
        a = {"checks": {k: {"state": "UNKNOWN", "reason": "缺逐日证据", "evidence_refs": []} for k in s.CHECKS},
            "alternative_explanations": ["新叙事或高位抱团"], "counterevidence": ["后续开板无法回封待核"],
            "cancel_conditions": ["相对强度与回封不成立"], "role_identity_confirmed": False,
            "fund_causality_confirmed": False,
            "confirmation_plan": [{"planned_at": "2026-10-13T09:35:00+08:00", "status": "PENDING", "requirement": "核对实际开盘和同池竞争"}]}
        return {"rows": [p], "target_date": DAYS[3], "formed_at": "2026-10-12T16:00:00+08:00"}, {"rows": [{"code": "002242", "short_term_assessment": a}]}

    def test_unknown_review_is_valid_and_supported_without_gates_is_rejected(self):
        context, draft = self.assessment()
        s.validate_assessments(context, draft)
        draft["rows"][0]["short_term_assessment"]["checks"]["launch_and_continuation"].update(state="SUPPORTED", evidence_refs=["daily_timeline"])
        with self.assertRaises(ValueError): s.validate_assessments(context, draft)

    def test_confirmation_is_future_pending_and_identity_cannot_be_certified(self):
        context, draft = self.assessment()
        draft["rows"][0]["short_term_assessment"]["confirmation_plan"][0]["status"] = "CONFIRMED"
        with self.assertRaises(ValueError): s.validate_assessments(context, draft)
        context, draft = self.assessment()
        draft["rows"][0]["short_term_assessment"]["role_identity_confirmed"] = True
        with self.assertRaises(ValueError): s.validate_assessments(context, draft)

    def test_missing_old_sidecar_is_compatible_but_new_freeze_requires_it(self):
        self.assertIsNone(s.checked_projection({}))
        with self.assertRaises(ValueError): s.checked_projection({}, required=True)

    def test_market_history_has_independent_scope_time_and_gap_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw = root/"raw.json"; e.write_new(raw, {"fixture": True})
            paths = []
            for day, name in ((DAYS[1], "旧题材"), (DAYS[2], "新题材")):
                path = root/(day+".json")
                e.write_new(path, {"signal_date": day, "mainline": name, "status": "INFERRED_SHORT_TERM_MAINLINE",
                    "scope": "market-fixture", "policy_version": "fixture", "source_hashes": e.bind(raw),
                    "generated_at": day+"T16:01:00+08:00", "knowledge_cutoff": day+"T16:00:00+08:00"})
                paths.append(path)
            result, _ = s._market_history(paths, DAYS, DAYS[2], e.stamp(e.now()))
            self.assertEqual(result["state"], "OBSERVED_MAINLINE_HYPOTHESIS_CHANGE")
            self.assertFalse(result["stock_narrative_or_market_causality_confirmed"])
            raw.write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError): s._market_history(paths, DAYS, DAYS[2], e.stamp(e.now()))


if __name__ == "__main__": unittest.main()
