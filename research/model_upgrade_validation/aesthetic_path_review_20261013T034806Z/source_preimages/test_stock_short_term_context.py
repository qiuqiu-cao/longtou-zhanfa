import copy
import json
from pathlib import Path
import tempfile
import unittest

import leader_role_engine as e
import stock_short_term_context as s


DAYS = ["2026-10-08", "2026-10-09", "2026-10-12", "2026-10-13"]


def row(day, boards, theme_id="topic", theme_scope="fixture"):
    return {"candidate_id": day+":002242", "code": "002242", "boards": boards,
        "signal_date": day, "scope_id": "fixture", "available_at": day+"T16:00:00+08:00",
        "decision_at": day+"T16:01:00+08:00", "actual_one_word_daily": 0,
        "roles": {"TEST": {"supporting_groups": ["theme", "dependency", "path", "pressure", "competition"]}},
        "peer_codes": ["600241"], "features": {"theme_return_ex_self_pp": 11.,
            "pressure_pre_event_core": 1., "pressure_survived": 1., "pressure_role_retained": 1., "actual_path_resealed": 1.},
        "source_groups": {
            "theme": {"value": {"theme_id": theme_id, "comparison_scope_id": theme_scope,
                "members": {"002242": {"return_pct": 10.}, "600241": {"return_pct": -1.}}}},
            "dependency": {"value": {"own_start": "2026-10-08", "anchor_code": "600241"}},
            "path": {"value": {"events": [{"state": "OPEN"}, {"state": "SEALED"}]}},
            "pressure": {"value": {"post_event_observation": {"state": "CORE"}}},
            "competition": {"value": {"peer_codes": ["600241"]}}}}


def cat(boards=6, topic=False):
    claim = {"kind": "NARRATIVE", "excerpt": "九阳股份晋级六连板", "source_id": "fixture"}
    if topic:
        claim.update(theme_id="topic", excerpt="九阳股份与同题材成员被作者讨论")
    return {"code": "002242", "name": "九阳股份", "boards": boards, "evidence": [claim]}


class ShortTermTests(unittest.TestCase):
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
