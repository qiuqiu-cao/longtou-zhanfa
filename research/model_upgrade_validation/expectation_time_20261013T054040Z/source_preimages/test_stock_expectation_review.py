import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import leader_role_engine as e
import leader_role_review as review
import stock_expectation_review as s


NOW = "2026-10-13T09:40:00+08:00"
FORMED = "2026-10-12T22:00:00+08:00"


def condition(cid, scope, expected, metric="retained", operator="EQ"):
    return {"condition_id": cid, "metric": metric, "scope": scope, "scope_id": "fixture",
        "comparison_codes": ["002242"] if scope == "STOCK" else ["600241"],
        "operator": operator, "expected": expected, "unit": "boolean"}


def plan():
    paths = {}
    for name, expected in (("normal", True), ("stronger", True), ("weaker", False)):
        paths[name] = {"state": "DEFINED", "reason": "在注册窗口核对价格路径及完整同身位对照",
            "conditions": [condition(name+"_stock", "STOCK", expected)]}
        if name != "normal":
            paths[name]["conditions"].append(condition(name+"_peers", "PEERS", expected))
    return {"checkpoint_id": "open", "baseline_kind": "AUTHOR_PREDECLARED", "planned_at": "2026-10-13T09:35:00+08:00",
        "observation_window": {"start_at": "2026-10-13T09:30:00+08:00", "end_at": "2026-10-13T09:35:00+08:00"},
        "status": "PENDING", "requirement": "实际价格路径与完整同身位范围对照", "paths": paths}


def record():
    return {"signal_date": "2026-10-12", "target_date": "2026-10-13", "formed_at": FORMED,
        "rows": [{"code": "002242", "short_term_assessment": {"confirmation_plan": [plan()]}}]}


class ExpectationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rec = record()
        self.source = self.root/"source.json"

    def tearDown(self):
        self.tmp.cleanup()

    def facts(self):
        return {"target_date": "2026-10-13", "prepared_at": FORMED,
            "rows": [{"code": "002242", "leader_role_evidence": {"scope_id": "fixture", "peer_codes": ["600241"],
                "roles": {"HEIGHT": {"supporting_groups": ["pool"]}, "THEME": {"supporting_groups": ["theme"]}},
                "source_groups": {"pool": {"value": {"codes": ["002242", "600241"]}},
                                  "theme": {"value": {"comparison_scope_id": "fixture", "members": {"002242": {}, "600241": {}}}}}}}]}

    def confirmation(self, values=None, changes=None):
        conditions = [c for p in self.rec["rows"][0]["short_term_assessment"]["confirmation_plan"][0]["paths"].values() for c in p["conditions"]]
        values = values or {c["condition_id"]: True for c in conditions}
        cells, proofs = {}, []
        for c in conditions:
            cid = c["condition_id"]
            if cid not in values:
                continue
            raw = {k: c[k] for k in ("metric", "scope", "scope_id", "comparison_codes", "unit")}
            raw.update(value=values[cid], observed_at="2026-10-13T09:35:00+08:00", available_at="2026-10-13T09:36:00+08:00")
            raw.update((changes or {}).get(cid, {}))
            cells[cid] = raw
        if self.source.exists():
            self.source.unlink()
        e.write_new(self.source, {"cells": cells})
        for cid, raw in cells.items():
            proofs.append({"condition_id": cid, "source_path": str(self.source.resolve()), "source_sha256": e.sha(self.source),
                "pointer": "/cells/"+cid, "raw_value": copy.deepcopy(raw)})
        return {"signal_date": self.rec["signal_date"], "target_date": self.rec["target_date"],
            "observed_at": "2026-10-13T09:35:00+08:00", "source_bindings": e.bind(self.source),
            "rows": [{"code": "002242", "checkpoints": [{"checkpoint_id": "open", "observations": proofs}]}]}

    def evaluated(self, confirmation):
        return s.evaluate_confirmation(self.rec, confirmation, NOW)["rows"][0]["checkpoints"][0]

    def test_complete_plan_and_all_unknown_preserve_abstention(self):
        s.validate_plans(self.facts(), self.rec)

    def test_unknown_paths_receive_source_observations_without_new_expectation(self):
        c = self.confirmation()
        proof = c["rows"][0]["checkpoints"][0]["observations"][0]
        proof.pop("condition_id")
        c["rows"][0]["checkpoints"][0]["observations"] = [proof]
        for path in self.rec["rows"][0]["short_term_assessment"]["confirmation_plan"][0]["paths"].values():
            path.update(state="UNKNOWN", reason="事前缺完整路径来源，弃判", conditions=[])
        out = self.evaluated(c)
        self.assertEqual(out["status"], "UNKNOWN")
        self.assertEqual(out["condition_states"], {})
        self.assertEqual(len(out["unscored_source_observations"]), 1)
        for path in self.rec["rows"][0]["short_term_assessment"]["confirmation_plan"][0]["paths"].values():
            path.update(state="UNKNOWN", reason="缺少同观察时点的完整来源，暂弃判", conditions=[])
        s.validate_plans(self.facts(), self.rec)

    def test_high_open_alone_and_changed_original_scope_are_rejected(self):
        self.rec["rows"][0]["short_term_assessment"]["confirmation_plan"][0]["paths"]["stronger"]["conditions"].pop()
        with self.assertRaisesRegex(ValueError, "external"):
            s.validate_plans(self.facts(), self.rec)
        self.rec = record()
        self.rec["rows"][0]["short_term_assessment"]["confirmation_plan"][0]["paths"]["stronger"]["conditions"][1]["comparison_codes"] = ["600812"]
        with self.assertRaisesRegex(ValueError, "scope"):
            s.validate_plans(self.facts(), self.rec)

    def test_unqualified_raw_pool_and_theme_do_not_authenticate_denominators(self):
        facts = self.facts()
        facts["rows"][0]["leader_role_evidence"]["roles"] = {}
        with self.assertRaisesRegex(ValueError, "scope"):
            s.validate_plans(facts, self.rec)
        facts = self.facts()
        facts["rows"][0]["leader_role_evidence"]["roles"]["THEME"]["supporting_groups"] = []
        condition = self.rec["rows"][0]["short_term_assessment"]["confirmation_plan"][0]["paths"]["stronger"]["conditions"][1]
        condition["scope"] = "THEME"
        with self.assertRaisesRegex(ValueError, "scope"):
            s.validate_plans(facts, self.rec)

    def test_duplicate_checkpoints_conditions_and_future_preconfirmation_rejected(self):
        self.rec["rows"][0]["short_term_assessment"]["confirmation_plan"].append(plan())
        with self.assertRaisesRegex(ValueError, "duplicate"):
            s.validate_plans(self.facts(), self.rec)
        self.rec = record()
        p = self.rec["rows"][0]["short_term_assessment"]["confirmation_plan"][0]
        p["paths"]["stronger"]["conditions"][0]["condition_id"] = "normal_stock"
        with self.assertRaisesRegex(ValueError, "condition_id"):
            s.validate_plans(self.facts(), self.rec)
        self.rec = record()
        self.rec["rows"][0]["short_term_assessment"]["confirmation_plan"][0]["status"] = "CONFIRMED"
        with self.assertRaisesRegex(ValueError, "PENDING"):
            s.validate_plans(self.facts(), self.rec)

    def test_overlapping_paths_are_ambiguous_without_tiebreak(self):
        result = self.evaluated(self.confirmation())
        self.assertEqual(result["status"], "AMBIGUOUS")
        self.assertEqual(result["path_states"]["weaker"], "FAIL")

    def test_only_weaker_path_matches_and_immutable_record_retained(self):
        before = copy.deepcopy(self.rec)
        values = {c["condition_id"]: False for p in plan()["paths"].values() for c in p["conditions"]}
        result = self.evaluated(self.confirmation(values))
        self.assertEqual(result["status"], "WEAKER")
        self.assertEqual(before, self.rec)

    def test_missing_condition_and_wrong_value_type_remain_unknown(self):
        c = self.confirmation({"normal_stock": False, "stronger_stock": False, "weaker_stock": False})
        result = self.evaluated(c)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertIn("weaker_peers", result["missing_condition_ids"])
        c = self.confirmation({"normal_stock": 1, "stronger_stock": 1, "stronger_peers": 1, "weaker_stock": 1, "weaker_peers": 1})
        self.assertEqual(self.evaluated(c)["status"], "UNKNOWN")

    def test_empty_sources_mutated_raw_pointer_and_hash_rejected(self):
        for defect in ("empty", "raw", "pointer", "hash"):
            c = self.confirmation()
            proof = c["rows"][0]["checkpoints"][0]["observations"][0]
            if defect == "empty": c["source_bindings"] = {}
            if defect == "raw": proof["raw_value"]["value"] = False
            if defect == "pointer": proof["pointer"] = "/cells/not_here"
            if defect == "hash": proof["source_sha256"] = "bad"
            with self.assertRaises(ValueError): self.evaluated(c)

    def test_json_pointer_rejects_negative_indices_and_malformed_escapes(self):
        for pointer in ("/cells/-1", "/cells/01", "/a~2b"):
            with self.assertRaises(ValueError):
                s._pointer({"cells": [1, 2], "a~2b": 3}, pointer)
        self.assertEqual(s._pointer({"a/b": {"~key": 4}}, "/a~1b/~0key"), 4)

    def test_observation_and_availability_future_window_range_and_unit_rejected(self):
        for change in ({"observed_at": "2026-10-13T09:41:00+08:00"},
                       {"available_at": "2026-10-13T09:41:00+08:00"},
                       {"observed_at": "2026-10-13T09:25:00+08:00"},
                       {"comparison_codes": ["600812"]}, {"unit": "percent"},
                       {"metric": "open_premium"}):
            c = self.confirmation(changes={"normal_stock": change})
            with self.assertRaises(ValueError): self.evaluated(c)

    def test_changed_expectation_unknown_ids_and_duplicate_rows_rejected(self):
        c = self.confirmation()
        c["rows"][0]["paths"] = plan()["paths"]
        with self.assertRaises(ValueError): self.evaluated(c)
        c = self.confirmation()
        c["rows"][0]["checkpoints"][0]["observations"][0]["condition_id"] = "new"
        with self.assertRaises(ValueError): self.evaluated(c)
        c = self.confirmation()
        c["rows"].append(copy.deepcopy(c["rows"][0]))
        with self.assertRaises(ValueError): self.evaluated(c)

    def test_legacy_is_source_observation_only_and_cannot_add_paths(self):
        c = self.confirmation()
        proof = c["rows"][0]["checkpoints"][0]["observations"][0]
        proof.pop("condition_id")
        self.rec["rows"][0].pop("short_term_assessment")
        c["rows"] = [{"code": "002242", "observations": [proof]}]
        out = s.evaluate_confirmation(self.rec, c, NOW)
        self.assertEqual(out["rows"][0]["status"], "LEGACY_OBSERVATION_ONLY_NO_PREDECLARED_EXPECTATION")
        c["rows"][0]["checkpoints"] = [{"checkpoint_id": "invented"}]
        with self.assertRaises(ValueError): s.evaluate_confirmation(self.rec, c, NOW)

    def test_unobserved_future_checkpoint_stays_pending(self):
        c = self.confirmation()
        future = plan()
        future.update(checkpoint_id="later", planned_at="2026-10-13T10:00:00+08:00",
            observation_window={"start_at": "2026-10-13T09:55:00+08:00", "end_at": "2026-10-13T10:00:00+08:00"})
        self.rec["rows"][0]["short_term_assessment"]["confirmation_plan"].append(future)
        out = s.evaluate_confirmation(self.rec, c, NOW)
        self.assertEqual(out["rows"][0]["checkpoints"][1]["status"], "PENDING")

    def test_partial_observation_retains_original_cohort_denominator(self):
        c = self.confirmation()
        self.rec["rows"].append({"code": "600241"})
        out = s.evaluate_confirmation(self.rec, c, NOW)
        self.assertEqual(out["registered_pool_size"], 2)
        self.assertEqual(out["observed_codes"], ["002242"])
        self.assertEqual(out["unobserved_codes"], ["600241"])
        self.assertEqual(out["unobserved_rows"][0]["status"], "UNKNOWN")
        self.assertEqual(out["observation_coverage"], .5)

    def test_no_observations_future_upper_and_claimed_market_consensus_rejected(self):
        c = self.confirmation()
        c["rows"][0]["checkpoints"] = []
        with self.assertRaises(ValueError): self.evaluated(c)
        c = self.confirmation()
        c["observed_at"] = "2026-10-13T09:41:00+08:00"
        with self.assertRaises(ValueError): self.evaluated(c)
        self.rec["rows"][0]["short_term_assessment"]["confirmation_plan"][0]["baseline_kind"] = "MARKET_CONSENSUS"
        with self.assertRaisesRegex(ValueError, "consensus"):
            s.validate_plans(self.facts(), self.rec)

    def test_review_confirm_integration_uses_shanghai_and_preserves_original(self):
        c = self.confirmation()
        rp, cp, output = self.root/"record.json", self.root/"confirm.json", self.root/"append.json"
        facts = self.root/"facts.json"; e.write_new(facts, self.facts())
        self.rec.update(facts_path=str(facts.resolve()), facts_sha256=e.sha(facts), source_bindings=e.bind(facts))
        e.write_new(rp, self.rec); e.write_new(cp, c)
        original_hash = e.sha(rp)
        receipt = [{"path": str(rp.resolve()), "sha256": original_hash}]
        with patch.object(e, "now", return_value=NOW), patch.object(review.original, "registrations", return_value=receipt):
            result = review.confirm(rp, cp, output)
        self.assertEqual(result["expectation_review"]["rows"][0]["checkpoints"][0]["status"], "AMBIGUOUS")
        self.assertEqual(original_hash, e.sha(rp))
        self.assertFalse(result["original_prediction_changed"])

    def test_confirm_rejects_legacy_copies_and_retrospectively_inserted_paths(self):
        c = self.confirmation()
        rp, cp = self.root/"record.json", self.root/"confirm.json"
        legacy = copy.deepcopy(self.rec)
        legacy["rows"][0].pop("short_term_assessment")
        e.write_new(rp, legacy); e.write_new(cp, c)
        receipt = [{"path": str(rp.resolve()), "sha256": e.sha(rp)}]
        rp.write_text(json.dumps(self.rec), encoding="utf-8")
        with patch.object(e, "now", return_value=NOW), patch.object(review.original, "registrations", return_value=receipt):
            with self.assertRaisesRegex(ValueError, "changed"):
                review.confirm(rp, cp, self.root/"append.json")
            copied = self.root/"copy.json"; e.write_new(copied, self.rec)
            with self.assertRaisesRegex(ValueError, "registration"):
                review.confirm(copied, cp, self.root/"copy_append.json")


if __name__ == "__main__": unittest.main()
