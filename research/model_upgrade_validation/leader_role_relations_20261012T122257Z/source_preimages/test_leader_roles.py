import copy
from datetime import date, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import leader_role_engine as e
import leader_role_research as r
import leader_role_review as review


class RolesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        dates = [date(2026, 8, 3)+timedelta(days=i) for i in range(90)]
        self.days = [d.isoformat() for d in dates if d.weekday() < 5]
        self.cal = self.root/"calendar.json"
        e.write_new(self.cal, {"status": "verified", "source": "synthetic_fixture_only", "sessions": self.days})
        self.source = self.root/"raw.json"
        self.source_data = {"formed_at": "2026-10-10T10:00:00+08:00", "groups": {
            "pool": {"complete_within_scope": True, "codes": ["000001", "000002"], "boards": {"000001": 3, "000002": 2}},
            "price": {"turnover": .15, "signal_one_word": 1., "price": 10.},
            "path": {"event_sequence_complete": True, "events": [
                {"observed_at": "2026-10-09T11:25:00+08:00", "state": "OPEN"},
                {"observed_at": "2026-10-09T13:05:00+08:00", "state": "SEALED"}]},
            "theme": {"membership_verified": True, "complete_within_scope": True, "granularity": "specific",
                "theme_id": "fixture", "codes": ["000001", "000002"], "members": {
                    "000001": {"amount": 100., "return_pct": 10., "boards": 3},
                    "000002": {"amount": 50., "return_pct": 5., "boards": 2}}}}}
        e.write_new(self.source, self.source_data)
        self.row = {"candidate_id": "2026-10-09:000001", "signal_date": "2026-10-09", "code": "000001",
            "name": "fixture", "boards": 3, "scope_id": "fixture", "decision_at": "2026-10-11T10:00:00+08:00",
            "available_at": "2026-10-10T10:00:00+08:00", "actual_one_word_daily": 0., "groups": {}}
        for kind in self.source_data["groups"]:
            self.row["groups"][kind] = {"source_path": str(self.source), "source_sha256": e.sha(self.source),
                "pointer": ["groups", kind], "value": self.source_data["groups"][kind], "status": "VERIFIED",
                "observed_at": "2026-10-09T15:00:00+08:00", "available_at": "2026-10-10T10:00:00+08:00"}
        self.dataset = {"version": e.VERSION, "formed_at": e.now(), "calendar_path": str(self.cal),
            "calendar_sha256": e.sha(self.cal), "source_bindings": e.bind(self.source), "rows": [self.row]}

    def test_roles_and_multi_identity(self):
        row = e.analyse(self.dataset, "STRICT")["rows"][0]
        self.assertEqual(set(row["roles"]), set(e.ROLES))
        self.assertTrue(row["roles"]["HEIGHT"]["baseline_call"])
        self.assertEqual(row["roles"]["THEME"]["state"], "CANDIDATE")
        self.assertEqual(row["roles"]["TURNOVER"]["state"], "CANDIDATE")
        self.assertFalse(row["role_identity_automatically_certified"])

    def test_saved_first_board_quotes_units_ties_and_no_target_fields(self):
        roster = [{"code": "000001", "boards": 1, "price_cents": 1234, "amount": 100,
                   "turnover_pct": 2.5, "target_promoted": True},
                  {"code": "000002", "boards": 1, "price_cents": 1234, "amount": 300,
                   "turnover_pct": 7.5}]
        facts = e.closing_price_facts(roster, {})
        self.assertEqual(facts["000001"]["price"], 12.34)
        self.assertEqual(facts["000001"]["turnover"], .025)
        self.assertEqual(facts["000001"]["signal_amount"], 100)
        self.assertEqual(facts["000001"]["price_same_height_percentile"], .75)
        self.assertEqual(facts["000001"]["signal_amount_same_height_percentile"], .5)
        self.assertEqual(set(facts["000001"]), set(e.RELAY_FIELDS))
        for key in ("signal_one_word", "turnover_vs_prev", "signal_amplitude"):
            self.assertIsNone(facts["000001"].get(key))

    def test_first_board_projection_preserves_original_relay_features(self):
        original = {key: float(i) for i, key in enumerate(e.RELAY_FIELDS)}
        relay = {"000001": {"features": original}}
        roster = [{"code": "000001", "boards": 3, "price_cents": 99999, "amount": 100,
                   "turnover_pct": 99},
                  {"code": "000002", "boards": 1, "price_cents": 1000, "amount": 10,
                   "turnover_pct": 2}]
        before = copy.deepcopy(relay)
        facts = e.closing_price_facts(roster, relay)
        self.assertEqual(facts["000001"], original)
        self.assertEqual(relay, before)
        self.assertEqual(facts["000002"]["height_gap"], 2)
        self.assertEqual(facts["000002"]["same_height_rivals"], 0)
        self.assertEqual(facts["000002"]["is_space_leader"], 0)

    def test_invalid_quotes_stay_unknown_and_incomplete_ranks_not_imputed(self):
        roster = [{"code": "000001", "boards": 1, "price_cents": 0, "amount": -1,
                   "turnover_pct": True},
                  {"code": "000002", "boards": 1, "price_cents": 1000, "amount": 10,
                   "turnover_pct": 2}]
        facts = e.closing_price_facts(roster, {})
        for key in ("price", "signal_amount", "turnover", "log_signal_amount"):
            self.assertIsNone(facts["000001"][key])
        for key in ("price_same_height_percentile", "signal_amount_same_height_percentile", "turnover_same_height_percentile"):
            self.assertIsNone(facts["000002"][key])
        with self.assertRaisesRegex(ValueError, "nonduplicated"):
            e.closing_price_facts([roster[0], roster[0]], {})

    def test_lunch_excluded_and_proxy_separate(self):
        row = e.analyse(self.dataset, "STRICT")["rows"][0]
        self.assertEqual(row["features"]["actual_path_open_minutes"], 10.)
        self.assertEqual(row["actual_one_word_daily"], 0.)
        self.assertEqual(row["low_turnover_proxy"], 1.)

    def test_missing_path_not_volume_confirmation(self):
        del self.row["groups"]["path"]
        row = e.analyse(self.dataset, "STRICT")["rows"][0]
        self.assertEqual(row["roles"]["TURNOVER"]["state"], "UNKNOWN")

    def test_late_strict_but_diagnostic_retained(self):
        self.row["decision_at"] = "2026-10-09T16:30:00+08:00"
        self.assertFalse(e.analyse(self.dataset, "STRICT")["rows"][0]["role_eligible"]["HEIGHT"])
        self.assertTrue(e.analyse(self.dataset, "DIAGNOSTIC")["rows"][0]["role_eligible"]["HEIGHT"])

    def test_source_tamper_rejected(self):
        self.source.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash"):
            e.analyse(self.dataset, "STRICT")

    def test_projection_tamper_rejected(self):
        cell = copy.deepcopy(self.row["groups"]["pool"])
        cell["value"]["boards"]["000001"] = 8
        self.row["groups"]["pool"] = cell
        with self.assertRaisesRegex(ValueError, "differs"):
            e.analyse(self.dataset, "STRICT")

    def test_normalized_fact_bundle_checks_bound_upstream_sources(self):
        self.source_data["source_bindings"] = {str(self.root/"missing_raw.json"): "a"*64}
        self.source.write_text(json.dumps(self.source_data), encoding="utf-8")
        for cell in self.row["groups"].values():
            cell["source_sha256"] = e.sha(self.source)
        self.dataset["source_bindings"] = e.bind(self.source)
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            e.analyse(self.dataset, "DIAGNOSTIC")

    def test_wrong_group_date_rejected(self):
        self.row["signal_date"] = "2026-10-08"
        self.row["candidate_id"] = "2026-10-08:000001"
        with self.assertRaises(ValueError):
            e.analyse(self.dataset, "STRICT")

    def test_fake_trusted_adapter_rejected(self):
        self.row["groups"]["pool"]["adapter"] = "SAVED_BASE_ALLOWLIST"
        p = self.root/"dataset.json"; e.write_new(p, self.dataset)
        with self.assertRaisesRegex(ValueError, "trusted"):
            e.load_dataset(p)

    def test_label_unknowns_preserved(self):
        p = self.root/"packet.json"; e.write_new(p, r.masked_packet(self.dataset))
        labels = r.label_template(self.dataset, p)
        l = self.root/"labels.json"; e.write_new(l, labels)
        loaded = r.load_labels(l, self.dataset)
        self.assertEqual(len(loaded), 11)
        self.assertTrue(all(v["value"] is None for v in loaded.values()))

    def test_masked_packet_cannot_link_model_or_future_source_archives(self):
        data = copy.deepcopy(self.dataset)
        data["source_bindings"]["operator_only_future_labels.json"] = "a"*64
        data["source_bindings"]["operator_only_model_scores.json"] = "b"*64
        before = copy.deepcopy(data)
        packet = r.masked_packet(data)
        self.assertEqual(packet["source_bindings"], {})
        self.assertNotIn("operator_only_future", json.dumps(packet))
        for group in packet["rows"][0]["groups"].values():
            self.assertNotIn("source_path", group)
            self.assertNotIn("pointer", group)
        self.assertEqual(data, before)
        self.assertIn("source_path", data["rows"][0]["groups"]["pool"])

    def test_matching_reviews_reference_reconstructed_masked_facts(self):
        p = self.root/"packet.json"; e.write_new(p, r.masked_packet(self.dataset))
        labels = r.label_template(self.dataset, p)
        cell = labels["rows"][0]; moment = e.now()
        cell.update(value=True, available_at=moment, reviews=[{
            "reviewer_id": name, "verdict": True, "model_masked": True,
            "future_masked": True, "independent_of_prediction": True,
            "reviewed_at": moment, "evidence_refs": [str(p.resolve())],
            "basis": "Complete scoped roster has height3 above height2."}
            for name in ("independent_a", "independent_b")])
        labels["formed_at"] = e.now()
        good = self.root/"reviewed.json"; e.write_new(good, labels)
        self.assertTrue(r.load_labels(good, self.dataset)[(self.row["candidate_id"], "HEIGHT")]["value"])
        cell["reviews"][0]["evidence_refs"] = [str(self.source)]
        bad = self.root/"unmasked_ref.json"; e.write_new(bad, labels)
        with self.assertRaisesRegex(ValueError, "masked fact"):
            r.load_labels(bad, self.dataset)

    def test_actual_review_import_preserves_context_exposure_and_disagreement(self):
        dataset = self.root/"dataset.json"; e.write_new(dataset, self.dataset)
        packet = self.root/"packet.json"; e.write_new(packet, r.masked_packet(self.dataset))
        paths = []
        for name, height in (("a", True), ("b", True)):
            moment = e.now()
            receipt = {"reviewer_id": name, "packet_path": packet.resolve().as_posix(), "packet_sha256": e.sha(packet),
                "review_started_at": moment, "reviewed_at": moment, "model_masked": True, "future_masked": True,
                "independent_of_prediction": True, "rows": [{"candidate_id": self.row["candidate_id"],
                    "role": role, "verdict": height if role == "HEIGHT" else None,
                    "basis": "synthetic fixture scoped height evidence or missing required role facts",
                    "evidence_refs": [packet.resolve().as_posix()]} for role in e.ROLES]}
            path = self.root/(name+".json"); e.write_new(path, receipt); paths.append(path)
        good = r.merge_identity_reviews(dataset, packet, paths, self.root/"known.json")
        self.assertTrue(good["rows"][0]["value"])
        exposed = e.read(paths[1]); exposed["future_masked"] = False
        exposed_path = self.root/"exposed.json"; e.write_new(exposed_path, exposed)
        unqualified = r.merge_identity_reviews(dataset, packet, [paths[0], exposed_path], self.root/"unqualified.json")
        self.assertIsNone(unqualified["rows"][0]["value"])
        self.assertEqual(unqualified["rows"][0]["unknown_reason"], "REVIEW_CONTEXT_NOT_BLINDED")
        disagreement = e.read(paths[1]); disagreement["rows"][0]["verdict"] = False
        different = self.root/"disagreed.json"; e.write_new(different, disagreement)
        disputed = r.merge_identity_reviews(dataset, packet, [paths[0], different], self.root/"disputed.json")
        self.assertIsNone(disputed["rows"][0]["value"])
        self.assertEqual(disputed["rows"][0]["unknown_reason"], "REVIEW_DISAGREEMENT")
        with self.assertRaisesRegex(ValueError, "different actual"):
            r.merge_identity_reviews(dataset, packet, [paths[0], paths[0]], self.root/"duplicate.json")

    def test_label_cannot_omit_failed_or_unknown(self):
        p = self.root/"packet.json"; e.write_new(p, r.masked_packet(self.dataset))
        labels = r.label_template(self.dataset, p); labels["rows"].pop()
        l = self.root/"labels.json"; e.write_new(l, labels)
        with self.assertRaisesRegex(ValueError, "omits"):
            r.load_labels(l, self.dataset)

    def test_known_label_requires_independent_review(self):
        p = self.root/"packet.json"; e.write_new(p, r.masked_packet(self.dataset))
        labels = r.label_template(self.dataset, p); labels["rows"][0]["value"] = True
        l = self.root/"labels.json"; e.write_new(l, labels)
        with self.assertRaisesRegex(ValueError, "independent"):
            r.load_labels(l, self.dataset)

    def test_role_metrics_abstention_and_unknown_truth(self):
        m = r.binary_metrics({"a": None, "b": False, "c": True}, {"a": True, "b": False, "c": None})
        self.assertEqual(m["registered"], 3)
        self.assertEqual(m["unknown_labels"], 1)
        self.assertEqual(m["complete_positive_Recall"], 0.)
        self.assertEqual(m["positive_abstentions"], 1)
        self.assertIsNone(m["Brier"])

    def diagnostic_record(self):
        return {"candidate_id": self.row["candidate_id"], "signal_date": self.row["signal_date"],
            "code": self.row["code"], "track": "DIAGNOSTIC", "fold": "fixture",
            "role_truth": dict.fromkeys(e.ROLES),
            "C1": {"calls": dict.fromkeys(e.ROLES), "scores": dict.fromkeys(e.ROLES)},
            "role_input_eligible": dict.fromkeys(e.ROLES, False),
            "missing_role_inputs": {role: ["MISSING_SOURCE"] for role in e.ROLES},
            "BASE_promotion": False, "C2_promotion": False, "promotion_truth": True}

    def append_fixture_group(self, kind, value):
        self.source_data["groups"][kind] = value
        self.source.write_text(json.dumps(self.source_data), encoding="utf-8")
        for cell in self.row["groups"].values():
            cell["source_sha256"] = e.sha(self.source)
        self.row["groups"][kind] = {"value": value, "status": "VERIFIED", "source_path": str(self.source),
            "source_sha256": e.sha(self.source), "pointer": ["groups", kind],
            "observed_at": "2026-10-09T15:00:00+08:00", "available_at": "2026-10-10T10:00:00+08:00"}
        self.dataset["source_bindings"] = e.bind(self.source)

    def test_cross_cycle_requires_actual_pre_and_post_pressure_observations(self):
        value = {"event_id": "fixture", "pre_event_role": "THEME", "started_at": "2026-10-08T10:00:00+08:00",
            "ended_at": "2026-10-09T10:00:00+08:00", "post_event_competition_verified": True,
            "pre_cycle_id": "old", "post_cycle_id": "new"}
        self.append_fixture_group("pressure", value)
        weak = e.analyse(self.dataset, "STRICT")["rows"][0]
        self.assertIn("INCOMPLETE_PRESSURE", weak["roles"]["CROSS_CYCLE"]["missing"])
        self.assertIsNone(weak["features"]["pressure_survived"])
        value.update(pre_event_observation={"code": "000001", "state": "CORE", "relative_rank": 3,
            "price": 10., "observed_at": "2026-10-08T09:35:00+08:00"},
            post_event_observation={"code": "000001", "state": "CORE", "relative_rank": 1,
            "price": 11., "observed_at": "2026-10-09T15:00:00+08:00"})
        value["post_competition"] = {"complete_within_scope": True, "peer_codes": ["000002", "000003"],
            "prior_at": "2026-10-08T09:35:00+08:00", "current_at": "2026-10-09T15:00:00+08:00",
            "prior": {"000001": "STRONG", "000002": "STRONG", "000003": "STRONG"},
            "current": {"000001": "STRONG", "000002": "FAILED", "000003": "COMPETING"}}
        self.append_fixture_group("pressure", value)
        facts = e.analyse(self.dataset, "STRICT")["rows"][0]
        self.assertEqual(facts["features"]["pressure_rank_improvement"], 2)
        self.assertAlmostEqual(facts["features"]["pressure_price_change_pct"], 10.)
        self.assertEqual(facts["features"]["pressure_cycle_changed"], 1.)
        self.assertFalse(facts["role_identity_automatically_certified"])
        value["post_competition"]["current"].pop("000003")
        self.append_fixture_group("pressure", value)
        incomplete = e.analyse(self.dataset, "STRICT")["rows"][0]
        self.assertIsNone(incomplete["features"]["pressure_survived"])

    def test_supplement_divergence_can_precede_anchor_break_and_future_break_is_rejected(self):
        value = {"anchor_code": "000002", "same_specific_theme": True, "anchor_identity_source": "synthetic_prior_review",
            "anchor_stable": True, "anchor_start": "2026-10-06", "own_start": "2026-10-08",
            "anchor_stage": "DIVERGENCE", "anchor_break_day": "2026-10-09"}
        self.append_fixture_group("dependency", value)
        facts = e.analyse(self.dataset, "STRICT")["rows"][0]
        self.assertTrue(facts["role_eligible"]["SUPPLEMENT"])
        self.assertEqual(facts["features"]["dependency_anchor_is_divergence"], 1.)
        self.assertEqual(facts["features"]["dependency_start_after_anchor_break_sessions"], -1)
        self.assertEqual(facts["roles"]["SUPPLEMENT"]["state"], "CANDIDATE")
        value["anchor_break_day"] = "2026-10-12"
        self.append_fixture_group("dependency", value)
        with self.assertRaisesRegex(ValueError, "dependency date follows signal"):
            e.analyse(self.dataset, "DIAGNOSTIC")

    def test_promotion_miss_cannot_be_rewritten_as_role_false_negative(self):
        case = self.diagnostic_record()
        d = r.role_error_diagnosis([case])
        self.assertEqual(d["role_cells"], 11)
        self.assertEqual(d["cases"][0]["promotion_evaluation"]["BASE"], "FALSE_NEGATIVE")
        self.assertTrue(all(v["evaluation"] == "TRUTH_UNKNOWN" for v in d["cases"][0]["roles"].values()))

    def test_saved_role_diagnosis_rejects_changed_prediction_matrix(self):
        protocol = self.root/"protocol.json"; e.write_new(protocol, {"source_bindings": {}})
        matrix = self.root/"full_candidate_comparison.json"; e.write_new(matrix, [self.diagnostic_record()])
        report = self.root/"report.json"
        e.write_new(report, {"protocol_sha256": e.sha(protocol), "predictions_sha256": e.sha(matrix),
            "tracks": {"DIAGNOSTIC": {"folds": {"fixture": {"C1_fit_logs": {}}}}}})
        result = r.diagnose_saved_comparison(report, protocol, self.root/"diagnosis.json")
        self.assertEqual(result["candidate_records"], 1)
        self.assertFalse(result["deployment_allowed"])
        matrix.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            r.diagnose_saved_comparison(report, protocol, self.root/"tampered.json")

    def test_role_diagnosis_preserves_multirole_abstention_and_missing_inputs(self):
        case = self.diagnostic_record()
        case["role_truth"].update(HEIGHT=True, SUPPLEMENT=True, CROSS_CYCLE=False)
        case["C1"]["calls"].update(HEIGHT=True, CROSS_CYCLE=True)
        case["role_input_eligible"]["HEIGHT"] = True
        d = r.role_error_diagnosis([case], {("DIAGNOSTIC", "fixture"):
            {"SUPPLEMENT": {"status": "INSUFFICIENT_KNOWN_CLASSES", "attempted": False}}})
        roles = d["cases"][0]["roles"]
        self.assertEqual(roles["HEIGHT"]["evaluation"], "TRUE_POSITIVE")
        self.assertEqual(roles["SUPPLEMENT"]["evaluation"], "POSITIVE_ABSTENTION")
        self.assertIn("REQUIRED_ROLE_INPUTS_UNAVAILABLE", roles["SUPPLEMENT"]["reasons"])
        self.assertIn("ROLE_HEAD_NOT_FITTED:INSUFFICIENT_KNOWN_CLASSES", roles["SUPPLEMENT"]["reasons"])
        self.assertEqual(roles["CROSS_CYCLE"]["evaluation"], "FALSE_POSITIVE")
        self.assertIn("ROLE_CALL_WITHOUT_REQUIRED_INPUTS", roles["CROSS_CYCLE"]["reasons"])

    def test_training_member_role_score_rejected(self):
        model = r.RoleModel(); model.training_ids["HEIGHT"] = [self.row["candidate_id"]]
        row = e.analyse(self.dataset, "STRICT")["rows"][0]
        with self.assertRaisesRegex(ValueError, "training member"):
            model.predict([row], reject_training_members=True)

    def test_frozen_role_score_explanation_reconstructs_actual_logit(self):
        x = np.zeros((40, len(e.FEATURES)))
        index = e.FEATURES.index("price")
        x[:20, index] = -1.; x[20:, index] = 1.
        pipeline = r._pipeline().fit(x, np.asarray([0]*20+[1]*20))
        model = r.RoleModel(); model.models["HEIGHT"] = pipeline
        row = e.analyse(self.dataset, "STRICT")["rows"][0]
        result = model.predict([row])[row["candidate_id"]]
        explanation = result["explanations"]["HEIGHT"]
        self.assertAlmostEqual(explanation["intercept"]+explanation["sum_feature_contributions"],
                               float(pipeline.decision_function(r._matrix([row]))[0]), places=10)
        self.assertTrue(explanation["not_causal_effect"])
        self.assertTrue(explanation["not_calibrated_probability"])
        self.assertEqual(explanation["top_factors"][0]["feature"], "price")
        self.assertIsNone(result["explanations"]["SUPPLEMENT"])

    def test_no_role_labels_means_no_fit(self):
        row = e.analyse(self.dataset, "STRICT")["rows"][0]
        labels = {(row["candidate_id"], role): {"value": None, "available_at": e.now()} for role in e.ROLES}
        m = r.RoleModel().fit([row], labels, "2026-10-12T00:00:00+08:00", self.days, "DIAGNOSTIC")
        self.assertFalse(m.models)
        self.assertTrue(all(not log["attempted"] for log in m.logs.values()))

    def test_calendar_horizon_exact(self):
        self.assertEqual(e.future_day(self.days, "2026-10-09", 3), "2026-10-14")

    def test_capacity_needs_non_limit_members(self):
        row = e.analyse(self.dataset, "STRICT")["rows"][0]
        self.assertFalse(row["role_eligible"]["CAPACITY"])
        self.assertFalse(row["role_eligible"]["TREND"])

    def test_complete_competition_missing_peer_not_failure(self):
        value = {"complete_within_scope": True, "peer_codes": ["000002"],
                 "prior": {"000001": "STRONG", "000002": "STRONG"}, "current": {"000001": "STRONG"},
                 "prior_at": "2026-10-09T10:00:00+08:00", "current_at": "2026-10-09T14:00:00+08:00"}
        self.assertFalse(e._valid_group("competition", value, self.row))

    def test_bootstrap_abstention_counts_fn(self):
        records = [{"candidate_id": str(i), "signal_date": day,
            "role_truth": {role: True for role in e.ROLES}, "baseline_roles": {role: None for role in e.ROLES},
            "C1": {"calls": {role: True for role in e.ROLES}}} for i, day in enumerate(self.days[:6])]
        self.assertEqual(r.paired_macro_interval(records, self.days[:6]), [1., 1.])

    def test_confirmation_future_rejected(self):
        rec = self.root/"record.json"; conf = self.root/"confirmation.json"
        e.write_new(rec, {"signal_date": "2026-10-09", "target_date": "2026-10-12", "formed_at": "2026-10-11T10:00:00+08:00", "rows": [{"code": "000001"}]})
        e.write_new(conf, {"signal_date": "2026-10-09", "target_date": "2026-10-12", "observed_at": "2026-10-13T10:00:00+08:00", "rows": [], "source_bindings": {}})
        with self.assertRaisesRegex(ValueError, "actual"):
            review.confirm(rec, conf, self.root/"append.json")

    def test_augment_rebuild_and_original_cutoff_preserved(self):
        parent = self.root/"parent.json"; output = self.root/"augmented.json"
        e.write_new(parent, self.dataset)
        data = e.augment(parent, output)
        loaded = e.load_dataset(output)
        self.assertEqual(data["rows"], loaded["rows"])
        self.assertEqual(loaded["rows"][0]["original_decision_at"], self.row["decision_at"])
        self.assertGreater(e.stamp(loaded["rows"][0]["decision_at"]), e.stamp(self.row["decision_at"]))

    def test_rich_evidence_cannot_replace_price(self):
        parent, bank = self.root/"parent.json", self.root/"bank.json"
        e.write_new(parent, self.dataset)
        e.write_new(bank, {"version": e.VERSION, "formed_at": e.now(), "rows": [
            {"candidate_id": self.row["candidate_id"], "groups": {"price": self.row["groups"]["price"]}}]})
        with self.assertRaisesRegex(ValueError, "protected"):
            e.augment(parent, self.root/"bad.json", evidence_path=bank)

    def test_augmented_group_tamper_rejected(self):
        parent, output = self.root/"parent.json", self.root/"augmented.json"
        e.write_new(parent, self.dataset)
        data = e.augment(parent, output)
        data["rows"][0]["boards"] = 5
        output.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "projection"):
            e.load_dataset(output)

    def test_partial_current_dataset_not_strict_available(self):
        self.row["available_at"] = "2026-10-12T16:00:00+08:00"
        row = e.analyse(self.dataset, "STRICT")["rows"][0]
        self.assertFalse(row["role_eligible"]["HEIGHT"])

    def promotion_data(self):
        row = e.analyse(self.dataset, "STRICT")["rows"][0]
        data = copy.deepcopy(self.dataset)
        data["adapter"] = {"kind": "AUGMENTED"}
        data["baseline_binding"] = {"original_codes": ["000001"]}
        data["baseline_promotion"] = {row["candidate_id"]: {"call": True, "raw_score": .8,
            "truth": None, "origin": "SOURCE_BOUND_PREAUCTION", "available_at": "2026-10-10T10:00:00+08:00"}}
        return data, row

    def test_C2_missing_role_falls_back_same_BASE(self):
        data, row = self.promotion_data()
        roles = {row["candidate_id"]: {"scores": {role: None for role in e.ROLES}}}
        pred = r.promotion_predictions(data, [row], roles, None)
        self.assertEqual(pred["ROLE_C2"][0]["raw_score"], .8)
        self.assertTrue(pred["ROLE_C2"][0]["prediction"])
        self.assertTrue(pred["ROLE_C2"][0]["fallback"])

    def test_C2_complete_roles_use_registered_width(self):
        data, row = self.promotion_data()
        class FixedFixture:
            n_features_in_ = len(e.RELAY_FIELDS)+len(e.ROLES)
            def predict_proba(self, x):
                self.x = x
                return np.asarray([[.7, .3]])
        model = FixedFixture()
        roles = {row["candidate_id"]: {"scores": {role: .6 for role in e.ROLES}}}
        pred = r.promotion_predictions(data, [row], roles, model)
        self.assertFalse(pred["ROLE_C2"][0]["prediction"])
        self.assertIsNone(pred["ROLE_C2"][0]["fallback"])
        self.assertEqual(model.x.shape[1], model.n_features_in_)

    def test_C2_cannot_delete_original_candidate(self):
        data, row = self.promotion_data(); data["baseline_binding"]["original_codes"].append("000002")
        with self.assertRaisesRegex(ValueError, "complete"):
            r.promotion_predictions(data, [row], {}, None)

    def test_original_gate_adapter_no_new_window_is_not_passed(self):
        value = r.assess_promotion([], {}, [], self.days)
        self.assertFalse(value["passed"])

    def test_original_gate_adapter_preserves_pure_modules(self):
        import relay_context_research as context
        original = context.ARMS.copy()
        facts = {"code": "000001", "boards": 2, "actual_one_word_daily": 0., "prediction": True, "raw_score": .8}
        record = {"signal_date": self.days[0], "target_dates": {"1": self.days[1]},
            "baseline_binding": {"original_codes": ["000001"]}, "promotion_predictions": {"BASE": [facts], "ROLE_C2": [dict(facts)]}}
        result = r.assess_promotion([record], {self.days[0]: {"000001": True}}, self.days[:5], self.days)
        self.assertFalse(result["passed"])
        self.assertEqual(context.ARMS, original)
        self.assertIsNone(result["probability_Brier"])

    def test_invalidation_delayed_observation_and_censoring(self):
        key = self.row["candidate_id"]
        rec = {"rows": [{"candidate_id": key}]}
        def event(h, truth, state):
            return {"registration_sha256": "fixture", "horizon": h, "rows": [{"candidate_id": key,
                "role_truth": {role: truth for role in e.ROLES}, "target_role_states": {role: state for role in e.ROLES}}]}
        result = r.invalidation_latency({"fixture": rec}, [event(0, True, "CANDIDATE"),
            event(3, False, "CANDIDATE"), event(5, False, "INVALIDATED")], self.days)
        self.assertEqual(result["mean_observed_delay_sessions"], 2.)
        self.assertEqual(result["observed_losses"], 11)
        self.assertIsNone(result["exact_intraday_loss_time_or_latency"])

    def test_unknown_initial_identity_cannot_claim_loss(self):
        key = self.row["candidate_id"]
        event = {"registration_sha256": "fixture", "horizon": 3, "rows": [{"candidate_id": key,
            "role_truth": {role: False for role in e.ROLES}, "target_role_states": {role: "INVALIDATED" for role in e.ROLES}}]}
        value = r.invalidation_latency({"fixture": {"rows": [{"candidate_id": key}]}}, [event], self.days)
        self.assertEqual(value["observed_losses"], 0)
        self.assertEqual(value["unknown_or_censored"], 11)

    def test_foreign_role_settlement_rejected(self):
        protocol = self.root/"p.json"; e.write_new(protocol, {})
        with self.assertRaisesRegex(ValueError, "registered"):
            r.verify_settlement({"registration_sha256": "not_registered", "protocol_sha256": e.sha(protocol)}, protocol, {})

    def test_abstaining_baseline_precision_gate_explicit_reference(self):
        a = r.binary_metrics({"a": True}, {"a": True})
        b = r.binary_metrics({"a": None}, {"a": True})
        self.assertTrue(r.metric_not_lower(a, b, "Precision"))
        self.assertIsNone(b["Precision"])

    def test_rich_competition_appended_from_exact_source(self):
        parent, bank, raw = self.root/"parent.json", self.root/"bank.json", self.root/"competition.json"
        value = {"complete_within_scope": True, "peer_codes": ["000002"],
            "prior": {"000001": "COMPETING", "000002": "STRONG"},
            "current": {"000001": "STRONG", "000002": "FAILED"},
            "prior_at": "2026-10-09T10:00:00+08:00", "current_at": "2026-10-09T14:00:00+08:00"}
        e.write_new(parent, self.dataset)
        e.write_new(raw, {"formed_at": "2026-10-10T10:00:00+08:00", "competition": value})
        cell = {"source_path": str(raw), "source_sha256": e.sha(raw), "pointer": ["competition"], "value": value,
            "status": "VERIFIED", "observed_at": "2026-10-09T14:00:00+08:00", "available_at": "2026-10-10T10:00:00+08:00"}
        e.write_new(bank, {"version": e.VERSION, "formed_at": e.now(), "rows": [{"candidate_id": self.row["candidate_id"], "groups": {"competition": cell}}]})
        out = self.root/"aug.json"; e.augment(parent, out, evidence_path=bank)
        row = e.analyse(e.load_dataset(out), "STRICT")["rows"][0]
        self.assertTrue(row["role_eligible"]["POSITION"])
        self.assertEqual(row["features"]["competition_failed_share"], 1.)
        self.assertEqual(row["roles"]["POSITION"]["state"], "CANDIDATE")

    def test_multirole_observation_does_not_certify_identity(self):
        row = e.analyse(self.dataset, "STRICT")["rows"][0]
        p = {row["candidate_id"]: {"calls": {role: role in ("HEIGHT", "TURNOVER") for role in e.ROLES},
            "scores": {role: .8 if role == "HEIGHT" else .6 for role in e.ROLES}}}
        result = r.role_observations([row], p)[0]
        self.assertEqual(result["primary_role"], "HEIGHT")
        self.assertEqual(result["auxiliary_roles"], ["TURNOVER"])
        self.assertFalse(result["identity_certified"])

    def test_exact_promotion_successor_rejected_before_source_adapter(self):
        record = {"baseline_binding": {"compatibility_path": "fixture", "target_date": "2026-10-13"},
                  "source_bindings": {}, "dataset_path": "fixture", "signal_date": "2026-10-09",
                  "target_dates": {"1": "2026-10-13"}}
        with patch.object(e, "load_dataset", return_value={"calendar_path": str(self.cal)}):
            with self.assertRaisesRegex(ValueError, "successor"):
                r.reconstruct_promotion_outcomes(record, "no-source-needed")


if __name__ == "__main__":
    unittest.main()
