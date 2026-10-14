import copy
from datetime import date, timedelta
import json
from pathlib import Path
import tempfile
import unittest

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

    def test_training_member_role_score_rejected(self):
        model = r.RoleModel(); model.training_ids["HEIGHT"] = [self.row["candidate_id"]]
        row = e.analyse(self.dataset, "STRICT")["rows"][0]
        with self.assertRaisesRegex(ValueError, "training member"):
            model.predict([row], reject_training_members=True)

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


if __name__ == "__main__":
    unittest.main()
