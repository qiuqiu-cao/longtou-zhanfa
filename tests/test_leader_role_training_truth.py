"""Original-label lineage, maturity, masking and downstream cohort checks."""
import copy
from datetime import date, timedelta
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import pandas as pd

import leader_role_engine as e
import leader_role_research as r
import leader_role_training_truth as t


class TrainingTruthTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.days = [(date(2024, 1, 2)+timedelta(days=i)).isoformat()
                     for i in range(30) if (date(2024, 1, 2)+timedelta(days=i)).weekday() < 5]
        self.cal = self.root/"calendar.json"
        e.write_new(self.cal, {"status": "verified", "source": "synthetic_fixture_only", "sessions": self.days})
        self.raw = self.root/"raw.json"
        e.write_new(self.raw, {"fixture": "synthetic_original_labels"})
        self.rows, records = [], []
        for i in range(12):
            signal, code = self.days[i], f"{i+1:06d}"
            values = dict.fromkeys(e.RELAY_FIELDS)
            values.update(boards=2, price=10.)
            self.rows.append({"candidate_id": signal+":"+code, "signal_date": signal,
                "code": code, "name": "synthetic", "boards": 2, "decision_at": signal+"T16:00:00+08:00",
                "scope_id": "synthetic_scope", "groups": {"price": {"value": values}}})
            records.append(values | {"date": signal, "code": code, "target": float(i % 2),
                "target_date": self.days[i+1], "official_target_date": self.days[i+1],
                "label_valid": True, "promotion_label_status": "synthetic_valid"})
        self.frame = pd.DataFrame(records)
        self.dataset = {"version": e.VERSION, "formed_at": e.now(), "rows": self.rows,
            "calendar_path": str(self.cal), "calendar_sha256": e.sha(self.cal),
            "source_bindings": e.bind(self.raw) | e.bind(self.cal),
            "baseline_promotion": {self.rows[-1]["candidate_id"]: {"call": True, "raw_score": .7, "truth": True}}}
        self.prepared = self.root/"prepared.joblib"
        joblib.dump({"relay": self.frame}, self.prepared)
        self.receipt = self.root/"receipt.json"
        self.captured = "2024-02-01T16:00:00+08:00"
        e.write_new(self.receipt, {"formed_at": self.captured, "sources": e.bind(self.raw),
            "prepared_path": str(self.prepared), "prepared_sha256": e.sha(self.prepared)})
        self.parent = self.root/"dataset.json"
        e.write_new(self.parent, self.dataset)

    def attach(self):
        return t.attach(self.parent, self.receipt, self.root/"attachment")

    def test_actual_attachment_preserves_identity_groups_and_BASE(self):
        data = self.attach()
        loaded = e.load_dataset(self.root/"attachment/history_with_original_training_truth.json")
        self.assertEqual(loaded, data)
        self.assertEqual(data["rows"], self.dataset["rows"])
        self.assertEqual(data["baseline_promotion"], self.dataset["baseline_promotion"])
        self.assertEqual(len(data["promotion_training_truth"]), 12)
        self.assertTrue(all(c["value"] is not None for c in data["promotion_training_truth"].values()))

    def test_unknown_original_labels_retained_not_imputed_or_dropped(self):
        frame = self.frame.copy()
        frame.loc[0, "label_valid"] = False
        frame.loc[1, "target"] = float("nan")
        frame.loc[2, "official_target_date"] = self.days[9]
        cells = t.project(frame, self.dataset, self.captured, e.now())
        self.assertEqual(len(cells), 12)
        self.assertEqual(sum(c["value"] is None for c in cells.values()), 3)

    def test_future_or_source_prematurity_labels_not_admitted(self):
        cells = t.project(self.frame, self.dataset, self.days[3]+"T14:59:00+08:00", e.now())
        self.assertIsNone(cells[self.rows[3]["candidate_id"]]["value"])
        data = self.attach()
        cutoff = self.days[3]+"T15:00:00+08:00"
        self.assertNotIn(self.rows[2]["candidate_id"], t.targets(data, cutoff, "DIAGNOSTIC"))
        self.assertEqual(len(t.targets(data, cutoff, "DIAGNOSTIC")), 2)

    def test_strict_actual_availability_excludes_historical_projection(self):
        data = self.attach()
        self.assertEqual(t.targets(data, "2024-02-02T00:00:00+08:00", "STRICT"), {})
        self.assertEqual(len(t.targets(data, "2024-02-02T00:00:00+08:00", "DIAGNOSTIC")), 12)
        with self.assertRaisesRegex(ValueError, "track"):
            t.targets(data, "2024-02-02T00:00:00+08:00", "OTHER")

    def test_full_cohort_and_original_price_mismatches_rejected(self):
        for frame, message in ((self.frame.iloc[1:], "omits"),
                (pd.concat([self.frame, self.frame.iloc[:1]]), "duplicates")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                t.project(frame, self.dataset, self.captured, e.now())
        frame = self.frame.copy(); frame.loc[0, "price"] = 10.01
        with self.assertRaisesRegex(ValueError, "source differs"):
            t.project(frame, self.dataset, self.captured, e.now())

    def test_rehashed_forged_target_cannot_replace_original_truth(self):
        data = self.attach(); binding = data["training_truth_binding"]
        bank = e.read(binding["path"]); key = self.rows[0]["candidate_id"]
        bank["rows"][key]["value"] = not bank["rows"][key]["value"]
        forged = self.root/"rehashed_forgery.json"; e.write_new(forged, bank)
        data["training_truth_binding"] = {"path": str(forged), "sha256": e.sha(forged)}
        data["source_bindings"] |= e.bind(forged)
        data["promotion_training_truth"] = bank["rows"]
        with self.assertRaisesRegex(ValueError, "original source replay"):
            t.targets(data, "2024-02-02T00:00:00+08:00", "DIAGNOSTIC")

    def test_attachment_mutation_of_role_features_or_BASE_is_rejected(self):
        data = self.attach()
        for field in ("rows", "baseline_promotion"):
            changed = copy.deepcopy(data)
            if field == "rows": changed["rows"][0]["groups"]["price"]["value"]["price"] = 10.01
            else: changed["baseline_promotion"] = {}
            with self.subTest(field=field), self.assertRaises(ValueError):
                t.load_attachment(changed)

    def test_downstream_uses_original_labels_keeps_gap_and_all_role_OOF_requirement(self):
        data = self.attach()
        scores = {row["candidate_id"]: {"scores": dict.fromkeys(e.ROLES, .5)} for row in self.rows}
        cutoff = self.days[10]+"T00:00:00+08:00"
        selected, y = r.downstream_training_rows(self.rows, scores, data, cutoff, self.days, "DIAGNOSTIC")
        self.assertEqual([row["candidate_id"] for row in selected], [row["candidate_id"] for row in self.rows[:5]])
        self.assertEqual(y, [False, True, False, True, False])
        scores[self.rows[0]["candidate_id"]]["scores"]["THEME"] = None
        selected, _ = r.downstream_training_rows(self.rows, scores, data, cutoff, self.days, "DIAGNOSTIC")
        self.assertNotIn(self.rows[0], selected)
        self.assertEqual(r.downstream_training_rows(self.rows, scores, data, cutoff, self.days, "STRICT"), ([], []))

    def test_training_labels_do_not_enter_masked_role_packet(self):
        data = self.attach()
        packet = r.masked_packet(data)
        text = json.dumps(packet)
        self.assertNotIn("promotion_training_truth", text)
        self.assertNotIn("mature_at", text)
        self.assertNotIn("original_promotion_training_truth", text)
        self.assertFalse(packet["future_outcomes_included"])


if __name__ == "__main__":
    unittest.main()
