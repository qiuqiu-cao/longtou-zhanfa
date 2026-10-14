"""Focused inference isolation and new complete-pool admission policy checks."""
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

import model_champion_runtime as runtime


def native_result(module="relay"):
    def stats(accuracy=.70, recall=.65, f1=.68):
        base = {"accuracy": accuracy, "complete_positive_recall": recall, "F1": f1,
                "scoring_coverage": 1., "registered_rows": 200,
                "known_outcome_rows": 200, "scored_rows": 200,
                "macro_f1": f1,
                "by_class": {k: {"complete_recall": recall} for k in ("IMPROVE", "CONTINUE", "WORSEN")}}
        return {"all_candidate": base} if module == "relay" else base
    return {"module": module, "official_window_size": 120, "selected_arm": "C1", "upgrade_eligible": True,
            "summary": {"registered_dates": 120, "evidence_class": "REAL_FORWARD_SAVED_RECORDS",
                        "arms": {"BASE": stats(), "C1": stats(.74), "C2": stats(.78)}},
            "gates": {a: {"passed": True, "checks": {"mature_fixed_120_node": True,
                        "family4_interval_lower_above0": True}} for a in ("C1", "C2")}}


class SelectionTests(unittest.TestCase):
    counts = {"BASE": 29, "C1": 30, "C2": 31}

    def test_accuracy_winner_replaces_original_c1_priority_only_in_new_result(self):
        native = native_result()
        old = deepcopy(native)
        result = runtime.choose_champion(native, self.counts)
        self.assertEqual(result["selected_arm"], "C2")
        self.assertEqual(native, old)
        self.assertFalse(result["automatic_switch"])

    def test_tie_keeps_incumbent_before_recall_or_complexity(self):
        native = native_result()
        native["summary"]["arms"]["C2"]["all_candidate"]["accuracy"] = .74
        native["summary"]["arms"]["C2"]["all_candidate"]["complete_positive_recall"] = .80
        self.assertEqual(runtime.choose_champion(native, self.counts, "C1")["selected_arm"], "C1")

    def test_tie_without_incumbent_uses_complete_recall_then_f1(self):
        native = native_result()
        a = native["summary"]["arms"]["C2"]["all_candidate"]
        a.update(accuracy=.74, complete_positive_recall=.66)
        self.assertEqual(runtime.choose_champion(native, self.counts)["selected_arm"], "C2")
        a.update(complete_positive_recall=.65, F1=.69)
        self.assertEqual(runtime.choose_champion(native, self.counts)["selected_arm"], "C2")

    def test_tie_uses_fewer_fields_then_stable_id(self):
        native = native_result()
        native["summary"]["arms"]["C2"] = deepcopy(native["summary"]["arms"]["C1"])
        self.assertEqual(runtime.choose_champion(native, self.counts)["selected_arm"], "C1")
        self.assertEqual(runtime.choose_champion(native, {"BASE": 29, "C1": 30, "C2": 30})["selected_arm"], "C1")

    def test_less_than_three_points_keeps_base(self):
        native = native_result()
        for arm in ("C1", "C2"):
            native["summary"]["arms"][arm]["all_candidate"]["accuracy"] = .729
        self.assertEqual(runtime.choose_champion(native, self.counts)["selected_arm"], "BASE")

    def test_accuracy_does_not_override_recall_f1_coverage_or_dropped_rows(self):
        for key, value in (("complete_positive_recall", .64), ("F1", .67),
                           ("scoring_coverage", .99), ("registered_rows", 199),
                           ("scored_rows", 199), ("known_outcome_rows", 199)):
            with self.subTest(key=key):
                native = native_result()
                native["summary"]["arms"]["C2"]["all_candidate"][key] = value
                self.assertEqual(runtime.choose_champion(native, self.counts)["selected_arm"], "C1")

    def test_hand_passed_flag_cannot_replace_all_original_gates(self):
        native = native_result()
        for arm in ("C1", "C2"):
            native["gates"][arm]["checks"]["family4_interval_lower_above0"] = False
        self.assertEqual(runtime.choose_champion(native, self.counts)["selected_arm"], "BASE")

    def test_history_or_immature_node_cannot_admit_challenge(self):
        for field, value in (("registered_dates", 119), ("evidence_class", "HISTORICAL_OR_MIXED_DIAGNOSTIC")):
            native = native_result()
            native["summary"][field] = value
            self.assertEqual(runtime.choose_champion(native, self.counts)["selected_arm"], "BASE")

    def test_every_market_class_recall_must_not_decline(self):
        native = native_result("market")
        native["summary"]["arms"]["C2"]["by_class"]["WORSEN"]["complete_recall"] = .64
        self.assertEqual(runtime.choose_champion(native, self.counts)["selected_arm"], "C1")

    def test_unknown_arm_rejected(self):
        with self.assertRaises(ValueError):
            runtime.choose_champion(native_result(), self.counts, "RANDOM")


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.protocol = self.root / "fixed" / "protocol_BEFORE_EFFECTS.json"
        self.protocol.parent.mkdir()
        self.protocol.write_text("{}", encoding="utf-8")
        self.shadow_root = self.protocol.parent / "shadow"
        self.shadow_root.mkdir()
        self.now = datetime(2026, 10, 8, 12, tzinfo=timezone.utc)
        self.pack = {"signal_date": "2026-10-08", "available_at": self.now.isoformat(),
                     "feature_archive": str(self.root / "features"), "manifest_sha256": "f"*64,
                     "sources": {str(self.protocol): runtime.shadow.file_sha256(self.protocol)},
                     "market": {"status": "partial", "features": {"x": 1.}},
                     "relay": [{"code": "000001", "name": "A", "boards": 2, "actual_one_word_daily": 0,
                                "features": {"x": 1.}},
                               {"code": "000002", "name": "B", "boards": 3, "actual_one_word_daily": 1,
                                "features": {"x": 2.}}]}
        self.models = {}
        for arm in runtime.ARMS:
            d = self.protocol.parent / "models" / "relay" / arm
            d.mkdir(parents=True)
            weight, manifest = d / "model.joblib", d / "manifest.json"
            weight.write_bytes(arm.encode())
            manifest.write_text("{}", encoding="utf-8")
            self.models[arm] = {"path": str(manifest), "sha256": runtime.shadow.file_sha256(manifest),
                                "model_path": str(weight), "model_sha256": runtime.shadow.file_sha256(weight),
                                "manifest": {"fields": ["x"], "classes": ["0", "1"]}}

    def mocks(self, pointer=None, actual_pointer=False):
        stack = ExitStack()
        stack.enter_context(patch.object(runtime.shadow, "_protocol", return_value=(
            self.protocol, {"modules": {"relay": {}}}, {"sessions": ["2026-10-08", "2026-10-09"]}, self.shadow_root)))
        stack.enter_context(patch.object(runtime, "load_feature_archive", return_value=self.pack))
        stack.enter_context(patch.object(runtime.shadow, "now_utc", return_value=self.now))
        stack.enter_context(patch.object(runtime.shadow, "require_current_closed_signal"))
        stack.enter_context(patch.object(runtime.shadow, "validate_cutoff"))
        stack.enter_context(patch.object(runtime.shadow, "_models", return_value=self.models))
        if not actual_pointer:
            stack.enter_context(patch.object(runtime.shadow, "_pointer", return_value=pointer or {"modules": {}}))
        return stack

    def test_only_explicit_base_is_deserialized_and_directory_intruder_is_ignored(self):
        intruder = self.protocol.parent / "models" / "relay" / "INTRUDER"
        intruder.mkdir()
        (intruder / "model.joblib").write_bytes(b"arbitrary")
        estimator = Mock(classes_=[0, 1])
        estimator.predict_proba.return_value = np.array([[.2, .8], [.5, .5]])
        with self.mocks(), patch.object(runtime.shadow.joblib, "load", return_value=estimator) as load:
            result = runtime.predict(self.protocol, "relay", self.root / "features")
        load.assert_called_once_with(self.models["BASE"]["model_path"])
        self.assertEqual(result["selected_arm"], "BASE")
        self.assertEqual(result["forecast"]["ranking"], ["000001", "000002"])
        self.assertFalse(result["forecast"]["candidates"][1]["predicted_positive"])
        self.assertEqual(result["estimators_deserialized"], 1)
        self.assertFalse(result["accuracy_improvement_proved"])

    def test_unknown_active_arm_fails_before_model_load(self):
        with self.mocks({"modules": {"relay": {"active_arm": "GARBAGE"}}}), patch.object(runtime.shadow.joblib, "load") as load:
            with self.assertRaisesRegex(ValueError, "unknown active arm"):
                runtime.predict(self.protocol, "relay", self.root / "features")
        load.assert_not_called()

    def test_hash_replacement_fails_before_model_load(self):
        Path(self.models["BASE"]["model_path"]).write_bytes(b"tampered")
        with self.mocks(), patch.object(runtime.shadow.joblib, "load") as load:
            with self.assertRaises(ValueError):
                runtime.predict(self.protocol, "relay", self.root / "features")
        load.assert_not_called()

    def test_retired_or_arbitrary_path_is_refused(self):
        retired = self.root / "retired" / "model.joblib"
        retired.parent.mkdir()
        retired.write_bytes(b"BASE")
        self.models["BASE"]["model_path"] = str(retired)
        with self.mocks(), patch.object(runtime.shadow.joblib, "load") as load:
            with self.assertRaisesRegex(ValueError, "retired/arbitrary"):
                runtime.predict(self.protocol, "relay", self.root / "features")
        load.assert_not_called()

    def test_active_challenge_requires_original_registration(self):
        with self.mocks({"modules": {"relay": {"active_arm": "C1"}}}), patch.object(runtime.shadow.joblib, "load") as load:
            with self.assertRaisesRegex(ValueError, "source-bound current registration"):
                runtime.predict(self.protocol, "relay", self.root / "features")
        load.assert_not_called()

    def test_invalid_active_audit_fails_without_writing_rollback(self):
        active = self.shadow_root / "active_pointer.json"
        active.write_text(json.dumps({"modules": {"relay": {"active_arm": "C1"}}}), encoding="utf-8")
        before = active.read_bytes()
        with self.mocks(actual_pointer=True), patch.object(runtime.shadow.joblib, "load") as load:
            with self.assertRaisesRegex(ValueError, "append-only audit"):
                runtime.predict(self.protocol, "relay", self.root / "features")
        self.assertEqual(active.read_bytes(), before)
        self.assertFalse((self.shadow_root / "control_audit.jsonl").exists())
        load.assert_not_called()

    def test_active_proof_failure_is_not_swallowed_or_replaced_by_base(self):
        with self.mocks({"modules": {"relay": {"active_arm": "C1"}}}), \
             patch.object(runtime, "_prove_active", side_effect=ValueError("native 120 gate failed")), \
             patch.object(runtime.shadow.joblib, "load") as load:
            with self.assertRaisesRegex(ValueError, "native 120 gate"):
                runtime.predict(self.protocol, "relay", self.root / "features", "registration.json")
        load.assert_not_called()

    def test_all_missing_features_remain_unknown_without_deserialization(self):
        for candidate in self.pack["relay"]:
            candidate["features"]["x"] = None
        with self.mocks(), patch.object(runtime.shadow.joblib, "load") as load:
            result = runtime.predict(self.protocol, "relay", self.root / "features")
        load.assert_not_called()
        self.assertEqual(len(result["forecast"]["candidates"]), 2)
        self.assertEqual(result["forecast"]["status"], "UNKNOWN")

    def test_existing_output_and_fixed_protocol_output_are_refused_before_prediction(self):
        existing = self.root / "existing.json"
        existing.write_text("original", encoding="utf-8")
        for output in (existing, self.protocol.parent / "new_prediction.json"):
            with patch.object(runtime, "predict") as predict:
                with self.assertRaises(ValueError):
                    runtime.main(["predict", "--protocol", str(self.protocol), "--module", "relay",
                                  "--features", str(self.root / "features"), "--output", str(output)])
            predict.assert_not_called()
        self.assertEqual(existing.read_text(encoding="utf-8"), "original")

    def test_serialization_failure_creates_no_output(self):
        output = self.root / "new.json"
        with patch.object(runtime, "predict", return_value={"unexpected": float("nan")}):
            with self.assertRaises(ValueError):
                runtime.main(["predict", "--protocol", str(self.protocol), "--module", "relay",
                              "--features", str(self.root / "features"), "--output", str(output)])
        self.assertFalse(output.exists())

    def test_partial_write_failure_removes_only_new_file(self):
        output = self.root / "new.json"
        original_open = Path.open
        class FailingWriter:
            def __init__(self, handle):
                self.handle = handle
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.handle.close()
            def write(self, data):
                self.handle.write(data[:2])
                raise OSError("disk write failure")
        def fail_open(path, mode="r", *args, **kwargs):
            handle = original_open(path, mode, *args, **kwargs)
            return FailingWriter(handle) if path == output and mode == "xb" else handle
        with patch.object(runtime, "predict", return_value={"valid": "value"}), patch.object(Path, "open", fail_open):
            with self.assertRaisesRegex(OSError, "disk write failure"):
                runtime.main(["predict", "--protocol", str(self.protocol), "--module", "relay",
                              "--features", str(self.root / "features"), "--output", str(output)])
        self.assertFalse(output.exists())
        self.assertTrue(self.protocol.exists())


if __name__ == "__main__":
    unittest.main()
