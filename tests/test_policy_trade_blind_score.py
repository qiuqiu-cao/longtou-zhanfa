import copy
import unittest
import numpy as np

from complete_trade_dataset import TARGET_DEFINITION, SELECTION_SCOPE
from full_pool_rank_blind_vault import _canonical_hash
from policy_trade_blind_score import score


class ProfitBase:
    def predict_proba(self, matrix):
        p = np.clip(matrix[:, 0], .01, .99)
        return np.column_stack((1 - p, p))


class RiskBase:
    def predict_proba(self, matrix):
        p = np.clip(1 - matrix[:, 0], .01, .99)
        return np.column_stack((1 - p, p))


class IdentityCalibrator:
    def predict_proba(self, matrix):
        p = 1 / (1 + np.exp(-matrix[:, 0]))
        return np.column_stack((1 - p, p))


def fixture():
    rows = [{"decision_id": "d1", "signal_date": "2025-01-01", "board_group": "FIRST_BOARD",
             "emotion_active": True, "selected_features": {"x": .8}},
            {"decision_id": "d2", "signal_date": "2025-01-02", "board_group": "CONSECUTIVE_BOARD",
             "emotion_active": True, "selected_features": {"x": 1.2}}]
    ids_hash = _canonical_hash(["d1", "d2"])
    labels_hash = "b" * 64
    support = {"x": {"minimum": 0.0, "maximum": 1.0}}
    card = {"status": "DEVELOPMENT_MODEL_ONLY", "deployment_allowed": False,
            "blind_label_access_control": "SEPARATE_FEATURE_AND_LABEL_VAULT",
            "blind_label_fields_accessed": False, "source_payload_including_blind_rows_parsed": False,
            "out_of_distribution_gate_frozen": True, "generated_at": "2025-02-01T10:00:00+08:00",
            "policy_id": "p1", "feature_allowlist": ["x"], "feature_support_envelope": support,
            "blind_decision_ids_sha256": ids_hash, "blind_labels_sha256_commitment": labels_hash}
    bundle = {"feature_allowlist": ["x"], "feature_support_envelope": support,
              "blind_decision_ids_sha256": ids_hash, "blind_labels_sha256_commitment": labels_hash,
              "profit": {"base": ProfitBase(), "calibrator": IdentityCalibrator()},
              "risk": {"base": RiskBase(), "calibrator": IdentityCalibrator()}}
    threshold = {"status": "VALIDATION_THRESHOLD_FROZEN", "deployment_allowed": False,
                 "blind_rows_accessed": False, "selection_policy_id": "p1",
                 "chosen_thresholds": {"profit_threshold": .5, "large_loss_threshold": .5},
                 "validated_at": "2025-02-02T10:00:00+08:00"}
    view = {"kind": "POLICY_TRADE_BLIND_FEATURES", "target_fields_absent": True,
            "target_definition": TARGET_DEFINITION, "selection_scope": SELECTION_SCOPE,
            "generalization_scope": "FROZEN_SELECTION_POLICY_CONDITIONAL_ONLY", "policy_id": "p1",
            "account_kind": "paper", "feature_allowlist": ["x"], "decision_count": 2,
            "decision_ids_sha256": ids_hash, "rows": rows}
    return bundle, card, threshold, view


class PolicyTradeBlindScoreTests(unittest.TestCase):
    def test_scores_only_inside_support_and_freezes_predictions(self):
        result = score(*fixture(), "2025-02-03T10:00:00+08:00")
        self.assertEqual(result["status"], "POLICY_BLIND_PREDICTIONS_FROZEN")
        self.assertTrue(result["rows"][0]["policy_selected"])
        self.assertIsNone(result["rows"][1]["complete_trade_profit_probability"])
        self.assertEqual(result["out_of_support_count"], 1)
        self.assertFalse(result["blind_labels_loaded"])

    def test_target_field_leak_is_rejected(self):
        bundle, card, threshold, view = fixture()
        view["rows"][0]["complete_trade_profit_label"] = True
        with self.assertRaises(ValueError):
            score(bundle, card, threshold, view, "2025-02-03T10:00:00+08:00")

    def test_unfrozen_threshold_is_rejected(self):
        bundle, card, threshold, view = fixture()
        threshold["status"] = "NO_VALIDATED_THRESHOLD"
        with self.assertRaises(ValueError):
            score(bundle, card, threshold, view, "2025-02-03T10:00:00+08:00")

    def test_policy_id_must_match(self):
        bundle, card, threshold, view = fixture()
        view["policy_id"] = "other"
        with self.assertRaises(ValueError):
            score(bundle, card, threshold, view, "2025-02-03T10:00:00+08:00")

    def test_support_definition_must_match_bundle(self):
        bundle, card, threshold, view = fixture()
        bundle["feature_support_envelope"] = copy.deepcopy(bundle["feature_support_envelope"])
        bundle["feature_support_envelope"]["x"]["maximum"] = 2
        with self.assertRaises(ValueError):
            score(bundle, card, threshold, view, "2025-02-03T10:00:00+08:00")

    def test_threshold_must_predate_score(self):
        bundle, card, threshold, view = fixture()
        with self.assertRaises(ValueError):
            score(bundle, card, threshold, view, "2025-02-01T12:00:00+08:00")
