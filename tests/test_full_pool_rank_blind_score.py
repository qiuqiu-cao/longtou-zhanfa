import copy
import unittest
import numpy as np

from full_pool_rank_blind_score import score
from full_pool_rank_blind_vault import _canonical_hash
from full_pool_rank_dataset import TARGET_DEFINITION, SELECTION_SCOPE


class Base:
    def predict_proba(self, matrix):
        p = np.clip(matrix[:, 0], .01, .99)
        return np.column_stack((1 - p, p))


class Calibrator:
    def predict_proba(self, matrix):
        p = 1 / (1 + np.exp(-matrix[:, 0]))
        return np.column_stack((1 - p, p))


def fixture():
    rows = [{"candidate_id": "a", "signal_date": "2025-01-01", "boards": 1,
             "board_group": "FIRST_BOARD", "selected_features": {"x": .2}},
            {"candidate_id": "b", "signal_date": "2025-01-01", "boards": 2,
             "board_group": "CONSECUTIVE_BOARD", "selected_features": {"x": .8}}]
    ids_hash = _canonical_hash(["a", "b"])
    labels_hash = "b" * 64
    view = {"kind": "FULL_POOL_RANK_BLIND_FEATURES", "target_fields_absent": True,
            "target_definition": TARGET_DEFINITION, "selection_scope": SELECTION_SCOPE,
            "feature_allowlist": ["x"], "candidate_ids_sha256": ids_hash,
            "candidate_count": 2, "rows": rows}
    card = {"status": "FULL_POOL_RANK_DEVELOPMENT_MODEL_ONLY", "deployment_allowed": False,
            "blind_label_access_control": "SEPARATE_FEATURE_AND_LABEL_VAULT",
            "blind_label_fields_accessed": False, "source_payload_including_blind_rows_parsed": False,
            "generated_at": "2025-02-01T10:00:00+08:00", "feature_allowlist": ["x"],
            "blind_candidate_ids_sha256": ids_hash, "blind_labels_sha256_commitment": labels_hash}
    bundle = {"feature_allowlist": ["x"], "blind_candidate_ids_sha256": ids_hash,
              "blind_labels_sha256_commitment": labels_hash,
              "model": {"base": Base(), "calibrator": Calibrator()}}
    return bundle, card, view


class BlindScoreTests(unittest.TestCase):
    def test_scores_all_candidates_without_labels(self):
        bundle, card, view = fixture()
        result = score(bundle, card, view, "2025-02-02T10:00:00+08:00")
        self.assertEqual(result["status"], "BLIND_RANK_PREDICTIONS_FROZEN")
        self.assertFalse(result["blind_labels_loaded"])
        self.assertIsNone(result["blind_metrics"])
        self.assertEqual(sum(row["is_daily_top1"] for row in result["rows"]), 1)
        self.assertEqual(next(row for row in result["rows"] if row["is_daily_top1"])["candidate_id"], "b")

    def test_target_field_leak_is_rejected(self):
        bundle, card, view = fixture()
        view["rows"][0]["opportunity_proxy_label"] = True
        with self.assertRaises(ValueError):
            score(bundle, card, view, "2025-02-02T10:00:00+08:00")

    def test_commitment_mismatch_is_rejected(self):
        bundle, card, view = fixture()
        view["candidate_ids_sha256"] = "c" * 64
        with self.assertRaises(ValueError):
            score(bundle, card, view, "2025-02-02T10:00:00+08:00")

    def test_feature_schema_mismatch_is_rejected(self):
        bundle, card, view = fixture()
        view["feature_allowlist"] = ["y"]
        with self.assertRaises(ValueError):
            score(bundle, card, view, "2025-02-02T10:00:00+08:00")

    def test_model_must_predate_score(self):
        bundle, card, view = fixture()
        with self.assertRaises(ValueError):
            score(bundle, card, view, "2025-01-01T10:00:00+08:00")

    def test_rank_score_is_not_complete_trade_probability(self):
        bundle, card, view = fixture()
        result = score(bundle, card, view, "2025-02-02T10:00:00+08:00")
        self.assertFalse(result["score_is_complete_trade_probability"])
