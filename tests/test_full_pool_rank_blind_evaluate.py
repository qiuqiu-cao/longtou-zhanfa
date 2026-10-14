import copy
import unittest

from full_pool_rank_blind_evaluate import evaluate
from full_pool_rank_blind_vault import _canonical_hash
from full_pool_rank_dataset import TARGET_DEFINITION, SELECTION_SCOPE


def fixture(days=100):
    predictions, labels = [], []
    for day in range(days):
        date = f"2025-{1 + day // 28:02d}-{1 + day % 28:02d}"
        for within in range(2):
            candidate_id = f"{day}:{within}"
            # Model chooses first board; it wins 75% while highest-board baseline wins 25%.
            first_wins = day % 4 != 0
            label = first_wins if within == 0 else not first_wins
            predictions.append({"candidate_id": candidate_id, "signal_date": date,
                                "boards": within + 1,
                                "board_group": "FIRST_BOARD" if within == 0 else "CONSECUTIVE_BOARD",
                                "opportunity_proxy_score": .8 if within == 0 else .2,
                                "is_daily_top1": within == 0})
            labels.append({"candidate_id": candidate_id, "signal_date": date,
                           "opportunity_proxy_label": label,
                           "label_available_at": "2026-01-01T15:00:00+08:00"})
    ids_hash = _canonical_hash(sorted(row["candidate_id"] for row in labels))
    labels_hash = _canonical_hash(labels)
    pred = {"status": "BLIND_RANK_PREDICTIONS_FROZEN", "predictions_frozen": True,
            "blind_labels_loaded": False, "blind_metrics": None, "selection_scope": SELECTION_SCOPE,
            "target_definition": TARGET_DEFINITION, "scored_at": "2026-02-01T10:00:00+08:00",
            "candidate_ids_sha256": ids_hash, "blind_labels_sha256_commitment": labels_hash,
            "rows": predictions}
    vault = {"kind": "FULL_POOL_RANK_BLIND_LABEL_VAULT",
             "do_not_open_before_predictions_frozen": True, "selection_scope": SELECTION_SCOPE,
             "target_definition": TARGET_DEFINITION, "sealed_at": "2026-01-15T10:00:00+08:00",
             "candidate_ids_sha256": ids_hash, "labels_sha256": labels_hash, "rows": labels}
    return pred, vault


class BlindEvaluateTests(unittest.TestCase):
    def test_passes_only_after_frozen_predictions_and_100_sessions(self):
        pred, vault = fixture()
        result = evaluate(pred, vault, "2026-02-02T10:00:00+08:00")
        self.assertEqual(result["status"], "BLIND_FULL_POOL_RANK_GATE_PASSED")
        self.assertEqual(result["blind_top1_proxy_rate"], .75)
        self.assertEqual(result["blind_baseline_proxy_rate"], .25)
        self.assertTrue(result["blind_labels_untouched_until_evaluation"])
        self.assertFalse(result["can_claim_complete_trade_win_probability"])

    def test_fewer_than_100_sessions_fails_gate(self):
        pred, vault = fixture(99)
        result = evaluate(pred, vault, "2026-02-02T10:00:00+08:00")
        self.assertEqual(result["status"], "BLIND_FULL_POOL_RANK_GATE_FAILED")

    def test_label_commitment_mismatch_is_rejected(self):
        pred, vault = fixture()
        vault["rows"][0]["opportunity_proxy_label"] = not vault["rows"][0]["opportunity_proxy_label"]
        with self.assertRaises(ValueError):
            evaluate(pred, vault, "2026-02-02T10:00:00+08:00")

    def test_missing_candidate_is_rejected(self):
        pred, vault = fixture()
        pred["rows"].pop()
        with self.assertRaises(ValueError):
            evaluate(pred, vault, "2026-02-02T10:00:00+08:00")

    def test_top_one_marker_must_match_scores(self):
        pred, vault = fixture()
        pred["rows"][0]["is_daily_top1"] = False
        pred["rows"][1]["is_daily_top1"] = True
        with self.assertRaises(ValueError):
            evaluate(pred, vault, "2026-02-02T10:00:00+08:00")

    def test_vault_cannot_be_opened_before_predictions(self):
        pred, vault = fixture()
        vault["sealed_at"] = "2026-02-02T11:00:00+08:00"
        with self.assertRaises(ValueError):
            evaluate(pred, vault, "2026-02-03T10:00:00+08:00")

    def test_both_groups_are_reported(self):
        pred, vault = fixture()
        result = evaluate(pred, vault, "2026-02-02T10:00:00+08:00")
        self.assertEqual(set(result["group_metrics"]), {"FIRST_BOARD", "CONSECUTIVE_BOARD"})
