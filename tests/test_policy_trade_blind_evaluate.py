import unittest

from complete_trade_dataset import TARGET_DEFINITION, SELECTION_SCOPE
from full_pool_rank_blind_vault import _canonical_hash
from policy_trade_blind_evaluate import evaluate


def fixture(days=100):
    predictions, labels = [], []
    for day in range(days):
        signal_date = f"2025-{1 + day // 28:02d}-{1 + day % 28:02d}"
        for within in range(2):
            decision_id = f"{day}:{within}"
            selected = within == 0
            if selected:
                ret = .03 if day % 4 != 0 else (-.08 if day % 20 == 0 else -.01)
            else:
                ret = -.08 if day % 10 == 0 else .01 if day % 4 == 0 else -.04
            group = "FIRST_BOARD" if day % 2 == 0 else "CONSECUTIVE_BOARD"
            predictions.append({"decision_id": decision_id, "signal_date": signal_date,
                                "board_group": group, "emotion_active": True,
                                "inside_frozen_feature_support": True,
                                "complete_trade_profit_probability": .8 if selected else .4,
                                "large_loss_probability": .05 if selected else .2,
                                "policy_selected": selected})
            labels.append({"decision_id": decision_id, "signal_date": signal_date,
                           "board_group": group, "complete_trade_profit_label": ret > 0,
                           "large_loss_label": ret <= -.07, "complete_trade_net_return": ret,
                           "closed_at": f"2026-01-{1 + day % 28:02d}T15:00:00+08:00"})
    ids_hash = _canonical_hash(sorted(row["decision_id"] for row in labels))
    labels_hash = _canonical_hash(labels)
    pred = {"status": "POLICY_BLIND_PREDICTIONS_FROZEN", "predictions_frozen": True,
            "blind_labels_loaded": False, "blind_metrics": None, "target_definition": TARGET_DEFINITION,
            "selection_scope": SELECTION_SCOPE, "policy_id": "p1", "account_kind": "paper",
            "scored_at": "2026-02-01T10:00:00+08:00", "decision_ids_sha256": ids_hash,
            "blind_labels_sha256_commitment": labels_hash, "rows": predictions}
    vault = {"kind": "POLICY_TRADE_BLIND_LABEL_VAULT", "do_not_open_before_predictions_frozen": True,
             "target_definition": TARGET_DEFINITION, "selection_scope": SELECTION_SCOPE,
             "policy_id": "p1", "account_kind": "paper", "sealed_at": "2026-01-15T10:00:00+08:00",
             "execution_independently_verified": True, "decision_ids_sha256": ids_hash,
             "labels_sha256": labels_hash, "rows": labels}
    return pred, vault


class PolicyTradeBlindEvaluateTests(unittest.TestCase):
    def test_valid_100_trade_blind_gate_passes(self):
        pred, vault = fixture()
        result = evaluate(pred, vault, "2026-02-02T10:00:00+08:00")
        self.assertEqual(result["status"], "BLIND_COMPLETE_TRADE_GATE_PASSED")
        self.assertEqual(result["metrics"]["completed_trades"], 100)
        self.assertEqual(result["metrics"]["win_rate"], .75)
        self.assertEqual(result["metrics"]["big_loss_rate"], .05)
        self.assertTrue(result["outperforms_baseline"])
        self.assertTrue(result["execution_basis_verified"])

    def test_fewer_than_100_selected_fails(self):
        pred, vault = fixture(99)
        result = evaluate(pred, vault, "2026-02-02T10:00:00+08:00")
        self.assertEqual(result["status"], "BLIND_COMPLETE_TRADE_GATE_FAILED")

    def test_label_return_consistency_is_enforced(self):
        pred, vault = fixture()
        vault["rows"][0]["complete_trade_profit_label"] = True
        vault["labels_sha256"] = _canonical_hash(vault["rows"])
        pred["blind_labels_sha256_commitment"] = vault["labels_sha256"]
        with self.assertRaises(ValueError):
            evaluate(pred, vault, "2026-02-02T10:00:00+08:00")

    def test_commitment_mismatch_is_rejected(self):
        pred, vault = fixture()
        vault["labels_sha256"] = "c" * 64
        with self.assertRaises(ValueError):
            evaluate(pred, vault, "2026-02-02T10:00:00+08:00")

    def test_multiple_selected_same_day_is_rejected(self):
        pred, vault = fixture()
        pred["rows"][1]["policy_selected"] = True
        with self.assertRaises(ValueError):
            evaluate(pred, vault, "2026-02-02T10:00:00+08:00")

    def test_groups_reported_separately(self):
        pred, vault = fixture()
        result = evaluate(pred, vault, "2026-02-02T10:00:00+08:00")
        self.assertGreater(result["group_metrics"]["FIRST_BOARD"]["completed_trades"], 0)
        self.assertGreater(result["group_metrics"]["CONSECUTIVE_BOARD"]["completed_trades"], 0)

    def test_account_drawdown_not_substituted_by_trade_sequence(self):
        pred, vault = fixture()
        result = evaluate(pred, vault, "2026-02-02T10:00:00+08:00")
        self.assertIsNone(result["account_max_drawdown"])
        self.assertIn("REQUIRES", result["account_drawdown_status"])
