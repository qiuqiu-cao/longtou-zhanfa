import unittest

from complete_trade_dataset import TARGET_DEFINITION, SELECTION_SCOPE
from policy_trade_blind_vault import prepare


def fixture():
    parts = {name: [] for name in ("train", "calibration", "validation", "blind")}
    rows = []
    for segment in parts:
        for index in range(2):
            decision_id = segment + str(index)
            parts[segment].append(decision_id)
            rows.append({"decision_id": decision_id, "signal_date": "2025-01-0" + str(index + 1),
                         "board_group": "FIRST_BOARD" if index == 0 else "CONSECUTIVE_BOARD",
                         "emotion_active": True, "selected_features": {"x": index},
                         "training_eligible": True, "complete_trade_profit_label": bool(index),
                         "large_loss_label": not bool(index), "complete_trade_net_return": .01 if index else -.08,
                         "closed_at": "2025-01-10T15:00:00+08:00", "issues": []})
    dataset = {"training_ready": True, "generalization_scope": "FROZEN_SELECTION_POLICY_CONDITIONAL_ONLY",
               "policy_id": "p1", "kind": "paper", "execution_independently_verified": False,
               "feature_allowlist": ["x"], "rows": rows}
    split = {"status": "READY_FOR_MODEL_DEVELOPMENT", "blind_labels_not_read_by_audit": True,
             "partitions": parts}
    return dataset, split


class PolicyTradeBlindVaultTests(unittest.TestCase):
    def test_separates_features_labels_and_development(self):
        dataset, split = fixture()
        dev, dev_split, features, labels = prepare(dataset, split, "2025-02-01T00:00:00+08:00")
        forbidden = {"complete_trade_profit_label", "large_loss_label", "complete_trade_net_return", "closed_at"}
        self.assertTrue(all(not (forbidden & set(row)) for row in features["rows"]))
        self.assertTrue(all(forbidden <= set(row) for row in labels["rows"]))
        self.assertTrue(dev["blind_rows_removed"])
        self.assertNotIn("blind", dev_split["partitions"])

    def test_blind_ids_are_not_in_development(self):
        dataset, split = fixture()
        dev, _, _, _ = prepare(dataset, split, "2025-02-01T00:00:00+08:00")
        self.assertFalse({row["decision_id"] for row in dev["rows"]} & set(split["partitions"]["blind"]))

    def test_shared_commitments_match(self):
        dataset, split = fixture()
        dev, dev_split, features, labels = prepare(dataset, split, "2025-02-01T00:00:00+08:00")
        self.assertEqual(dev["blind_labels_sha256_commitment"], labels["labels_sha256"])
        self.assertEqual(dev_split["blind_decision_ids_sha256"], features["decision_ids_sha256"])

    def test_invalid_blind_trade_is_rejected(self):
        dataset, split = fixture()
        blind = split["partitions"]["blind"][0]
        next(row for row in dataset["rows"] if row["decision_id"] == blind)["training_eligible"] = False
        with self.assertRaises(ValueError):
            prepare(dataset, split, "2025-02-01T00:00:00+08:00")

    def test_overlap_is_rejected(self):
        dataset, split = fixture()
        split["partitions"]["blind"][0] = split["partitions"]["train"][0]
        with self.assertRaises(ValueError):
            prepare(dataset, split, "2025-02-01T00:00:00+08:00")
