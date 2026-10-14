import unittest

from full_pool_rank_blind_vault import prepare
from full_pool_rank_dataset import TARGET_DEFINITION, SELECTION_SCOPE


def fixture():
    rows = []
    parts = {"train": [], "calibration": [], "validation": [], "blind": []}
    for segment in parts:
        for index in range(2):
            candidate_id = segment + str(index)
            parts[segment].append(candidate_id)
            rows.append({"candidate_id": candidate_id, "signal_date": "2025-01-0" + str(index + 1),
                         "boards": index + 1, "board_group": "FIRST_BOARD" if index == 0 else "CONSECUTIVE_BOARD",
                         "selected_features": {"boards": index + 1}, "label_available": True,
                         "opportunity_proxy_label": bool(index), "b2_close_mark_net_return_proxy": .01,
                         "label_available_at": "2025-01-10T15:00:00+08:00", "issues": []})
    dataset = {"training_ready": True, "full_pool_scope_established": True,
               "target_definition": TARGET_DEFINITION, "selection_scope": SELECTION_SCOPE,
               "feature_allowlist": ["boards"], "rows": rows}
    split = {"status": "READY_FOR_FULL_POOL_RANK_DEVELOPMENT", "blind_labels_not_summarized": True,
             "same_signal_date_kept_together": True, "partitions": parts}
    return dataset, split


class FullPoolRankBlindVaultTests(unittest.TestCase):
    def test_blind_features_contain_no_target_fields(self):
        dataset, split = fixture()
        development, dev_split, features, labels = prepare(dataset, split, "2025-02-01T00:00:00+08:00")
        forbidden = {"opportunity_proxy_label", "b2_close_mark_net_return_proxy", "label_available_at", "label_available"}
        self.assertTrue(all(not (forbidden & set(row)) for row in features["rows"]))
        self.assertTrue(features["target_fields_absent"])
        self.assertTrue(all("opportunity_proxy_label" in row for row in labels["rows"]))
        self.assertTrue(development["blind_rows_removed"])
        self.assertNotIn("blind", dev_split["partitions"])

    def test_development_contains_no_blind_ids(self):
        dataset, split = fixture()
        development, _, _, _ = prepare(dataset, split, "2025-02-01T00:00:00+08:00")
        ids = {row["candidate_id"] for row in development["rows"]}
        self.assertFalse(ids & set(split["partitions"]["blind"]))

    def test_commitment_is_shared_without_exposing_labels(self):
        dataset, split = fixture()
        development, dev_split, features, labels = prepare(dataset, split, "2025-02-01T00:00:00+08:00")
        self.assertEqual(development["blind_labels_sha256_commitment"], labels["labels_sha256"])
        self.assertEqual(dev_split["blind_candidate_ids_sha256"], features["candidate_ids_sha256"])

    def test_unresolved_blind_row_is_rejected(self):
        dataset, split = fixture()
        blind_id = split["partitions"]["blind"][0]
        next(row for row in dataset["rows"] if row["candidate_id"] == blind_id)["label_available"] = False
        with self.assertRaises(ValueError):
            prepare(dataset, split, "2025-02-01T00:00:00+08:00")

    def test_overlap_is_rejected(self):
        dataset, split = fixture()
        split["partitions"]["blind"][0] = split["partitions"]["train"][0]
        with self.assertRaises(ValueError):
            prepare(dataset, split, "2025-02-01T00:00:00+08:00")
