import copy
import unittest

from full_pool_rank_dataset import TARGET_DEFINITION, SELECTION_SCOPE
from full_pool_rank_model import train


def fixture():
    rows, partitions = [], {name: [] for name in ("train", "calibration", "validation", "blind")}
    segment_sizes = {"train": 20, "calibration": 8, "validation": 8, "blind": 8}
    offset = 0
    for segment, day_count in segment_sizes.items():
        for day_index in range(day_count):
            day = f"2025-{1 + (offset + day_index) // 28:02d}-{1 + (offset + day_index) % 28:02d}"
            for within, boards in enumerate((1, 2, 3, 4)):
                candidate_id = f"{segment}:{day_index}:{within}"
                label = within == (day_index % 4)
                rows.append({"candidate_id": candidate_id, "signal_date": day, "boards": boards,
                             "board_group": "FIRST_BOARD" if boards == 1 else "CONSECUTIVE_BOARD",
                             "label_available": True, "opportunity_proxy_label": label, "issues": [],
                             "selected_features": {"signal": 1.0 if label else 0.0, "boards": boards}})
                partitions[segment].append(candidate_id)
        offset += day_count
    dataset = {"training_ready": True, "full_pool_scope_established": True,
               "target_definition": TARGET_DEFINITION, "selection_scope": SELECTION_SCOPE,
               "feature_allowlist": ["signal", "boards"],
               "blind_rows_removed": True, "blind_label_fields_present": False,
               "rows": [row for row in rows if row["candidate_id"] not in partitions["blind"]]}
    blind_ids = list(partitions.pop("blind"))
    split = {"status": "READY_FOR_FULL_POOL_RANK_DEVELOPMENT", "partitions": partitions,
             "blind_labels_not_summarized": True, "same_signal_date_kept_together": True,
             "blind_access_control": "SEPARATE_FEATURE_AND_LABEL_VAULT",
             "source_dataset_with_blind_labels_not_required_for_training": True,
             "blind_count": len(blind_ids), "blind_candidate_ids_sha256": "a" * 64,
             "blind_labels_sha256_commitment": "b" * 64}
    return dataset, split


class FullPoolRankModelTests(unittest.TestCase):
    def test_trains_full_pool_development_ranker_without_blind(self):
        dataset, split = fixture()
        card, bundle = train(dataset, split)
        self.assertEqual(card["status"], "FULL_POOL_RANK_DEVELOPMENT_MODEL_ONLY")
        self.assertTrue(card["full_pool_training_scope"])
        self.assertFalse(card["rank_score_named_probability"])
        self.assertFalse(card["blind_feature_matrix_built"])
        self.assertIsNone(card["blind_metrics"])
        self.assertFalse(card["source_payload_including_blind_rows_parsed"])
        self.assertEqual(bundle["blind_candidate_count"], 32)

    def test_validation_is_measured_as_daily_top_one(self):
        dataset, split = fixture()
        card, _ = train(dataset, split)
        selected = card["candidate_models"][card["chosen_model"]]
        self.assertEqual(selected["session_count"], 8)
        self.assertIn("daily_top1_proxy_hit_rate", selected)
        self.assertIn("mean_reciprocal_rank", selected)

    def test_height_baseline_is_prefixed_and_deterministic(self):
        dataset, split = fixture()
        card, _ = train(dataset, split)
        baseline = card["validation_height_baseline"]
        self.assertTrue(baseline["frozen_rule"])
        self.assertEqual(baseline["name"], "highest_board_then_candidate_id")
        self.assertEqual(baseline["session_count"], 8)

    def test_missing_development_feature_is_rejected(self):
        dataset, split = fixture()
        dataset["rows"][0]["selected_features"].pop("signal")
        with self.assertRaises(ValueError):
            train(dataset, split)

    def test_blind_identity_overlap_is_rejected(self):
        dataset, split = fixture()
        split["partitions"]["blind"] = [split["partitions"]["train"][0]]
        with self.assertRaises(ValueError):
            train(dataset, split)

    def test_proxy_cannot_be_rebranded_as_profit(self):
        dataset, split = fixture()
        dataset["target_definition"] = "complete_trade_profit"
        with self.assertRaises(ValueError):
            train(dataset, split)

    def test_incomplete_dataset_is_rejected(self):
        dataset, split = fixture()
        dataset["training_ready"] = False
        with self.assertRaises(ValueError):
            train(dataset, split)

    def test_same_day_partition_invariant_is_required(self):
        dataset, split = fixture()
        split["same_signal_date_kept_together"] = False
        with self.assertRaises(ValueError):
            train(dataset, split)

    def test_duplicate_development_identity_is_rejected(self):
        dataset, split = fixture()
        split["partitions"]["train"].append(split["partitions"]["train"][0])
        with self.assertRaises(ValueError):
            train(dataset, split)
