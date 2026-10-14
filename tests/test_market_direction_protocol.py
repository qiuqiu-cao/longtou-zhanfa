"""Frozen shared-contract checks using synthetic temporary protocols only."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from market_direction_research import digest, validate_protocol


class MarketDirectionProtocolIntegrity(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.sequence_path = Path(self.temporary.name) / "synthetic_sequence.json"
        self.shared = {
            "input_channels": [
                "sh_a_return_pct", "sz_a_return_pct", "limit_count",
                "down_limit_count", "failed_rate", "max_board",
                "relay_promotion_rate", "yesterday_premium",
            ],
            "signal_dates": ["2024-01-02", "2024-06-28"],
            "periods": {
                "DEVELOPMENT_DIAGNOSTIC": ["2024-01-02", "2024-02-29"],
                "EVALUATION_DIAGNOSTIC": ["2024-03-01", "2024-06-28"],
            },
            "input_hashes": {"SYNTHETIC_OBSERVATIONS_NOT_READ.csv": "a" * 64},
            "protected_sha256": {"SYNTHETIC_PROTECTED_NOT_READ.py": "b" * 64},
            "preprocessing": "synthetic training-only median, indicators and scaling",
            "training": "synthetic closed-target window contract",
            "one_class_LR": "synthetic frozen Laplace train-prior contract",
            "current_all_missing": "unknown, retain the original date",
            "LR": {
                "C": 1.0, "solver": "lbfgs", "max_iter": 2000,
                "random_state": 20261007,
                "threshold": "raw positive class score >0.5; tie nonpositive",
            },
            "KNN": {
                "neighbors": 21,
                "weights": "uniform",
                "distance": "squared Euclidean on training-only transformed levels",
                "distance_ties": "ascending training target date",
                "cutoff": "positive neighbor fraction >0.5",
            },
            "bootstrap": {
                "draws": 2000, "seed": 20261007, "block_official_sessions": 5,
                "sequence_primary_challenges": 4,
                "tails": [0.00625, 0.99375],
                "missing_dates": "retain full official grid before circular block resampling",
            },
            "production_upgrade_allowed": False,
        }
        self.stage = {
            "stage": 1,
            "arms": ["LR120", "KNN120"],
            "baseline": "synthetic saved Ridge sign baseline",
            "id": "SYNTHETIC_PROTOCOL_CHECK_ONLY",
        }
        self.sequence_path.write_text(
            json.dumps({"shared": self.shared, "stages": [self.stage]}),
            encoding="utf-8",
        )
        self.protocol = {
            **copy.deepcopy(self.shared),
            **copy.deepcopy(self.stage),
            "sequence_path": str(self.sequence_path),
            "sequence_sha256": digest(self.sequence_path),
        }

    def test_unchanged_shared_and_stage_contract_is_accepted(self):
        validate_protocol(copy.deepcopy(self.protocol))

    def test_changed_shared_fields_are_rejected_without_reading_inputs(self):
        changes = [
            ("signal_dates", ["2024-02-01", "2024-06-28"]),
            ("periods", {"EVALUATION_DIAGNOSTIC": ["2024-01-02", "2024-06-28"]}),
            ("input_hashes", {"SYNTHETIC_OTHER_NOT_READ.csv": "c" * 64}),
            ("protected_sha256", {}),
            ("preprocessing", "fit scaling on the entire series"),
            ("training", "allow labels after the signal date"),
            ("one_class_LR", "pretend the classifier was fitted"),
            ("current_all_missing", "replace unavailable facts with zero"),
        ]
        for field, replacement in changes:
            with self.subTest(field=field):
                altered = copy.deepcopy(self.protocol)
                altered[field] = replacement
                with self.assertRaises(ValueError):
                    validate_protocol(altered)

        # Keep the previously checked neighbor count unchanged: every other
        # KNN setting must still be bound to the original sequence contract.
        knn_changes = {
            "weights": "distance",
            "distance": "unscaled Manhattan distance",
            "distance_ties": "random order",
            "cutoff": "positive neighbor fraction >0.4",
        }
        for field, replacement in knn_changes.items():
            with self.subTest(knn_field=field):
                altered = copy.deepcopy(self.protocol)
                altered["KNN"][field] = replacement
                self.assertEqual(altered["KNN"]["neighbors"], 21)
                with self.assertRaises(ValueError):
                    validate_protocol(altered)

    def test_missing_shared_field_is_rejected(self):
        altered = copy.deepcopy(self.protocol)
        del altered["periods"]
        with self.assertRaises(ValueError):
            validate_protocol(altered)


if __name__ == "__main__":
    unittest.main()
