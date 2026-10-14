"""Integration boundaries: original scope and byte-bound peer lists survive."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import research_indicator_pipeline as pipeline


class IntakeBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.path = self.directory / "candidate_theme_intake.csv"
        self.meta_path = self.directory / "manifest.json"
        self.rows = [dict(signal_date="2026-09-30", code="000011", name="甲", boards="3",
                          provider_same_height_codes='["002242"]'),
                     dict(signal_date="2026-09-30", code="002242", name="乙", boards="3",
                          provider_same_height_codes='["000011"]'),
                     dict(signal_date="2026-09-30", code="600825", name="丙", boards="7",
                          provider_same_height_codes="[]")]
        self.write()

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        with self.path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        self.meta = dict(recorded_at="2026-10-02T19:21:00+00:00", signal_date="2026-09-30",
                         target_date="2026-10-08", provider_pool_count=len(self.rows),
                         artifact_hashes={self.path.name: pipeline.sha(self.path)})
        pipeline.dump(self.meta_path, self.meta)

    def test_complete_original_group_is_retained(self):
        _, _, _, rows = pipeline.load_intake(self.directory)
        self.assertEqual([r["code"] for r in rows], ["000011", "002242", "600825"])

    def test_bytes_changed_without_receipt_rejected(self):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "hash"):
            pipeline.load_intake(self.directory)

    def test_missing_peer_rejected_even_with_updated_file_hash(self):
        self.rows[0]["provider_same_height_codes"] = "[]"
        self.write()
        with self.assertRaisesRegex(ValueError, "denominator"):
            pipeline.load_intake(self.directory)

    def test_wrong_signal_date_rejected(self):
        self.rows[1]["signal_date"] = "2026-09-29"
        self.write()
        with self.assertRaisesRegex(ValueError, "signal date"):
            pipeline.load_intake(self.directory)

    def test_registry_subset_cannot_silently_drop_original_candidate(self):
        registry = self.directory / "registry.json"
        plan = dict(signal_date=self.meta["signal_date"], target_date=self.meta["target_date"],
                    candidates=[dict(code=r["code"], boards=int(r["boards"])) for r in self.rows[:-1]])
        pipeline.dump(registry, {"plan": plan})
        with self.assertRaisesRegex(ValueError, "all original relay"):
            pipeline.registered_scope(registry, self.meta, self.rows)

    def test_registry_scope_validation_does_not_modify_original(self):
        registry = self.directory / "registry.json"
        plan = dict(signal_date=self.meta["signal_date"], target_date=self.meta["target_date"],
                    candidates=[dict(code=r["code"], boards=int(r["boards"])) for r in self.rows])
        pipeline.dump(registry, {"plan": plan})
        before = registry.read_bytes()
        result = pipeline.registered_scope(registry, self.meta, self.rows)
        self.assertEqual(result["candidate_count"], 3)
        self.assertEqual(registry.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
