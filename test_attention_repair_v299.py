"""A missing response is supplemented later without deleting the failure."""
import unittest
from attention_repair_v299 import merged
from attention_observations_v299 import ENDPOINTS, query


class AttentionSupplementTests(unittest.TestCase):
    def fixture(self):
        parent = {"scope": {"codes": ["600825"]}, "recorded_at": "2026-10-02T23:40:00+08:00",
            "observations": [{"code": "600825", "errors": {"related": "ISOLATED TEST TIMEOUT"}}]}
        entries = [{"code": "600825", "kind": k, "method": "POST", "url": query("600825", k)[0],
            "query": query("600825", k)[1], "requested_at": parent["recorded_at"], "available_at": parent["recorded_at"],
            "status": "UNAVAILABLE", "error": "ISOLATED TEST GAP"} for k in ENDPOINTS]
        replacement = dict(entries[-1]); replacement["requested_at"] = replacement["available_at"] = "2026-10-02T23:41:00+08:00"
        return parent, entries, replacement

    def test_valid_parent_fields_cannot_be_overwritten(self):
        p,e,r = self.fixture(); r["kind"] = "latest"
        with self.assertRaisesRegex(ValueError, "explicit parent gaps only"):
            merged(p,e,[r],"2026-10-02T23:42:00+08:00")

    def test_supplement_cannot_be_backdated_or_duplicated(self):
        p,e,r = self.fixture(); r["requested_at"] = "2026-10-02T23:39:00+08:00"
        with self.assertRaisesRegex(ValueError, "precede parent completion"):
            merged(p,e,[r],"2026-10-02T23:42:00+08:00")
        r["requested_at"] = "2026-10-02T23:41:00+08:00"
        with self.assertRaisesRegex(ValueError, "without duplicates"):
            merged(p,e,[r,r],"2026-10-02T23:42:00+08:00")

    def test_failed_supplement_remains_a_gap_and_keeps_original_failure_metadata(self):
        p,e,r = self.fixture()
        result = merged(p,e,[r],"2026-10-02T23:42:00+08:00")
        self.assertIsNone(result["observations"][0]["related"])
        self.assertEqual(result["scope"],p["scope"])
        self.assertTrue(result["supplement"]["parent_failures_retained"])
        self.assertFalse(result["supplement"]["original_valid_fields_overwritten"])


if __name__ == "__main__":
    unittest.main()
