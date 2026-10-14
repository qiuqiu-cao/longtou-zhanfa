import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from verify_focus_notice_reviews import verify


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FocusNoticeReviewVerificationTests(unittest.TestCase):
    def make_case(self, root):
        pdf, text = root / "a.pdf", root / "a.txt"
        pdf.write_bytes(b"pdf")
        text.write_text("text", encoding="utf-8")
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"notices": [{
            "notice_id": "N1", "pdf_path": str(pdf), "text_path": str(text),
            "pdf_sha256": digest(pdf), "text_sha256": digest(text),
            "received_at": "2026-09-07T10:00:00+08:00",
        }]}), encoding="utf-8")
        ledger = root / "reviews.csv"
        fields = ["notice_id", "pdf_sha256", "text_sha256", "reviewed_at",
                  "visual_pages_reviewed", "conclusion", "entry_authorized"]
        with ledger.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow({"notice_id": "N1", "pdf_sha256": digest(pdf),
                             "text_sha256": digest(text),
                             "reviewed_at": "2026-09-07T11:00:00+08:00",
                             "visual_pages_reviewed": "1",
                             "conclusion": "ACTIVE_REGULATORY_WARNING",
                             "entry_authorized": "false"})
        return manifest, ledger, text

    def test_valid_hash_bound_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, ledger, _ = self.make_case(Path(tmp))
            self.assertTrue(verify(manifest, ledger)["valid"])

    def test_tampered_source_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, ledger, text = self.make_case(Path(tmp))
            text.write_text("changed", encoding="utf-8")
            report = verify(manifest, ledger)
            self.assertFalse(report["valid"])
            self.assertTrue(any(x.startswith("TEXT_HASH_MISMATCH") for x in report["errors"]))


if __name__ == "__main__":
    unittest.main()
