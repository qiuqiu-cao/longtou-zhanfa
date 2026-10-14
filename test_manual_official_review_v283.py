from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bind_manual_official_review_v283 import validate_candidate_scope, source_review
from holiday_official_evidence_v283 import sha


class Pages:
    def __len__(self): return 2
    def __getitem__(self, index): return self
    def get_text(self): return "TEST_ONLY 13.14万元"
    def close(self): pass


class ManualReviewTests(unittest.TestCase):
    def test_exact_scope_requires_every_candidate_without_duplicates(self):
        frozen = {"plan":{"candidates":[{"code":"000678"},{"code":"600825"}]}}
        policy = dict(all_candidates_certified_clean=False, rank_change_authorized=False,
                      trade_authorized=False, original_frozen_calls_change_authorized=False,
                      probability_adjustment=None)
        manual = dict(candidates=[{"code":"000678"},{"code":"600825"}], policy=policy)
        validate_candidate_scope(manual, frozen)
        for codes in (["000678"], ["000678","000678"]):
            changed = deepcopy(manual);changed["candidates"] = [{"code":c} for c in codes]
            with self.assertRaisesRegex(ValueError, "denominator"):
                validate_candidate_scope(changed, frozen)

    def test_evidence_binding_cannot_adjust_probability_or_authorize_trade(self):
        frozen = {"plan":{"candidates":[{"code":"000678"}]}}
        policy = dict(all_candidates_certified_clean=False, rank_change_authorized=False,
                      trade_authorized=False, original_frozen_calls_change_authorized=False,
                      probability_adjustment=None)
        for key,value in (("probability_adjustment",-.1),("trade_authorized",True),
                          ("all_candidates_certified_clean",True),("original_frozen_calls_change_authorized",True)):
            manual = dict(candidates=[{"code":"000678"}],policy={**policy,key:value})
            with self.assertRaises(ValueError):validate_candidate_scope(manual,frozen)

    def source(self, directory):
        path=Path(directory)/"test.pdf";path.write_bytes(b"TEST_ONLY_PDF_PLACEHOLDER")
        source=dict(pdf_path=str(path),pdf_sha256=sha(path),
            fulltext_available_at="2026-10-02T17:11:00+08:00",url="TEST_ONLY",
            announcement_date="2026-09-29")
        row=dict(code="000678",announcement_id="1",source_pages=[1],anchors=["13.14"])
        return row,{("000678","1"):source}

    def test_future_source_cannot_be_bound_as_already_reviewed(self):
        with tempfile.TemporaryDirectory() as directory:
            row,evidence=self.source(directory)
            with self.assertRaisesRegex(ValueError,"not yet available"):
                source_review(row,evidence,"2026-10-02T16:43:00+08:00")

    def test_page_and_numeric_anchor_must_match_original_source(self):
        with tempfile.TemporaryDirectory() as directory:
            row,evidence=self.source(directory)
            with patch("bind_manual_official_review_v283.fitz.open",return_value=Pages()):
                result=source_review(row,evidence,"2026-10-02T17:20:00+08:00")
                self.assertEqual(result["publication_clock_status"],"DATE_ONLY")
                for changed in ({**row,"source_pages":[3]},{**row,"anchors":["999.99"]}):
                    with self.assertRaises(ValueError):
                        source_review(changed,evidence,"2026-10-02T17:20:00+08:00")

    def test_changed_pdf_is_rejected_before_claim_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            row,evidence=self.source(directory)
            Path(evidence[("000678","1")]["pdf_path"]).write_bytes(b"CHANGED_TEST_ONLY")
            with self.assertRaisesRegex(ValueError,"PDF hash mismatch"):
                source_review(row,evidence,"2026-10-02T17:20:00+08:00")


if __name__=="__main__":unittest.main()
