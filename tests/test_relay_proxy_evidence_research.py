import unittest
import numpy as np
import pandas as pd
from relay_proxy_evidence_research import ambiguous_mask, apply_mask


class EvidenceTests(unittest.TestCase):
    def row(self, **changes):
        return dict(date="2025-01-02", code="000001", first_seal_minute=np.nan,
                    last_seal_minute=np.nan, bomb_count=np.nan, turnover=.02,
                    signal_one_word=0., target=1., **changes)

    def test_all_known_support_retained(self):
        r = self.row(); r.update(first_seal_minute=0., last_seal_minute=0., bomb_count=0., signal_one_word=1.)
        frame = pd.DataFrame([r]); changed, mask = apply_mask(frame)
        self.assertFalse(mask.iloc[0]); pd.testing.assert_frame_equal(changed, frame)

    def test_missing_support_masks_both_original_values(self):
        r = self.row(); other = dict(r, signal_one_word=1.)
        changed, mask = apply_mask(pd.DataFrame([r, other]))
        self.assertTrue(mask.all()); self.assertTrue(changed.signal_one_word.isna().all())

    def test_each_observed_counterexample_preserves_false(self):
        rows = []
        for field, value in (("first_seal_minute", .01), ("last_seal_minute", .01),
                             ("bomb_count", 1.), ("turnover", .03)):
            r = self.row(); r[field] = value; rows.append(r)
        self.assertFalse(ambiguous_mask(pd.DataFrame(rows)).any())

    def test_new_ohlc_recipe_unchanged(self):
        r = self.row(); r["date"] = "2026-09-01"
        frame = pd.DataFrame([r]); changed, mask = apply_mask(frame)
        self.assertFalse(mask.any()); pd.testing.assert_frame_equal(changed, frame)

    def test_no_target_or_identity_dependency(self):
        a = pd.DataFrame([self.row()]); b = a.copy(); b["target"] = 0.; b["actual_one_word_daily"] = 1.
        pd.testing.assert_series_equal(ambiguous_mask(a), ambiguous_mask(b))

    def test_rows_other_fields_and_original_not_mutated(self):
        original = pd.DataFrame([self.row(), self.row()]); before = original.copy(deep=True)
        changed, _ = apply_mask(original)
        pd.testing.assert_frame_equal(original, before)
        pd.testing.assert_frame_equal(changed.drop(columns="signal_one_word"), before.drop(columns="signal_one_word"))


if __name__ == "__main__": unittest.main()
