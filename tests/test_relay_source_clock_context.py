import unittest
import numpy as np
import pandas as pd
from reported_seal_archive_v275 import clock
from relay_source_clock_context import clock_context, annotate


class ArchiveClockSemanticsTests(unittest.TestCase):
    def test_auction_adjacent_well_formed_stays_unaccepted(self):
        for value in [92501, "09:25:04", 92504.0, "09:29:59"]:
            context = clock_context(value)
            self.assertTrue(context["semantics"].startswith("AUCTION_ADJACENT_UNRESOLVED"))
            self.assertIsNotNone(context["structural_seconds"])
            self.assertTrue(np.isnan(clock(value)))

    def test_original_valid_boundary_is_preserved(self):
        for value in [92500, "09:30:00", "11:30:00", "13:00:00", "15:00:00"]:
            self.assertTrue(clock_context(value)["semantics"].startswith("ORIGINAL_CONTRACT"))
            self.assertFalse(np.isnan(clock(value)))

    def test_malformed_is_distinct_from_source_semantics_unknown(self):
        for value in ["abc", "09:25:61", 256000, "92501.5"]:
            self.assertIsNone(clock_context(value)["structural_seconds"])
        self.assertEqual(clock_context(None)["semantics"], "MISSING")

    def test_all_original_columns_and_unknown_training_authorization_preserved(self):
        original = pd.DataFrame([dict(date="2026-09-01", code="000001", first_seal=92501, last_seal=100000,
            boards=2, reported_seal_status="UNAVAILABLE_OR_INVALID", arbitrary_original=17.)])
        output = annotate(original)
        pd.testing.assert_frame_equal(output[original.columns], original)
        self.assertTrue(output.reported_clock_adjacent_unresolved.iloc[0])
        self.assertFalse(output.context_authorizes_new_validity_or_training.iloc[0])
        with self.assertRaises(ValueError):
            annotate(pd.concat([original, original], ignore_index=True))


if __name__ == "__main__":
    unittest.main()
