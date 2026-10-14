import unittest
import pandas as pd
from emotion_evidence_coverage import profile


class EvidenceCoverageTest(unittest.TestCase):
    def test_derived_state_never_substitutes_for_path(self):
        result = profile(pd.DataFrame({"emotion_state": ["修复"], "repair_close_retention": [""]}))
        self.assertEqual(result["dimensions"]["repair_close_quality"]["jointly_nonmissing"], 0)
        self.assertFalse(result["substitution_of_eod_for_intraday_allowed"])

    def test_zero_is_observed_not_missing(self):
        frame = pd.DataFrame({"promotion_1to2": [0], "promotion_2to3": [0],
                              "promotion_3to4": [0], "space_feedback": [0]})
        result = profile(frame)
        self.assertEqual(result["dimensions"]["height_feedback"]["jointly_nonmissing"], 1)
        self.assertFalse(result["dimensions"]["height_feedback"]["source_verified_by_this_audit"])
