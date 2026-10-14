import unittest
from closing_label_audit_v290 import label_status


class LabelNecessaryConditionTests(unittest.TestCase):
    def test_unknown_original_target_is_not_filled_by_next_day_evidence(self):
        self.assertEqual(label_status(float('nan'),False,'NECESSARY_CONDITION_ONLY','CLOSE_LIMIT_CONTRADICTION'),
            'ORIGINAL_LABEL_UNRESOLVED')

    def test_signal_and_next_day_contradictions_have_distinct_causes(self):
        self.assertEqual(label_status(1.,True,'CLOSE_LIMIT_CONTRADICTION','NECESSARY_CONDITION_ONLY'),
            'POSITIVE_CONTINUATION_SIGNAL_CONTRADICTION')
        self.assertEqual(label_status(1.,True,'NECESSARY_CONDITION_ONLY','CLOSE_LIMIT_CONTRADICTION'),
            'POSITIVE_NEXT_CLOSE_CONTRADICTION')

    def test_equal_high_close_cannot_certify_a_positive_label(self):
        self.assertEqual(label_status(1.,True,'NECESSARY_CONDITION_ONLY','NECESSARY_CONDITION_ONLY'),
            'POSITIVE_NECESSARY_CONDITIONS_ONLY')

    def test_unverified_next_price_cannot_certify_a_negative_label(self):
        self.assertEqual(label_status(0.,True,'NECESSARY_CONDITION_ONLY','UNVERIFIED_SOURCE'),
            'NEGATIVE_REQUIRES_LIMIT_PRICE_VERIFICATION')


if __name__=='__main__':unittest.main()
