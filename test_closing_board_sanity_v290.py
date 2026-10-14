import unittest
from closing_board_sanity_v290 import classify


class ClosingBoardNecessaryTests(unittest.TestCase):
    def row(self,high,close):
        return dict(date='2026-08-05',code='001229',price_reconciled=True,
            original_pool_close_agrees=True,observed_ohlc=dict(high=high,close=close))

    def test_one_cent_contradiction_is_not_a_rounded_limit_pass(self):
        r=classify(self.row(42.33,42.32))
        self.assertEqual(r['status'],'CLOSE_LIMIT_CONTRADICTION')
        self.assertFalse(r['necessary_close_high_passed']);self.assertFalse(r['close_limit_confirmed'])

    def test_high_equals_close_is_only_necessary_not_a_limit_or_height_confirmation(self):
        r=classify(self.row(42.33,42.33))
        self.assertTrue(r['necessary_close_high_passed'])
        for key in ['close_limit_confirmed','board_height_confirmed','trade_authorized']:self.assertFalse(r[key])
        self.assertIsNone(r['official_upper_limit_price'])

    def test_subcent_precision_mismatch_remains_unknown(self):
        self.assertEqual(classify(self.row(42.325,42.32))['status'],'UNRESOLVED_PRICE_PRECISION')

    def test_conflicting_price_source_cannot_be_used_to_reject_a_board(self):
        row=self.row(42.33,42.32);row['price_reconciled']=False
        r=classify(row);self.assertEqual(r['status'],'UNVERIFIED_SOURCE')
        self.assertIsNone(r['necessary_close_high_passed'])

    def test_future_label_changes_cannot_change_the_ledger(self):
        row=self.row(42.33,42.32);before=classify(row)
        row.update(target=1,next_close=100.,next_high=101.,label_valid=False)
        self.assertEqual(classify(row),before)


if __name__=='__main__':unittest.main()
