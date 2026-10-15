import unittest

import pandas as pd

import short_term_attempt_reconstruction as r


class AttemptReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.days=["2026-08-03","2026-08-04","2026-08-05","2026-08-06"]

    def frame(self):
        return pd.DataFrame([dict(signal_date=self.days[0],code="000001",name="甲",pool_assignment="CLOSING_LIMIT_UP",board_height=1,pre_close=10.,limit_price_10=11.,high=11.,close=11.),
            dict(signal_date=self.days[1],code="000001",name="甲",pool_assignment="CLOSING_LIMIT_UP",board_height=2,pre_close=11.,limit_price_10=12.10,high=12.10,close=12.09),
            dict(signal_date=self.days[2],code="000001",name="甲",pool_assignment="CLOSING_LIMIT_UP",board_height=3,pre_close=12.09,limit_price_10=13.30,high=13.30,close=13.30)])

    def test_one_cent_below_close_is_retained_as_failed_attempt(self):
        f,q=r.reconstruct(self.frame(),self.days)
        self.assertFalse(len(q))
        self.assertEqual(f.pool_assignment.tolist(),["CLOSING_LIMIT_UP","FAILED_BOARD_OBSERVATION","CLOSING_LIMIT_UP"])
        self.assertEqual(f.board_height.tolist(),[1,2,1])
        self.assertEqual(f.original_board_height.tolist(),[1,2,3])
        self.assertTrue(f.iloc[1].reclassified)

    def test_missing_official_session_does_not_extend_a_streak(self):
        f,_=r.reconstruct(self.frame().iloc[[0,2]],self.days)
        self.assertEqual(f.board_height.tolist(),[1,1])

    def test_future_suffix_targets_and_input_order_do_not_change_prefix(self):
        frame=self.frame();full,_=r.reconstruct(frame,self.days)
        prefix,_=r.reconstruct(frame.iloc[:2].assign(target=999).sample(frac=1,random_state=2),self.days)
        pd.testing.assert_frame_equal(full.iloc[:2].reset_index(drop=True),prefix.reset_index(drop=True))

    def test_near_cap_without_exact_touch_is_not_called_failed_attempt(self):
        f,q=r.reconstruct(self.frame().iloc[:1].assign(high=10.99,close=10.98),self.days)
        self.assertTrue(f.empty)
        self.assertEqual(q.iloc[0].qualification_reason,"NO_EXACT_UPPER_PRICE_TOUCH_NOT_AN_ATTEMPT")

    def test_over_cap_regime_and_noncent_values_keep_scope_unknown(self):
        for high,reason in [(11.01,"PRICE_ABOVE_ASSUMED_10PCT_CAP_REGIME_UNRESOLVED"),(11.001,"INVALID_OR_NONCENT_PRICE")]:
            f,q=r.reconstruct(self.frame().iloc[:1].assign(high=high),self.days)
            self.assertTrue(f.empty)
            self.assertEqual(q.iloc[0].qualification_reason,reason)
            self.assertNotIn(self.days[0],f.attrs["failed_scope_complete_dates"])

    def test_decimal_half_up_cap_is_recomputed(self):
        f,q=r.reconstruct(self.frame().iloc[:1].assign(pre_close=2.05,limit_price_10=2.26,high=2.26,close=2.26),self.days)
        self.assertEqual(len(f),1)
        self.assertTrue(q.empty)


if __name__=="__main__":unittest.main()
