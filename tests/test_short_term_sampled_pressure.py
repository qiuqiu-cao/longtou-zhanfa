import unittest

import numpy as np
import pandas as pd

import short_term_sampled_pressure as p


class SampledPressureTests(unittest.TestCase):
    def setUp(self):
        day="2026-08-03";code="000001"
        self.day,self.code=day,code
        times=sorted(p.expected_times(day))
        close=np.repeat(11.,48);close[:5]=10.5;close[12:15]=10.7
        self.bars=pd.DataFrame(dict(code=code,minute=[day+" "+t+":00" for t in times],open=close,high=close,low=close,close=close,
            volume_shares=np.repeat(10000.,48),amount=close*10000))
        self.bars.loc[[5,15],"low"]=10.6
        self.ref=pd.DataFrame([dict(symbol="sz000001",date=day,open=10.5,close=11.,high=11.,low=10.5,volume=4800.)])

    def test_separate_intervals_preserve_pressure_order_without_tick_claims(self):
        x=p.describe(self.bars,self.ref,self.code,self.day,11.)
        self.assertTrue(x["price_path_qualified"])
        self.assertTrue(x["volume_path_qualified"])
        self.assertEqual(x["sampled_cap_close_then_below_intervals"],1)
        self.assertEqual(x["sampled_cap_close_recovery_transitions"],2)
        self.assertEqual(x["sampled_intervals_after_last_below_close"],33)
        self.assertAlmostEqual(x["sampled_final_six_volume_fraction"],.125)
        self.assertIsNone(x["real_reseal_count"])
        self.assertIsNone(x["verified_absorption"])
        self.assertFalse(x["continuous_seal_verified"])

    def test_future_day_other_code_and_label_columns_are_ignored(self):
        a=p.describe(self.bars,self.ref,self.code,self.day,11.)
        changed=pd.concat([self.bars,self.bars.assign(minute=self.bars.minute.str.replace(self.day,"2026-08-04"),low=.1),self.bars.assign(code="000002",low=.01)])
        changed["future_target"]=999;changed["model_score"]=1
        z=p.describe(changed,self.ref,self.code,self.day,11.)
        self.assertEqual(a,z)

    def test_missing_official_interval_cannot_be_interpolated(self):
        x=p.describe(self.bars.drop(index=20),self.ref,self.code,self.day,11.)
        self.assertFalse(x["price_path_qualified"])
        self.assertTrue(all(x[k] is None for k in p.FIELDS))

    def test_duplicate_bar_and_daily_price_conflict_reject_path(self):
        for bars,ref in [(pd.concat([self.bars,self.bars.iloc[:1]]),self.ref),(self.bars,self.ref.assign(high=12.))]:
            self.assertFalse(p.describe(bars,ref,self.code,self.day,11.)["price_path_qualified"])

    def test_volume_disagreement_keeps_price_facts_and_volume_unknown(self):
        x=p.describe(self.bars,self.ref.assign(volume=200.),self.code,self.day,11.)
        self.assertTrue(x["price_path_qualified"])
        self.assertFalse(x["volume_path_qualified"])
        self.assertTrue(all(x[k] is None for k in p.VOLUME_FIELDS))

    def test_one_missing_peer_prevents_entire_cohort_relative_rank(self):
        f=pd.DataFrame([dict(date=self.day,code="000001",boards=2,daily_one_word=0,price_path_qualified=True,sampled_negative_prior_bar_peak_drawdown=-.03),
            dict(date=self.day,code="000002",boards=2,daily_one_word=0,price_path_qualified=False,sampled_negative_prior_bar_peak_drawdown=np.nan)])
        x=p.cohort_relative(f)
        self.assertTrue(x.sampled_same_height_pressure_rank.isna().all())
        self.assertFalse(x.sampled_same_height_comparison_complete.any())


if __name__=="__main__":unittest.main()
