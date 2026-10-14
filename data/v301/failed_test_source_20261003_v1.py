import copy
import unittest
import numpy as np
import pandas as pd
from exchange_turnover_intervals_v301 import CHANNELS,PROTOCOL,predict,score


class TurnoverIntervalIntegrity(unittest.TestCase):
    def sample(self):
        sessions=pd.bdate_range('2026-01-05',periods=10).strftime('%Y-%m-%d').tolist()
        obs=pd.DataFrame({k:np.arange(10,dtype=float)*100+1000 for k in CHANNELS},index=sessions)
        protocol=copy.deepcopy(PROTOCOL);protocol.update(signal_start=sessions[3],signal_end=sessions[7],
            lookback_official_sessions=4,minimum_known_observations=2)
        return sessions,obs,protocol

    def test_future_levels_or_extra_labels_cannot_change_earlier_bands(self):
        s,o,p=self.sample();a=predict(o,s,p)
        o.loc[s[6]:,CHANNELS]=999999;o['next_amount']=999999
        b=predict(o,s,p)
        pd.testing.assert_frame_equal(a[a.signal_date.lt(s[6])],b[b.signal_date.lt(s[6])])
        self.assertTrue(a.max_known_observation_date.le(a.signal_date).all())

    def test_missing_target_does_not_skip_to_later_available_day(self):
        s,o,p=self.sample();o.loc[s[5],CHANNELS]=np.nan
        f=score(predict(o,s,p),o);rows=f[f.signal_date.eq(s[4])]
        self.assertTrue(rows.target_date.eq(s[5]).all());self.assertFalse(rows.evaluated.any())
        self.assertEqual(len(f),5*2*2)

    def test_no_difference_across_missing_day_or_current_zero_fill(self):
        s,o,p=self.sample();o.loc[s[4],CHANNELS]=np.nan
        f=predict(o,s,p)
        current=f[f.signal_date.eq(s[4]) & f.arm.eq('PERSISTENCE_DELTA')]
        self.assertTrue(current.status.eq('UNKNOWN_CURRENT_LEVEL').all())
        later=f[f.signal_date.eq(s[5]) & f.arm.eq('PERSISTENCE_DELTA')]
        self.assertTrue(later.training_n.eq(2).all())

    def test_insufficient_history_has_no_interval(self):
        s,o,p=self.sample();p['minimum_known_observations']=60
        f=predict(o,s,p)
        self.assertTrue(f.status.eq('INSUFFICIENT_HISTORY').all())
        self.assertTrue(f[['low','median','high']].isna().all().all())

    def test_physical_nonnegative_bounds_and_uncalibrated_nominal_band(self):
        s,o,p=self.sample();o.loc[:,CHANNELS]=[10000,0,10000,0,10000,0,10000,0,10000,0]
        f=predict(o,s,p)
        self.assertTrue(f.low.dropna().ge(0).all());self.assertTrue(f.high.dropna().ge(0).all())
        self.assertNotIn('probability',f)
        self.assertFalse(f.forward_evidence.any())

    def test_unknown_final_successor_and_scope_never_relabels_original_pool(self):
        s,o,p=self.sample();p['signal_end']=s[-1]
        f=score(predict(o,s,p),o);last=f[f.signal_date.eq(s[-1])]
        self.assertFalse(last.evaluated.any());self.assertTrue(last.target_date.isna().all())
        self.assertFalse(PROTOCOL['scope_matches_original_non_st_pool'])
        self.assertFalse(PROTOCOL['pure_blind_test']);self.assertFalse(PROTOCOL['core_upgrade_authorized'])
        self.assertIn('CDR_SCOPE_UNRESOLVED',PROTOCOL['scope']['shsz_stock_category_amount_yi'])


if __name__=='__main__':unittest.main()
