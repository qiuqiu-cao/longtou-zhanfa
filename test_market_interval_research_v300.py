import copy
import unittest
import numpy as np
import pandas as pd
from market_interval_research_v300 import (
    METRICS, SLOTS, PROTOCOL, forecasts, score, interval_score, bounded,
    prepare, relay_observations, bootstrap, validate_sessions,
)


class MarketIntervalIntegrity(unittest.TestCase):
    def sample(self):
        sessions=pd.bdate_range('2026-01-01',periods=9).strftime('%Y-%m-%d').tolist()
        obs=pd.DataFrame({k:np.arange(9,dtype=float)/20 for k in METRICS},index=sessions)
        for k in ['limit_count','down_limit_count','max_board']: obs[k]=np.arange(9,dtype=float)+1
        reasons=pd.DataFrame('KNOWN',index=sessions,columns=list(METRICS))
        protocol=copy.deepcopy(PROTOCOL)
        protocol.update(signal_start=sessions[3],signal_end=sessions[6],
                        lookback_official_sessions=4,minimum_known_observations=2)
        return sessions,obs,reasons,protocol

    def test_future_observations_do_not_change_prior_intervals_or_training(self):
        s,o,r,p=self.sample(); a=forecasts(o,r,s,p)
        o.loc[s[6]:]=100000; r.loc[s[6]:]='FUTURE_CHANGED'
        b=forecasts(o,r,s,p)
        pd.testing.assert_frame_equal(a[a.signal_date.lt(s[6])],b[b.signal_date.lt(s[6])])
        self.assertTrue((a.max_known_observation_date<=a.signal_date).all())

    def test_exact_successor_not_next_available_row(self):
        s,o,r,p=self.sample(); o.loc[s[5]]=np.nan
        pred=forecasts(o,r,s,p); scored=score(pred,o,r)
        previous=scored[scored.signal_date.eq(s[4])]
        self.assertTrue(previous.target_date.eq(s[5]).all())
        self.assertTrue(previous.actual.isna().all())
        self.assertFalse(previous.evaluated.any())
        self.assertEqual(len(pred),4*8*2)

    def test_missing_current_delta_unknown_but_level_remains(self):
        s,o,r,p=self.sample(); o.loc[s[4],'limit_count']=np.nan
        f=forecasts(o,r,s,p); f=f[f.signal_date.eq(s[4]) & f.metric.eq('limit_count')].set_index('arm')
        self.assertEqual(f.loc['PERSISTENCE_DELTA','status'],'UNKNOWN_CURRENT_LEVEL')
        self.assertEqual(f.loc['LEVEL','status'],'KNOWN_DIAGNOSTIC_INTERVAL')

    def test_no_increment_across_a_missing_official_day(self):
        s,o,r,p=self.sample(); o.loc[s[4],'limit_count']=np.nan
        f=forecasts(o,r,s,p); g=f[(f.signal_date.eq(s[5])) & f.metric.eq('limit_count')].set_index('arm')
        self.assertEqual(g.loc['PERSISTENCE_DELTA','training_n'],2)
        self.assertEqual(g.loc['LEVEL','training_n'],3)

    def test_short_history_is_not_padded_or_zero_filled(self):
        s,o,r,p=self.sample(); p['minimum_known_observations']=100
        f=forecasts(o,r,s,p)
        self.assertTrue(f.status.eq('INSUFFICIENT_HISTORY').all())
        self.assertTrue(f[['low','median','high']].isna().all().all())

    def test_wider_band_penalized_even_if_hit(self):
        self.assertEqual(float(interval_score(5,4,6)),2)
        self.assertEqual(float(interval_score(5,0,10)),10)
        self.assertEqual(float(interval_score(9,4,6)),32)
        with self.assertRaises(ValueError): interval_score(1,2,0)

    def test_bounds_counts_and_rates_preserve_units(self):
        np.testing.assert_array_equal(bounded([-.2,.4,1.3],'failed_rate'),[0,.4,1])
        np.testing.assert_array_equal(bounded([-4,2.4,7.1],'limit_count'),[0,2.4,8])
        np.testing.assert_array_equal(bounded([-1.5,0,1.5],'sh_a_return_pct'),[-1.5,0,1.5])

    def market(self,s):
        return pd.DataFrame({'date':s,'data_status':'verified','limit_count':10.,'consecutive_count':2.,
            'failed_count':5.,'failed_rate':1/3,'down_limit_count':0.,'max_board':3.,
            'yesterday_premium':.01,'previous_limit_feedback_coverage':1.})

    def test_actual_all_two_plus_rate_not_overall_or_mid_proxy(self):
        s=['2026-01-05','2026-01-06','2026-01-07']
        market=self.market(s).set_index('date')
        pool=pd.DataFrame({'date':[s[0],s[0],s[1],s[1],s[2],s[2]],
            'code':['000001','600001','000001','600002','000001','600002'],
            'boards':[2,5,3,2,4,3]})
        value,complete=relay_observations(market,pool)
        self.assertTrue(complete.all()); self.assertTrue(np.isnan(value.iloc[0]))
        self.assertEqual(value.iloc[1],.5); self.assertEqual(value.iloc[2],1)
        market.loc[s[1],'consecutive_count']=3
        value,_=relay_observations(market,pool)
        self.assertTrue(value.isna().all())

    def test_partial_feedback_not_a_zero_return_label(self):
        s=['2026-01-05','2026-01-06','2026-01-07']; market=self.market(s)
        market.loc[1,'previous_limit_feedback_coverage']=.99
        market.loc[1,'data_status']='partial'
        pool=pd.DataFrame({'date':np.repeat(s,2),'code':['000001','600001']*3,'boards':[2,2,3,3,4,4]})
        indexes={k:pd.Series([100,101,102],index=s) for k in ['sh000002','sz399107']}
        obs,reasons,_=prepare(market,s,indexes,pool)
        self.assertTrue(np.isnan(obs.loc[s[1],'yesterday_premium']))
        self.assertEqual(obs.loc[s[1],'down_limit_count'],0)
        self.assertEqual(obs.loc[s[1],'limit_count'],10)
        self.assertEqual(reasons.loc[s[1],'yesterday_premium'],'PREVIOUS_POOL_OR_FULL_FEEDBACK_MISSING')

    def test_calendar_grid_enforced_and_final_target_unknown(self):
        s,o,r,p=self.sample(); p['signal_end']=s[-1]
        f=forecasts(o,r,s,p); z=score(f,o,r); last=z[z.signal_date.eq(s[-1])]
        self.assertTrue(last.actual.isna().all()); self.assertTrue(last.target_date.isna().all())
        with self.assertRaises(ValueError): forecasts(o.drop(s[5]),r,s,p)
        with self.assertRaises(ValueError): validate_sessions(s+s[:1])

    def test_above_band_is_numerical_not_economic_outperformance(self):
        s,o,r,p=self.sample(); f=forecasts(o,r,s,p)
        o.loc[s[4],'down_limit_count']=100
        g=score(f,o,r); g=g[g.signal_date.eq(s[3]) & g.metric.eq('down_limit_count')]
        self.assertTrue(g.position.eq('ABOVE_BAND').all())
        self.assertNotIn('super_expected',g.columns)

    def test_official_grid_bootstrap_retains_holes(self):
        dates=['2026-01-05','2026-01-06','2026-01-07','2026-01-08']
        pair=pd.Series([1.,-1.],index=[dates[0],dates[3]])
        spec={**PROTOCOL['bootstrap'],'draws':100,'block_official_sessions':2}
        a=bootstrap(pair,dates,spec); b=bootstrap(pair,dates,spec)
        self.assertEqual(a,b); self.assertEqual(a['official_dates'],4); self.assertEqual(a['pairs'],2)

    def test_ten_requirements_no_live_or_sixty_claim(self):
        self.assertEqual(len(SLOTS),10); self.assertEqual(len(METRICS),8)
        self.assertEqual(sum(not channels for _,channels,_ in SLOTS),3)
        for k in ['pure_blind_test','genuine_historical_availability_certified','core_upgrade_authorized',
                  'stable_60_proved','probability_calibration_proved','original_oct8_plan_may_be_changed']:
            self.assertFalse(PROTOCOL[k])


if __name__=='__main__': unittest.main()
