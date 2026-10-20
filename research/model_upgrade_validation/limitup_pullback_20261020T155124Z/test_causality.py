"""Safety checks for genuinely new history reconstruction, without fitting."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('pullback_study',HERE/'study.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def toy():
    sessions=pd.bdate_range('2024-01-01',periods=100).strftime('%Y-%m-%d').tolist()
    rows=[];previous=100.
    for i,d in enumerate(sessions):
        close={75:110.,76:107.,77:106.,78:116.6,79:128.26}.get(i,100. if i<75 else previous)
        low=min(previous,close)-1.;high=max(previous,close)
        rows.append(dict(date=d,code='600001',open=previous,high=high,low=low,close=close,pre_close=previous,volume=1000. if i in (75,78,79) else 400.,float_cap=close*1e7))
        previous=close
    bars=pd.DataFrame(rows)
    attempts=pd.DataFrame([dict(signal_date=sessions[i],code='600001',pool_assignment='CLOSING_LIMIT_UP') for i in (75,78,79)])
    sig=pd.DataFrame([dict(date=sessions[79],code='600001',boards=2,price=128.26)])
    return sig,bars,attempts,sessions


class Causality(unittest.TestCase):
    def test_observed_sequence(self):
        s,b,a,c=toy();f,_=m.build_features(s,b,a,c);r=f.iloc[0]
        self.assertEqual(r.pullback_has_anchor,1.)
        self.assertEqual(r.anchor_date,c[75]);self.assertEqual(r.launch_date,c[78])
        self.assertEqual(r.pullback_days,2.)
        self.assertAlmostEqual(r.pullback_mean_volume_ratio,.4)
        self.assertAlmostEqual(r.relaunch_volume_ratio,2.5)
        self.assertLess(r.history_latest_date,r.launch_date)

    def test_future_path_changes_do_not_change_signal_features(self):
        s,b,a,c=toy();before,_=m.build_features(s,b,a,c)
        b.loc[b.date.gt(c[79]),['open','high','low','close','pre_close','volume','float_cap']]=99999.
        a=pd.concat([a,pd.DataFrame([dict(signal_date=c[90],code='600001',pool_assignment='CLOSING_LIMIT_UP')])],ignore_index=True)
        after,_=m.build_features(s,b,a,c)
        pd.testing.assert_frame_equal(before,after)

    def test_ordinary_positive_candle_is_not_limitup_anchor(self):
        s,b,a,c=toy();a=a[a.signal_date.ne(c[75])]
        f,_=m.build_features(s,b,a,c)
        self.assertEqual(f.iloc[0].pullback_has_anchor,0.)
        self.assertTrue(pd.isna(f.iloc[0].anchor_date))

    def test_missing_pullback_day_stays_unknown(self):
        s,b,a,c=toy();b=b[b.date.ne(c[76])]
        f,_=m.build_features(s,b,a,c)
        self.assertTrue(pd.isna(f.iloc[0].pullback_has_anchor))
        self.assertEqual(f.iloc[0].pullback_status,'ANCHOR_TO_LAUNCH_PATH_GAP')

    def test_price_contradiction_is_not_usable_shape(self):
        s,b,a,c=toy();s['price']=128.27
        f,_=m.build_features(s,b,a,c)
        self.assertFalse(f.iloc[0].current_run_verified)
        self.assertTrue(pd.isna(f.iloc[0].pullback_has_anchor))

    def test_unknown_volume_is_not_zero_contraction(self):
        s,b,a,c=toy();b.loc[b.date.eq(c[76]),'volume']=np.nan
        f,_=m.build_features(s,b,a,c)
        self.assertEqual(f.iloc[0].pullback_has_anchor,1.)
        self.assertTrue(pd.isna(f.iloc[0].pullback_mean_volume_ratio))


if __name__=='__main__':unittest.main()
