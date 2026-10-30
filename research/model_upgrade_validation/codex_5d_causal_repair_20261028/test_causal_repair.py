import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from causal_engine import FEATURES, engineer, limit_prices, snapshot_inputs


def fixture():
    dates=pd.bdate_range('2025-01-01',periods=38).strftime('%Y-%m-%d').tolist()
    rows=[]
    for code in ['000001','000002','000003']:
        prior=10.
        for i,day in enumerate(dates):
            close={27:11.,28:12.1,29:13.31,30:14.64}.get(i,10.) if code=='000001' else 10.
            op=13.9 if i==30 and code=='000001' else close
            rows.append(dict(date=day,code=code,name=code,open=op,high=max(op,close),
                low=min(op,close),close=close,pre_close=prior,volume=10_000_000.,
                amount=close*10_000_000.,turnover=15.,float_cap=close*1e8,
                st=False,new_listing=False,source='Sina unadjusted daily via AKShare fallback'))
            prior=close
    return pd.DataFrame(rows),dates,{'000001':'a','000002':'a'}


class CausalityTests(unittest.TestCase):
    def test_today_and_future_outcomes_cannot_change_features_or_membership(self):
        raw,dates,imap=fixture()
        changed=raw.copy()
        m=changed.date.ge(dates[30])
        changed.loc[m,'close']=changed.loc[m,'open']*.91
        changed.loc[m,'low']=changed.loc[m,'close']
        changed.loc[m,'float_cap']=changed.loc[m,'close']*1e8
        changed.loc[m,'amount']*=10
        a,b=engineer(raw,dates,imap),engineer(changed,dates,imap)
        a=a[a.date.le(dates[30])].reset_index(drop=True)
        b=b[b.date.le(dates[30])].reset_index(drop=True)
        pd.testing.assert_frame_equal(a[FEATURES+['eligible']],b[FEATURES+['eligible']])

    def test_prediction_snapshot_matches_historical_transform(self):
        raw,dates,imap=fixture()
        hist=engineer(raw,dates,imap)
        realtime=engineer(snapshot_inputs(raw,'000001',dates[30]),dates,imap)
        a=hist[hist.date.eq(dates[30])].reset_index(drop=True)
        b=realtime[realtime.date.eq(dates[30])].reset_index(drop=True)
        pd.testing.assert_frame_equal(a[FEATURES+['eligible']],b[FEATURES+['eligible']])

    def test_crash_gene_observable_only_after_crash(self):
        raw,dates,imap=fixture()
        m=raw.code.eq('000001') & raw.date.eq(dates[30])
        raw.loc[m,['close','low','float_cap']]=[11.98,11.98,11.98e8]
        d=engineer(raw,dates,imap)
        target=d[d.code.eq('000001')].set_index('date')
        self.assertEqual(target.loc[dates[30],'has_bad_crash_gene'],0.)
        self.assertEqual(target.loc[dates[31],'has_bad_crash_gene'],1.)

    def test_volume_sources_have_same_vwap(self):
        raw,dates,imap=fixture()
        equivalent=raw.copy()
        equivalent['source']='Eastmoney unadjusted daily via AKShare'
        equivalent['volume']/=100
        a,b=engineer(raw,dates,imap),engineer(equivalent,dates,imap)
        np.testing.assert_allclose(a.vwap,b.vwap,equal_nan=True)

    def test_missing_orderflow_and_unknown_industry_not_filled(self):
        raw,dates,imap=fixture()
        d=engineer(raw,dates,imap)
        self.assertTrue(d.tail_seal_momentum.isna().all())
        self.assertTrue(d.loc[d.code.eq('000003'),'ind_max_streak'].isna().all())

    def test_suspension_gap_breaks_streak_and_prior_context(self):
        raw,dates,imap=fixture()
        raw=raw[~(raw.code.eq('000001') & raw.date.eq(dates[28]))]
        d=engineer(raw,dates,imap)
        row=d[d.code.eq('000001') & d.date.eq(dates[29])].iloc[0]
        self.assertFalse(row.eligible)
        self.assertTrue(pd.isna(row.prev_is_zt))
        self.assertEqual(row.board_streak,0)

    def test_decimal_half_up_not_bankers(self):
        up,down=limit_prices([10.05,10.15],[.10,.10])
        np.testing.assert_allclose(up,[11.06,11.17])
        np.testing.assert_allclose(down,[9.05,9.14])

    def test_tplusone_limit_down_delay_and_cash_account(self):
        import run_backtest as runner
        dates=['2025-01-02','2025-01-03','2025-01-06']
        bars=pd.DataFrame([
            dict(date=dates[0],code='000001',open=10.,close=10.,upper=11.,lower=9.),
            dict(date=dates[1],code='000001',open=9.,close=9.,upper=11.,lower=9.),
            dict(date=dates[2],code='000001',open=9.5,close=9.6,upper=9.9,lower=8.1)])
        signals=pd.DataFrame([dict(date=dates[0],code='000001')])
        previous=runner.HERE
        with tempfile.TemporaryDirectory() as tmp:
            runner.HERE=Path(tmp)
            try:
                result=runner.portfolio(signals,bars,dates)
                trades=pd.read_csv(Path(tmp)/'portfolio_conditional_trades.csv')
                equity=pd.read_csv(Path(tmp)/'portfolio_conditional_daily.csv')
                self.assertEqual(trades.exit_date.iloc[0],dates[2])
                self.assertTrue((trades.exit_date>trades.entry_date).all())
                self.assertTrue((equity.cash>=0).all())
                self.assertEqual(result['closed_trades'],1)
                self.assertEqual(result['open_positions'],0)
                self.assertLess(result['max_drawdown_pct'],0)
            finally:
                runner.HERE=previous


if __name__=='__main__':
    unittest.main(verbosity=2)
