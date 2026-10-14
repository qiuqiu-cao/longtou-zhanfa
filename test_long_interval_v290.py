import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import pandas as pd
import archive_long_interval_v290 as engine


class LongIntervalEvidenceTests(unittest.TestCase):
    def fixture(self):
        bars=pd.DataFrame([dict(code='000678',minute='2026-09-30 '+time+':00',
            open=10.,high=11.,low=10.,close=10.5,volume_shares=100.,amount=1050.) for time in engine.TIMES])
        bars.loc[bars.index[-2:],['open','high','low','close']]=11.
        bars.loc[bars.index[-2:],'amount']=1100.
        daily=pd.DataFrame([dict(symbol='sz000678',date='2026-09-30',open=10.,close=11.,high=11.,low=10.,volume=8.)])
        return bars,daily

    def check(self,bars,daily,cap=11.,word=0.):return engine.check_signal(bars,daily,'000678','2026-09-30',cap,word)

    def test_thirty_minute_grid_cannot_authorize_finer_path_or_fill(self):
        bars,daily=self.fixture();r=self.check(bars,daily)
        self.assertEqual(r['observed_bars'],8)
        self.assertEqual(r['interval_minutes'],30)
        self.assertTrue(r['price_reconciled']);self.assertTrue(r['grid_complete'])
        self.assertEqual(r['final_cap_price_bar_run'],2)
        for key in ['one_minute_evidence','auction_verified','continuous_seal_verified','actual_fill_verified','eligible_for_frozen_forward']:
            self.assertFalse(r[key])
        self.assertIsNone(r['real_reseal_count'])

    def test_future_dates_and_other_codes_cannot_change_signal_features(self):
        bars,daily=self.fixture();original=self.check(bars,daily)
        future=bars.copy();future.minute=future.minute.str.replace('2026-09-30','2026-10-08')
        other=bars.copy();other.code='600825';other[['open','high','low','close']]=1000.
        self.assertEqual(self.check(pd.concat([bars,future,other]),daily),original)

    def test_one_missing_interval_blocks_path_and_full_day_vwap(self):
        bars,daily=self.fixture();r=self.check(bars.drop(index=2),daily)
        self.assertTrue(r['price_reconciled']);self.assertFalse(r['grid_complete'])
        self.assertEqual(r['missing_times'],['11:00'])
        self.assertIsNone(r['final_cap_price_bar_run']);self.assertIsNone(r['full_day_vwap'])

    def test_one_cent_high_above_original_cap_does_not_get_tolerance_relaxation(self):
        bars,daily=self.fixture();bars.loc[0,'high']=11.01;daily.loc[0,'high']=11.01
        r=self.check(bars,daily)
        self.assertTrue(r['price_reconciled']);self.assertTrue(r['grid_complete'])
        self.assertTrue(r['original_pool_close_agrees']);self.assertFalse(r['closing_price_is_daily_high'])
        self.assertIsNone(r['cap_price_bar_share']);self.assertIsNone(r['negative_max_pullback'])

    def test_volume_conflict_does_not_delete_price_path_or_create_vwap(self):
        bars,daily=self.fixture();daily.loc[0,'volume']=100.
        r=self.check(bars,daily)
        self.assertTrue(r['price_reconciled']);self.assertEqual(r['final_cap_price_bar_run'],2)
        self.assertFalse(r['volume_reconciled']);self.assertIsNone(r['full_day_vwap'])

    def test_missing_or_out_of_bar_amount_blocks_vwap(self):
        for value in [float('nan'),1e9]:
            bars,daily=self.fixture();bars.loc[3,'amount']=value
            r=self.check(bars,daily);self.assertIsNone(r['full_day_vwap'])

    def test_duplicate_and_outside_session_time_are_not_silently_dropped(self):
        bars,daily=self.fixture();r=self.check(pd.concat([bars,bars.iloc[[2]]]),daily)
        self.assertFalse(r['grid_complete']);self.assertFalse(r['price_reconciled'])
        bars.loc[3,'minute']='2026-09-30 09:25:00';r=self.check(bars,daily)
        self.assertFalse(r['grid_complete']);self.assertFalse(r['auction_verified'])

    def test_original_word_conflict_is_retained_as_conflict(self):
        bars,daily=self.fixture();r=self.check(bars,daily,word=1.)
        self.assertFalse(r['original_word_agrees'])
        self.assertTrue(r['original_pool_close_agrees'])

    def test_adjusted_only_daily_payload_and_duplicates_fail(self):
        with self.assertRaisesRegex(ValueError,'unadjusted day list'):
            engine.daily_parse({'data':{'sz000678':{'qfqday':[['2026-09-30',10,11,11,10,8]]}}},'sz000678')
        row=['2026-09-30',10,11,11,10,8]
        with self.assertRaisesRegex(ValueError,'duplicate or unordered'):
            engine.daily_parse({'data':{'sz000678':{'day':[row,row]}}},'sz000678')

    def test_unavailable_interval_keeps_independent_reference_and_original_signal(self):
        class Response:
            def __init__(self,text):
                self.text=text;self.content=text.encode();self.status_code=200;self.url='https://example.test'
            def raise_for_status(self):pass
            def json(self):return json.loads(self.text)
        payload={'data':{'sz000678':{'day':[['2026-09-30',10,11,11,10,8]]}}}
        signals=pd.DataFrame([dict(code='000678',date='2026-09-30',price=11.,daily_true_one_word=0.)])
        with tempfile.TemporaryDirectory(prefix='long_interval_') as temporary:
            target=Path(temporary)
            for kind in ['interval','daily']:(target/(kind+'_raw')).mkdir()
            with patch.object(engine.requests,'get',side_effect=[Response('var _data=(null);'),Response(json.dumps(payload))]):
                audits,frames,checks=engine.fetch('000678',signals,target)
            self.assertEqual(set(frames),{'daily'})
            self.assertEqual(len(checks),1);self.assertTrue(checks[0]['original_pool_close_agrees'])
            self.assertFalse(checks[0]['price_reconciled'])
            self.assertEqual([r['status'] for r in audits],['UNAVAILABLE','RAW_PARSED'])


if __name__=='__main__':unittest.main()
