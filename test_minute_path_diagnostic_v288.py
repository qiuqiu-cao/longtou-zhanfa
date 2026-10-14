import unittest
import json
import shutil
import tempfile
from pathlib import Path
import pandas as pd
from archive_september_5min_v287 import expected_times
from minute_path_diagnostic_v288 import features, pick, evaluate_picks
from minute_path_diagnostic_v288 import ROOT, sha
from audit_minute_path_diagnostic_v288 import run as audit_run


class MinutePathCausalTests(unittest.TestCase):
    def fixture(self):
        bars = pd.DataFrame([dict(code='000678',minute='2026-09-30 '+t+':00',
            open=10.,high=11.,low=10.,close=10.5,volume_shares=100.,amount=1050.)
            for t in sorted(expected_times('2026-09-30'))])
        bars.loc[bars.index[-3:],['open','high','low','close']] = 11.
        bars.loc[bars.index[-3:],'amount'] = 1100.
        daily = pd.DataFrame([dict(symbol='sz000678',date='2026-09-30',open=10.,close=11.,
            high=11.,low=10.,volume=48.)])
        return bars,daily

    def test_observed_bar_run_does_not_invent_continuous_seal_or_reseal_order(self):
        bars,daily = self.fixture()
        r = features(bars,daily,'000678','2026-09-30',11.)
        self.assertEqual(r['final_cap_price_bar_run'],3)
        self.assertFalse(r['continuous_seal_verified'])
        self.assertIsNone(r['real_reseal_count'])
        self.assertFalse(r['actual_fill_verified'])

    def test_future_path_changes_cannot_change_signal_features(self):
        bars,daily = self.fixture()
        original = features(bars,daily,'000678','2026-09-30',11.)
        future = bars.copy();future.minute = future.minute.str.replace('2026-09-30','2026-10-08')
        future[['open','high','low','close','amount']] = 1000000.
        self.assertEqual(features(pd.concat([bars,future]),daily,'000678','2026-09-30',11.),original)

    def test_one_missing_interval_blocks_full_path_even_when_daily_price_matches(self):
        bars,daily = self.fixture()
        r = features(bars.drop(index=7),daily,'000678','2026-09-30',11.)
        self.assertTrue(r['price_reconciled'])
        self.assertFalse(r['price_path_complete'])
        self.assertIsNone(r['final_cap_price_bar_run'])

    def test_volume_mismatch_keeps_price_path_but_blocks_full_day_vwap(self):
        bars,daily = self.fixture();daily.loc[0,'volume'] = 200.
        r = features(bars,daily,'000678','2026-09-30',11.)
        self.assertTrue(r['price_path_complete'])
        self.assertIsNone(r['full_day_vwap'])

    def ranks(self):
        return pd.DataFrame([
            dict(date='2026-09-30',code='000678',boards=4,probability=.8,daily_true_one_word=0.,
                price_path_complete=True,final_cap_price_bar_run=1,cap_price_bar_share=.1,negative_max_pullback=-.1),
            dict(date='2026-09-30',code='600825',boards=7,probability=.3,daily_true_one_word=0.,
                price_path_complete=True,final_cap_price_bar_run=40,cap_price_bar_share=.8,negative_max_pullback=-.02)])

    def test_labels_and_future_prices_do_not_change_ranking(self):
        frame = self.ranks()
        before = pick(frame,'TRUE_NONWORD','FINAL_CAP_PRICE_RUN')
        frame['target'] = [1,0];frame['next_open_gap'] = [100,-100];frame['label_valid'] = [False,True]
        pd.testing.assert_frame_equal(before,pick(frame,'TRUE_NONWORD','FINAL_CAP_PRICE_RUN'))
        self.assertEqual(before.code.iloc[0],'600825')

    def test_partial_candidate_path_falls_back_entire_date_not_good_data_subset(self):
        frame = self.ranks();frame.loc[0,'price_path_complete'] = False
        result = pick(frame,'TRUE_NONWORD','FINAL_CAP_PRICE_RUN')
        self.assertEqual(result.code.iloc[0],'000678')
        self.assertFalse(result.path_rule_used.iloc[0])
        self.assertEqual(result.fallback_reason.iloc[0],'ENTIRE_POOL_PATH_INCOMPLETE')

    def test_other_candidate_missing_label_blocks_scoring_even_if_picked_label_known(self):
        picks = pick(self.ranks(),'TRUE_NONWORD','CONTROL')
        labels = pd.DataFrame([dict(date='2026-09-30',code='000678',label_valid=True,target=1.),
            dict(date='2026-09-30',code='600825',label_valid=False,target=float('nan'))])
        scored = evaluate_picks(picks,labels)
        self.assertEqual(scored.target.iloc[0],1.)
        self.assertFalse(scored.evaluated.iloc[0])

    def test_candidate_input_order_cannot_change_tied_choice(self):
        frame = self.ranks()
        frame.loc[:,'final_cap_price_bar_run'] = 1
        pd.testing.assert_frame_equal(pick(frame,'ALL_RELAY','FINAL_CAP_PRICE_RUN'),
            pick(frame.iloc[::-1],'ALL_RELAY','FINAL_CAP_PRICE_RUN'))


class ArchivedMinutePathIntegrityTests(unittest.TestCase):
    def copy_archive(self,target):
        source = ROOT/'results/v288/minute_path_diagnostic_20261002_v1'
        for name in ['specification.json','report.json','features_NO_NEXT_SESSION_LABELS.csv',
                'ranked_choices_BEFORE_LABEL_JOIN.csv','scored_choices_LABEL_ONLY.csv']:
            shutil.copyfile(source/name,target/name)

    def test_forged_summary_is_rejected_even_when_input_hashes_still_match(self):
        with tempfile.TemporaryDirectory(prefix='minute_path_audit_') as temporary:
            directory = Path(temporary);self.copy_archive(directory)
            report = json.loads((directory/'report.json').read_text(encoding='utf-8'))
            report['summaries']['TRUE_NONWORD']['SHALLOW_PULLBACK']['successes'] = 13
            (directory/'report.json').write_text(json.dumps(report),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'summary does not match'):
                audit_run(directory,directory/'audit')
            self.assertFalse((directory/'audit').exists())

    def test_rehashed_forged_feature_is_rejected_by_price_path_reconstruction(self):
        with tempfile.TemporaryDirectory(prefix='minute_path_audit_') as temporary:
            directory = Path(temporary);self.copy_archive(directory)
            path = directory/'features_NO_NEXT_SESSION_LABELS.csv'
            frame = pd.read_csv(path,dtype={'code':str})
            frame.loc[0,'final_cap_price_bar_run'] = 999
            frame.to_csv(path,index=False,encoding='utf-8-sig')
            spec_path = directory/'specification.json'
            spec = json.loads(spec_path.read_text(encoding='utf-8'))
            spec['artifact_hashes'][path.name] = sha(path)
            spec_path.write_text(json.dumps(spec),encoding='utf-8')
            report = json.loads((directory/'report.json').read_text(encoding='utf-8'))
            report['source_hashes'] = {str(spec_path.resolve()):sha(spec_path)}
            (directory/'report.json').write_text(json.dumps(report),encoding='utf-8')
            with self.assertRaises(AssertionError):
                audit_run(directory,directory/'audit')
            self.assertFalse((directory/'audit').exists())


if __name__ == '__main__': unittest.main()
