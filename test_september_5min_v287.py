import unittest
import pandas as pd
from archive_september_5min_v287 import check_signal, expected_times


class FiveMinuteEvidenceTests(unittest.TestCase):
    def fixture(self):
        bars=pd.DataFrame([dict(code="000678",minute="2026-09-30 "+t+":00",
            open=10.,high=11.,low=10.,close=10.5,volume_shares=100.,amount=1050.)
            for t in sorted(expected_times("2026-09-30"))])
        bars.loc[bars.index[-1],"close"]=11.
        daily=pd.DataFrame([dict(symbol="sz000678",date="2026-09-30",open=10.,close=11.,
                                 high=11.,low=10.,volume=48.)])
        return bars,daily

    def check(self,bars,daily):return check_signal(bars,daily,"000678","2026-09-30")

    def test_complete_five_minute_grid_does_not_authorize_auction_or_tick_claims(self):
        bars,daily=self.fixture();r=self.check(bars,daily)
        self.assertEqual(r["interval_minutes"],5)
        self.assertTrue(r["grid_complete"])
        self.assertTrue(r["price_reconciled"])
        self.assertFalse(r["auction_verified"])
        self.assertFalse(r["tick_path_verified"])
        self.assertFalse(r["eligible_for_original_forward_forecast"])

    def test_missing_internal_bar_is_not_imputed_from_price_agreement(self):
        bars,daily=self.fixture();bars=bars.drop(index=8)
        r=self.check(bars,daily)
        self.assertTrue(r["price_reconciled"])
        self.assertFalse(r["grid_complete"])
        self.assertEqual(r["observed_bars"],47)
        self.assertEqual(len(r["missing_times"]),1)

    def test_first_partial_day_cannot_be_called_price_reconciled(self):
        bars,daily=self.fixture();bars=bars.iloc[10:].copy();bars.loc[bars.index[0],"open"]=10.2
        self.assertFalse(self.check(bars,daily)["price_reconciled"])

    def test_volume_mismatch_or_missing_amount_blocks_vwap(self):
        bars,daily=self.fixture();daily.loc[0,"volume"]=1000.
        self.assertIsNone(self.check(bars,daily)["vwap_observed_bars"])
        bars,daily=self.fixture();bars.loc[3,"amount"]=float("nan")
        self.assertIsNone(self.check(bars,daily)["vwap_observed_bars"])

    def test_duplicate_and_unexpected_time_fail_price_integrity(self):
        bars,daily=self.fixture();double=pd.concat([bars,bars.iloc[[4]]],ignore_index=True)
        self.assertFalse(self.check(double,daily)["price_reconciled"])
        bars.loc[3,"minute"]="2026-09-30 09:25:00"
        self.assertFalse(self.check(bars,daily)["price_reconciled"])

    def test_future_day_bars_do_not_change_signal_features(self):
        bars,daily=self.fixture();expected=self.check(bars,daily)
        future=bars.copy();future.minute=future.minute.str.replace("2026-09-30","2026-10-08")
        future.amount=1e12
        self.assertEqual(self.check(pd.concat([bars,future],ignore_index=True),daily),expected)


if __name__=="__main__":unittest.main()
