import unittest
from pathlib import Path
import json
import tempfile
import pandas as pd
from archive_signal_minutes_v286 import reconcile, expected_minutes


class MinuteReconciliationTests(unittest.TestCase):
    def fixture(self):
        # Explicitly synthetic, incomplete two-bar fixture; no outcome labels.
        bars=pd.DataFrame([
            dict(code="000678",minute="2026-09-30 09:31:00",open=10.,high=11.,low=10.,close=10.5,volume_shares=100.,amount=1050.),
            dict(code="000678",minute="2026-09-30 15:00:00",open=10.5,high=11.,low=10.5,close=11.,volume_shares=100.,amount=1100.)])
        daily=pd.DataFrame([dict(symbol="sz000678",date="2026-09-30",open=10.,high=11.,low=10.,close=11.,volume=2.)])
        return bars,daily

    def check(self,bars,daily):
        return reconcile(bars,daily,"2026-09-30","000678")

    def test_ohlc_agreement_does_not_imply_complete_minutes_or_auction(self):
        bars,daily=self.fixture();result=self.check(bars,daily)
        self.assertEqual(result["status"],"PRICE_RECONCILED_MINUTE_GAPS_RETAINED")
        self.assertEqual(len(result["missing_minutes"]),236)
        self.assertFalse(result["minute_grid_complete"])
        self.assertFalse(result["auction_0925_available"])
        self.assertFalse(result["eligible_for_original_frozen_forecast"])

    def test_missing_amount_cannot_produce_vwap(self):
        bars,daily=self.fixture();bars.loc[0,"amount"]=float("nan")
        self.assertIsNone(self.check(bars,daily)["vwap_over_observed_bars"])

    def test_price_disagreement_rejects_price_evidence(self):
        bars,daily=self.fixture();daily.loc[0,"close"]=11.1
        result=self.check(bars,daily)
        self.assertEqual(result["status"],"RECONCILIATION_FAILED")
        self.assertIsNone(result["vwap_over_observed_bars"])

    def test_duplicate_or_out_of_session_timestamp_is_not_complete(self):
        bars,daily=self.fixture()
        doubled=pd.concat([bars,bars.iloc[[0]]],ignore_index=True)
        self.assertEqual(self.check(doubled,daily)["status"],"RECONCILIATION_FAILED")
        bars.loc[0,"minute"]="2026-09-30 09:25:00"
        self.assertEqual(self.check(bars,daily)["status"],"RECONCILIATION_FAILED")

    def test_amount_volume_inconsistency_blocks_vwap(self):
        bars,daily=self.fixture();bars.loc[0,"amount"]=99999.
        result=self.check(bars,daily)
        self.assertFalse(result["amount_volume_implied_price_within_bar_bounds"])
        self.assertIsNone(result["vwap_over_observed_bars"])

    def test_volume_disagreement_keeps_explicit_inferred_unit_status(self):
        bars,daily=self.fixture();daily.loc[0,"volume"]=20.
        result=self.check(bars,daily)
        self.assertFalse(result["volume_agrees_at_inferred_scale_100"])
        self.assertEqual(result["volume_to_daily_raw_scale_inferred"],10.)
        self.assertFalse(result["actual_fill_verified"])

    def test_other_session_bars_do_not_change_signal_day_measurements(self):
        bars,daily=self.fixture();original=self.check(bars,daily)
        future=bars.copy();future["minute"]=future.minute.str.replace("2026-09-30","2026-10-08")
        future["amount"]=1e12
        self.assertEqual(self.check(pd.concat([bars,future],ignore_index=True),daily),original)

    def test_closing_auction_grid_is_explicit(self):
        grid=expected_minutes("2026-09-30")
        self.assertEqual(len(grid),238)
        self.assertIn("15:00",grid)
        self.assertNotIn("14:58",grid)
        self.assertNotIn("09:25",grid)


class RawMinuteAuditTests(unittest.TestCase):
    def test_real_archive_matches_raw_bytes_and_all_candidates(self):
        from audit_minute_archive_v286 import run
        root=Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            result=run(root/"data/v286/frozen_pool_minutes_20261002_v1",
                root/"data/v271/expectation_registry/2026-09-30_2c74dc406eef7dcf.json",
                Path(directory)/"audit")
        self.assertEqual(result["candidate_denominator"],11)
        self.assertEqual(result["volume_mismatch_codes"],["603188","603200"])
        self.assertFalse(result["accuracy_claim_allowed"])

    def test_edited_volume_claim_is_rejected_even_when_raw_hashes_match(self):
        from audit_minute_archive_v286 import run
        root=Path(__file__).resolve().parent
        original=root/"data/v286/frozen_pool_minutes_20261002_v1"
        report=json.loads((original/"report.json").read_text(encoding="utf-8"))
        row=next(r for r in report["candidate_checks"] if r["code"]=="603188")
        row["volume_agrees_at_inferred_scale_100"]=True
        with tempfile.TemporaryDirectory() as directory:
            modified=Path(directory)/"changed";modified.mkdir()
            (modified/"report.json").write_text(json.dumps(report),encoding="utf-8")
            (modified/"minute_bars_UNRECONCILED.csv").write_bytes((original/"minute_bars_UNRECONCILED.csv").read_bytes())
            with self.assertRaisesRegex(ValueError,"differ from reconstructed"):
                run(modified,root/"data/v271/expectation_registry/2026-09-30_2c74dc406eef7dcf.json",
                    Path(directory)/"audit")


if __name__=="__main__":unittest.main()
