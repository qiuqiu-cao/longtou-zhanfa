import unittest

import pandas as pd

from build_current_review_snapshot_v52 import build


class CurrentReviewSnapshotV52Tests(unittest.TestCase):
    def fixtures(self):
        closing = pd.DataFrame([{
            "signal_date": "2026-09-08", "code": "603626", "name": "科森科技",
            "boards": 2, "first_seal_time": 93311, "last_seal_time": None,
            "failed_seal_count": 12, "close": 21.69, "amount": 1_000_000,
            "turnover": 10.7, "provider_industry": "消费电子",
            "classification_status": "RECONCILED_QUOTE_CLOSE_OVERRIDES_FAILED_POOL",
        }])
        failed = pd.DataFrame([{
            "code": "000001", "name": "平安银行", "close": 10, "pct": 1,
            "amount_wan": 100, "turnover": 2,
            "classification_status": "INFERRED_TOUCHED_LIMIT_FROM_DAILY_HIGH_TWO_SOURCE_WHERE_AVAILABLE",
        }])
        manifest = {"signal_date": "2026-09-08", "closing_count": 1, "failed_count": 1,
                    "created_at": "2026-09-08T20:00:00+08:00",
                    "complete_within_archived_roster": True}
        raw_failed = pd.DataFrame([{"代码": "603626", "首次封板时间": 93311,
                                    "炸板次数": 12, "涨停统计": "2/2", "所属行业": "消费电子"}])
        sse = pd.DataFrame([{"A_STOCK_CODE": "603626", "SEC_NAME_CN": "科森科技",
                             "LIST_DATE": "20140210"}])
        szse = pd.DataFrame([{
            "code": "000001", "name": "平安银行", "valid_from": "2026-09-08",
            "valid_to": "2026-09-08", "exchange": "SZSE", "board": "MAINBOARD",
            "listing_status": "LISTED", "st_status": "NORMAL",
            "price_limit_regime": "MAINBOARD_10", "source_type": "szse_official",
            "source_ref": "official", "source_timestamp": "2026-09-08T12:00:00+00:00",
            "data_status": "verified_import",
        }])
        return closing, failed, manifest, raw_failed, sse, szse

    def test_reconciled_close_remains_closing_and_security_is_verified(self):
        frame, report = build(*self.fixtures(), "2026-09-08", "2026-09-08T12:00:00+00:00")
        close = frame[frame.pool_assignment.eq("CLOSING_LIMIT_UP")].iloc[0]
        self.assertEqual(close.code, "603626")
        self.assertEqual(close.boards, 2)
        self.assertEqual(close.eligibility_status, "VERIFIED_ELIGIBLE")
        self.assertEqual(report["mainboard_board_level_counts"], {"2": 1})
        self.assertTrue(report["main_and_failed_pools_disjoint"])

    def test_overlap_is_rejected(self):
        closing, failed, manifest, raw_failed, sse, szse = self.fixtures()
        failed.loc[0, "code"] = "603626"
        with self.assertRaisesRegex(ValueError, "overlap"):
            build(closing, failed, manifest, raw_failed, sse, szse,
                  "2026-09-08", "2026-09-08T12:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
