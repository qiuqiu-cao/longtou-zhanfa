import unittest

import pandas as pd

from theme_profit_coverage_audit_v197 import audit, build_rows


class ThemeProfitCoverageAuditV197Test(unittest.TestCase):
    def predictions(self):
        base = {
            "promotion_label": 0, "fill_status": "FILLED_OPEN_PROXY",
            "profit_target_available": 1, "executable_positive_B_plus_2": 0,
            "net_return_B_plus_2": -.02, "big_loss_B_plus_1": 0,
        }
        return pd.DataFrame([
            dict(base, signal_date="2026-01-05", code="A"),
            dict(base, signal_date="2026-01-05", code="B"),
        ])

    def test_missing_archive_row_stays_unknown_not_weak(self):
        archive = pd.DataFrame([{
            "signal_date": "2026-01-05", "code": "A",
            "theme_status": "verified_same_day_group",
            "point_in_time_available_for_next_session": True,
            "theme_factor_model_eligible": False,
            "full_market_completeness": "not_proven",
            "driver_claim_status": "inferred", "expectation_status": "unknown",
        }])
        rows = build_rows(self.predictions(), archive).set_index("code")
        self.assertEqual(rows.loc["A", "theme_roster_available_preopen"], 1)
        self.assertEqual(rows.loc["B", "theme_roster_available_preopen"], 0)
        self.assertEqual(rows.loc["B", "theme_missing_is_weak"], 0)

    def test_late_roster_is_not_point_in_time_available(self):
        archive = pd.DataFrame([{
            "signal_date": "2026-01-05", "code": "A",
            "theme_status": "verified_same_day_group",
            "point_in_time_available_for_next_session": False,
            "theme_factor_model_eligible": False,
            "full_market_completeness": "not_proven",
            "driver_claim_status": "verified", "expectation_status": "unknown",
        }, {
            "signal_date": "2026-01-05", "code": "B",
            "theme_status": "unavailable",
            "point_in_time_available_for_next_session": False,
            "theme_factor_model_eligible": False,
            "full_market_completeness": "not_proven",
            "driver_claim_status": "unavailable", "expectation_status": "unknown",
        }])
        rows = build_rows(self.predictions(), archive)
        self.assertEqual(rows.theme_roster_available_preopen.sum(), 0)

    def test_zero_model_eligible_rows_forbid_factor_and_entry(self):
        archive = pd.DataFrame([{
            "signal_date": "2026-01-05", "code": code,
            "theme_status": "verified_same_day_group" if code == "A" else "unavailable",
            "point_in_time_available_for_next_session": code == "A",
            "theme_factor_model_eligible": False,
            "full_market_completeness": "not_proven",
            "driver_claim_status": "inferred", "expectation_status": "unknown",
        } for code in ["A", "B"]])
        report = audit(build_rows(self.predictions(), archive), bootstrap_repetitions=20)
        self.assertFalse(report["governance"]["theme_roster_direct_profit_factor_allowed"])
        self.assertFalse(report["governance"]["direct_stock_reranking_allowed"])
        self.assertFalse(report["governance"]["entry_authorized"])


if __name__ == "__main__":
    unittest.main()
