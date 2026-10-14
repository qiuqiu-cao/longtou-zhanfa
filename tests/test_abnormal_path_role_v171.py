import unittest

import pandas as pd

from abnormal_path_role_v171 import build


ASOF = "2026-09-17T16:00:00+08:00"


def candidates():
    return pd.DataFrame([
        {"signal_date": "2026-09-17", "code": "600001", "name": "HIGH",
         "boards": 4, "pool_assignment": "CLOSING_LIMIT_UP"},
        {"signal_date": "2026-09-17", "code": "000001", "name": "CLOSE",
         "boards": 2, "pool_assignment": "CLOSING_LIMIT_UP"},
        {"signal_date": "2026-09-17", "code": "000002", "name": "FIRST",
         "boards": 1, "pool_assignment": "CLOSING_LIMIT_UP"},
    ])


def evidence():
    base = {
        "signal_date": "2026-09-17", "available_at": "2026-09-17T15:30:00+08:00",
        "rule_status": "verified", "benchmark_status": "official_reconciled",
        "reset_status": "verified", "corporate_action_status": "verified",
        "security_status": "verified", "coverage_complete": True,
        "indicator_status": "verified", "distance_200_pp": 100,
        "scenario_limit_boards_to_100": 2,
        "scenario_status": "verified_inputs_counterfactual",
        "active_theme_path_status": "verified",
        "competition_identity_status": "verified", "theme_support_ex_self": 2,
        "same_height_rivals": 0,
    }
    return pd.DataFrame([
        dict(base, code="600001", rule_id="SSE_MAIN_20260706",
             benchmark_code="sh000002", distance_100_pp=20,
             competition_identity="height_anchor"),
        dict(base, code="000001", rule_id="SZSE_MAIN_20260706",
             benchmark_code="sz399107", distance_100_pp=1,
             competition_identity="supplement"),
        dict(base, code="000002", rule_id="SZSE_MAIN_20260706",
             benchmark_code="sz399107", distance_100_pp=30,
             competition_identity="pioneer"),
    ])


class AbnormalPathRoleTests(unittest.TestCase):
    def test_closest_distance_and_path_potential_are_separate(self):
        report = build(candidates(), evidence(), ASOF)
        self.assertEqual(report["rule_distance_closest"]["code"], "000001")
        self.assertEqual(report["hundred_percent_abnormal_path_potential"]["code"], "600001")
        self.assertFalse(report["distance_is_positive_ranking_factor"])
        self.assertFalse(report["buy_signal"])
        self.assertIsNone(report["calibrated_probability"])

    def test_missing_reset_removes_both_verified_distance_and_path_role(self):
        frame = evidence()
        frame["reset_status"] = "unavailable"
        report = build(candidates(), frame, ASOF)
        self.assertIsNone(report["rule_distance_closest"])
        self.assertIsNone(report["hundred_percent_abnormal_path_potential"])
        self.assertEqual(report["verified_distance_count"], 0)

    def test_first_board_cannot_be_path_role_from_distance_alone(self):
        frame = evidence()
        frame["distance_100_pp"] = frame["distance_100_pp"].astype(float)
        frame.loc[frame.code.eq("000002"), "distance_100_pp"] = 0.1
        frame.loc[frame.code.eq("000002"), "competition_identity"] = "height_anchor"
        report = build(candidates(), frame, ASOF)
        self.assertEqual(report["rule_distance_closest"]["code"], "000002")
        self.assertNotEqual(report["hundred_percent_abnormal_path_potential"]["code"], "000002")

    def test_future_evidence_is_rejected(self):
        frame = evidence()
        frame["available_at"] = "2026-09-17T16:01:00+08:00"
        with self.assertRaisesRegex(ValueError, "after asof"):
            build(candidates(), frame, ASOF)

    def test_missing_evidence_fails_closed_for_every_candidate(self):
        report = build(candidates(), None, ASOF)
        self.assertEqual(len(report["candidate_records"]), 3)
        self.assertTrue(all(row["diagnostic"]["status"] == "UNAVAILABLE"
                            for row in report["candidate_records"]))
        self.assertIsNone(report["hundred_percent_abnormal_path_potential"])
        self.assertFalse(report["plain_price_doubling_is_sufficient"])


if __name__ == "__main__":
    unittest.main()
