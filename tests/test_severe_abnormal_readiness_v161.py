import unittest

import pandas as pd

from severe_abnormal_readiness_v161 import audit


def benchmark(status="official_reconciled"):
    dates = pd.bdate_range("2026-06-01", periods=40).strftime("%Y-%m-%d")
    return pd.DataFrame({"date": dates, "close": range(100, 140), "data_status": status})


def library():
    return {
        "candidate_universe_status": "verified_full_market",
        "reset_chronology_status": "verified_complete",
        "reset_events": [{
            "publication_date": "2026-07-01", "reset_effective_date": "2026-07-02",
            "status": "verified",
        }],
        "events": [{
            "code": "600001", "event_date": "2026-06-26", "signal_date": "2026-06-26",
            "target_pct": 100, "official_confirmed": True, "training_label_ready": True,
            "label": 1, "candidate_group": "FIRST_BOARD",
            "feature_available_at": "2026-06-26T15:05:00+08:00",
            "outcome_observed_at": "2026-07-01T15:05:00+08:00",
            "target_horizon_sessions": 3, "right_censored": False,
        }],
        "negative_labels": [{
            "code": "600002", "event_date": "2026-06-25", "signal_date": "2026-06-25",
            "training_label_ready": True, "label": 0,
            "candidate_group": "CONSECUTIVE_BOARD",
            "feature_available_at": "2026-06-25T15:05:00+08:00",
            "outcome_observed_at": "2026-06-30T15:05:00+08:00",
            "target_horizon_sessions": 3, "right_censored": False,
        }],
    }


def daily(include_adjustment=True):
    row = {
        "date": "2026-06-26", "code": "600001", "status_verified": True,
        "st": False, "new_listing": False, "universe_complete": True,
        "price_limit_status": "verified_limit_applicable",
    }
    if include_adjustment:
        row["adjustment_status"] = "official_reconciled"
    return pd.DataFrame([row])


class SevereAbnormalReadinessV161Tests(unittest.TestCase):
    def test_synthetic_complete_inputs_pass_schema_and_support_gates(self):
        report = audit(
            library(), {"sh000002": benchmark()}, daily(),
            minimum_positive_labels=1, minimum_negative_labels=1,
            minimum_sessions=2, minimum_group_labels=1,
        )
        self.assertEqual(report["status"], "READY_FOR_TIME_ISOLATED_CALIBRATION")
        self.assertTrue(report["calibrated_hit_probability_allowed"])
        self.assertFalse(report["missing_values_treated_as_zero"])

    def test_provider_only_benchmark_blocks_probability_but_not_path_diagnostic(self):
        report = audit(
            library(), {"sh000002": benchmark("provider_observed_not_official_reconciled")},
            daily(), minimum_positive_labels=1, minimum_negative_labels=1,
            minimum_sessions=2, minimum_group_labels=1,
        )
        self.assertFalse(report["calibrated_hit_probability_allowed"])
        self.assertTrue(report["diagnostic_counterfactual_path_rating_allowed"])
        self.assertIn("benchmark_official_reconciliation", report["failed_gates"])

    def test_missing_corporate_action_field_fails_closed(self):
        report = audit(
            library(), {"sh000002": benchmark()}, daily(include_adjustment=False),
            minimum_positive_labels=1, minimum_negative_labels=1,
            minimum_sessions=2, minimum_group_labels=1,
        )
        self.assertFalse(report["calibrated_hit_probability_allowed"])
        self.assertIn("stock_corporate_action_treatment", report["failed_gates"])

    def test_missing_reset_chronology_fails_closed(self):
        payload = library()
        payload.pop("reset_events")
        report = audit(
            payload, {"sh000002": benchmark()}, daily(),
            minimum_positive_labels=1, minimum_negative_labels=1,
            minimum_sessions=2, minimum_group_labels=1,
        )
        self.assertIn("official_publication_reset_chronology", report["failed_gates"])


if __name__ == "__main__":
    unittest.main()
