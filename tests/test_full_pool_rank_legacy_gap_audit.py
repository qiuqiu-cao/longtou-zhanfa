import unittest
import pandas as pd

from full_pool_rank_legacy_gap_audit import FEATURES, audit


def frames():
    base = {feature: 1 for feature in FEATURES}
    features = pd.DataFrame([{**base, "code": "1", "signal_date": "2025-01-02", "boards": 1,
                              "board_group": "FIRST_BOARD",
                              "feature_available_not_before": "2025-01-02T15:05:00+08:00",
                              "feature_status": "historical_reconstruction_not_forward_archive",
                              "pool_scope": "available_nonst_mainboard_history_not_complete_market"},
                             {**base, "code": "600001", "signal_date": "2025-01-03", "boards": 2,
                              "board_group": "CONSECUTIVE_BOARD",
                              "feature_available_not_before": "2025-01-03T15:05:00+08:00",
                              "feature_status": "historical_reconstruction_not_forward_archive",
                              "pool_scope": "available_nonst_mainboard_history_not_complete_market"}])
    outcomes = pd.DataFrame([{"code": "1", "signal_date": "2025-01-02",
                              "exit_date_LABEL_ONLY": "2025-01-06",
                              "price_path_state": "PRICE_OPPORTUNITY_PROXY_ONLY",
                              "net_close_mark_LABEL_ONLY": None, "entry_one_price_LABEL_ONLY": False},
                             {"code": "600001", "signal_date": "2025-01-03",
                              "exit_date_LABEL_ONLY": "2025-01-07",
                              "price_path_state": "PRICE_OPPORTUNITY_PROXY_ONLY",
                              "net_close_mark_LABEL_ONLY": .03, "entry_one_price_LABEL_ONLY": False}])
    manifest = {"status": "DIAGNOSTIC_ONLY", "historical_reconstruction_only": True,
                "original_point_in_time_capture_available": False}
    return features, outcomes, manifest


class LegacyGapAuditTests(unittest.TestCase):
    def test_reports_salvageable_proxy_without_claiming_model_readiness(self):
        result = audit(*frames())
        self.assertEqual(result["matched_rows"], 2)
        self.assertEqual(result["target_proxy_counts"]["resolved"], 2)
        self.assertEqual(result["target_proxy_counts"]["positive"], 1)
        self.assertTrue(result["usable_for_exploratory_proxy_research"])
        self.assertFalse(result["usable_for_locked_model_stack"])
        self.assertFalse(result["complete_trade_evidence"])

    def test_no_touch_without_mark_is_resolved_negative(self):
        result = audit(*frames())
        self.assertEqual(result["target_proxy_counts"]["negative"], 1)

    def test_future_feature_timestamp_is_counted_not_hidden(self):
        features, outcomes, manifest = frames()
        features.loc[0, "feature_available_not_before"] = "2025-01-03T09:00:00+08:00"
        result = audit(features, outcomes, manifest)
        self.assertEqual(result["feature_time_at_or_before_close_rows"], 1)

    def test_duplicate_key_is_rejected(self):
        features, outcomes, manifest = frames()
        features = pd.concat([features, features.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            audit(features, outcomes, manifest)

    def test_missing_required_feature_is_rejected(self):
        features, outcomes, manifest = frames()
        features = features.drop(columns=[FEATURES[0]])
        with self.assertRaises(ValueError):
            audit(features, outcomes, manifest)

    def test_board_group_can_be_derived_from_integer_height(self):
        features, outcomes, manifest = frames()
        result = audit(features.drop(columns=["board_group"]), outcomes, manifest)
        self.assertEqual(result["board_group_source"], "derived_from_integer_boards")
        self.assertEqual(result["valid_board_and_group_rows"], 2)

    def test_inventory_reconciliation_exposes_left_censoring(self):
        features, outcomes, manifest = frames()
        inventory = features[["code", "signal_date", "boards", "board_group"]].copy()
        inventory.loc[1, "boards"] = None
        result = audit(features, outcomes, manifest, inventory)
        self.assertEqual(result["inventory_reconciliation"]["inventory_left_censored_but_feature_height_assigned"], 1)
        self.assertIn("LEFT_CENSORED_IDENTITY_ASSIGNED_IN_LEGACY_FEATURES", result["blocking_gaps"])
