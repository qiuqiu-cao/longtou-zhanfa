import copy
import unittest

from complete_trade_dataset import TARGET_DEFINITION as TRADE_TARGET
from full_pool_rank_dataset import TARGET_DEFINITION as RANK_TARGET, SELECTION_SCOPE
from leader_model_stack_audit import audit


SHA = "a" * 64


def evidence():
    return {
        "full_pool_ranker": {"dataset_ready": True, "selection_scope": SELECTION_SCOPE,
                             "target_definition": RANK_TARGET, "actual_trade_target": False,
                             "purged_split_ready": True, "blind_labels_untouched_until_evaluation": True,
                             "blind_label_access_control": "SEPARATE_FEATURE_AND_LABEL_VAULT",
                             "blind_evaluation_status": "BLIND_FULL_POOL_RANK_GATE_PASSED",
                             "blind_session_count": 120, "blind_top1_proxy_rate": .62,
                             "blind_baseline_proxy_rate": .51, "baseline_frozen_before_blind": True,
                             "first_and_relay_reported": True,
                             "frozen_at": "2026-01-01T10:00:00+08:00"},
        "policy_conditional_trade_models": {"complete_trade_dataset_ready": True,
                             "purged_split_ready": True, "selection_scope": SELECTION_SCOPE,
                             "target_definition": TRADE_TARGET, "execution_basis_verified": True,
                             "generalization_scope": "FROZEN_SELECTION_POLICY_CONDITIONAL_ONLY",
                             "profit_model_calibrated": True, "large_loss_model_calibrated": True,
                             "validation_complete_trades": 40, "out_of_distribution_gate_frozen": True,
                             "blind_label_access_control": "SEPARATE_FEATURE_AND_LABEL_VAULT",
                             "blind_labels_untouched_until_evaluation": True,
                             "blind_evaluation_status": "BLIND_COMPLETE_TRADE_GATE_PASSED",
                             "blind_completed_trades": 120, "blind_win_rate": .72,
                             "blind_large_loss_rate": .08, "blind_mean_net_return": .012,
                             "blind_profit_factor": 1.4, "blind_outperforms_baseline": True,
                             "selection_policy_id": "policy-v1",
                             "frozen_at": "2026-01-01T10:00:00+08:00"},
        "frozen_threshold": {"grid_frozen_before_model_output": True, "status": "VALIDATION_THRESHOLD_FROZEN",
                             "validation_selected_trades": 35, "validation_win_rate": .72,
                             "validation_large_loss_rate": .08, "validation_mean_net_return": .015,
                             "active_weekly_signals_within_3_to_5": True,
                             "selection_policy_id": "policy-v1",
                             "frozen_at": "2025-12-31T10:00:00+08:00",
                             "validated_at": "2026-01-02T10:00:00+08:00"},
        "stack_lock": {"selection_policy_id": "policy-v1",
                       "frozen_at": "2026-01-03T10:00:00+08:00",
                       "rank_model_sha256": SHA, "profit_model_sha256": SHA,
                       "large_loss_model_sha256": SHA, "threshold_sha256": SHA,
                       "feature_schema_sha256": SHA, "rank_score_named_probability": False,
                       "complete_trade_probability_from_rank_proxy": False},
    }


class LeaderModelStackAuditTests(unittest.TestCase):
    def test_valid_stack_can_only_feed_locked_paper(self):
        result = audit(evidence(), "2026-01-04T10:00:00+08:00")
        self.assertEqual(result["status"], "MODEL_STACK_READY_FOR_LOCKED_BLIND_PAPER")
        self.assertTrue(result["can_feed_daily_decision_contract"])
        self.assertFalse(result["live_trading_authorized"])
        self.assertFalse(result["guarantee"])

    def test_proxy_cannot_be_declared_actual_trade_target(self):
        item = evidence()
        item["full_pool_ranker"]["actual_trade_target"] = True
        result = audit(item, "2026-01-04T10:00:00+08:00")
        self.assertIn("rank_target_is_proxy", result["failed_gates"])

    def test_ranker_must_beat_prefrozen_baseline(self):
        item = evidence()
        item["full_pool_ranker"]["blind_top1_proxy_rate"] = .50
        result = audit(item, "2026-01-04T10:00:00+08:00")
        self.assertIn("rank_beats_frozen_baseline", result["failed_gates"])

    def test_trade_model_must_be_policy_conditional(self):
        item = evidence()
        item["policy_conditional_trade_models"]["generalization_scope"] = "ALL_POOL"
        result = audit(item, "2026-01-04T10:00:00+08:00")
        self.assertIn("trade_scope_policy_conditional", result["failed_gates"])

    def test_support_gate_is_mandatory(self):
        item = evidence()
        item["policy_conditional_trade_models"]["out_of_distribution_gate_frozen"] = False
        result = audit(item, "2026-01-04T10:00:00+08:00")
        self.assertIn("trade_support_gate", result["failed_gates"])

    def test_seventy_percent_is_exact_threshold_gate(self):
        item = evidence()
        item["frozen_threshold"]["validation_win_rate"] = .699
        result = audit(item, "2026-01-04T10:00:00+08:00")
        self.assertIn("threshold_win_gate", result["failed_gates"])

    def test_component_policy_ids_must_match(self):
        item = evidence()
        item["frozen_threshold"]["selection_policy_id"] = "other"
        result = audit(item, "2026-01-04T10:00:00+08:00")
        self.assertIn("common_selection_policy", result["failed_gates"])

    def test_hash_format_is_exact(self):
        item = evidence()
        item["stack_lock"]["threshold_sha256"] = "short"
        result = audit(item, "2026-01-04T10:00:00+08:00")
        self.assertIn("stack_components_hashed", result["failed_gates"])

    def test_freeze_cannot_be_after_use(self):
        item = copy.deepcopy(evidence())
        item["stack_lock"]["frozen_at"] = "2026-01-05T10:00:00+08:00"
        result = audit(item, "2026-01-04T10:00:00+08:00")
        self.assertIn("stack_time_sequence", result["failed_gates"])

    def test_rank_score_cannot_be_presented_as_probability(self):
        item = evidence()
        item["stack_lock"]["rank_score_named_probability"] = True
        result = audit(item, "2026-01-04T10:00:00+08:00")
        self.assertIn("no_probability_conflation", result["failed_gates"])
