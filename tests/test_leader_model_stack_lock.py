import unittest

from complete_trade_dataset import TARGET_DEFINITION as TRADE_TARGET
from full_pool_rank_dataset import TARGET_DEFINITION as RANK_TARGET, SELECTION_SCOPE
from full_pool_rank_blind_vault import _canonical_hash
from leader_model_stack_lock import assemble


A, B, C = "a" * 64, "b" * 64, "c" * 64


def fixture():
    rank_dataset = {"status": "FULL_POOL_RANK_DATASET_READY", "selection_scope": SELECTION_SCOPE,
                    "target_definition": RANK_TARGET, "actual_trade_target": False,
                    "feature_allowlist": ["boards", "emotion"]}
    rank_split = {"status": "READY_FOR_FULL_POOL_RANK_DEVELOPMENT"}
    rank_model = {"generated_at": "2026-01-01T10:00:00+08:00", "model_artifact_sha256": A,
                  "rank_score_named_probability": False}
    rank_blind = {"evaluated_at": "2026-01-03T10:00:00+08:00",
                  "status": "BLIND_FULL_POOL_RANK_GATE_PASSED",
                  "blind_labels_untouched_until_evaluation": True,
                  "blind_label_access_control": "SEPARATE_FEATURE_AND_LABEL_VAULT",
                  "blind_session_count": 100, "blind_top1_proxy_rate": .65,
                  "blind_baseline_proxy_rate": .50, "baseline_frozen_before_blind": True,
                  "first_and_relay_reported": True}
    trade_dataset = {"status": "POLICY_CONDITIONAL_DATASET_READY", "policy_id": "p1",
                     "selection_scope": SELECTION_SCOPE, "target_definition": TRADE_TARGET,
                     "generalization_scope": "FROZEN_SELECTION_POLICY_CONDITIONAL_ONLY",
                     "feature_allowlist": ["x"]}
    trade_split = {"status": "READY_FOR_MODEL_DEVELOPMENT"}
    model_result = {"chosen": "regularized_logistic", "candidates": {"regularized_logistic": {"n": 40}}}
    support = {"x": {"minimum": 0, "maximum": 1}}
    trade_model = {"generated_at": "2026-01-01T10:00:00+08:00", "model_artifact_sha256": B,
                   "policy_id": "p1", "profit_model": model_result, "risk_model": model_result,
                   "feature_support_envelope": support, "out_of_distribution_gate_frozen": True}
    threshold = {"status": "VALIDATION_THRESHOLD_FROZEN", "frozen_at": "2025-12-31T10:00:00+08:00",
                 "validated_at": "2026-01-02T10:00:00+08:00", "grid_frozen_before_model_output": True,
                 "selection_policy_id": "p1", "validation_observations": 40,
                 "validation_selected_trades": 35, "validation_win_rate": .72,
                 "validation_large_loss_rate": .08, "validation_mean_net_return": .01,
                 "active_weekly_signals_within_3_to_5": True}
    trade_blind = {"evaluated_at": "2026-01-03T11:00:00+08:00",
                   "status": "BLIND_COMPLETE_TRADE_GATE_PASSED", "policy_id": "p1",
                   "execution_basis_verified": True, "blind_labels_untouched_until_evaluation": True,
                   "blind_label_access_control": "SEPARATE_FEATURE_AND_LABEL_VAULT",
                   "outperforms_baseline": True,
                   "metrics": {"completed_trades": 120, "win_rate": .72, "big_loss_rate": .08,
                               "mean_net_return": .012, "profit_factor": 1.4}}
    schema = {"rank_features": rank_dataset["feature_allowlist"],
              "trade_features": trade_dataset["feature_allowlist"], "trade_support": support}
    hashes = {"rank_model_artifact": A, "trade_model_artifact": B,
              "threshold_report": C, "feature_schema": _canonical_hash(schema)}
    return [rank_dataset, rank_split, rank_model, rank_blind, trade_dataset, trade_split,
            trade_model, threshold, trade_blind, hashes, "2026-01-04T10:00:00+08:00"]


class LeaderModelStackLockTests(unittest.TestCase):
    def test_assembles_ready_but_not_deployed_stack(self):
        result = assemble(*fixture())
        self.assertEqual(result["status"], "MODEL_STACK_READY_FOR_LOCKED_BLIND_PAPER")
        self.assertFalse(result["deployment_allowed"])
        self.assertEqual(result["evidence"]["policy_conditional_trade_models"]["selection_policy_id"], "p1")

    def test_policy_id_mismatch_is_rejected(self):
        args = fixture()
        args[7]["selection_policy_id"] = "other"
        with self.assertRaises(ValueError):
            assemble(*args)

    def test_component_after_lock_is_rejected(self):
        args = fixture()
        args[3]["evaluated_at"] = "2026-01-05T10:00:00+08:00"
        with self.assertRaises(ValueError):
            assemble(*args)

    def test_feature_schema_hash_must_match(self):
        args = fixture()
        args[9]["feature_schema"] = "d" * 64
        with self.assertRaises(ValueError):
            assemble(*args)

    def test_failed_rank_blind_keeps_stack_diagnostic(self):
        args = fixture()
        args[3]["status"] = "BLIND_FULL_POOL_RANK_GATE_FAILED"
        result = assemble(*args)
        self.assertEqual(result["status"], "DIAGNOSTIC_ONLY")
        self.assertIn("rank_purged_blind", result["stack_audit"]["failed_gates"])

    def test_trade_target_cannot_be_substituted(self):
        args = fixture()
        args[4]["target_definition"] = "mark_to_market"
        result = assemble(*args)
        self.assertIn("trade_target_actual_complete", result["stack_audit"]["failed_gates"])
