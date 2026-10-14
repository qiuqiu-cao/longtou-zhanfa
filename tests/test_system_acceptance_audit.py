import copy
import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from system_acceptance_audit import audit, current_evidence


class AcceptanceAuditTests(unittest.TestCase):
    def passing(self):
        return dict(
            data=dict(trading_days=300,universe_complete=True,point_in_time_status_complete=True,
                      cross_theme_competition_ready=True,
                      theme_catalyst_archive_ready=True,
                      severe_abnormal_path_ready=True,
                      emotion_path_ready=True,
                      same_height_group_validation_ready=True,
                      first_board_dual_target_ready=True,
                      five_factor_history_ready=True,
                      all_observed_board_levels_reported=True,
                      board_observations={str(i):100 for i in range(1,6)}),
            blind_complete_trades=dict(complete_trade_dataset_ready=True,purged_split_ready=True,
                      development_model_ready=True,validation_threshold_frozen=True,can_rank_full_pool=True,
                      model_stack_compatible=True,
                      independent_blind_window=True,execution_basis_verified=True,
                      completed_trades=120,win_rate=.72,big_loss_rate=.08,mean_net_return=.01,
                      profit_factor=1.3,account_max_drawdown=-.12,frozen_max_drawdown_limit=.15,
                      outperforms_baseline=True,group_metrics={"FIRST_BOARD":{"completed_trades":50},
                                                               "CONSECUTIVE_BOARD":{"completed_trades":70}}),
            historical_execution_replay=dict(candidate_universe_coverage_complete=True,
                      candidate_universe_coverage_rate=1.0,
                      selection_coverage_status="COMPLETE_CANDIDATE_UNIVERSE", selected_days=100),
            frozen_forward=dict(frozen_before_period=True,no_retuning=True,completed_trades=60,
                      win_rate=.70,big_loss_rate=.10,mean_net_return=.005,
                      account_max_drawdown=-.10,frozen_max_drawdown_limit=.15))

    def test_all_gates_produce_forward_validated(self):
        result=audit(self.passing())
        self.assertEqual(result["status"],"FORWARD_VALIDATED")
        self.assertTrue(result["deployment_allowed"])

    def test_historical_only_produces_paper_ready(self):
        evidence=self.passing()
        evidence["frozen_forward"]["completed_trades"]=59
        result=audit(evidence)
        self.assertEqual(result["status"],"PAPER_TRADE_READY")
        self.assertFalse(result["deployment_allowed"])

    def test_missing_value_never_becomes_zero_pass(self):
        evidence=self.passing()
        evidence["blind_complete_trades"]["big_loss_rate"]=None
        result=audit(evidence)
        self.assertEqual(result["status"],"DIAGNOSTIC_ONLY")
        self.assertIn("large_loss_rate",result["failed_gates"])

    def test_point_estimate_cannot_replace_execution_evidence(self):
        evidence=self.passing()
        evidence["blind_complete_trades"]["execution_basis_verified"]=False
        result=audit(evidence)
        self.assertEqual(result["status"],"DIAGNOSTIC_ONLY")

    def test_incomplete_intraday_pool_must_fail_closed(self):
        evidence = self.passing()
        evidence["historical_execution_replay"] = dict(
            candidate_universe_coverage_complete=False,
            candidate_universe_coverage_rate=.4,
            selection_coverage_status="INCOMPLETE_DEVELOPMENT_OVERRIDE", selected_days=10)
        result = audit(evidence)
        self.assertIn("historical_full_pool_intraday_coverage", result["failed_gates"])
        self.assertIn("incomplete_intraday_universe_fails_closed", result["failed_gates"])
        evidence["historical_execution_replay"].update(
            selection_coverage_status="BLOCKED_INCOMPLETE_CANDIDATE_UNIVERSE", selected_days=0)
        result = audit(evidence)
        self.assertIn("historical_full_pool_intraday_coverage", result["failed_gates"])
        self.assertNotIn("incomplete_intraday_universe_fails_closed", result["failed_gates"])

    def test_full_pool_security_audit_can_upgrade_only_status_coverage_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "results" / "first_board_inventory_20260905").mkdir(parents=True)
            (root / "results" / "full_pool_outcomes_20260905").mkdir(parents=True)
            (root / "results" / "emotion_broken_feedback_20260905").mkdir(parents=True)
            (root / "results" / "v10" / "intraday_forward").mkdir(parents=True)
            (root / "results" / "v16" / "full_pool_security_audit_20260906").mkdir(parents=True)
            (root / "results" / "first_board_inventory_20260905" / "manifest.json").write_text(
                json.dumps({"candidate_universe_complete": False, "historical_reconstruction_only": True,
                            "candidate_rows": 2}), encoding="utf-8")
            pd.DataFrame({"boards": [1, 2]}).to_csv(
                root / "results" / "first_board_inventory_20260905" / "candidate_observations.csv", index=False)
            (root / "results" / "full_pool_outcomes_20260905" / "manifest.json").write_text(
                json.dumps({"complete_trade_count": 0}), encoding="utf-8")
            (root / "results" / "emotion_broken_feedback_20260905" / "report.json").write_text(
                json.dumps({"days": 300}), encoding="utf-8")
            (root / "results" / "v10" / "intraday_forward" / "intraday_forward_monitor.json").write_text(
                json.dumps({}), encoding="utf-8")
            (root / "results" / "v16" / "full_pool_security_audit_20260906" / "security_eligibility_audit.json").write_text(
                json.dumps({"candidate_rows": 2, "verified_coverage": 1.0, "eligible_coverage": 1.0}),
                encoding="utf-8")
            evidence, hashes = current_evidence(root)
            self.assertTrue(evidence["data"]["point_in_time_status_complete"])
            self.assertFalse(evidence["data"]["universe_complete"])
            self.assertTrue(any("full_pool_security_audit" in path for path in hashes))

    def test_matching_full_pool_audit_wins_over_newer_cutoff_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "results" / "first_board_inventory_20260905").mkdir(parents=True)
            (root / "results" / "full_pool_outcomes_20260905").mkdir(parents=True)
            (root / "results" / "emotion_broken_feedback_20260905").mkdir(parents=True)
            (root / "results" / "v10" / "intraday_forward").mkdir(parents=True)
            (root / "results" / "v17" / "full_pool_proxy_walkforward_cutoff_20260831_v5").mkdir(parents=True)
            for name in ("full_pool_security_audit_20260906", "full_pool_security_audit_cutoff_20260831"):
                (root / "results" / "v16" / name).mkdir(parents=True)
            (root / "results" / "first_board_inventory_20260905" / "manifest.json").write_text(
                json.dumps({"candidate_universe_complete": False, "historical_reconstruction_only": True,
                            "candidate_rows": 3}), encoding="utf-8")
            pd.DataFrame({"boards": [1, 2, 3]}).to_csv(
                root / "results" / "first_board_inventory_20260905" / "candidate_observations.csv", index=False)
            (root / "results" / "full_pool_outcomes_20260905" / "manifest.json").write_text(
                json.dumps({"complete_trade_count": 0}), encoding="utf-8")
            (root / "results" / "emotion_broken_feedback_20260905" / "report.json").write_text(
                json.dumps({"days": 300}), encoding="utf-8")
            (root / "results" / "v10" / "intraday_forward" / "intraday_forward_monitor.json").write_text(
                json.dumps({}), encoding="utf-8")
            (root / "results" / "v16" / "full_pool_security_audit_20260906" / "security_eligibility_audit.json").write_text(
                json.dumps({"candidate_rows": 3, "verified_coverage": 1.0, "eligible_coverage": 1.0}),
                encoding="utf-8")
            (root / "results" / "v16" / "full_pool_security_audit_cutoff_20260831" / "security_eligibility_audit.json").write_text(
                json.dumps({"candidate_rows": 2, "verified_coverage": 1.0, "eligible_coverage": 1.0}),
                encoding="utf-8")
            (root / "results" / "v17" / "full_pool_proxy_walkforward_cutoff_20260831_v5" / "report.json").write_text(
                json.dumps({"status": "EXPLORATORY_ONLY", "windows": [{"status": "OK"}],
                            "aggregate_selected": {"identity": {"summary": {
                                "selected_sessions": 1, "proxy_success_rate": .5,
                                "path_proxy": {"b2": {"positive_rate": .4, "large_loss_rate": .2}}}}}}),
                encoding="utf-8")
            evidence, hashes = current_evidence(root)
            self.assertTrue(evidence["data"]["point_in_time_status_complete"])
            self.assertEqual(evidence["research_proxy"]["walkforward_folds"], 1)
            self.assertEqual(evidence["research_proxy"]["walkforward_models"]["identity"]["b2_positive_rate"], .4)
            self.assertTrue(any("full_pool_security_audit_20260906" in path for path in hashes))


if __name__ == "__main__": unittest.main()
