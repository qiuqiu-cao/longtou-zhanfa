import copy
import unittest

from daily_primary_selector import FALSE_STRENGTH_FLAGS, select


ASOF = "2026-09-08T09:35:00+08:00"


def fixture():
    candidates = [{"candidate_id": "2026-09-07:000001", "code": "000001", "name": "甲",
                   "boards": 1, "board_group": "FIRST_BOARD", "pool_assignment": "CLOSING_LIMIT_UP",
                   "is_mainboard": True, "is_st": False, "is_delisting_period": False,
                   "no_price_limit": False, "security_status": "verified",
                   "limit_up_status": "verified", "identity_status": "verified"},
                  {"candidate_id": "2026-09-07:600001", "code": "600001", "name": "乙",
                   "boards": 3, "board_group": "CONSECUTIVE_BOARD", "pool_assignment": "CLOSING_LIMIT_UP",
                   "is_mainboard": True, "is_st": False, "is_delisting_period": False,
                   "no_price_limit": False, "security_status": "verified",
                   "limit_up_status": "verified", "identity_status": "verified"}]
    flags = {name: False for name in FALSE_STRENGTH_FLAGS}
    factors = {name: "VERIFIED" for name in {
        "turnover_absorption_quality", "theme_purity", "sector_leadership",
        "competitor_elimination", "announcement_soft_risk"}}
    return {"trade_date": "2026-09-08",
            "universe": {"status": "VERIFIED_FULL_CLOSING_LIMIT_POOL", "signal_date": "2026-09-07",
                         "universe_complete": True, "point_in_time_status_complete": True,
                         "all_observed_board_levels_reported": True, "failed_board_pool_mixed_with_main": False,
                         "closing_limit_count": 2, "board_level_counts": {"1": 1, "3": 1},
                         "failed_board_count": 1, "failed_board_candidate_ids": ["2026-09-07:000002"],
                         "emotion_anchor_ids": ["2026-09-07:600001"], "candidates": candidates},
            "ranking": {"status": "CURRENT_FULL_POOL_RANK_SCORES_FROZEN", "signal_date": "2026-09-07",
                        "scored_at": "2026-09-07T15:10:00+08:00", "score_is_complete_trade_probability": False,
                        "rows": [{"candidate_id": "2026-09-07:000001", "opportunity_proxy_score": .8,
                                  "is_daily_top1": True},
                                 {"candidate_id": "2026-09-07:600001", "opportunity_proxy_score": .6,
                                  "is_daily_top1": False}]},
            "same_height_comparison": {
                "status": "FROZEN_POINT_IN_TIME_COMPLETE", "signal_date": "2026-09-07",
                "seal_speed_or_zero_failed_seals_can_select_champion": False,
                "factor_use_status": "EVIDENCE_GATE_ONLY_NOT_POSITIVE_WEIGHT",
                "rows": [
                    {"candidate_id": "2026-09-07:000001", "height_group_id": "2026-09-07:H1",
                     "same_height_group_size": 1, "group_member_ids": ["2026-09-07:000001"],
                     "group_scope_complete": True, "five_factor_evidence_complete": True,
                     "factor_evidence_status": factors, "official_champion": False,
                     "evidence_available_at": "2026-09-07T18:00:00+08:00"},
                    {"candidate_id": "2026-09-07:600001", "height_group_id": "2026-09-07:H3",
                     "same_height_group_size": 1, "group_member_ids": ["2026-09-07:600001"],
                     "group_scope_complete": True, "five_factor_evidence_complete": True,
                     "factor_evidence_status": factors, "official_champion": False,
                     "evidence_available_at": "2026-09-07T18:00:00+08:00"}]},
            "first_board_dual_target": {
                "status": "FORWARD_VALIDATED_DUAL_TARGET", "signal_date": "2026-09-07",
                "targets_trained_separately": True,
                "b2_target": "EXECUTABLE_NET_RETURN_POSITIVE",
                "b1_risk_target": "NET_LOSS_AT_LEAST_7_PERCENT",
                "probabilities_calibrated_for_deployment": True,
                "rows": [{"candidate_id": "2026-09-07:000001",
                          "b2_positive_probability": .76, "b1_big_loss_probability": .07,
                          "inside_frozen_feature_support": True,
                          "available_at": "2026-09-07T15:10:00+08:00"}]},
            "top1_opening_confirmation": {"candidate_id": "2026-09-07:000001", "code": "000001",
                        "observed_at": "2026-09-08T09:34:30+08:00", "available_at": "2026-09-08T09:34:40+08:00",
                        "max_age_seconds": 60, "data_status": "verified", "source": "minute-feed",
                        "complete_trade_profit_probability": .76, "large_loss_probability": .07,
                        "b2_positive_probability": .76, "b1_big_loss_probability": .07,
                        "probability_target_scope": "FIRST_BOARD_B2_POSITIVE_AND_B1_BIG_LOSS",
                        "inside_frozen_feature_support": True, "auction_conditions_passed": True,
                        "opening_conditions_passed": True, "theme_sync_passed": True,
                        "competition_confirmation_passed": True, "false_strength_flags": flags,
                        "theme": {"name": "细分题材A", "granularity": "specific",
                                  "claim_status": "verified_business", "source": "公告",
                                  "available_at": "2026-09-07T18:00:00+08:00"},
                        "hard_risk_block": False, "execution_state": "BUYABLE",
                        "planned_entry_price": 10.0, "maximum_acceptable_price": 10.1,
                        "invalidation_price": 9.5},
            "preopen_announcement_binding": {
                "version": "v14-announcement-forward-1",
                "signal_date": "2026-09-07", "trade_date": "2026-09-08",
                "bound_at": "2026-09-08T08:30:00+08:00",
                "candidate_code": "000001", "candidate_name": "甲",
                "decision": "NO_HARD_RISK_FOUND", "coverage_complete": True,
                "evidence_assurance": "OFFICIAL_INDEX_VERIFIED_NO_ROWS",
                "binding_hash": "test-binding-hash",
            },
            "emotion": {"asof": ASOF, "entry_support_eligible": True, "market_regime": "升温",
                        "pattern": "ROTATION_OR_DIVERGENCE_HYPOTHESIS",
                        "dimension_states": {"money_effect": "SUPPORTIVE", "broken_loss_effect": "MIXED",
                                             "height_feedback": "SUPPORTIVE", "theme_divergence": "MIXED",
                                             "repair_quality": "SUPPORTIVE"}},
            "model_stack_lock": {"status": "MODEL_STACK_READY_FOR_LOCKED_BLIND_PAPER",
                         "frozen_at": "2026-09-01T10:00:00+08:00", "deployment_allowed": False,
                         "stack_audit": {"can_feed_daily_decision_contract": True},
                         "evidence": {"frozen_threshold": {"chosen_thresholds": {
                             "profit_threshold": .70, "large_loss_threshold": .10}}}},
            "system_acceptance": {"status": "PAPER_TRADE_READY"},
            "risk_policy": {"policy_id": "risk-v1", "frozen_at": "2026-09-01T09:00:00+08:00",
                         "available_at": "2026-09-01T09:01:00+08:00", "data_status": "verified",
                         "source": "account", "single_trade_risk_fraction": .005,
                         "gross_exposure_fraction": .50, "theme_exposure_fraction": .35,
                         "stress_loss_fraction": .08, "account_equity": 100000,
                         "existing_gross_exposure_cash": 0, "existing_theme_exposure_cash": 0,
                         "regime_position_caps": {"启动": .20, "升温": .30, "加速": .20, "修复": .15},
                         "maximum_adverse_dimensions": 1}}


class DailyPrimarySelectorTests(unittest.TestCase):
    def test_valid_case_returns_only_one_conditional_paper_primary(self):
        result = select(fixture(), ASOF)
        self.assertEqual(result["status"], "CONDITIONAL_PRIMARY_READY")
        self.assertEqual(result["mode"], "LOCKED_PAPER_PLAN")
        self.assertEqual(result["primary_choice"]["code"], "000001")
        self.assertEqual(result["primary_choice"]["max_shares"], 600)
        self.assertEqual(result["research_shortlist"]["first_board_choice"]["code"], "000001")
        self.assertEqual(result["research_shortlist"]["relay_choice"]["code"], "600001")
        self.assertEqual(result["research_shortlist"]["all_limit_up_choice"]["code"], "000001")
        self.assertFalse(result["research_shortlist"]["buy_signal"])
        self.assertFalse(result["order_submitted"])
        self.assertFalse(result["guarantee_of_premium"])

    def test_preopen_announcement_binding_is_mandatory(self):
        item = fixture()
        item.pop("preopen_announcement_binding")
        result = select(item, ASOF)
        self.assertEqual(result["status"], "CASH")
        self.assertIn("PREOPEN_ANNOUNCEMENT_DATA_UNAVAILABLE", result["blocking_reasons"])

    def test_preopen_hard_risk_forces_cash(self):
        item = fixture()
        item["preopen_announcement_binding"]["decision"] = "HARD_RISK_REJECT"
        result = select(item, ASOF)
        self.assertEqual(result["status"], "CASH")
        self.assertIn("PREOPEN_ANNOUNCEMENT_HARD_RISK", result["blocking_reasons"])

    def test_preopen_full_text_assurance_is_required(self):
        item = fixture()
        item["preopen_announcement_binding"]["evidence_assurance"] = (
            "INDEX_DATE_VERIFIED_FULL_TEXT_UNVERIFIED")
        result = select(item, ASOF)
        self.assertIn("PREOPEN_ANNOUNCEMENT_FULLTEXT_NOT_VERIFIED", result["blocking_reasons"])

    def test_preopen_binding_must_match_frozen_top1(self):
        item = fixture()
        item["preopen_announcement_binding"]["candidate_code"] = "600001"
        result = select(item, ASOF)
        self.assertIn("PREOPEN_ANNOUNCEMENT_BINDING_INVALID", result["blocking_reasons"])

    def test_forward_validated_changes_mode_not_direct_authorization(self):
        item = fixture()
        item["system_acceptance"]["status"] = "FORWARD_VALIDATED"
        result = select(item, ASOF)
        self.assertEqual(result["mode"], "CONDITIONAL_USER_DECISION")
        self.assertFalse(result["direct_recommendation"])

    def test_diagnostic_system_forces_cash(self):
        item = fixture()
        item["system_acceptance"]["status"] = "DIAGNOSTIC_ONLY"
        result = select(item, ASOF)
        self.assertEqual(result["status"], "CASH")
        self.assertIn("SYSTEM_ACCEPTANCE_NOT_READY", result["blocking_reasons"])
        # Research ranking remains visible, while trade permission stays cash.
        self.assertEqual(result["research_shortlist"]["all_limit_up_choice"]["code"], "000001")
        self.assertIsNone(result["primary_choice"])

    def test_pool_coverage_mismatch_forces_cash(self):
        item = fixture()
        item["universe"]["closing_limit_count"] = 3
        result = select(item, ASOF)
        self.assertIn("FULL_POOL_COVERAGE_NOT_VERIFIED", result["blocking_reasons"])

    def test_ranker_must_score_every_pool_candidate(self):
        item = fixture()
        item["ranking"]["rows"].pop()
        result = select(item, ASOF)
        self.assertIn("RANKING_DOES_NOT_COVER_EXACT_POOL", result["blocking_reasons"])
        self.assertEqual(result["research_shortlist"]["status"],
                         "UNAVAILABLE_RANKING_INTEGRITY_FAILED")
        self.assertIsNone(result["research_shortlist"]["all_limit_up_choice"])

    def test_failed_board_pool_must_be_explicit_and_disjoint(self):
        item = fixture()
        item["universe"]["failed_board_candidate_ids"] = ["2026-09-07:000001"]
        result = select(item, ASOF)
        self.assertIn("FULL_POOL_COVERAGE_NOT_VERIFIED", result["blocking_reasons"])

    def test_security_status_must_be_point_in_time_verified(self):
        item = fixture()
        item["universe"]["candidates"][0]["security_status"] = "inferred"
        result = select(item, ASOF)
        self.assertIn("INVALID_OR_DUPLICATE_POOL_CANDIDATE", result["blocking_reasons"])

    def test_top1_failure_means_cash_not_backup(self):
        item = fixture()
        item["top1_opening_confirmation"]["opening_conditions_passed"] = False
        result = select(item, ASOF)
        self.assertEqual(result["status"], "CASH")
        self.assertTrue(result["frozen_top1_failed_means_cash"])
        self.assertFalse(result["automatic_backup_allowed"])

    def test_out_of_support_probability_is_rejected(self):
        item = fixture()
        item["top1_opening_confirmation"]["inside_frozen_feature_support"] = False
        result = select(item, ASOF)
        self.assertIn("PROFIT_OR_LARGE_LOSS_GATE_FAILED", result["blocking_reasons"])

    def test_retreat_regime_forces_cash(self):
        item = fixture()
        item["emotion"]["market_regime"] = "退潮"
        result = select(item, ASOF)
        self.assertIn("EMOTION_HARD_BLOCK", result["blocking_reasons"])

    def test_observable_false_strength_flag_forces_cash(self):
        item = fixture()
        item["top1_opening_confirmation"]["false_strength_flags"]["below_vwap_expanding"] = True
        result = select(item, ASOF)
        self.assertIn("OBSERVABLE_FALSE_STRENGTH_RISK", result["blocking_reasons"])

    def test_price_above_maximum_is_rejected(self):
        item = fixture()
        item["top1_opening_confirmation"]["planned_entry_price"] = 10.2
        result = select(item, ASOF)
        self.assertIn("INVALID_ENTRY_PRICE_BOUNDS", result["blocking_reasons"])

    def test_broken_loss_adverse_is_hard_block(self):
        item = fixture()
        item["emotion"]["dimension_states"]["broken_loss_effect"] = "ADVERSE"
        result = select(item, ASOF)
        self.assertIn("EMOTION_HARD_BLOCK", result["blocking_reasons"])

    def test_confirmation_must_match_prefrozen_top1(self):
        item = fixture()
        item["top1_opening_confirmation"]["candidate_id"] = "2026-09-07:600001"
        item["top1_opening_confirmation"]["code"] = "600001"
        result = select(item, ASOF)
        self.assertIn("OPENING_CONFIRMATION_NOT_FOR_FROZEN_TOP1", result["blocking_reasons"])

    def test_same_height_group_must_cover_every_peer(self):
        item = fixture()
        second = copy.deepcopy(item["universe"]["candidates"][0])
        second.update(candidate_id="2026-09-07:000002", code="000002", name="丙")
        item["universe"]["candidates"].append(second)
        item["universe"]["closing_limit_count"] = 3
        item["universe"]["board_level_counts"] = {"1": 2, "3": 1}
        item["ranking"]["rows"].append({"candidate_id": second["candidate_id"],
                                         "opportunity_proxy_score": .7,
                                         "is_daily_top1": False})
        result = select(item, ASOF)
        self.assertIn("SAME_HEIGHT_COMPARISON_NOT_EXACT_FULL_POOL", result["blocking_reasons"])

    def test_fast_seal_cannot_replace_missing_five_factor_evidence(self):
        item = fixture()
        item["same_height_comparison"]["rows"][0]["five_factor_evidence_complete"] = False
        result = select(item, ASOF)
        self.assertIn("SAME_HEIGHT_GROUP_OR_FIVE_FACTOR_EVIDENCE_INCOMPLETE",
                      result["blocking_reasons"])

    def test_first_board_confirmation_must_match_both_frozen_targets(self):
        item = fixture()
        item["top1_opening_confirmation"]["b1_big_loss_probability"] = .05
        result = select(item, ASOF)
        self.assertIn("FIRST_BOARD_CONFIRMATION_NOT_BOUND_TO_DUAL_TARGET",
                      result["blocking_reasons"])
