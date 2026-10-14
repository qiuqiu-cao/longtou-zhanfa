import copy
import unittest

from daily_decision_contract import EMOTION_DIMENSIONS
from daily_decision_package_audit import audit


ASOF = "2026-09-07T09:35:00+08:00"


def condition(name):
    return {"condition_id": name, "metric": name, "unit": "ratio", "definition_id": name + "-v1",
            "operator": ">=", "threshold": .1, "window_seconds": 300, "max_age_seconds": 60}


def evidence(metric):
    return {"metric": metric, "value": .1, "source": "synthetic", "definition_id": metric + "-v1",
            "observed_at": ASOF, "available_at": ASOF, "data_status": "verified", "max_age_seconds": 300,
            "denominator": 100, "observed_count": 100, "missing_count": 0}


def report(primary=True):
    emotion = {dimension: {"interpretation": "有证据的解释", "trade_effect": "控制候选和仓位",
                           "invalidation": "同口径反馈反向即失效",
                           "evidence": [evidence(metric) for metric in metrics]}
               for dimension, metrics in EMOTION_DIMENSIONS.items()}
    candidate = {"code": "600001", "name": "甲", "boards": 2, "board_group": "CONSECUTIVE_BOARD",
                 "pool_assignment": "CLOSING_LIMIT_UP", "competition_identity": "turnover_core",
                 "height_role": "POTENTIAL_NEXT_HIGHEST",
                 "theme": {"name": "细分A", "granularity": "specific", "claim_status": "verified_business",
                           "source": "filing", "published_at": "2026-09-06T18:00:00+08:00",
                           "available_at": "2026-09-06T18:01:00+08:00"},
                 "continuation_logic": ["梯队共振"], "opposition_evidence": ["高开兑现"],
                 "competitor_invalidation": ["竞争者卡位"], "risk_factors": ["隔夜跳空"],
                 "auction_conditions": [condition("auction")], "opening_conditions": [condition("vwap")],
                 "cancel_conditions": [condition("cancel")],
                 "entry_plan": {"entry_start": "2026-09-07T09:35:01+08:00",
                                "entry_end": "2026-09-07T10:00:00+08:00", "min_price": 9.8,
                                "max_price": 10.1, "allowed_slippage": .001, "max_shares": 600,
                                "risk_policy_id": "r1"},
                 "premium_space": {"assessment": "MEDIUM", "evidence_status": "verified",
                                   "risk_reward_conclusion": "仍有条件性溢价但不能追高",
                                   "drivers": ["题材梯队共振"],
                                   "exhaustion_signals": ["高开跌破均价"]},
                 "anchor_entry_plan": {"buyability": "CONDITIONAL", "setup_type": "换手确认",
                                       "high_open_action": "等待承接", "flat_open_action": "观察转强",
                                       "low_open_action": "无弱转强则取消",
                                       "sector_break_action": "取消",
                                       "entry_trigger_ids": ["auction", "vwap"],
                                       "invalid_trigger_ids": ["cancel"], "max_price": 10.1,
                                       "expected_holding_sessions": [1, 3]},
                 "holding_scenarios": {"strong_continuation": {"action": "HOLD_TO_NEXT_REVIEW", "rule_ids": ["h1"]},
                                       "normal_divergence": {"action": "HOLD_IF_RULES_PASS", "rule_ids": ["h2"]},
                                       "trend_invalidation": {"action": "SELL_ALL", "rule_ids": ["s1"]}},
                 "execution_state": "CONDITIONAL", "model_probability": .75,
                 "probability_status": "calibrated_out_of_sample"}
    return {"report_date": "2026-09-07", "asof": ASOF, "stage": "OPENING_CONFIRMATION",
            "system_status": "PAPER_TRADE_READY",
            "data_quality": {"sources": ["synthetic"], "verified_fields": ["price"],
                             "unavailable_fields": ["level2"], "execution_limitation": "no queue proof",
                             "direct_recommendation_allowed": False},
            "universe_summary": {"closing_limit_count": 2, "first_board_count": 1,
                                 "board_level_counts": {"1": 1, "2": 1},
                                 "failed_board_pool_mixed_with_main": False,
                                 "all_observed_board_levels_reported": True},
            "emotion_analysis": emotion, "focus_candidates": [candidate] if primary else [],
            "candidate_groups": {"FIRST_BOARD": [], "CONSECUTIVE_BOARD": ["600001"] if primary else []},
            "roles": {"emotion_anchor": [], "leader_potential": ["600001"] if primary else [],
                      "current_buy_opportunity": ["600001"] if primary else []},
            "height_outlook": {"current_highest_board": 2, "current_anchor_codes": [],
                               "potential_next_highest_codes": ["600001"] if primary else [],
                               "role_overlap_allowed": True,
                               "anchor_premium_conclusion": "仅作条件性研究，不保证溢价",
                               "rotation_confirmations": ["次日主动晋级且板块同步"],
                               "failure_conditions": ["高度股断板并向中低位传导"]},
            "primary_choice": {"code": "600001", "conditional": True, "relative_advantage": "综合更优",
                               "trigger_summary": "全部条件通过", "on_failure": "CASH"} if primary else None,
            "final_decision": "CONDITIONAL_PRIMARY" if primary else "CASH"}


def selection(primary=True):
    return {"status": "CONDITIONAL_PRIMARY_READY" if primary else "CASH",
            "mode": "LOCKED_PAPER_PLAN" if primary else "CASH", "asof": ASOF,
            "trade_date": "2026-09-07", "direct_recommendation": False, "order_submitted": False,
            "primary_choice": {"code": "600001", "maximum_acceptable_price": 10.1, "max_shares": 600,
                               "rank_score_meaning": "opportunity_proxy_score_not_complete_trade_probability"}
                              if primary else None}


class DailyDecisionPackageAuditTests(unittest.TestCase):
    def test_matching_conditional_package_is_paper_ready(self):
        result = audit(report(), selection(), ASOF)
        self.assertEqual(result["status"], "PACKAGE_READY_FOR_LOCKED_PAPER")
        self.assertTrue(result["selector_and_narrative_match"])

    def test_different_primary_is_rejected(self):
        selected = selection()
        selected["primary_choice"]["code"] = "000001"
        result = audit(report(), selected, ASOF)
        self.assertIn("CONDITIONAL_DECISIONS_DIVERGE", result["issues"])

    def test_position_size_must_match_report(self):
        selected = selection()
        selected["primary_choice"]["max_shares"] = 700
        result = audit(report(), selected, ASOF)
        self.assertIn("ENTRY_OR_ROLE_FIELDS_DIVERGE", result["issues"])

    def test_cash_selector_requires_cash_narrative(self):
        result = audit(report(), selection(False), ASOF)
        self.assertIn("CASH_DECISIONS_DIVERGE", result["issues"])

    def test_matching_cash_package_is_valid(self):
        result = audit(report(False), selection(False), ASOF)
        self.assertEqual(result["status"], "PACKAGE_READY_FOR_LOCKED_PAPER")
        self.assertIsNone(result["primary_code"])

    def test_diagnostic_report_never_ready(self):
        item = report(False)
        item["system_status"] = "DIAGNOSTIC_ONLY"
        result = audit(item, selection(False), ASOF)
        self.assertIn("REPORT_SYSTEM_NOT_READY", result["issues"])
