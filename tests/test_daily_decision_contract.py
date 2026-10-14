import copy
import json
import tempfile
import unittest
from pathlib import Path

from daily_decision_contract import audit, run, EMOTION_DIMENSIONS


class DailyDecisionContractTests(unittest.TestCase):
    asof = "2026-09-07T09:35:00+08:00"

    def evidence(self, metric):
        return dict(metric=metric,value=.1,source="synthetic",definition_id=metric+"-v1",
                    observed_at="2026-09-07T09:35:00+08:00",available_at="2026-09-07T09:35:00+08:00",
                    data_status="verified",max_age_seconds=300,denominator=100,observed_count=100,missing_count=0)

    def condition(self, cid):
        return dict(condition_id=cid,metric=cid,unit="ratio",definition_id=cid+"-v1",
                    operator=">=",threshold=.1,window_seconds=300,max_age_seconds=60)

    def candidate(self, rumor=False):
        return dict(code="600001",name="甲",boards=1,board_group="FIRST_BOARD",
                    pool_assignment="CLOSING_LIMIT_UP",competition_identity="pioneer",
                    height_role="POTENTIAL_NEXT_HIGHEST",
                    theme=dict(name="细分设备",granularity="specific",
                               claim_status="rumor" if rumor else "verified_business",source="company filing",
                               published_at="2026-09-06T18:00:00+08:00",available_at="2026-09-06T18:01:00+08:00"),
                    continuation_logic=["先发且题材梯队仍在"],opposition_evidence=["首板身份尚未确认"],
                    competitor_invalidation=["同题材二板核心转弱"],risk_factors=["隔夜跳空与开板风险"],
                    auction_conditions=[self.condition("auction_amount_ratio")],
                    opening_conditions=[self.condition("close_vs_vwap")],
                    cancel_conditions=[dict(self.condition("failed_seal_count"),operator=">=",threshold=2)],
                    entry_plan=dict(entry_start="2026-09-07T09:35:01+08:00",entry_end="2026-09-07T10:00:00+08:00",
                                    min_price=10,max_price=10.5,allowed_slippage=.001,max_shares=1000,risk_policy_id="r1"),
                    premium_space=dict(assessment="MEDIUM",evidence_status="verified",
                                       drivers=["题材梯队与高度竞争支持"],
                                       exhaustion_signals=["高开兑现或核心转弱"],
                                       risk_reward_conclusion="确认后仍有覆盖隔夜风险的空间"),
                    anchor_entry_plan=dict(buyability="CONDITIONAL",setup_type="换手回封",
                                           high_open_action="高开超过边界则取消",
                                           flat_open_action="收回均价后观察回封",
                                           low_open_action="弱转强且板块同步才考虑",
                                           sector_break_action="立即取消",
                                           entry_trigger_ids=["auction_amount_ratio","close_vs_vwap"],
                                           invalid_trigger_ids=["failed_seal_count"],max_price=10.5,
                                           expected_holding_sessions=[1,3]),
                    holding_scenarios={
                        "strong_continuation":dict(action="HOLD_TO_NEXT_REVIEW",rule_ids=["h1"]),
                        "normal_divergence":dict(action="HOLD_IF_RULES_PASS",rule_ids=["h2"]),
                        "trend_invalidation":dict(action="SELL_ALL",rule_ids=["s1"])},
                    execution_state="CONDITIONAL",model_probability=None,probability_status="unavailable")

    def report(self, primary=False, system="DIAGNOSTIC_ONLY"):
        emotion={}
        for dimension,metrics in EMOTION_DIMENSIONS.items():
            emotion[dimension]=dict(interpretation="有证据的解释",trade_effect="控制候选和仓位",
                                    invalidation="同口径反馈反向即失效",
                                    evidence=[self.evidence(metric) for metric in metrics])
        focus=[self.candidate()] if primary else []
        return dict(report_date="2026-09-07",asof=self.asof,stage="OPENING_CONFIRMATION",
                    system_status=system,
                    data_quality=dict(sources=["synthetic"],verified_fields=["price"],unavailable_fields=["level2"],
                                      execution_limitation="no queue verification",direct_recommendation_allowed=False),
                    universe_summary=dict(closing_limit_count=2,first_board_count=1,
                                          board_level_counts={"1":1,"2":1},failed_board_pool_mixed_with_main=False,
                                          all_observed_board_levels_reported=True),emotion_analysis=emotion,
                    focus_candidates=focus,
                    candidate_groups={"FIRST_BOARD":["600001"] if primary else [],"CONSECUTIVE_BOARD":[]},
                    roles={"emotion_anchor":[],"leader_potential":["600001"] if primary else [],
                           "current_buy_opportunity":["600001"] if primary else []},
                    height_outlook=dict(current_highest_board=2,current_anchor_codes=[],
                                        potential_next_highest_codes=["600001"] if primary else [],
                                        role_overlap_allowed=True,
                                        anchor_premium_conclusion="当前锚点只作高度观察，潜在锚点需确认",
                                        rotation_confirmations=["当前最高标断板且候选主动晋级"],
                                        failure_conditions=["高中位负反馈扩散"]),
                    primary_choice=(dict(code="600001",conditional=True,relative_advantage="承接与价格边界更优",
                                         trigger_summary="全部冻结条件通过",on_failure="CASH") if primary else None),
                    final_decision="CONDITIONAL_PRIMARY" if primary else "CASH")

    def test_complete_cash_report_is_valid_but_not_entry(self):
        result=audit(self.report())
        self.assertEqual(result["status"],"VALID_CASH_REPORT")
        self.assertEqual(result["entry_permission"],"NO_ENTER")

    def test_paper_ready_opening_report_allows_only_conditional_plan(self):
        result=audit(self.report(primary=True,system="PAPER_TRADE_READY"))
        self.assertEqual(result["status"],"VALID_CONDITIONAL_REPORT")
        self.assertEqual(result["entry_permission"],"CONDITIONAL_ENTRY_PLAN_ONLY")

    def test_diagnostic_model_cannot_create_entry_permission(self):
        result=audit(self.report(primary=True))
        self.assertEqual(result["status"],"VALID_CONDITIONAL_REPORT")
        self.assertEqual(result["entry_permission"],"NO_ENTER")

    def test_missing_emotion_metric_invalidates_report(self):
        report=self.report()
        report["emotion_analysis"]["money_effect"]["evidence"].pop()
        result=audit(report)
        self.assertEqual(result["status"],"INVALID_REPORT")
        self.assertIn("MISSING_EMOTION_METRICS_MONEY_EFFECT",result["issues"])

    def test_unvalidated_probability_is_rejected(self):
        report=self.report(primary=True,system="PAPER_TRADE_READY")
        report["focus_candidates"][0].update(model_probability=.9,probability_status="in_sample")
        result=audit(report)
        self.assertEqual(result["status"],"INVALID_REPORT")
        self.assertTrue(any("UNVALIDATED_PROBABILITY" in item for item in result["issues"]))

    def test_rumor_cannot_receive_entry_permission(self):
        report=self.report(primary=True,system="PAPER_TRADE_READY")
        report["focus_candidates"][0]["theme"]["claim_status"]="rumor"
        result=audit(report)
        self.assertEqual(result["status"],"VALID_CONDITIONAL_REPORT")
        self.assertEqual(result["entry_permission"],"NO_ENTER")

    def test_entry_window_before_confirmation_is_rejected(self):
        report=self.report(primary=True,system="PAPER_TRADE_READY")
        report["focus_candidates"][0]["entry_plan"]["entry_start"]="2026-09-07T09:34:00+08:00"
        result=audit(report)
        self.assertEqual(result["status"],"INVALID_REPORT")
        self.assertTrue(any("INVALID_ENTRY_PLAN" in item for item in result["issues"]))

    def test_missing_height_outlook_is_rejected(self):
        report=self.report(primary=True,system="PAPER_TRADE_READY")
        del report["height_outlook"]
        result=audit(report)
        self.assertEqual(result["status"],"INVALID_REPORT")
        self.assertIn("INVALID_HEIGHT_OUTLOOK",result["issues"])

    def test_anchor_with_low_premium_space_cannot_receive_entry_permission(self):
        report=self.report(primary=True,system="PAPER_TRADE_READY")
        report["focus_candidates"][0]["premium_space"]["assessment"]="LOW"
        result=audit(report)
        self.assertEqual(result["status"],"VALID_CONDITIONAL_REPORT")
        self.assertEqual(result["entry_permission"],"NO_ENTER")

    def test_anchor_only_role_cannot_receive_entry_permission(self):
        report=self.report(primary=True,system="PAPER_TRADE_READY")
        report["focus_candidates"][0]["anchor_entry_plan"]["buyability"]="ANCHOR_ONLY"
        result=audit(report)
        self.assertEqual(result["status"],"VALID_CONDITIONAL_REPORT")
        self.assertEqual(result["entry_permission"],"NO_ENTER")

    def test_malformed_roles_and_negative_freshness_fail_closed(self):
        report=self.report()
        report["roles"]["emotion_anchor"]=[{"bad":"value"}]
        report["emotion_analysis"]["repair_quality"]["evidence"][0]["max_age_seconds"]=-1
        result=audit(report)
        self.assertEqual(result["status"],"INVALID_REPORT")
        self.assertIn("INVALID_ROLE_EMOTION_ANCHOR",result["issues"])

    def test_cli_artifact_is_immutable_and_hashed(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/"daily.json"
            target=Path(folder)/"audit.json"
            source.write_text(json.dumps(self.report(),ensure_ascii=False),encoding="utf-8")
            result=run(source,target)
            self.assertEqual(result["status"],"VALID_CASH_REPORT")
            self.assertEqual(len(result["source_hashes"]),2)
            with self.assertRaises(FileExistsError):
                run(source,target)


if __name__ == "__main__": unittest.main()
