import copy
import unittest
from datetime import date,timedelta

from complete_trade_dataset import build,TARGET_DEFINITION,SELECTION_SCOPE


class CompleteTradeDatasetTests(unittest.TestCase):
    asof="2026-09-08T15:05:00+08:00"

    def policy(self):
        return dict(policy_id="p-policy",frozen_at="2026-01-01T14:00:00+08:00",
                    available_at="2026-01-01T14:01:00+08:00",large_loss_threshold=-.07,
                    target_definition=TARGET_DEFINITION,selection_scope=SELECTION_SCOPE,
                    feature_allowlist=["emotion_score","theme_sync"])

    def observation(self,index=1,selected=True,boards=1,signal_date="2026-09-04"):
        code=str(600000+index)
        plan="p"+str(index) if selected else None
        meta=lambda name:dict(observed_at=signal_date+"T15:00:00+08:00",available_at=signal_date+"T15:01:00+08:00",
                              source="synthetic",definition_id=name+"-v1",data_status="verified")
        return dict(decision_id="d"+str(index),plan_id=plan,code=code,name="甲",boards=boards,
                    board_group="FIRST_BOARD" if boards==1 else "CONSECUTIVE_BOARD",
                    signal_date=signal_date,trade_date="2026-09-07",decision_at=signal_date+"T15:02:00+08:00",
                    available_at=signal_date+"T15:01:30+08:00",source="synthetic",data_status="verified",kind="paper",
                    emotion_active=True,
                    selection_state="FOCUS" if selected else "WATCH",
                    features={"emotion_score":.7,"theme_sync":.8,"future_label":1},
                    feature_evidence={"emotion_score":meta("emotion"),"theme_sync":meta("theme")})

    def account(self,indices=(1,),open_indices=()):
        closed=[dict(plan_id="p"+str(i),trade_id="t"+str(i),code=str(600000+i),opened_at="2026-09-07T09:36:00+08:00",
                     closed_at="2026-09-08T09:40:00+08:00",net_return=.02,net_profit=20,remaining_shares=0,
                     entry_cost=1000,sale_proceeds=1020) for i in indices]
        opened=[dict(plan_id="p"+str(i),trade_id="t"+str(i),code=str(600000+i),opened_at="2026-09-07T09:36:00+08:00",
                     remaining_shares=100,entry_cost=1000,sale_proceeds=0) for i in open_indices]
        entries=[dict(plan_id="p"+str(i),issues=[]) for i in (*indices,*open_indices)]
        risks=[dict(plan_id="p"+str(i),issues=[]) for i in (*indices,*open_indices)]
        exits=[dict(plan_id="p"+str(i),issues=[]) for i in (*indices,*open_indices)]
        return dict(kind="paper",status="ACCOUNTING_ONLY_NOT_STRATEGY_VALIDATED",closed_trades=closed,open_trades=opened,
                    entry_plan_audit=dict(entries=entries),risk_budget_audit=dict(plans=risks),
                    exit_plan_audit=dict(trades=exits),drawdown_status="COMPLETE_SUPPLIED_SESSIONS",account_max_drawdown=-.02)

    def funnel(self):
        return dict(account_kind="paper",status="PASS",complete_session_count=2,evaluation_session_count=2)

    def fills(self,indices=(1,)):
        return [dict(plan_id="p"+str(i),side="BUY") for i in indices]

    def test_complete_compliant_trade_gets_label_only(self):
        result=build([self.observation()],self.fills(),self.account(),self.funnel(),self.policy(),self.asof)
        row=result["rows"][0]
        self.assertTrue(row["training_eligible"])
        self.assertTrue(row["complete_trade_profit_label"])
        self.assertFalse(row["large_loss_label"])
        self.assertEqual(row["ignored_unregistered_features"],["future_label"])
        self.assertEqual(result["overall_metrics"]["maximum_consecutive_losses"],0)

    def test_unselected_and_unfilled_are_not_negative_labels(self):
        unselected=self.observation(selected=False)
        unfilled=self.observation(2,selected=True,boards=2)
        result=build([unselected,unfilled],[],self.account(indices=()),self.funnel(),self.policy(),self.asof)
        self.assertTrue(all(row["complete_trade_profit_label"] is None for row in result["rows"]))
        self.assertFalse(result["nonselected_rows_are_negative_labels"])
        self.assertFalse(result["unfilled_rows_are_negative_labels"])

    def test_future_feature_or_exit_issue_removes_training_label(self):
        observation=self.observation()
        observation["feature_evidence"]["theme_sync"]["available_at"]="2026-09-04T15:03:00+08:00"
        result=build([observation],self.fills(),self.account(),self.funnel(),self.policy(),self.asof)
        self.assertIsNone(result["rows"][0]["complete_trade_profit_label"])
        account=self.account()
        account["exit_plan_audit"]["trades"][0]["issues"]=["late"]
        result=build([self.observation()],self.fills(),account,self.funnel(),self.policy(),self.asof)
        self.assertIsNone(result["rows"][0]["complete_trade_profit_label"])
        self.assertIn("EXIT_AUDIT_NOT_PASSED",result["rows"][0]["issues"])

    def test_open_trade_stays_unlabeled(self):
        result=build([self.observation()],self.fills(),self.account(indices=(),open_indices=(1,)),
                     self.funnel(),self.policy(),self.asof)
        self.assertEqual(result["rows"][0]["outcome_state"],"OPEN")
        self.assertIsNone(result["rows"][0]["complete_trade_profit_label"])

    def test_one_hundred_policy_conditional_trades_can_prepare_training_only(self):
        start=date(2026,5,1)
        observations=[self.observation(i,boards=1 if i<=50 else 2,
                                       signal_date=(start+timedelta(days=i-1)).isoformat()) for i in range(1,101)]
        result=build(observations,self.fills(range(1,101)),self.account(range(1,101)),self.funnel(),self.policy(),self.asof)
        self.assertEqual(result["status"],"POLICY_CONDITIONAL_DATASET_READY")
        self.assertTrue(result["training_ready"])
        self.assertFalse(result["can_rank_unselected_full_pool"])
        self.assertEqual(result["generalization_scope"],"FROZEN_SELECTION_POLICY_CONDITIONAL_ONLY")

    def test_one_stock_policy_rejects_multiple_selected_same_signal_date(self):
        observations=[self.observation(1),self.observation(2,boards=2)]
        result=build(observations,self.fills((1,2)),self.account((1,2)),self.funnel(),self.policy(),self.asof)
        self.assertFalse(result["training_ready"])
        self.assertTrue(any(issue.startswith("MULTIPLE_SELECTED_CANDIDATES_SAME_SIGNAL_DATE")
                            for issue in result["global_issues"]))

    def test_invalid_policy_or_account_kind_rejected(self):
        policy=self.policy()
        policy["target_definition"]="promotion"
        with self.assertRaises(ValueError):
            build([self.observation()],self.fills(),self.account(),self.funnel(),policy,self.asof)
        account=self.account()
        account["kind"]="user_confirmed"
        with self.assertRaises(ValueError):
            build([self.observation()],self.fills(),account,self.funnel(),self.policy(),self.asof)

    def test_large_loss_definition_cannot_be_changed(self):
        policy=self.policy()
        policy["large_loss_threshold"]=-.10
        with self.assertRaises(ValueError):
            build([self.observation()],self.fills(),self.account(),self.funnel(),policy,self.asof)


if __name__=="__main__": unittest.main()
