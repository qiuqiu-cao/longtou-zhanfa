import copy
import unittest
from datetime import date,timedelta

from policy_threshold_validation import validate,TIE_BREAK


class ThresholdValidationTests(unittest.TestCase):
    def card(self,big_losses=2):
        predictions=[]
        start=date(2026,1,5)
        for i in range(40):
            predictions.append(dict(decision_id="d"+str(i),signal_date=(start+timedelta(days=i//4*7+i%4)).isoformat(),
                board_group="FIRST_BOARD" if i%2 else "CONSECUTIVE_BOARD",emotion_active=True,
                complete_trade_net_return=-.08 if i<big_losses else -.01 if i<max(4,big_losses) else .02,
                profit_probability=.8,large_loss_probability=.05,
                profit_label=0 if i<max(4,big_losses) else 1,large_loss_label=1 if i<big_losses else 0))
        return dict(status="DEVELOPMENT_MODEL_ONLY",blind_metrics=None,threshold_selected=False,can_rank_unselected_full_pool=False,
                    generated_at="2026-06-01T10:00:00+08:00",validation_predictions=predictions)

    def config(self):
        return dict(config_id="threshold-v1",frozen_at="2026-05-31T08:00:00+08:00",
                    available_at="2026-05-31T08:01:00+08:00",target_win_rate=.70,max_big_loss_rate=.10,
                    minimum_selected_trades=30,minimum_signals_per_active_week=3,
                    maximum_signals_per_active_week=5,profit_thresholds=[.5,.7,.9],
                    large_loss_thresholds=[.05,.10,.20],tie_break=TIE_BREAK)

    def test_feasible_grid_freezes_one_pair_without_deployment(self):
        result=validate(self.card(),self.config())
        self.assertEqual(result["status"],"VALIDATION_THRESHOLD_FROZEN")
        self.assertEqual(result["chosen_thresholds"]["selected_trades"],40)
        self.assertFalse(result["deployment_allowed"])
        self.assertIsNone(result["blind_metrics"])
        self.assertEqual(result["grid_size"],9)

    def test_no_feasible_pair_does_not_relax_target(self):
        result=validate(self.card(big_losses=8),self.config())
        self.assertEqual(result["status"],"NO_VALIDATED_THRESHOLD")
        self.assertIsNone(result["chosen_thresholds"])
        self.assertEqual(result["system_action"],"ABSTAIN_AND_KEEP_DIAGNOSTIC_ONLY")

    def test_inactive_emotion_rows_never_become_signals(self):
        card=self.card()
        for row in card["validation_predictions"][:20]:
            row["emotion_active"]=False
        result=validate(card,self.config())
        self.assertEqual(result["status"],"NO_VALIDATED_THRESHOLD")
        self.assertTrue(all(row["selected_trades"]<=20 for row in result["grid"]))

    def test_grid_must_be_frozen_before_model_result(self):
        config=self.config()
        config["available_at"]="2026-06-01T10:01:00+08:00"
        with self.assertRaisesRegex(ValueError,"unfrozen"):
            validate(self.card(),config)

    def test_challenge_targets_and_grid_are_not_mutable(self):
        for key,value in (("target_win_rate",.65),("max_big_loss_rate",.2),
                          ("minimum_signals_per_active_week",2)):
            config=self.config()
            config[key]=value
            with self.assertRaises(ValueError):
                validate(self.card(),config)

    def test_labels_must_match_complete_net_return(self):
        card=self.card()
        card["validation_predictions"][0]["profit_label"]=1
        with self.assertRaisesRegex(ValueError,"prediction row"):
            validate(card,self.config())

    def test_multiple_policy_trades_on_one_date_are_rejected(self):
        card=self.card()
        card["validation_predictions"][1]["signal_date"]=card["validation_predictions"][0]["signal_date"]
        with self.assertRaisesRegex(ValueError,"one-stock"):
            validate(card,self.config())


if __name__=="__main__": unittest.main()
