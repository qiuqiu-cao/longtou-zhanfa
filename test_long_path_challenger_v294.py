import unittest
import numpy as np
import pandas as pd

from long_path_challenger_v294 import binary, ranking, chronological, logit, SPEC, ARMS, fit_forecast
from long_path_features_v291 import path_features
from archive_long_interval_v290 import TIMES
from next_session_forecast_lab_v271 import FORBIDDEN


class LongPathIntegrity(unittest.TestCase):
    def rows(self):
        return pd.DataFrame({"date":["2026-03-02"]*3+["2026-03-03"]*2,
            "code":["000001","000002","000003","000001","000002"], "boards":[2,2,3,2,2],
            "daily_true_one_word":[0,0,1,0,0],"label_valid":[True,True,True,True,False],
            "target_original":[1,0,1,1,np.nan],"target_pressure_zero":[1,0,1,1,np.nan],
            "predicted":[False,False,True,True,False],"raw_score":[.4,.2,1.,.8,.1],"calibrated_score":[.4,.2,1.,.8,.1]})

    def test_missing_preserves_full_denominator(self):
        result=binary(self.rows())
        self.assertEqual((result["total_frozen_rows"],result["known"],result["unknown"]),(5,4,1))
        self.assertEqual((result["tp"],result["fp"],result["fn"],result["tn"]),(2,0,1,1))
        self.assertAlmostEqual(result["recall"],2/3)

    def test_top1_missing_peer_does_not_replace_choice(self):
        result,picks=ranking(self.rows())
        self.assertEqual(result["frozen_eligible_dates"],2)
        self.assertEqual(result["scored_complete_dates"],1)
        self.assertEqual(result["unknown_dates"],1)
        self.assertEqual(picks.date.tolist(),["2026-03-02"])

    def test_targets_cannot_change_ranking(self):
        rows=self.rows()
        _,before=ranking(rows)
        rows.loc[:2,"target_original"]=[0,1,0]
        _,after=ranking(rows)
        self.assertEqual(before.date.tolist(),after.date.tolist())
        self.assertEqual(before.promoted_next_day.tolist(),[1])
        self.assertEqual(after.promoted_next_day.tolist(),[0])

    def test_probability_one_in_final_bin(self):
        result=binary(self.rows())
        self.assertEqual(result["calibration_bins"][-1]["n"],1)
        self.assertEqual(sum(b["n"] for b in result["calibration_bins"]),result["known"])

    def test_pressure_keeps_same_predicted_rows(self):
        rows=self.rows()
        rows.loc[2,"target_pressure_zero"]=0
        original=binary(rows)
        pressure=binary(rows,"target_pressure_zero")
        self.assertEqual(original["total_frozen_rows"],pressure["total_frozen_rows"])
        self.assertEqual(original["correct"]-pressure["correct"],1)

    def test_allowlists_no_target_fields(self):
        self.assertTrue(all(not set(fields)&FORBIDDEN for fields in ARMS.values()))
        self.assertNotIn("daily_true_one_word",ARMS["BASE"])

    def test_logit_extremes_are_finite(self):
        self.assertTrue(np.isfinite(logit([0,1,.5])).all())

    def test_split_rejects_post_boundary_realizations(self):
        dates=list(pd.bdate_range("2025-10-01",periods=40).strftime("%Y-%m-%d"))+list(pd.bdate_range("2026-01-01",periods=30).strftime("%Y-%m-%d"))+["2026-03-02"]
        features=pd.DataFrame({"date":np.repeat(dates,2),"code":["000001","000002"]*len(dates)})
        labels=features.copy();labels["target_original"]=[0,1]*len(dates);labels["label_valid"]=True
        labels["target_date"]=(pd.to_datetime(labels.date)+pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
        labels.loc[labels.date.lt(SPEC["train_end_exclusive"]),"target_date"]="2026-01-01"
        with self.assertRaisesRegex(ValueError,"insufficient prior"):
            chronological(features,labels)

    def test_test_outcomes_cannot_change_fitted_scores(self):
        dates=list(pd.bdate_range("2025-10-01",periods=40).strftime("%Y-%m-%d"))+list(pd.bdate_range("2026-01-01",periods=30).strftime("%Y-%m-%d"))+["2026-03-02","2026-03-03"]
        features=pd.DataFrame({"date":np.repeat(dates,2),"code":["000001","000002"]*len(dates),"boards":[2,3]*len(dates),"daily_true_one_word":0})
        labels=features[["date","code"]].copy();labels["target_original"]=[0,1]*len(dates);labels["label_valid"]=True
        labels["target_date"]=(pd.to_datetime(labels.date)+pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
        before,log=fit_forecast(features,labels,["boards"])
        labels.loc[labels.date.ge(SPEC["test_start"]),"target_original"]=1-labels.loc[labels.date.ge(SPEC["test_start"]),"target_original"]
        after,new_log=fit_forecast(features,labels,["boards"])
        pd.testing.assert_frame_equal(before,after)
        self.assertEqual(log,new_log)

    def test_next_day_bars_cannot_enter_signal_path(self):
        bars=pd.DataFrame({"code":"000001","minute":["2026-03-02 "+t+":00" for t in TIMES],"close":np.linspace(9.,10.,8),"high":10.,"low":8.9,"volume_shares":100.})
        check={"code":"000001","date":"2026-03-02","price_reconciled":True,"grid_complete":True,"original_pool_close_agrees":True,"final_cap_price_bar_run":0,"cap_price_bar_share":0.,"full_day_vwap":9.5}
        before=path_features(bars,check)
        future=bars.copy();future["minute"]=future.minute.str.replace("2026-03-02","2026-03-03");future["close"]=1000.
        self.assertEqual(before,path_features(pd.concat([bars,future]),check))


if __name__=="__main__":unittest.main()
