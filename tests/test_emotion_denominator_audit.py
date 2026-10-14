import unittest
import pandas as pd
from emotion_denominator_audit import build, broken_feedback


class DenominatorTests(unittest.TestCase):
    def test_previous_broken_fixed_roster_missing_not_repair(self):
        rows=[]
        for code,day,price,prior in [("600001","2025-01-03",10,11),
                                      ("600001","2025-01-06",9,10),
                                      ("600002","2025-01-03",10,11)]:
            rows.append(dict(code=code,date=day,open=price,high=price,low=price,close=price,
                             pre_close=prior,volume=100,status_verified=True,st=False,new_listing=False))
        daily=pd.DataFrame(rows).set_index(["code","date"])
        result=broken_feedback(["600001","600002"],daily,"2025-01-03","2025-01-06")
        self.assertEqual(result["previous_broken_missing"],1)
        self.assertAlmostEqual(result["previous_broken_return_q10"],-.1)
        self.assertEqual(result["previous_broken_negative_share"],1)
        self.assertEqual(result["previous_broken_coverage"],.5)
        self.assertIsNone(broken_feedback(None,daily,"2025-01-03","2025-01-06")["previous_broken_denominator"])
        self.assertEqual(broken_feedback([],daily,"2025-01-03","2025-01-06")["previous_broken_denominator"],0)

    def test_missing_not_failure_and_future_not_used(self):
        dates=["2025-01-02","2025-01-03","2025-01-06"]
        obs=pd.DataFrame([dict(code=c,signal_date=dates[0],boards=1) for c in ["600001","600002"]])
        rows=[]
        for c,d,p,q in [("600001",dates[0],10,9.09),("600002",dates[0],10,9.09),("600001",dates[1],11,10)]:
            rows.append(dict(code=c,date=d,open=p,high=p,low=p,close=p,pre_close=q,volume=100,
                             status_verified=True,st=False,new_listing=False))
        daily=pd.DataFrame(rows).set_index(["code","date"])
        result=build(obs,daily,dates,dates[2]).iloc[0]
        self.assertEqual(result.all_missing,1)
        self.assertEqual(result.all_promotion_observed,1)
        self.assertIsNone(result.all_promotion_complete)
        self.assertEqual(result.all_promotion_lower_bound,.5)
        self.assertEqual(result.all_promotion_upper_bound,1)
        self.assertEqual(result.broken_today_count,0)
        future=rows[-1].copy()
        future.update(date=dates[2],close=999,high=999)
        augmented=pd.concat([daily,pd.DataFrame([future]).set_index(["code","date"])])
        pd.testing.assert_frame_equal(build(obs,daily,dates,dates[2]),build(obs,augmented,dates,dates[2]))


if __name__=="__main__":
    unittest.main()
