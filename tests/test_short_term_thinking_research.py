import unittest

import numpy as np
import pandas as pd

import short_term_thinking_research as r
from short_term_thinking_preprocessing import EmptyTrainingColumnGuard


class ResearchIntegrityTests(unittest.TestCase):
    def test_training_uses_last252_full_days_then_real_shape_filter(self):
        days=pd.bdate_range("2023-01-02",periods=302).strftime("%Y-%m-%d").tolist()
        full=pd.DataFrame([dict(date=d,code=str(k),target_date=days[i+1],target=float(k),label_valid=True,
            daily_one_word=float(k),signal_one_word=float(1-k)) for i,d in enumerate(days[:-1]) for k in (0,1)])
        # An apparently historical signal with a not-yet-mature target is out.
        full.loc[full.date.eq(days[-3]),"target_date"]=days[-1]
        train=r.training_subset(full,days[-1],"ST_B_ONLY")
        self.assertEqual(train.date.nunique(),252)
        self.assertTrue(train.daily_one_word.eq(0).all())
        self.assertTrue(train.signal_one_word.eq(1).all())
        self.assertTrue(train.target_date.lt(days[-1]).all())

    def test_weights_are_balanced_from_training_labels_only(self):
        train=pd.DataFrame(dict(date=["a","a","b","c"],target=[0.,0.,0.,1.]))
        weights=r.training_weights(train)
        self.assertAlmostEqual(weights.sum(),3.)
        self.assertAlmostEqual(weights[train.target.eq(0)].sum(),1.5)
        self.assertAlmostEqual(weights[train.target.eq(1)].sum(),1.5)

    def test_empty_column_guard_never_learns_from_future_observation(self):
        train=pd.DataFrame(dict(missing=[np.nan,np.nan],partial=[1.,np.nan]))
        guard=EmptyTrainingColumnGuard().fit(train)
        test=pd.DataFrame(dict(missing=[999.],partial=[np.nan]))
        transformed=guard.transform(test)
        self.assertEqual(transformed.missing.iloc[0],0.)
        self.assertTrue(np.isnan(transformed.partial.iloc[0]))
        self.assertTrue(train.missing.isna().all())

    def test_top2_preserves_unknown_label_and_tie_breaks_by_code(self):
        f=pd.DataFrame([dict(date="a",code=c,boards=2,score=.5,target=t,label_valid=True,daily_one_word=0)
            for c,t in [("2",1.),("1",np.nan),("3",0.)]])
        chosen=r.top_selection(f,2)
        self.assertEqual(chosen.code.tolist(),["1","2"])
        m=r.top_metrics(f,2)
        self.assertEqual((m["selected"],m["known"],m["groups_with_unresolved_truth"]),(2,1,1))

    def test_blocked_lower_limit_loss_is_in_risk_denominator(self):
        days=["2025-01-02","2025-01-03","2025-01-06","2025-01-07"]
        bars=pd.DataFrame([dict(date=days[0],code="1",open=10.,high=10.,low=10.,close=10.,volume=100.),
            dict(date=days[1],code="1",open=10.,high=10.5,low=9.8,close=10.,volume=100.),
            dict(date=days[2],code="1",open=9.,high=9.,low=9.,close=9.,volume=100.),
            dict(date=days[3],code="1",open=9.1,high=9.3,low=9.0,close=9.2,volume=100.)])
        log=r.proxy_returns(pd.DataFrame([dict(date=days[0],code="1")]),dict(sessions=days,bars=bars))
        self.assertEqual(log.B.iloc[0],days[1])
        self.assertIsNone(log.b1_net.iloc[0])
        self.assertEqual(log.status_b1.iloc[0],"SELL_BLOCKED_ONE_PRICE_DOWN_LIMIT")
        m=r.execution_metrics(log)
        self.assertEqual(m["b1"]["risk_loss_ge7pct_including_blocked"],1.)
        self.assertEqual(m["b1"]["sell_blocked"],1)
        self.assertEqual(m["actual_fills"],0)


if __name__=="__main__":unittest.main()
