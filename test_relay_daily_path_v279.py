import unittest
import numpy as np
import pandas as pd
from relay_daily_path_features_v279 import bar_features, attach, NEW_FIELDS
from relay_daily_path_challenger_v279 import eligible_nonword


CALENDAR=["2026-09-01","2026-09-02","2026-09-03","2026-09-04"]


def bars():
    return pd.DataFrame({"date":CALENDAR,"code":["600001"]*4,
        "open":[10.,11.,11.5,12.],"close":[10.,11.,12.1,13.31],
        "high":[10.,11.,12.1,13.31],"low":[10.,11.,11.,12.],
        "pre_close":[9.09,10.,11.,12.1],"volume":[100.,200.,300.,400.],
        "amount":[1000.,2000.,3000.,4000.],"turnover":[1.,2.,3.,4.]})


def candidate():
    return pd.DataFrame({"date":["2026-09-03"],"code":["600001"],"price":[12.1],"boards":[3],"signal_amount":[3000.]})


class DailyPathTests(unittest.TestCase):
    def test_pairing_keeps_only_one_outcome_column(self):
        frame=pd.DataFrame({'date':['2026-09-01'],'code':['600001'],'boards':[2],
                            'probability':[.8],'daily_true_one_word':[0],
                            'label_valid':[True],'target':[1],'promoted_next_day':[1]})
        picked=eligible_nonword(frame)
        self.assertEqual(picked.columns.tolist(),['date','promoted_next_day'])
        self.assertEqual(int(picked.promoted_next_day.sum()),1)
    def test_future_perturbation_cannot_change_current_features(self):
        original=attach(candidate(),bars(),CALENDAR)
        changed=bars();changed.loc[3,["open","close","high","low","amount","volume","turnover"]]=[1,500,500,1,1e12,1e12,99]
        later=attach(candidate(),changed,CALENDAR)
        pd.testing.assert_frame_equal(original[NEW_FIELDS],later[NEW_FIELDS])

    def test_ratio_uses_immediately_previous_session(self):
        f=bar_features(bars(),CALENDAR).set_index("date")
        self.assertEqual(f.loc["2026-09-03","daily_turnover_previous_session_ratio"],1.5)
        self.assertEqual(f.loc["2026-09-03","daily_first_open_after_one_word"],1.)
        self.assertEqual(f.loc["2026-09-04","daily_amount_previous3_ratio"],2.)

    def test_missing_session_does_not_get_bridged(self):
        f=bar_features(bars().drop(index=1),CALENDAR).set_index("date")
        self.assertTrue(np.isnan(f.loc["2026-09-03","daily_amount_previous_session_ratio"]))
        merged=attach(candidate(),bars().drop(index=1),CALENDAR)
        self.assertFalse(merged.iloc[0].episode_daily_complete)
        self.assertTrue(merged[[k for k in NEW_FIELDS if k.startswith('first_board_')]].isna().all().all())

    def test_first_board_fields_do_not_use_signal_turnover(self):
        f=attach(candidate(),bars(),CALENDAR).iloc[0]
        self.assertEqual(f.first_board_turnover,.01)
        self.assertEqual(f.first_board_true_one_word,1.)
        self.assertAlmostEqual(f.first_board_amount_to_signal_ratio,1/3)

    def test_wrong_close_invalidates_all_clues(self):
        c=candidate();c.loc[0,'price']=99
        f=attach(c,bars(),CALENDAR)
        self.assertTrue(f[NEW_FIELDS].isna().all().all())

    def test_duplicate_bar_and_unknown_calendar_date_rejected(self):
        with self.assertRaises(ValueError):bar_features(pd.concat([bars(),bars().iloc[:1]]),CALENDAR)
        with self.assertRaises(ValueError):bar_features(bars(),CALENDAR[:-1])


if __name__=='__main__':unittest.main()
