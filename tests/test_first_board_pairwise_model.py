import numpy as np
import pandas as pd
import first_board_pairwise_model as pair
import first_second_stage_model as stage


def test_unknowns_never_become_pairwise_negative_labels():
    rows=pd.DataFrame(dict(signal_date=['2025-01-02']*3,code=['000001','000002','000003'],f=[1.,0.,100.],promotion_next_LABEL_ONLY=[1.,0.,np.nan]))
    x,y,w,groups=pair.build_pairs(rows,['f'])
    assert len(x)==2
    assert set(y)=={0,1}
    assert groups.unknown.iloc[0]==1
    assert groups.complete.iloc[0]==3
    assert not (x==100).any()
    assert w.sum()==2


class Preference:
    def predict_proba(self,x):
        p=1/(1+np.exp(-x[:,-1]))
        return np.column_stack([1-p,p])


def test_complete_day_ranking_uses_all_rivals_and_no_future_labels():
    model=pair.PairRanker(Preference(),np.ones(3,dtype=bool),['f'])
    rows=pd.DataFrame(dict(signal_date=['2025-07-01']*3,code=['000001','000002','000003'],f=[1.,2.,0.],promotion_next_LABEL_ONLY=[0,1,0]))
    a=model.predict(rows)
    b=model.predict(rows.drop(columns=[stage.TARGET]))
    pd.testing.assert_frame_equal(a,b)
    assert a.opponents.eq(2).all()
    assert a.loc[a.ranking_score.idxmax(),'code']=='000002'
    assert a.ranking_status.eq('COHORT_PREFERENCE_NOT_EVENT_PROBABILITY').all()
