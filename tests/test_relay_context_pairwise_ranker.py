import numpy as np
import pandas as pd
import pytest
import relay_context_pairwise_ranker as ranker


def fixture():
    return pd.DataFrame(dict(date=["a"]*4+["b"],code=["1","2","3","4","1"],x=[4.,1.,np.nan,2.,9.],market=[7.]*4+[8.],target=[1.,0.,np.nan,1.,1.],label_valid=[True,True,False,True,True]))


def test_pairs_keep_shared_context_and_native_missing():
    a=np.array([[3.,7.,np.nan]]);b=np.array([[1.,7.,2.]])
    pair=ranker.pair_features(a,b)
    assert pair.shape==(1,9) and pair[0,1]==pair[0,4]==7 and pair[0,7]==0
    assert np.isnan(pair[0,2]) and pair[0,5]==2 and np.isnan(pair[0,8])


def test_unknown_targets_never_become_negative_and_day_mass_equal():
    x,y,w,groups=ranker.build_pairs(fixture(),["x","market"])
    assert len(y)==4 and np.sum(y)==2 and w.sum()==1
    assert groups.directed_pairs.tolist()==[4,0]
    assert groups.unknown_labels.tolist()==[1,0]
    assert groups.complete_nonword_candidates.tolist()==[4,1]


class FakeModel:
    classes_=np.array([0,1])
    def predict_proba(self,pairs):
        p=np.where(np.nan_to_num(pairs[:,0])>np.nan_to_num(pairs[:,2]),.8,.3)
        return np.column_stack([1-p,p])


def test_complete_opponents_and_complementarity_and_singleton_unknown():
    result=ranker.predict_cohort(FakeModel(),fixture(),["x","market"])
    assert len(result)==5 and result.opponent_count.tolist()==[3,3,3,3,0]
    assert result.pairwise_antisymmetry_error.max()<1e-15
    assert pd.isna(result.ranking_score.iloc[-1])


def test_outcomes_and_target_mutation_cannot_change_inference():
    f=fixture();a=ranker.predict_cohort(FakeModel(),f,["x","market"])
    f["target"]=1-f.target;f["label_valid"]=False
    pd.testing.assert_frame_equal(a,ranker.predict_cohort(FakeModel(),f,["x","market"]))


def test_candidate_input_order_does_not_change_prediction():
    f=fixture();a=ranker.predict_cohort(FakeModel(),f,["x","market"])
    z=ranker.predict_cohort(FakeModel(),f.iloc[::-1],["x","market"])
    pd.testing.assert_frame_equal(a,z)


def test_no_mixed_day_cannot_silently_train():
    f=fixture();f["target"]=1
    with pytest.raises(ValueError,match="no mixed day"):ranker.build_pairs(f,["x","market"])
