import numpy as np
import pytest
import relay_iteration_research as r


def stat(tp=20,fp=20):
    return dict(candidates=100,positive=40,tp=tp,fp=fp,missed_positives=40-tp,
        F05_numerator=5*tp,F05_denominator=5*tp+40-tp+4*fp)


def test_prefix_equal_improvement_keeps_final_or_earlier_tie():
    same=stat()
    assert r.select_round({30:same,60:same,120:same})==120
    better=stat(fp=18)
    assert r.select_round({30:better,60:better,120:same})==30


def test_prefix_recall_loss_cannot_hide_better_precision():
    assert r.select_round({30:stat(tp=0,fp=0),60:stat(tp=18,fp=5),120:stat()})==120


class HGB:
    classes_=np.array([0,1]);n_iter_=120
    def staged_predict_proba(self,x):
        for i in range(1,121):yield np.tile([1-i/121,i/121],(len(x),1))


class Transform:
    def transform(self,x):return x+1


class Pipeline:
    named_steps={'hgb':HGB()}
    def __getitem__(self,key):return Transform()


def test_public_prefix_generator_preserves_exact_requested_rounds_and_model():
    model=Pipeline();result=r.prefix_scores(model,np.array([[0.],[1.]]))
    assert sorted(result)==[30,60,120]
    np.testing.assert_array_equal(result[30],[30/121,30/121])
    assert model.named_steps['hgb'].n_iter_==120


def test_invalid_or_unsaved_rounds_cannot_be_extrapolated():
    for stages in [[],[0],[121],[60,30],[30,30]]:
        with pytest.raises(ValueError):r.prefix_scores(Pipeline(),np.array([[0.]]),stages)
