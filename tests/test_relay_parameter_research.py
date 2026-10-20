from fractions import Fraction
import numpy as np
import pandas as pd
import pytest
import relay_parameter_research as r


def stat(tp=20,fp=20,positive=40):
    fn=positive-tp;den=5*tp+fn+4*fp
    return dict(candidates=100,positive=positive,tp=tp,fp=fp,missed_positives=fn,F05_numerator=5*tp,F05_denominator=den)


def test_capacity_exact_tie_keeps_baseline_and_candidates_tie_prefers_compact():
    baseline=stat()
    assert r.select_capacity(dict(BASE=baseline,COMPACT=baseline,EXPANDED=baseline))=='BASE'
    better=stat(fp=18)
    assert r.select_capacity(dict(BASE=baseline,COMPACT=better,EXPANDED=better))=='COMPACT'


def test_low_false_positives_cannot_hide_recall_collapse():
    zero=stat(tp=0,fp=0)
    assert not r.guard(zero,stat())
    assert r.select_capacity(dict(BASE=stat(),COMPACT=zero,EXPANDED=zero))=='BASE'


def test_recall_guard_is_exact_and_denominator_cannot_drift():
    assert r.guard(stat(tp=19,fp=18),stat(tp=20,fp=20))
    assert not r.guard(stat(tp=18,fp=10),stat(tp=20,fp=20))
    wrong=stat();wrong['candidates']=99
    with pytest.raises(ValueError):r.guard(wrong,stat())


def test_threshold_uses_original_baseline_guard_not_cumulative_selected_loss():
    baseline=stat(tp=20,fp=20)
    fixed=stat(tp=19,fp=18)
    choices={t:fixed for t in r.THRESHOLDS}
    choices[.55]=stat(tp=18,fp=5)
    assert r.select_threshold(choices,baseline)==.5


def test_threshold_equal_scores_keep_half_or_fixed_nearest_then_higher_tie():
    base=stat();choices={t:base for t in r.THRESHOLDS}
    assert r.select_threshold(choices,base)==.5
    choices[.45]=choices[.55]=stat(fp=18)
    choices[.4]=choices[.6]=stat(fp=18)
    assert r.select_threshold(choices,base)==.55


def test_unknown_source_positive_counts_as_missed_and_threshold_is_strict():
    oof=pd.DataFrame(dict(date=['d']*4,code=list('abcd'),target=[1,1,0,0],label_valid=[True]*4,
        raw_score=[.6,.5,.7,.9],verified_current_price_shape=[np.nan,0,0,1]))
    s=r.decision_statistics(oof,.5)
    assert (s['tp'],s['fp'],s['missed_positives'])==(0,1,2)
    assert r.score_fraction(s)==Fraction(0)


def test_unknown_labels_cannot_be_inserted_into_training_selection():
    oof=pd.DataFrame(dict(date=['d']*2,code=['a','b'],target=[1,0],label_valid=[True,False],raw_score=[.6,.4],verified_current_price_shape=[0,0]))
    with pytest.raises(ValueError):r.decision_statistics(oof,.5)


def reference():
    return pd.DataFrame(dict(score=[.7,np.nan,np.nan],call=['PROMOTE','UNKNOWN','UNKNOWN'],verified_current_price_shape=[0.,1.,np.nan],target=[1,0,np.nan]))


def test_non_B_and_unknown_routes_never_gain_direction_from_new_parameters():
    old=reference();new=r.apply_head(old,np.array([.6,.9,.9]),.6,r.ARMS[1],'COMPACT')
    assert new.call.tolist()==['FAIL','UNKNOWN','UNKNOWN']
    assert new.score.iloc[1:].isna().all()
    assert old.call.tolist()==['PROMOTE','UNKNOWN','UNKNOWN']


def test_target_perturbation_does_not_change_raw_score_route_or_rank():
    old=reference();changed=old.copy();changed['target']=[0,1,1]
    a=r.apply_head(old,np.array([.7,.8,.9]),.5,r.ARMS[0],'EXPANDED')
    b=r.apply_head(changed,np.array([.7,.8,.9]),.5,r.ARMS[0],'EXPANDED')
    pd.testing.assert_frame_equal(a[['score','b_score','call','ranking_score']],b[['score','b_score','call','ranking_score']],check_exact=True)
