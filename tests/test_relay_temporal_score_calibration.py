import numpy as np
import pandas as pd
import pytest
import relay_temporal_score_calibration as r


def window():
    dates = pd.bdate_range('2024-01-01', periods=253).strftime('%Y-%m-%d')
    return pd.DataFrame([dict(date=day, code=str(code), target=code, target_date=dates[i+1], label_valid=True)
        for i,day in enumerate(dates[:-1]) for code in [0,1]])


def test_inner_target_must_be_mature_before_entire_validation_block():
    data = window(); splits = r.temporal_splits(data)
    assert [x.date.nunique() for x,y in splits] == [62,125,188]
    for train,valid in splits:
        assert train.target_date.max() < valid.date.min()
        assert len(set(train.date)&set(valid.date)) == 0
        assert valid.date.nunique() == 63
    # A late label is removed even if its signal date lies in the past.
    data.loc[data.date.eq(data.date.min()),'target_date']='2099-01-01'
    assert all(data.date.min() not in set(x.date) for x,y in r.temporal_splits(data))


def test_duplicate_or_incomplete_window_cannot_silently_enter_temporal_cv():
    data=window()
    with pytest.raises(ValueError): r.temporal_splits(data.iloc[2:])
    with pytest.raises(ValueError): r.temporal_splits(pd.concat([data,data.iloc[:1]],ignore_index=True))


def test_calibration_day_mass_does_not_balance_classes_or_renormalize_rows():
    frame=pd.DataFrame(dict(date=['a','a','b'],target=[0,1,0],label_valid=[True]*3))
    weights=r.calibration_weights(frame)
    np.testing.assert_allclose(weights,[.5,.5,1.])
    assert weights[frame.target.eq(0)].sum()==1.5
    assert weights.sum()==2


def test_calibration_never_uses_unknown_labels_as_negative():
    frame=pd.DataFrame(dict(date=['a','b'],target=[0,1],label_valid=[True,False]))
    with pytest.raises(ValueError):r.calibration_weights(frame)


def reference():
    return pd.DataFrame(dict(arm=[r.BASE_ARM]*3,verified_current_price_shape=[0.,1.,np.nan],
        score=[.7,np.nan,np.nan],b_score=[.7,.8,.9],ranking_score=[.7,.8,.9],
        call=['PROMOTE','UNKNOWN','UNKNOWN'],target=[1,0,np.nan]))


def test_new_calibration_only_changes_verified_B_route_and_keeps_original_rank():
    old=reference(); result=r.apply_head(old,[.7,.8,.9],[.5,.85,.95])
    assert result.call.tolist()==['FAIL','UNKNOWN','UNKNOWN']
    assert result.score.iloc[0]==.5 and result.score.iloc[1:].isna().all()
    pd.testing.assert_series_equal(result.ranking_score,old.ranking_score,check_exact=True)
    assert old.call.tolist()==['PROMOTE','UNKNOWN','UNKNOWN']


def test_target_perturbation_cannot_change_score_direction_or_rank():
    old=reference(); new=old.copy();new['target']=[0,1,1]
    a=r.apply_head(old,[.7,.8,.9],[.6,.9,.95]);b=r.apply_head(new,[.7,.8,.9],[.6,.9,.95])
    pd.testing.assert_frame_equal(a[['score','b_score','call','ranking_score']],b[['score','b_score','call','ranking_score']],check_exact=True)


def test_non_finite_out_of_range_raw_scores_are_rejected():
    for value in [np.nan,np.inf,-.1,1.1]:
        with pytest.raises(ValueError):r.logit_column([value])
    assert np.isfinite(r.logit_column([0.,1.])).all()


def test_reversing_calibrator_is_rejected_before_any_prediction():
    class Bad:
        coef_=np.array([[-1.]])
        def predict_proba(self,x):raise AssertionError('should never run')
    with pytest.raises(ValueError,match='slope'):r.calibrated_scores(Bad(),[.5])
