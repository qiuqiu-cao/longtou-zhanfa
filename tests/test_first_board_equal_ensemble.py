import numpy as np
import pandas as pd
import pytest
import first_board_equal_ensemble as ensemble


def fixture():
    rows=pd.DataFrame(dict(signal_date=["2024-11-01"]*3+["2024-11-04"]*2,
        code=["000001","000002","000003","000001","000002"],promotion_next_LABEL_ONLY=[1.,0.,np.nan,0.,1.]))
    for head,raw in {"BASE":[.8,.2,.2,.1,.9],"LINEAR":[.4,.8,.1,.5,.5],"ORDERED":[.6,.6,.1,.3,.7]}.items():
        rows["component_"+head+"_raw"]=raw
        rows["component_"+head+"_calibrated"]=np.array(raw)*.5
    return rows


def test_exact_equal_weights_and_complete_day_tie_ranks():
    rows=fixture()
    probability=ensemble.combine(rows,ensemble.ARMS[0])
    expected=rows[["component_"+head+"_calibrated" for head in ensemble.HEADS]].to_numpy().sum(axis=1)/3
    np.testing.assert_allclose(probability.model_score,expected,atol=1e-15)
    ranked=ensemble.combine(rows,ensemble.ARMS[1])
    np.testing.assert_allclose(ranked.model_score.iloc[:3],[(1+2/3+5/6)/3,(.5+1+5/6)/3,(.5+1/3+1/3)/3])
    assert ranked.calibrated_score.isna().all()


def test_future_labels_dates_and_unknown_candidates_do_not_change_signal_composition():
    rows=fixture()
    for arm in ensemble.ARMS:
        expected=ensemble.combine(rows,arm)
        changed=rows.copy();changed.promotion_next_LABEL_ONLY=[0.,1.,1.,1.,0.]
        for head in ensemble.HEADS:
            changed.loc[changed.signal_date.eq("2024-11-04"),"component_"+head+"_raw"]=.99
            changed.loc[changed.signal_date.eq("2024-11-04"),"component_"+head+"_calibrated"]=.99
        actual=ensemble.combine(changed,arm)
        np.testing.assert_allclose(expected.model_score.iloc[:3],actual.model_score.iloc[:3])
        assert len(actual)==5
    missing=rows.copy();missing.loc[0,"component_LINEAR_calibrated"]=np.nan
    with pytest.raises(ValueError,match="no silent"):
        ensemble.combine(missing,ensemble.ARMS[0])
    with pytest.raises(ValueError,match="Duplicate"):
        ensemble.combine(pd.concat([rows,rows.iloc[:1]],ignore_index=True),ensemble.ARMS[1])
