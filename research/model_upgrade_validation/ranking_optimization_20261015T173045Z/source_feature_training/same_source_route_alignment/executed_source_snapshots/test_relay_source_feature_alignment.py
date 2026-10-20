import numpy as np
import pandas as pd
import pytest

from relay_source_feature_alignment import align_routes,predict_with_same_source_routes
from relay_source_feature_research import ARM


def rows():
    return pd.DataFrame([
        dict(date='2026-09-04',code='000001',arm='CORRECTED_CAUSAL',verified_current_price_shape=1.,score=np.nan,call='UNKNOWN',evaluation_call='UNKNOWN',b_score=.8),
        dict(date='2026-09-04',code='000001',arm=ARM,verified_current_price_shape=1.,score=.9,call='PROMOTE',evaluation_call='PROMOTE',b_score=.85),
        dict(date='2026-09-04',code='000002',arm='CORRECTED_CAUSAL',verified_current_price_shape=0.,score=np.nan,call='UNKNOWN',evaluation_call='PROMOTE',b_score=.6),
        dict(date='2026-09-04',code='000002',arm=ARM,verified_current_price_shape=0.,score=.7,call='PROMOTE',evaluation_call='PROMOTE',b_score=.7),
        dict(date='2026-09-04',code='000003',arm='CORRECTED_CAUSAL',verified_current_price_shape=np.nan,score=np.nan,call='UNKNOWN',evaluation_call='UNKNOWN',b_score=.6),
        dict(date='2026-09-04',code='000003',arm=ARM,verified_current_price_shape=np.nan,score=np.nan,call='UNKNOWN',evaluation_call='UNKNOWN',b_score=.8),
    ])


def test_new_A_qualification_cannot_create_direction_from_BASE():
    source=rows();out=align_routes(source)
    assert out.loc[1,'aligned_evaluation_call']=='UNKNOWN'
    assert pd.isna(out.loc[1,'same_source_aligned_score'])
    assert out.loc[1,'call']=='PROMOTE'  # Original executed image remains.
    pd.testing.assert_frame_equal(out[source.columns],source,check_exact=True)


def test_verified_B_keeps_own_absolute_score_and_unknown_stays_unknown():
    out=align_routes(rows())
    assert out.loc[3,'aligned_evaluation_call']=='PROMOTE' and out.loc[3,'same_source_aligned_score']==.7
    assert out.loc[5,'aligned_evaluation_call']=='UNKNOWN' and pd.isna(out.loc[5,'effective_verified_shape'])


def test_B_direction_cannot_be_borrowed_from_other_score():
    source=rows();source.loc[3,'b_score']=.5
    with pytest.raises(ValueError,match='fixed absolute score'):
        align_routes(source)


def test_actual_inference_wrapper_cannot_give_new_A_or_unknown_directions():
    table=rows();baseline=table[table.arm.eq('CORRECTED_CAUSAL')].copy()
    current=table[table.arm.eq(ARM)][['date','code']].copy()
    shape=table[table.arm.eq(ARM)][['date','code','verified_current_price_shape']].copy()
    result=predict_with_same_source_routes(current,[.85,.7,.8],shape,baseline)
    assert result.call.tolist()==['UNKNOWN','PROMOTE','UNKNOWN']
    assert pd.isna(result.score.iloc[0]) and result.score.iloc[1]==.7 and pd.isna(result.score.iloc[2])
    assert result.ranking_score.tolist()==[.85,.7,.8]


def test_actual_inference_wrapper_strict_threshold_and_reference_identity():
    table=rows();baseline=table[table.arm.eq('CORRECTED_CAUSAL')].copy()
    current=table[table.arm.eq(ARM)][['date','code']].copy()
    shape=table[table.arm.eq(ARM)][['date','code','verified_current_price_shape']].copy()
    result=predict_with_same_source_routes(current,[.9,.5,.9],shape,baseline)
    assert result.call.iloc[1]=='FAIL'
    baseline.loc[baseline.code.eq('000001'),'evaluation_call']='FAIL'
    result=predict_with_same_source_routes(current,[.9,.5,.9],shape,baseline)
    assert result.call.iloc[0]=='FAIL'
