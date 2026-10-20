import numpy as np
import pandas as pd
import pytest

import relay_source_feature_research as research


def input_frame():
    return pd.DataFrame([dict(date='2026-01-05',code='000001',target_date='2026-01-06',target=1.,label_valid=True,
        causal_price_shape=np.nan,other_feature=4.,current_price_shape=np.nan,
        current_price_path_status='UNKNOWN_INCOMPLETE_CONTINUOUS_PATH',parent='UNKNOWN')])


def test_only_one_empty_feature_changes_all_metadata_and_labels_retained():
    old=input_frame();verified=pd.DataFrame([dict(date='2026-01-05',code='000001',verified_current_price_shape=0.)])
    result,changes=research.patched_feature_copy(old,['causal_price_shape','other_feature'],verified,1)
    assert result.causal_price_shape.iloc[0]==0 and pd.isna(old.causal_price_shape.iloc[0])
    pd.testing.assert_frame_equal(old.drop(columns='causal_price_shape'),result.drop(columns='causal_price_shape'),check_exact=True)
    assert len(changes)==1


def test_nonempty_or_wrong_key_never_overwritten():
    old=input_frame();old.loc[0,'causal_price_shape']=1.
    verified=pd.DataFrame([dict(date='2026-01-05',code='000001',verified_current_price_shape=0.)])
    with pytest.raises(ValueError,match='nonmissing'):
        research.patched_feature_copy(old,['causal_price_shape'],verified,1)
    assert old.causal_price_shape.iloc[0]==1
    old.loc[0,'causal_price_shape']=np.nan;verified.loc[0,'code']='000002'
    with pytest.raises(ValueError,match='outside fixed input'):
        research.patched_feature_copy(old,['causal_price_shape'],verified,1)


def test_unresolved_shape_stays_missing_and_never_certifies_parent_path():
    old=input_frame();verified=pd.DataFrame([dict(date='2026-01-05',code='000001',verified_current_price_shape=np.nan)])
    result,changes=research.patched_feature_copy(old,['causal_price_shape'],verified,0)
    assert pd.isna(result.causal_price_shape.iloc[0]) and changes.empty
    assert result.current_price_path_status.iloc[0]=='UNKNOWN_INCOMPLETE_CONTINUOUS_PATH'


def test_model_reuse_depends_on_complete_training_digest_only():
    assert research.choose_reuse('same','same')
    assert not research.choose_reuse('old','new')


def test_verified_shape_is_separate_from_original_metadata():
    old=input_frame();verified=pd.DataFrame([dict(date='2026-01-05',code='000001',verified_current_price_shape=0.)])
    shape=research.verified_shapes(old,verified)
    assert shape.verified_current_price_shape.iloc[0]==0
    assert pd.isna(old.current_price_shape.iloc[0])


def route_frame():
    return pd.DataFrame([dict(date='2026-10-15',code=f'{i:06}',name='x',boards=2,target_date='2026-10-16',target=np.nan,label_valid=False,
        daily_one_word=0.,current_price_shape=np.nan,review_cycle_state='UNKNOWN',current_price_path_status='UNKNOWN') for i in [1,2,3]])


def test_fixed_route_keeps_unknown_and_A_BASE_separate():
    frame=route_frame()
    shape=frame[['date','code']].copy();shape['verified_current_price_shape']=[0.,1.,np.nan]
    base=pd.Series([.1,.9,.3],index=pd.MultiIndex.from_frame(frame[['date','code']]))
    result=research.route_predictions(frame,np.array([.5,.2,.95]),shape,base)
    assert result.call.tolist()==['FAIL','PROMOTE','UNKNOWN']
    assert result.ranking_score.tolist()==[.5,.2,.95]
    assert result.score.iloc[1]==.9 and pd.isna(result.score.iloc[2])
    assert result.target.isna().all() and not result.label_valid.any()


def test_top3_full_slot_counts_unknown_and_empty_slots_and_dates():
    frame=pd.DataFrame([dict(date='2026-01-05',code='000002',daily_one_word=0.,ranking_score=.5,target=1,label_valid=True,evaluation_call='PROMOTE'),
        dict(date='2026-01-05',code='000001',daily_one_word=0.,ranking_score=.5,target=np.nan,label_valid=False,evaluation_call='UNKNOWN'),
        dict(date='2026-01-06',code='000003',daily_one_word=1.,ranking_score=.9,target=1,label_valid=True,evaluation_call='PROMOTE')])
    stats,picks=research.full_slot_ranking(frame,3)
    assert stats['full_slots']==6 and stats['empty_slots']==4 and stats['unknown_selected']==1
    assert stats['hits']==1 and stats['full_slot_hit_rate']==1/6
    assert picks.code.tolist()==['000001','000002']


def test_fixed_cost1_weights_same_old_formula():
    frame=pd.DataFrame([dict(date='2026-01-05',target=0),dict(date='2026-01-05',target=1),dict(date='2026-01-06',target=0)])
    w=research.original.weights(frame,1.)
    assert abs(w.sum()-2)<1e-12
    assert abs(w[frame.target.eq(1)].sum()-w[frame.target.eq(0)].sum())<1e-12
