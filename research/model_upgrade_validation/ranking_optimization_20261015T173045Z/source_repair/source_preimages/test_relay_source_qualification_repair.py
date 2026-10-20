import hashlib
import numpy as np
import pandas as pd
import pytest

import relay_source_qualification_repair as repair


def qualification(shape=0):
    return pd.DataFrame([dict(date='2026-09-07',code='000001',verified_current_price_shape=shape,
        qualification_status='VERIFIED' if shape is not None else 'UNKNOWN',qualification_source='VERIFIED_RAW',
        qualification_PIT_eligible=False,qualification_late_data=True)])


def predictions(arm='CORRECTED_CAUSAL',shape=np.nan,score=.6):
    return pd.DataFrame([dict(date='2026-09-07',code='000001',arm=arm,current_price_shape=shape,
        call='UNKNOWN',score=np.nan,b_score=score,ranking_score=12.5,
        daily_one_word=0,target=1.,label_valid=True)])


def test_attach_preserves_all_original_columns_and_absolute_fixed_call():
    original=predictions()
    result=repair.attach_qualifications(original,qualification())
    pd.testing.assert_frame_equal(result[original.columns],original)
    assert result.loc[0,'verified_current_price_shape']==0
    assert result.loc[0,'recovered_B_call']=='PROMOTE'
    assert np.isnan(result.loc[0,'score'])


def test_nonmissing_original_shape_never_overwritten():
    with pytest.raises(ValueError,match='nonmissing original shape'):
        repair.attach_qualifications(predictions(shape=1),qualification())


def test_unknown_not_automatically_filled_zero_or_negative():
    result=repair.attach_qualifications(predictions(),qualification(None))
    assert pd.isna(result.loc[0,'verified_current_price_shape'])
    assert result.loc[0,'recovered_B_call']=='UNKNOWN'
    assert not result.loc[0,'source_repair_applied']


def test_missing_source_reference_retains_unknown():
    q=qualification();q.loc[0,'code']='000002'
    result=repair.attach_qualifications(predictions(),q)
    assert pd.isna(result.loc[0,'verified_current_price_shape'])
    assert result.loc[0,'recovered_B_call']=='UNKNOWN'


@pytest.mark.parametrize('arm',['PAIRWISE','PAIRWISE_HGB_FIXED','BASE','ST_DUAL','LAGGED'])
def test_pairwise_and_reference_head_never_get_new_direction(arm):
    original=predictions(arm)
    if arm=='BASE': original.loc[0,'call']='FAIL';original.loc[0,'score']=.4
    result=repair.attach_qualifications(original,qualification())
    assert result.loc[0,'recovered_B_call']==original.loc[0,'call']
    pd.testing.assert_frame_equal(result[original.columns],original)


def test_strict_threshold_half_is_fail_but_pairwise_preference_not_used():
    result=repair.attach_qualifications(predictions(score=.5),qualification())
    assert result.loc[0,'recovered_B_call']=='FAIL'
    no_score=predictions().drop(columns='b_score')
    assert repair.attach_qualifications(no_score,qualification()).loc[0,'recovered_B_call']=='UNKNOWN'


def test_verified_one_price_does_not_create_b_call():
    result=repair.attach_qualifications(predictions(),qualification(1))
    assert result.loc[0,'verified_current_price_shape']==1
    assert result.loc[0,'recovered_B_call']=='UNKNOWN'


def test_old_direction_not_reclassified_when_already_decided():
    original=predictions();original.loc[0,'call']='FAIL'
    assert repair.attach_qualifications(original,qualification()).loc[0,'recovered_B_call']=='FAIL'


def patch():
    return dict(signal_date='2026-09-07',date='2026-09-07',code='000001',field='volume',
        proposed_value=123.,source_paths=['raw.json'])


def raw_reference():
    from pathlib import Path
    return {('2026-09-07','2026-09-07','000001',str(Path('raw.json').resolve())):dict(volume=123.)}


def test_field_repair_does_not_mutate_original_or_mix_relative_volume():
    old={('2026-09-07','000001'):dict(close=11.)}
    output=repair.apply_missing_fields(old,[patch()],raw_reference())
    assert output[('2026-09-07','000001')]['volume']==123
    assert 'volume' not in old[('2026-09-07','000001')]
    assert set(output[('2026-09-07','000001')])=={'close','volume'}


def test_field_repair_rejects_nonmissing_overwrite():
    old={('2026-09-07','000001'):dict(volume=999.)}
    with pytest.raises(ValueError,match='overwrite nonmissing'):
        repair.apply_missing_fields(old,[patch()],raw_reference())


def test_unreferenced_raw_and_value_mismatch_rejected():
    with pytest.raises(ValueError,match='unreferenced source'):
        repair.apply_missing_fields({},[patch()],{})
    changed=patch();changed['proposed_value']=456.
    with pytest.raises(ValueError,match='candidate/raw mismatch'):
        repair.apply_missing_fields({},[changed],raw_reference())


def test_hash_change_rejected(tmp_path):
    source=tmp_path/'raw.json';source.write_text('{}',encoding='utf-8')
    expected=hashlib.sha256(source.read_bytes()).hexdigest()
    repair._bindings({str(source):expected})
    source.write_text('{"changed":true}',encoding='utf-8')
    with pytest.raises(ValueError,match='source hash mismatch'):
        repair._bindings({str(source):expected})


def test_only_official_same_adjacent_dates_allowed():
    sessions=['2026-09-03','2026-09-04','2026-09-07','2026-09-08']
    assert repair.validate_reference_date('2026-09-07','2026-09-04',sessions)=='2026-09-04'
    for day in ['2026-09-03','2026-09-06','2026-09-08']:
        with pytest.raises(ValueError,match='nonadjacent or future'):
            repair.validate_reference_date('2026-09-07',day,sessions)


def test_actual_nonclosing_and_missing_volume_remain_unknown():
    current=dict(open=10.5,high=11,low=10.4,close=10.9,volume=100.)
    assert repair.classify_current(current,dict(close=10.))==(None,'OBSERVED_NONCLOSING_10PCT_PRICE_PATH')
    current['close']=11;current['volume']=None
    assert repair.classify_current(current,dict(close=10.))[0] is None


def test_evaluation_keeps_abstained_positives_in_recall_and_unknown_outcomes():
    frame=pd.concat([predictions(),predictions()],ignore_index=True)
    frame.loc[1,['code','target','label_valid']]=['000002',np.nan,False]
    frame['recovered_B_call']=['PROMOTE','UNKNOWN']
    table=repair.evaluate_attachment(frame)
    original=table[(table.period=='2026Sep')&table['mode'].eq('ORIGINAL_FROZEN_CALL')].iloc[0]
    recovered=table[(table.period=='2026Sep')&table['mode'].eq('SOURCE_REPAIRED_RETROSPECTIVE')].iloc[0]
    assert original.Recall==0 and original.unknown_outcomes==1
    assert recovered.Recall==1 and recovered.unknown_outcomes==1
