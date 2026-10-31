import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

path=Path(__file__).resolve().parents[1]/'research/first_board_quant_price_volume_20261031/quant_factors.py'
spec=importlib.util.spec_from_file_location('daily_vwap_distribution',path)
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)

def fixture():
    p=np.arange(22);c=np.full(22,10.)
    return pd.DataFrame(dict(code='000001',date=[f'd{i}' for i in p],p=p,open=c,high=c+1,low=c-1,close=c,pre_close=c,volume=100.,amount=1000.))

def test_constant_prices_real_zero_not_missing():
    v,s=f.window_values(fixture());np.testing.assert_allclose(v,[0,0,0,0],atol=1e-14)
    assert s.startswith('COMPLETE')

def test_weighted_distribution_direct_definition_and_excludes_signal_vwap():
    w=fixture();w.loc[16:20,'amount']=1050.
    v,s=f.window_values(w)
    assert v[1]==pytest.approx(np.log(1.05))
    assert v[2]==pytest.approx(np.log(10/10.5)) and v[3]==pytest.approx(.25)
    w.loc[21,'amount']=900.;np.testing.assert_array_equal(v,f.window_values(w)[0])

def test_price_and_volume_scale_invariance():
    w=fixture();w.loc[16:20,'amount']=1050.
    v=f.window_values(w)[0]
    z=w.copy();z[['open','high','low','close','pre_close','amount']]*=2
    np.testing.assert_allclose(v,f.window_values(z)[0],atol=1e-14)
    z=w.copy();z[['volume','amount']]*=10
    np.testing.assert_allclose(v,f.window_values(z)[0],atol=1e-14)

def test_missing_reference_and_wrong_unit_never_auto_repaired():
    w=fixture();w.loc[10,'pre_close']=8.
    assert f.window_values(w)[1]=='REFERENCE_DISCONTINUITY'
    w=fixture();w['volume']/=100
    assert f.window_values(w)[1]=='AMOUNT_VOLUME_UNIT_OR_PRICE_MISMATCH'
    w=fixture().drop(index=8)
    assert f.window_values(w)[1]=='MISSING_OFFICIAL_SESSION_WINDOW'

def test_future_rows_do_not_change_prior_signal_and_label_columns_rejected():
    w=fixture();keys=pd.DataFrame({'signal_date':['d21'],'code':['000001']})
    a=f.build(w,keys)
    future=w.tail(1).assign(date='d22',p=22,amount=1099.)
    pd.testing.assert_frame_equal(a,f.build(pd.concat([w,future]),keys))
    with pytest.raises(ValueError):f.build(w,keys.assign(target=1))
