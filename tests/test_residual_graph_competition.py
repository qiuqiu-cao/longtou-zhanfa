import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/first_board_residual_graph_20261031'))
from graph_features import build_day,FIELDS

def test_same_peer_competition_and_input_order():
    rng=np.random.default_rng(18);market=rng.normal(size=60);v=rng.normal(size=60)
    r=np.column_stack([market+v,market+v*.8,market+v*.6]+[market+rng.normal(size=60) for _ in range(4)])
    codes=list('abcdefg');sc=dict(a=.7,b=.2,c=-.1)
    a,_=build_day(r,codes,list('abc'),sc);b,_=build_day(r[:,::-1],codes[::-1],list('cba'),sc)
    pd.testing.assert_frame_equal(a.set_index('code').sort_index(),b.set_index('code').sort_index())
    assert a.loc[a.code.eq('a'),'peer_count'].iloc[0]>0

def test_missing_and_zero_variance_unknown():
    r=np.ones((60,5));r[0,0]=np.nan
    a,_=build_day(r,list('abcde'),list('ab'),dict(a=1.,b=0.))
    assert a[FIELDS].isna().all().all() and a.peer_count.eq(0).all()
