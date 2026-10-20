import numpy as np
import pandas as pd
import pytest
import relay_monthly_oof_alignment_research as r


def test_monthly_cutoff_precedes_signal_and_does_not_use_block_start():
    assert r.month_cutoff('2025-07-23')=='2025-07-01'
    assert r.month_cutoff('2025-08-04')=='2025-08-01'


def test_exact252_is_required(monkeypatch):
    monkeypatch.setattr(r.history,'training_rows',lambda *args:pd.DataFrame({'date':['2024-01-01'],'target_date':['2024-01-02']}))
    with pytest.raises(ValueError,match='exact252'):r.fixed_month_training(pd.DataFrame(),'2025-01-01')


def test_complete_monthly_oof_unknown_preserved_and_source_refresh_verified():
    full=pd.DataFrame(dict(date=['2025-01-02','2025-02-03','2025-03-03'],code=['000001']*3,
        target_date=['2025-01-03',np.nan,'2025-03-04'],target=[1.,np.nan,0.],label_valid=[True,False,True],
        frozen_original_nonword_identity=[0.,0.,0.]))
    shape=full[r.old.KEYS].copy();shape['verified_current_price_shape']=[0.,0.,np.nan]
    plan={'validation_dates':[['2025-01-02'],['2025-02-03'],['2025-03-03']]}
    raw=full[r.old.KEYS].copy();raw['raw_score']=[.6,.7,.8]
    raw['first_stage_model_cutoff']=['2025-01-01','2025-02-01','2025-03-01']
    out=r.assemble_oof(full,shape,plan,raw)
    assert len(out)==3 and out.target.isna().sum()==1
    assert out.call.tolist()==['PROMOTE','PROMOTE','UNKNOWN']
    raw.loc[1,'first_stage_model_cutoff']='2025-01-01'
    with pytest.raises(ValueError,match='refresh'):r.assemble_oof(full,shape,plan,raw)


def test_attach_facts_uses_frozen_complete_pool_not_union_nan_columns():
    full=pd.DataFrame({'date':['2025-01-02'],'code':['000001']})
    for col in r.old.INDIVIDUAL:full[col]=2.
    ref=full[['date','code','boards']].copy();ref['evaluation_call']='PROMOTE';ref['b_score']=.6
    for col in r.old.INDIVIDUAL[1:]:ref[col]=np.nan
    out=r.attach_inference(ref,full)
    assert out[r.old.INDIVIDUAL].eq(2).all().all()
    assert ref.turnover.isna().all()
