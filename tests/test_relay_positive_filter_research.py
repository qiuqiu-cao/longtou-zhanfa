from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
import relay_positive_filter_research as r


def frame():
    return pd.DataFrame(dict(date=['2025-07-01']*4+['2025-07-02'],
        code=['000001','000002','000003','000004','000005'],
        target_date=['2025-07-02']*4+['2025-07-03'],
        target=[1.,0.,np.nan,1.,0.],label_valid=[True,True,False,True,True],
        daily_one_word=[0.,0.,0.,0.,0.],verified_current_price_shape=[0.,0.,0.,np.nan,0.],
        base_call=['PROMOTE','PROMOTE','PROMOTE','UNKNOWN','FAIL'],
        call=['PROMOTE','PROMOTE','PROMOTE','UNKNOWN','FAIL'],
        ranking_score=[.8,.7,.9,.6,.4]))


def test_unknown_and_nonpositive_routes_inherit():
    f=frame();out=r.apply_filter(f,[.8,.1,.2,np.nan,np.nan],.3)
    assert out.call.tolist()==['PROMOTE','FAIL','FAIL','UNKNOWN','FAIL']
    assert out.ranking_score.equals(f.ranking_score)
    assert out.base_call.equals(f.base_call)
    assert f.call.tolist()==['PROMOTE','PROMOTE','PROMOTE','UNKNOWN','FAIL']


def test_explicit_no_filter_includes_zero_model_score():
    f=frame();out=r.apply_filter(f,[0.,0.,0.,np.nan,np.nan],0.)
    assert out.call.equals(f.call)
    assert not out.filter_applied.any()


def test_strict_threshold_boundary():
    f=frame();out=r.apply_filter(f,[.3,.30000001,.2,np.nan,np.nan],.3)
    assert out.call.iloc[:2].tolist()==['FAIL','PROMOTE']


def test_filtered_positive_and_unknown_count_complete_denominator():
    out=r.apply_filter(frame(),[.1,.1,.8,np.nan,np.nan],.3)
    stat=r.statistics(out)
    assert (stat['candidates'],stat['known'],stat['unknown_outcomes'])==(5,4,1)
    assert stat['tp']==0 and stat['fp']==0
    assert stat['missed_positives']==2
    assert stat['abstained_positives']==1
    assert stat['Precision'] is None and stat['Recall']==0


def test_strict_label_maturity_and_duplicates():
    f=frame();f.loc[2,['target','label_valid']]=[1.,True]
    f.loc[0,'target_date']='2025-07-03'
    train=r.select_meta_rows(f,'2025-07-03')
    assert train.code.tolist()==['000002','000003']
    with pytest.raises(ValueError,match='duplicate'):
        r.select_meta_rows(pd.concat([f,f.iloc[[0]]]),'2025-07-03')


def stat(tp,fp,pos=100):
    return dict(tp=tp,fp=fp,positives=pos)


def test_full_recall_guard_prevents_positive_subset_illusion():
    table={t:stat(40,40) for t in r.THRESHOLDS}
    table[.2]=stat(36,20)
    table[.3]=stat(34,10)  # Better Precision but six pp lost: reject.
    assert r.choose_threshold(table)==.2


def test_equal_precision_prefers_lower_threshold():
    table={t:stat(40,40) for t in r.THRESHOLDS}
    table[.2]=stat(36,20);table[.3]=stat(36,20)
    assert r.choose_threshold(table)==.2


def test_no_valid_or_no_positive_candidate_uses_bypass():
    table={t:stat(40,40) for t in r.THRESHOLDS}
    table[.4]=stat(0,0)
    assert r.choose_threshold(table)==0.
    assert r.qualification(pd.DataFrame({'target':[1,0],'date':['2025-01-01']*2}))['qualified'] is False


def test_positive_top3_keeps_unknown_and_empty_slots():
    f=frame();f.loc[1,'call']='FAIL'
    counts,picks=r.positive_top3(f)
    assert counts['full_slots']==6 and counts['selected']==2
    assert counts['empty_slots']==4 and counts['unknown_selected']==1
    assert counts['hits']==1
    assert picks.code.tolist()==['000003','000001']


def test_invalid_raw_and_eligible_missing_scores_rejected():
    for values in [[np.nan],[1.01],[-.01]]:
        with pytest.raises(ValueError):r.logit(values)
    assert np.isfinite(r.logit([0.,1.])).all()
    with pytest.raises(ValueError,match='missing'):
        r.apply_filter(frame(),[np.nan,.4,.5,np.nan,np.nan],.3)


def test_train_only_preprocessing_and_save_reload(tmp_path):
    # One isolated synthetic unit fit; not an empirical study/model selection.
    dates=np.repeat(pd.date_range('2024-01-01',periods=20).strftime('%Y-%m-%d'),2)
    y=np.tile([0.,1.],20)
    f=pd.DataFrame(dict(date=dates,code=[f'{i:06d}' for i in range(40)],
        target_date=dates,target=y,label_valid=True,daily_one_word=0.,
        verified_current_price_shape=0.,base_call='PROMOTE',base_score_logit=y+.1))
    for c in r.INDIVIDUAL:f[c]=y*2
    f['turnover']=np.nan  # A wholly empty train feature must retain schema.
    model,receipt=r.fit_head(f,r.ARMS[1],tmp_path,'SYNTHETIC_UNIT','PREFIX')
    median=model.named_steps['imputer'].statistics_.copy()
    center=model.named_steps['scale'].mean_.copy()
    future=f.copy();future['log_signal_amount']=1e8
    values=r.scores(model,future,r.ARMS[1])
    np.testing.assert_array_equal(median,model.named_steps['imputer'].statistics_)
    np.testing.assert_array_equal(center,model.named_steps['scale'].mean_)
    saved=joblib.load(receipt['model_path'])
    np.testing.assert_array_equal(values,r.scores(saved['model'],future,r.ARMS[1]))
    assert model.named_steps['logistic'].class_weight is None
    assert model.named_steps['logistic'].C==1


def test_block_bootstrap_uses_known_minus_positive_negative_count():
    f=frame();parts=[]
    for arm in [r.BASE_ARM]+r.ARMS:
        one=f.copy();one['arm']=arm;one['evaluation_call']=one.call;parts.append(one)
    result=r.bootstrap_intervals(pd.concat(parts,ignore_index=True))
    assert len(result)>0
    assert result.finite_replicates.max()==2000
    assert result[['lower','upper']].fillna(0).to_numpy().sum()==0
