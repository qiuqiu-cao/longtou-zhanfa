import importlib.util
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import pytest

BASE=Path(__file__).resolve().parents[1]/'research/first_board_dynamic_rank_20261030'
sys.path.insert(0,str(BASE))
import dynamic_features as features
import pairwise_residual as residual
import run_experiment as runner
import theme_sources as sources


def fixture():
    days=['2026-01-05','2026-01-06','2026-01-07']
    def row(day,code,topic,minute=10):
        return dict(signal_date=day,code=code,themes=[topic],first_seal_minutes=minute,
                    source_path='synthetic',source_sha256='hash',source_received_at='2026-02-01')
    provider=pd.DataFrame([row(days[0],'000002','P:T'),row(days[0],'000003','P:T'),row(days[0],'000005','P:OTHER'),
                           row(days[1],'000001','P:T',10),row(days[1],'000004','P:T',20),row(days[1],'000005','P:OTHER')])
    records=[]
    for day in days[:2]:
        for code in ['000001','000002','000003','000004','000005']:
            limit = code in (['000002','000003','000005'] if day==days[0] else ['000001','000004','000005'])
            height= {'000002':4,'000003':1,'000005':5}.get(code,0) if day==days[0] else int(limit)
            ret = {'000002':-.1,'000003':-.05}.get(code,.1) if day==days[1] else .1
            records.append(dict(date=day,code=code,p=days.index(day),eligible=True,valid_price=True,
                                regular_price=True,limit=limit,height=height,day_return=ret))
    return pd.DataFrame([{'signal_date':days[1],'code':'000001'}]),provider,pd.DataFrame(records),days


def test_fixed_cohort_keeps_today_failed_stocks_and_old_anchor():
    x,p,s,days=fixture();r=features.build(x,p,s,days).iloc[0]
    assert r.prior_cohort_relative_return==pytest.approx(-.075-(-.05/3))
    assert r.prior_cohort_relative_survival==pytest.approx(-1/3)
    assert r.prior_anchor_relative_return==pytest.approx(-.1)
    assert r.prior_anchor_height_gap==3
    detail=json.loads(r.theme_details_json)[0]
    assert detail['prior_cohort_codes']==['000002','000003']
    assert detail['prior_anchor_codes']==['000002']
    assert r.relative_reported_first_seal==1


def test_missing_member_price_is_unknown_not_dropped_or_failed():
    x,p,s,days=fixture();s=s[~(s.date.eq(days[1]) & s.code.eq('000003'))]
    r=features.build(x,p,s,days).iloc[0]
    assert pd.isna(r.prior_cohort_relative_return)
    assert pd.isna(r.prior_cohort_relative_survival)
    assert json.loads(r.theme_details_json)[0]['prior_cohort_status']=='PARTIAL_OR_UNKNOWN_CURRENT_PRICES'


def test_prefix_invariance_and_no_label_input():
    x,p,s,days=fixture();baseline=features.build(x,p,s,days)
    later=p.iloc[[0]].copy();later.signal_date=days[2];later['themes']=[['P:FUTURE']]
    pd.testing.assert_frame_equal(baseline,features.build(x,pd.concat([p,later]),s,days))
    with pytest.raises(ValueError):features.build(x.assign(outcome=1),p,s,days)


def test_missing_theme_neutral_and_time_ties_not_false_leadership():
    x,p,s,days=fixture();p.loc[p.code.eq('000004'),'first_seal_minutes']=10
    assert features.build(x,p,s,days).iloc[0].relative_reported_first_seal==0
    r=features.build(x,p[~p.code.eq('000001')],s,days).iloc[0]
    assert r[features.FIELDS].isna().all()
    assert r.context_status=='UNKNOWN_THEME_OR_PRICE_CONTEXT'


def test_source_namespaces_dedup_and_exact_no_name_aliases():
    assert sources.topics('T','T+A、A')==['C:A','C:T','P:T']
    assert sources.topics('阳','洋')==['C:洋','P:阳']
    assert pd.isna(sources.seal_time(0,'2026-01-06'))


def test_scale_uses_all_candidates_unknown_label_does_not_remove_row():
    pool=pd.DataFrame({'signal_date':['2026-01-06']*3,'code':['000001','000002','000003'],
                       's1_score':[.5,.5,.5],'feature':[0.,1.,9.]})
    y=pool[['signal_date','code']].copy();y[residual.TARGET]=[0,1,np.nan]
    m=residual.fit(pool,y,['feature'])
    assert m['means'][0]==pytest.approx(10/3)
    assert m['pairs']==1 and m['pair_days']==1
    assert m['coefficients'][0]>0
    missing=pool.iloc[[0]].copy();missing['feature']=np.nan
    assert residual.score(missing,m)[1][0]==0


def test_constant_and_empty_features_fall_back_exactly():
    x=pd.DataFrame({'signal_date':['2026-01-06']*3,'code':['000003','000001','000002'],
                    's1_score':[.4,.4,.6],'f':[np.nan]*3})
    y=x[features.KEYS].copy();y[residual.TARGET]=[0,1,0]
    m=residual.fit(x,y,['f']);v,delta,status=residual.score(x,m)
    assert m['status']=='FALLBACK' and np.array_equal(v,x.s1_score)
    assert residual.choose(x.assign(research_score=v),'research_score').equals(residual.choose(x))
    assert residual.choose(x).code.tolist()==['000002','000001','000003']


def test_monthly_window_excludes_late_labels_but_keeps_unknown_candidates():
    days=['2026-01-05','2026-01-06','2026-01-07','2026-01-08']
    x=pd.DataFrame({'signal_date':[days[0],days[0],days[1],days[2]],'code':['000001','000002','000003','000004']})
    y=x.copy();y['label_end']=[days[1],days[3],days[2],days[3]];y[residual.TARGET]=[1,0,0,1]
    train,labels,cutoff,selected=runner.maturity_window(x,y,days,days[3])
    assert cutoff==days[2] and len(train)==3
    assert set(train.code)=={'000001','000002','000003'}
    assert set(labels.code)=={'000001','000003'}
    y.loc[y.code.eq('000004'),residual.TARGET]=0
    assert runner.maturity_window(x,y,days,days[3])[1].equals(labels)


def test_negative_anchor_names_do_not_enter_positive_stage_by_string_alone():
    x,p,s,days=fixture()
    p.loc[p.signal_date.eq(days[0]) & p.code.eq('000002'),'themes']=[['P:OTHER']]
    r=features.build(x,p,s,days).iloc[0]
    assert json.loads(r.theme_details_json)[0]['prior_anchor_codes']==[]
    assert r.prior_anchor_height_gap==0
