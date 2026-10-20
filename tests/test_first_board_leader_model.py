from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
import joblib
from sklearn.dummy import DummyClassifier

import first_board_leader_model as fb


def sample_inputs(tmp_path, missing=None, change_future=False):
    days=pd.bdate_range("2024-01-02",periods=30).strftime("%Y-%m-%d").tolist()
    rows=[]
    for code in ("000001","600001"):
        price=10.
        for i,day in enumerate(days):
            previous=price
            if i in (5,6,7,8):
                price=fb.limit_cents([previous])[0]/100
            elif i==15 and code=="600001":
                price=fb.limit_cents([previous])[0]/100-.01
            if change_future and i>=10:
                price=round(previous*.99,2)
            if (code,i)==missing:continue
            rows.append(dict(date=day,code=code,name=code,open=previous,high=max(price,previous),low=min(price,previous),close=price,pre_close=previous,volume=100000.,amount=1000000.,turnover=5.,float_cap=20000000.,st=False,new_listing=False,status_verified=True,source="synthetic_test_fixture"))
    frame=pd.DataFrame(rows)
    frame.to_csv(tmp_path/"bars.csv",index=False)
    frame.iloc[:0].to_csv(tmp_path/"recovered.csv",index=False)
    (tmp_path/"calendar.json").write_text(json.dumps({"sessions":days}))
    pd.DataFrame([dict(code=c,listing_date="2000-01-01") for c in ("000001","600001")]).to_csv(tmp_path/"security.csv",index=False)
    return dict(bars=tmp_path/"bars.csv",recovered=tmp_path/"recovered.csv",calendar=tmp_path/"calendar.json",security_intervals=tmp_path/"security.csv"),days


def test_exact_continuous_targets_and_first_identity(tmp_path):
    inputs,days=sample_inputs(tmp_path)
    rows,_,_,_=fb.build_dataset(inputs)
    assert len(rows)==2
    assert set(rows.signal_date)=={days[5]}
    assert rows.reach4_LABEL_ONLY.eq(1).all()
    assert rows.reach5_LABEL_ONLY.eq(0).all()
    assert rows.max_height_5_LABEL_ONLY.eq(4).all()
    assert rows.cohort_leader_LABEL_ONLY.eq(1).all()
    assert not any("LABEL" in f or "future" in f for f in fb.FEATURES)


def test_missing_official_session_is_unknown_and_never_skipped(tmp_path):
    inputs,days=sample_inputs(tmp_path,missing=("000001",6))
    rows,_,_,_=fb.build_dataset(inputs)
    candidate=rows[(rows.code=="000001") & (rows.signal_date==days[5])].iloc[0]
    assert pd.isna(candidate.reach4_LABEL_ONLY)
    assert pd.isna(candidate.cohort_leader_LABEL_ONLY)
    assert not ((rows.code=="000001") & (rows.signal_date==days[7])).any()


def test_future_prices_do_not_change_signal_features(tmp_path):
    a=tmp_path/"a";a.mkdir();b=tmp_path/"b";b.mkdir()
    first,days=sample_inputs(a)
    changed,_=sample_inputs(b,change_future=True)
    x=fb.build_dataset(first)[0]
    y=fb.build_dataset(changed)[0]
    x=x[x.signal_date==days[5]].set_index("code")[fb.FEATURES]
    y=y[y.signal_date==days[5]].set_index("code")[fb.FEATURES]
    pd.testing.assert_frame_equal(x,y)


def test_unknown_labels_remain_in_topk_denominator():
    rows=pd.DataFrame(dict(signal_date=["2025-01-02"]*3,code=["000001","000002","000003"],score=[.9,.8,.7],reach4_LABEL_ONLY=[1,np.nan,0]))
    result=fb.metric(rows,"score",3)
    assert result["selected"]==3
    assert result["known"]==2
    assert result["unknown"]==1
    assert result["precision"]==.5


def test_source_conflicts_refused(tmp_path):
    inputs,_=sample_inputs(tmp_path)
    frame=pd.read_csv(inputs["bars"],dtype={"code":str}).iloc[:1].copy()
    frame["close"]+=.01
    frame.to_csv(inputs["recovered"],index=False)
    with pytest.raises(ValueError,match="conflicting duplicate"):
        fb.build_dataset(inputs)


def dummy_bundle(tmp_path):
    model=DummyClassifier(strategy="prior").fit(pd.DataFrame(np.zeros((2,len(fb.FEATURES))),columns=fb.FEATURES),[0,1])
    path=tmp_path/"bundle.joblib"
    joblib.dump(dict(model=model,features=fb.FEATURES,knowledge_cutoff="2023-12-31"),path)
    return path


def test_inference_rebuilds_full_pool_and_excludes_label_columns(tmp_path):
    inputs,days=sample_inputs(tmp_path)
    bundle=dummy_bundle(tmp_path)
    out=tmp_path/"ranking.csv"
    fb.rank_bars(bundle,inputs,days[5],out)
    result=pd.read_csv(out,dtype={"code":str})
    assert set(result.code)=={"000001","600001"}
    assert not any("LABEL_ONLY" in c for c in result)
    assert result.probability_status.eq("UNCALIBRATED").all()
    with pytest.raises(FileExistsError):
        fb.rank_bars(bundle,inputs,days[5],out)


def test_daily_hit_rate_does_not_count_unknown_as_failure():
    rows=pd.DataFrame(dict(signal_date=["2025-01-02","2025-01-03"],code=["000001","000002"],score=[.9,.8],reach4_LABEL_ONLY=[1,np.nan]))
    m=fb.metric(rows,"score",1)
    assert m["daily_unknown_dates"]==1
    assert m["daily_any_hit_rate_complete_dates"]==1
    assert m["daily_confirmed_hit_fraction_all_registered_dates"]==.5


def test_simulation_cancels_limit_open_without_fallback(tmp_path):
    inputs,days=sample_inputs(tmp_path)
    rows,bars,sessions,_=fb.build_dataset(inputs)
    entry=bars.date.eq(days[6])&bars.code.eq("000001")
    for c in ("open","high","low","close"):
        bars.loc[entry,c]=bars.loc[entry,"cap"]/100
    rows["test_score"]=[.9,.8]
    trades,_,report=fb.simulation(rows,bars,sessions,"test_score")
    assert report["closed_trades"]==0
    assert report["selected"]==1
    assert trades.iloc[0].status=="OPEN_AT_UPPER_LIMIT_NO_QUEUE_ASSUMPTION"
    assert report["actual_trade_count"]==0


def test_separate_stage_labels_use_only_one_next_official_session(tmp_path):
    import first_second_stage_model as stage
    inputs,days=sample_inputs(tmp_path)
    rows,_,_,_=stage.build_rows(inputs)
    first=rows[rows.stage.eq(1)];second=rows[rows.stage.eq(2)]
    assert len(first)==2 and len(second)==2
    assert first.signal_date.eq(days[5]).all()
    assert second.signal_date.eq(days[6]).all()
    assert first[stage.TARGET].eq(1).all()
    assert second[stage.TARGET].eq(1).all()
    # A later missing day must not invalidate an already observable next-day event.
    missing=tmp_path/'missing';missing.mkdir()
    inputs,_=sample_inputs(missing,missing=("000001",9))
    later=stage.build_rows(inputs)[0]
    assert later[(later.code=="000001")&later.stage.eq(1)][stage.TARGET].iloc[0]==1


def test_new_stage_features_causal_at_both_first_and_second_close(tmp_path):
    import first_second_stage_model as stage
    a=tmp_path/'a';a.mkdir();b=tmp_path/'b';b.mkdir()
    x,days=sample_inputs(a)
    y,_=sample_inputs(b,change_future=True)
    before=stage.build_rows(x)[0].set_index(['signal_date','stage','code'])
    after=stage.build_rows(y)[0].set_index(['signal_date','stage','code'])
    pd.testing.assert_frame_equal(before[stage.VARIANTS['STAGE_CYCLE_GENE']],after[stage.VARIANTS['STAGE_CYCLE_GENE']])


def test_second_recheck_does_not_turn_a_failed_first_pick_into_success():
    import first_second_stage_model as stage
    first=pd.DataFrame(dict(signal_date=['2025-07-01','2025-07-02'],code=['000001','000002'],name=['A','B'],label_end=['2025-07-02','2025-07-03'],model_score=[.7,.8],calibrated_score=[.6,.7],promotion_next_LABEL_ONLY=[0,1]))
    second=pd.DataFrame(dict(signal_date=['2025-07-02','2025-07-03'],code=['000001','000002'],label_end=['2025-07-03','2025-07-04'],model_score=[.9,.8],calibrated_score=[.8,.7],promotion_next_LABEL_ONLY=[1,1]))
    rows,result=stage.chain_recheck(first,second)
    assert pd.isna(rows.iloc[0].second_model_score)
    assert result['first_promoted']==1
    assert result['continued_to_third']==1
    assert result['conditional_second_to_third']==1
    assert result['full_first_to_third_rate']==.5


def test_new_close_workbench_rechecks_previous_pick_without_rewriting_it(tmp_path):
    import first_second_stage_model as stage
    import first_second_stage_workbench as work
    inputs,days=sample_inputs(tmp_path)
    model=DummyClassifier(strategy='prior').fit(pd.DataFrame(np.zeros((2,62)),columns=stage.VARIANTS['STAGE_CYCLE_GENE']),[0,1])
    head=dict(model=model,calibrator=None,features=stage.VARIANTS['STAGE_CYCLE_GENE'],train_prior=.2,calibration_status='PROBABILITY_NOT_QUALIFIED')
    bundle=tmp_path/'two_heads.joblib';joblib.dump(dict(heads={1:head,2:head},knowledge_cutoff='2023-12-31'),bundle)
    first=tmp_path/'first_facts.csv';second=tmp_path/'second_facts.csv'
    initial=work.prepare(inputs,days[5],first)
    assert not any('LABEL_ONLY' in c or 'model_score' in c for c in initial)
    old=tmp_path/'first_scores.csv';work.recheck(bundle,first,old)
    previous_hash=fb.sha(old)
    work.prepare(inputs,days[6],second)
    updated=work.recheck(bundle,second,tmp_path/'second_scores.csv',old)
    assert updated.was_previous_first_pick.sum()==1
    assert updated.loc[updated.was_previous_first_pick,'code'].iloc[0]=='000001'
    assert fb.sha(old)==previous_hash
    assert updated.calibrated_score.isna().all()
