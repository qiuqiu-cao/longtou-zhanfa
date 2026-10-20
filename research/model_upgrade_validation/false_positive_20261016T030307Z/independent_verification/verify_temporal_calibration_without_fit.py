"""Independent saved-artifact verification. No estimator is fitted here."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
REPO=HERE.parents[3]
sys.path.insert(0,str(REPO))  # Resolve saved transformer classes; never fit.
EXEC=ROOT/'execution'
ARM='TEMPORAL_OOF_CALIBRATED_SOURCE_HEAD'
BASE='SOURCE_PATCHED_FEATURE_CLASSIFIER'
REFERENCE=REPO/'research/model_upgrade_validation/ranking_optimization_20261015T173045Z/source_feature_training/same_source_route_alignment/complete26439_predictions_WITH_SAME_SOURCE_ROUTES.csv'
PERIODS=[('2025H2','2025-07-01','2025-12-31'),('2026JanAug','2026-01-01','2026-08-31'),('2026Sep','2026-09-01','2026-09-30'),('2026Oct','2026-10-01','2026-10-15'),('HISTORICAL_POOLED','2025-07-01','2026-09-30')]
HASH_CACHE={}

def sha(path):
    key=str(Path(path).resolve())
    if key not in HASH_CACHE: HASH_CACHE[key]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return HASH_CACHE[key]

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def known(data):return data.label_valid.astype(str).str.lower().eq('true')&data.target.isin([0,1])

def matrix(group):
    k=group[known(group)]
    p=k.evaluation_call.eq('PROMOTE');n=k.evaluation_call.eq('FAIL');y=k.target.eq(1)
    tp=int((p&y).sum());fp=int((p&~y).sum());fn=int((n&y).sum());tn=int((n&~y).sum())
    ap=int((~p&~n&y).sum());an=int((~p&~n&~y).sum());positives=int(y.sum());negatives=int((~y).sum())
    rate=lambda a,z:a/z if z else np.nan
    return dict(candidates=len(group),known=len(k),unknown_outcomes=len(group)-len(k),positives=positives,negatives=negatives,
        TP=tp,FP=fp,FN_called=fn,TN=tn,abstained_positives=ap,abstained_negatives=an,
        positive_signals_all=int(group.evaluation_call.eq('PROMOTE').sum()),positive_signals_unknown=int(group.loc[~known(group),'evaluation_call'].eq('PROMOTE').sum()),
        Recall=rate(tp,positives),Precision=rate(tp,tp+fp),F1=rate(2*tp,positives+tp+fp),FPR=rate(fp,negatives),
        wrong_share_in_known_positive_signals=rate(fp,tp+fp))

def assert_scores(actual,expected,name,tol=1e-14):
    a=np.asarray(actual,dtype=float);z=np.asarray(expected,dtype=float)
    np.testing.assert_allclose(a,z,rtol=0,atol=tol,equal_nan=True,err_msg=name)
    finite=np.isfinite(a)&np.isfinite(z)
    return float(np.max(np.abs(a[finite]-z[finite]))) if finite.any() else 0.

def main():
    result_path=EXEC/'calibration_fixed_result.json'
    complete_path=EXEC/'all30216_predictions_BEFORE_EVALUATION.csv'
    if not result_path.exists() or not complete_path.exists():raise FileNotFoundError('wait for full fixed execution; do not poll or fit')
    receipt_path=HERE/'temporal_calibration_independent_receipt.json'
    if receipt_path.exists():raise FileExistsError('immutable independent verification already exists')
    protocol=read(ROOT/'protocol_BEFORE_ANY_FIT.json')
    result=read(result_path)
    sources=protocol['source_bindings']|result['source_bindings']
    for path,expected in sources.items():assert sha(path)==expected,('source drift',path)
    old=pd.read_csv(REFERENCE,dtype={'code':str},low_memory=False)
    all_heads=pd.read_csv(complete_path,dtype={'code':str},low_memory=False)
    assert len(old)==26439 and len(all_heads)==30216
    pd.testing.assert_frame_equal(old,all_heads.iloc[:len(old)][old.columns],check_exact=True,check_dtype=False)
    assert all_heads.groupby('arm').size().eq(3777).all() and all_heads.arm.nunique()==8
    assert not all_heads.duplicated(['date','code','arm']).any()
    assert all_heads.date.max()=='2026-10-15'
    new=all_heads[all_heads.arm.eq(ARM)].copy()
    baseline=all_heads[all_heads.arm.eq(BASE)].copy()
    base_idx=baseline.set_index(['date','code'])
    new_idx=new.set_index(['date','code']).loc[base_idx.index]
    for col in ['target_date','target','label_valid','daily_one_word','frozen_original_nonword_identity','ranking_score','verified_current_price_shape']:
        pd.testing.assert_series_equal(base_idx[col],new_idx[col],check_names=False,check_exact=True)
    assert new_idx.event_probability_validated.astype(str).str.lower().eq('false').all()
    eligible=new_idx.verified_current_price_shape.eq(0)
    pd.testing.assert_frame_equal(new_idx.loc[~eligible,['score','call']],base_idx.loc[~eligible,['score','call']],check_exact=True)
    own=np.where(new_idx.loc[eligible,'b_score'].to_numpy()>.5,'PROMOTE','FAIL')
    assert np.array_equal(new_idx.loc[eligible,'call'].to_numpy(),own)
    for c in ['evaluation_call','recovered_B_call','aligned_evaluation_call']:
        assert new_idx[c].equals(new_idx.call)
    current=new[new.date.eq('2026-10-15')]
    assert len(current)==6 and current.target.isna().all() and not known(current).any()
    # Frozen reference metadata lacks target_date here. Preserve that field;
    # append the expected next session from already-bound official sessions.
    calendar_path=REPO/'research/model_upgrade_validation/ranking_optimization_20261015T173045Z/source_feature_training/prepared.joblib'
    assert str(calendar_path.resolve()) in sources and sha(calendar_path)==sources[str(calendar_path.resolve())]
    sessions=joblib.load(calendar_path)['sessions']
    expected_target=min(day for day in sessions if day>'2026-10-15')
    assert expected_target=='2026-10-16'
    current=current.copy();current['expected_target_from_bound_official_sessions']=expected_target
    current['original_target_date_was_missing']=current.target_date.isna()
    sep=new[new.date.between('2026-09-01','2026-09-30')&new.daily_one_word.eq(0)]
    assert (len(sep),int(known(sep).sum()),int(sep.target.eq(1).sum()),int((~known(sep)).sum()))==(143,107,40,36)
    data=joblib.load(ROOT/'fixed_input_BEFORE_ANY_FIT.joblib')
    full=data['full'];fields=data['fields']
    assert len(full)==9315 and len(fields)==92 and len(data['plans'])==16
    blocks=[]; maps=[]; replay=[];inner_total=0;map_total=0
    for plan in data['plans']:
        month=plan['month'];cutoff=plan['cutoff'];prefix=EXEC/month
        assert sha(plan['final_model_path'])==plan['final_model_sha256']
        valid_outer=known(full)&full.date.lt(cutoff)&full.target_date.lt(cutoff)
        prior=full[valid_outer].copy();days=sorted(prior.date.unique())[-252:]
        window=prior[prior.date.isin(days)].copy()
        assert len(days)==252 and window.target_date.max()<cutoff
        all_b=[]
        for block in range(1,4):
            validation_days=days[63*block:63*(block+1)]
            first=validation_days[0]
            training=window[window.date.lt(first)&window.target_date.lt(first)]
            validation=window[window.date.isin(validation_days)]
            assert training.target.nunique()==2 and training.target_date.max()<first
            assert set(training.date).isdisjoint(validation.date)
            assert validation.date.nunique()==63 and validation.target_date.max()<cutoff
            path=Path(str(prefix)+f'_inner{block}')
            started=read(str(path)+'_STARTED.json');completed=read(str(path)+'_COMPLETED.json')
            saved=joblib.load(str(path)+'_model.joblib')
            assert saved['features']==fields and saved['cutoff']==first
            assert started['max_train_target']==training.target_date.max() and started['validation_first']==first
            assert saved['training_digest']==started['training_digest']
            assert sha(str(path)+'_model.joblib')==completed['model_sha256']
            inner_params=saved['model'].named_steps['hgb'].get_params()
            assert all(inner_params[k]==v for k,v in protocol['inner_estimator'].items())
            record=pd.read_csv(str(path)+'_all_validation_scores_BEFORE_EFFECTS.csv',dtype={'code':str},low_memory=False)
            assert not record.duplicated(['date','code']).any()
            expected=validation[['date','code','target_date','target','label_valid','frozen_original_nonword_identity']].reset_index(drop=True)
            pd.testing.assert_frame_equal(record[expected.columns],expected,check_dtype=False,check_exact=True)
            with threadpool_limits(limits=2):
                probabilities=saved['model'].predict_proba(validation[fields])[:,list(saved['model'].classes_).index(1)]
            error=assert_scores(record.raw_score,probabilities,'inner validation saved score replay')
            all_b.append(record[record.frozen_original_nonword_identity.eq(0)])
            blocks.append(dict(month=month,block=block,training_rows=len(training),training_days=training.date.nunique(),
                maximum_training_target=training.target_date.max(),validation_first=first,validation_last=validation.date.max(),
                validation_days=63,validation_rows=len(validation),outer_cutoff=cutoff,max_validation_target=validation.target_date.max(),
                no_train_validation_overlap=True,strict_target_maturity=True,maximum_saved_score_error=error))
            inner_total+=1
        oof=pd.read_csv(str(prefix)+'_OOF_calibration_input_BEFORE_FIT.csv',dtype={'code':str},low_memory=False)
        expected_oof=pd.concat(all_b,ignore_index=True)
        pd.testing.assert_frame_equal(oof.drop(columns='calibration_weight'),expected_oof,check_exact=True,check_dtype=False)
        assert not oof.duplicated(['date','code']).any() and known(oof).all() and oof.target_date.max()<cutoff
        assert oof.frozen_original_nonword_identity.eq(0).all()
        expected_weight=1/oof.groupby('date').date.transform('size')
        assert_scores(oof.calibration_weight,expected_weight,'uncosted temporal calibration weight')
        masses=oof.groupby('date').calibration_weight.sum()
        counts=oof.groupby('date').size().reindex(masses.index)
        assert (np.abs(masses-1)<=counts*np.finfo(float).eps).all(), 'CSV day mass exceeds accumulation precision'
        maximum_day_mass_error=float(np.max(np.abs(masses-1)))
        cstart=read(str(prefix)+'_calibrator_STARTED.json')
        assert sha(str(prefix)+'_OOF_calibration_input_BEFORE_FIT.csv')==cstart['input_sha256']
        saved_map=joblib.load(str(prefix)+'_calibrator.joblib');mapper=saved_map['calibrator']
        assert mapper.coef_.shape==(1,1) and mapper.coef_[0,0]>0
        n_iter=int(np.max(mapper.n_iter_));limit=int(mapper.get_params()['max_iter'])
        assert n_iter<limit,('calibrator not converged',month,n_iter,limit)
        assert all(mapper.get_params()[k]==v for k,v in protocol['calibration']['estimator'].items())
        completed=read(str(prefix)+'_COMPLETED.json')
        assert sha(str(prefix)+'_calibrator.joblib')==completed['calibrator_sha256']
        saved_final=joblib.load(plan['final_model_path'])
        test=full[full.date.str[:7].eq(month)].copy()
        with threadpool_limits(limits=2):
            raw=saved_final['model'].predict_proba(test[fields])[:,list(saved_final['model'].classes_).index(1)]
        compare_base=base_idx.loc[pd.MultiIndex.from_frame(test[['date','code']])]
        compare_new=new_idx.loc[pd.MultiIndex.from_frame(test[['date','code']])]
        raw_error=assert_scores(raw,compare_base.b_score,'exact frozen final source classifier')
        assert_scores(compare_new.raw_b_score_before_calibration,raw,'attached uncalibrated raw scores')
        clipped=np.maximum(1e-12,np.minimum(1-1e-12,raw))
        logits=np.log(clipped/(1-clipped)).reshape(-1,1)
        with threadpool_limits(limits=2):adjusted=mapper.predict_proba(logits)[:,list(mapper.classes_).index(1)]
        adjusted_error=assert_scores(compare_new.b_score,adjusted,'saved temporal score scale mapping')
        maps.append(dict(month=month,cutoff=cutoff,rows=len(oof),days=oof.date.nunique(),max_target=oof.target_date.max(),
            positives=int(oof.target.eq(1).sum()),negative_mass=float(oof.loc[oof.target.eq(0),'calibration_weight'].sum()),
            positive_mass=float(oof.loc[oof.target.eq(1),'calibration_weight'].sum()),coefficient=float(mapper.coef_[0,0]),
            maximum_CSV_day_mass_accumulation_error=maximum_day_mass_error,
            intercept=float(mapper.intercept_[0]),n_iter=n_iter,max_iter=limit,converged=True,positive_slope=True,
            final_classifier_exact_reuse=True,max_raw_score_replay_error=raw_error,max_map_replay_error=adjusted_error))
        for r,a,z in zip(test.itertuples(index=False),raw,adjusted):replay.append(dict(month=month,date=r.date,code=r.code,raw_score_replayed=float(a),calibrated_scale_replayed=float(z)))
        map_total+=1
    assert inner_total==48 and map_total==16 and result['inner_HGB_fits']==48 and result['calibrator_fits']==16 and result['old_final_classifier_refits']==0
    records=[];rank_records=[];selected=[]
    for period,first,last in PERIODS:
        for arm,frame in [(BASE,baseline),(ARM,new)]:
            sample=frame[frame.date.between(first,last)]
            for scope,part in [('FULL_POOL',sample),('ORIGINAL_NONWORD',sample[sample.daily_one_word.eq(0)])]:
                records.append(dict(period=period,arm=arm,scope=scope,**matrix(part)))
            for k in [1,2,3]:
                bpool=sample[sample.daily_one_word.eq(0)].copy()
                chosen=bpool.sort_values(['date','ranking_score','code'],ascending=[True,False,True],na_position='last').groupby('date',sort=True).head(k)
                n_known=int(known(chosen).sum());hits=int((known(chosen)&chosen.target.eq(1)).sum());slots=k*sample.date.nunique()
                rank_records.append(dict(period=period,arm=arm,k=k,signal_days=sample.date.nunique(),full_slots=slots,selected=len(chosen),known_selected=n_known,unknown_selected=len(chosen)-n_known,hits=hits,empty_slots=slots-len(chosen),full_slot_hit_rate=hits/slots))
                for row in chosen.itertuples(index=False):selected.append(dict(period=period,arm=arm,k=k,date=row.date,code=row.code,target=row.target,label_valid=row.label_valid,ranking_score=row.ranking_score))
    stat=pd.DataFrame(records);ranks=pd.DataFrame(rank_records);picks=pd.DataFrame(selected)
    pa=picks[picks.arm.eq(BASE)].drop(columns='arm').reset_index(drop=True)
    pz=picks[picks.arm.eq(ARM)].drop(columns='arm').reset_index(drop=True)
    pd.testing.assert_frame_equal(pa,pz,check_exact=True)
    official=pd.read_csv(EXEC/'full_same_scope_classification_metrics.csv',low_memory=False)
    for row in official.itertuples(index=False):
        own=stat[stat.period.eq(row.period)&stat.arm.eq(row.arm)&stat.scope.eq(row.scope)].iloc[0]
        for mine,theirs in [('TP','tp'),('FP','fp'),('Recall','Recall'),('Precision','Precision'),('F1','F1')]:
            assert np.isclose(own[mine],getattr(row,theirs),rtol=0,atol=1e-14,equal_nan=True),(row.period,row.arm,row.scope,mine)
    published=pd.read_csv(EXEC/'same_full_slot_Top1Top2Top3.csv',low_memory=False)
    for row in published.itertuples(index=False):
        own=ranks[ranks.period.eq(row.period)&ranks.arm.eq(row.arm)&ranks.k.eq(row.k)].iloc[0]
        for key in ['signal_days','full_slots','selected','known_selected','unknown_selected','hits','empty_slots','full_slot_hit_rate']:
            assert np.isclose(own[key],getattr(row,key),rtol=0,atol=1e-14,equal_nan=True),(row.period,row.arm,row.k,key)
    table=stat[stat.scope.eq('ORIGINAL_NONWORD')].set_index(['period','arm'])
    a=table.loc[('HISTORICAL_POOLED',BASE)];z=table.loc[('HISTORICAL_POOLED',ARM)]
    guards=dict(pooled_Precision_gain_at_least3pp=bool(z.Precision>=a.Precision+.03),pooled_FP_drop_at_least10percent=bool(z.FP<=a.FP*.9),pooled_F1_not_lower=bool(z.F1>=a.F1))
    for period,first,last in PERIODS[:3]:
        aa=table.loc[(period,BASE)];zz=table.loc[(period,ARM)]
        guards[period+'_FP_not_higher']=bool(zz.FP<=aa.FP)
        guards[period+'_Recall_loss_within2p5pp']=bool(zz.Recall>=aa.Recall-.025)
    guards['all_complete_Topk_picks_identical']=True
    assert guards==result['guards'] and all(guards.values())==result['fixed_screen_pass']
    HERE.mkdir(parents=True,exist_ok=True)
    stat.to_csv(HERE/'独立复算_完整池与原非一字混淆矩阵.csv',index=False,encoding='utf-8-sig')
    ranks.to_csv(HERE/'独立复算_完整槽位Top1Top2Top3.csv',index=False,encoding='utf-8-sig')
    picks.to_csv(HERE/'独立复算_原样排序预选与未知.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(blocks).to_csv(HERE/'独立核验_48时间块成熟边界与重放.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(maps).to_csv(HERE/'独立核验_16校准映射权重与收敛.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(replay).to_csv(HERE/'独立核验_全部3777原模型与校准映射重放.csv',index=False,encoding='utf-8-sig')
    current.to_csv(HERE/'独立核验_当前6只target16保持未知.csv',index=False,encoding='utf-8-sig')
    receipt=dict(formed_at=datetime.now(timezone.utc).isoformat(),verification_new_fits=0,no_fetch=True,no_new_market_outcomes_read=True,old_target_state_untouched=True,
        original_26439_all_columns_exact=True,all_30216_complete8heads=True,current6_target16_unmatured=True,september_original_B=[143,107,40,36],
        independent_inner_saved_model_replays=48,independent_final_classifier_replays=16,independent_calibrator_replays=16,
        all16_calibrators_positive_slope_and_converged=True,original_day_rankings_and_Topk_all_identical=True,guards=guards,fixed_screen_pass=result['fixed_screen_pass'],
        calibrated_map_is_not_probability_validation=True,source_hashes_verified=len(sources),
        source_bindings={str(Path(p).resolve()):sha(p) for p in [Path(__file__),ROOT/'protocol_BEFORE_ANY_FIT.json',ROOT/'fixed_input_BEFORE_ANY_FIT.joblib',REFERENCE,complete_path,result_path]},
        output_bindings={str(p.resolve()):sha(p) for p in HERE.glob('*.csv')})
    receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(verified=True,new_fits=0,source_hashes=len(sources),fixed_screen_pass=result['fixed_screen_pass']),ensure_ascii=False))
    print(stat[stat.scope.eq('ORIGINAL_NONWORD')][['period','arm','TP','FP','Recall','Precision','F1']].to_string(index=False))

if __name__=='__main__':main()
