"""Independent saved-model replay and training qualification. Never fits."""
from pathlib import Path
from fractions import Fraction
from datetime import datetime
import ast
import hashlib
import json
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
REPO=ROOT.parents[2]
sys.path.insert(0,str(REPO))
import model_upgrade_history as history
import relay_local_recovery as recovery

PRIOR=REPO/'research/model_upgrade_validation/false_positive_20261016T030307Z'
SOURCE=REPO/'research/model_upgrade_validation/ranking_optimization_20261015T173045Z/source_feature_training'
RECOVERY=REPO/'research/local_repair_20261016T051940Z/recovery_manifest.json'
MAIN=REPO/'relay_positive_filter_research.py'
KEYS=['date','code']
INDIVIDUAL=['boards','height_gap','same_height_rivals','turnover','amplitude','log_signal_amount',
    'price_same_height_percentile','signal_amount_same_height_percentile','turnover_same_height_percentile']
ARMS=['POSITIVE_FILTER_SCORE_ONLY','POSITIVE_FILTER_INDIVIDUAL_CONTEXT']
THRESHOLDS=[0.]+[n/10 for n in range(1,10)]


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csv(p):return pd.read_csv(p,dtype={'code':str},float_precision='round_trip',low_memory=False)
def eligible(f):return f.daily_one_word.eq(0)&f.verified_current_price_shape.eq(0)&f.base_call.eq('PROMOTE')
def known(f):return f.label_valid.fillna(False).astype(bool)&f.target.isin([0,1])
def logits(raw):
    value=np.clip(np.asarray(raw,float),1e-12,1-1e-12)
    return np.log(value)-np.log1p(-value)


def stats(f,score,t):
    e=eligible(f).to_numpy();reject=e&(np.asarray(score)<=t) if t else np.zeros(len(f),bool)
    calls=np.where(reject,'FAIL',f.base_call)
    k=known(f).to_numpy();y=f.target.eq(1).to_numpy()
    tp=int((k&y&(calls=='PROMOTE')).sum());fp=int((k&~y&(calls=='PROMOTE')).sum())
    positive=int((k&y).sum())
    return dict(candidates=len(f),known=int(k.sum()),unknown_outcomes=int((~k).sum()),
        positives=positive,tp=tp,fp=fp,missed_positives=positive-tp)


def choose(statistics):
    base=statistics[0.]
    def metric(s,name):
        tp,fp,pos=s['tp'],s['fp'],s['positives']
        n,d=(tp,tp+fp) if name=='P' else (tp,pos) if name=='R' else (2*tp,tp+fp+pos)
        return Fraction(n,d) if d else None
    good=[]
    for t in THRESHOLDS[1:]:
        new=statistics[t]
        p,r,f,bp,br,bf=[metric(s,n) for s,n in [(new,'P'),(new,'R'),(new,'F'),(base,'P'),(base,'R'),(base,'F')]]
        if None not in [p,r,f,bp,br,bf] and p>bp and r>=br-Fraction(1,20) and f>=bf and new['fp']*10<=base['fp']*9:
            good.append(t)
    return min(good,key=lambda t:(-metric(statistics[t],'P'),t)) if good else 0.


def predict(model,f,columns):
    output=np.full(len(f),np.nan)
    e=eligible(f).to_numpy()
    if model is None:output[e]=0.
    elif e.any():
        with threadpool_limits(limits=2):
            output[e]=model.predict_proba(f.loc[e,columns])[:,list(model.named_steps['logistic'].classes_).index(1)]
    return output


def verify_model(path,rows,columns,bindings):
    saved=joblib.load(path)
    model=saved['model'];imputer=model.named_steps['imputer'];scale=model.named_steps['scale'];lr=model.named_steps['logistic']
    assert saved['features']==columns and saved['weights']=='Every row weight1; no class balancing'
    assert lr.C==1 and lr.class_weight is None and lr.solver=='lbfgs' and lr.max_iter==2000 and lr.random_state==20261016
    assert saved['estimator']['class_weight'] is None
    assert int(lr.n_iter_.max())<2000 and np.isfinite(lr.coef_).all() and np.isfinite(lr.intercept_).all()
    assert imputer.strategy=='median' and imputer.add_indicator is True and imputer.keep_empty_features is True
    x=rows[columns].to_numpy(float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        median=np.nanmedian(x,axis=0)
    median=np.where(np.isnan(median),0.,median)
    np.testing.assert_allclose(imputer.statistics_,median,atol=1e-14,rtol=0)
    np.testing.assert_array_equal(imputer.indicator_.features_,np.flatnonzero(np.isnan(x).any(axis=0)))
    imputed=imputer.transform(rows[columns])
    np.testing.assert_allclose(scale.mean_,imputed.mean(axis=0),atol=1e-12,rtol=1e-14)
    np.testing.assert_allclose(scale.var_,imputed.var(axis=0),atol=1e-10,rtol=1e-12)
    digest=hashlib.sha256(pd.util.hash_pandas_object(rows[KEYS+['target_date','target']+columns],index=False).values.tobytes()).hexdigest()
    assert saved['training_digest']==digest and saved['max_training_target']==rows.target_date.max()
    assert saved['rows']==len(rows) and saved['positives']==int(rows.target.eq(1).sum()) and saved['negatives']==int(rows.target.eq(0).sum())
    stem=str(path).removesuffix('_model.joblib')
    started_path=Path(stem+'_FIT_STARTED.json');completed_path=Path(stem+'_FIT_COMPLETED.json')
    started,completed=read(started_path),read(completed_path)
    assert started['training_digest']==completed['training_digest']==digest
    assert completed['model_sha256']==sha(path) and completed['n_iter']==int(lr.n_iter_.max())
    for p in [path,started_path,completed_path]:bindings[str(p.resolve())]=sha(p)
    return model,dict(n_iter=int(lr.n_iter_.max()),train_rows=len(rows),train_days=rows.date.nunique(),
        training_digest=digest,imputer_statistics_train_only=True,scaler_statistics_train_only=True,
        every_row_weight1=True,class_weight_none=True,features=columns)


def run():
    completed_path=ROOT/'execution/run_COMPLETED_BEFORE_EVALUATION.json'
    completed=read(completed_path)
    recovery_status=recovery.verify_sources(completed['source_bindings'],RECOVERY)
    protocol_path=ROOT/'protocol_BEFORE_ANY_FIT.json';protocol=read(protocol_path)
    assert protocol['features'][ARMS[0]]==['base_score_logit'] and protocol['features'][ARMS[1]]==['base_score_logit']+INDIVIDUAL
    assert protocol['sample_weights']=='Every row1; class_weight=None' and protocol['thresholds']==THRESHOLDS
    tree=ast.parse(MAIN.read_text(encoding='utf-8'))
    fit_calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='fit']
    assert len(fit_calls)==1 and len(fit_calls[0].args)==2 and not fit_calls[0].keywords
    data_path=ROOT/'fixed_input_BEFORE_ANY_FIT.joblib';data=joblib.load(data_path)
    previous_path=PRIOR/'fixed_input_BEFORE_ANY_FIT.joblib';previous=joblib.load(previous_path)
    pd.testing.assert_frame_equal(data['full'],previous['full'],check_exact=True)
    assert data['fields']==previous['fields'] and len(data['fields'])==92 and len(data['full'])==9315
    full,fields,shapes=data['full'],data['fields'],data['shapes']
    shape_saved=joblib.load(SOURCE/'prepared.joblib')['verified_shapes']
    pd.testing.assert_frame_equal(shapes,shape_saved,check_exact=True)
    reference_path=REPO/'research/model_upgrade_validation/iteration_research_20261016T041226Z/execution/all41547_predictions_BEFORE_EVALUATION.csv'
    # The immutable reference is explicitly bound by the reviewed source; locate
    # its unique exact filename from the frozen protocol to avoid guessed roots.
    reference_candidates=[Path(p) for p in protocol['source_bindings'] if Path(p).name=='all41547_predictions_BEFORE_EVALUATION.csv']
    assert len(reference_candidates)==1
    reference_path=reference_candidates[0]
    refs=csv(reference_path);base=refs[refs.arm.eq(protocol['baseline'])].copy()
    assert len(base)==3777
    identity_path=REPO/'research/model_upgrade_validation/short_term_thinking_20261015T071000Z/dataset.joblib'
    historical_identity=joblib.load(identity_path)['features'].set_index(KEYS).daily_one_word
    frozen_identity=refs[refs.arm.eq('BASE')].set_index(KEYS).daily_one_word
    for r in full.itertuples():
        key=(r.date,r.code)
        expected=frozen_identity.loc[key] if key in frozen_identity.index else historical_identity.loc[key]
        assert pd.isna(expected) and pd.isna(r.frozen_original_nonword_identity) or expected==r.frozen_original_nonword_identity
    bindings={str(p.resolve()):sha(p) for p in [completed_path,protocol_path,data_path,previous_path,reference_path,identity_path,RECOVERY,MAIN,Path(__file__)]}
    model_records=[];cache_records=[];qualification_records=[]
    max_cache_error=0.;max_meta_error=0.;model_count=0
    for plan in data['plans']:
        month,cutoff=plan['month'],plan['cutoff']
        eligible_mature=known(full)&full.date.lt(cutoff)&full.target_date.lt(cutoff)
        mature=full[eligible_mature];days=sorted(mature.date.unique())[-252:]
        window=mature[mature.date.isin(days)].copy()
        assert len(days)==252 and history._frame_digest(window,fields)==plan['training_digest']
        oof_blocks=[]
        for block in (1,2,3):
            vd=days[63*block:63*(block+1)];first=vd[0]
            past=window[window.date.lt(first)&window.target_date.lt(first)]
            model_path=PRIOR/f'execution/{month}_inner{block}_model.joblib'
            cache_path=PRIOR/f'execution/{month}_inner{block}_all_validation_scores_BEFORE_EFFECTS.csv'
            saved=joblib.load(model_path);cache=csv(cache_path)
            assert sha(model_path)==protocol['source_bindings'][str(model_path.resolve())]
            assert saved['training_digest']==history._frame_digest(past,fields) and saved['features']==fields
            assert past.target_date.max()<first and saved['cutoff']==first
            complete=full[full.date.isin(vd)].copy()
            with threadpool_limits(limits=2):
                raw=saved['model'].predict_proba(complete[fields])[:,list(saved['model'].classes_).index(1)]
            complete['raw_score']=raw
            original=complete.set_index(KEYS).loc[pd.MultiIndex.from_frame(cache[KEYS])]
            err=float(np.max(np.abs(original.raw_score.to_numpy()-cache.raw_score.to_numpy())))
            max_cache_error=max(max_cache_error,err)
            assert err<=1e-14
            complete=complete.merge(shapes,on=KEYS,validate='one_to_one')
            complete['daily_one_word']=complete.frozen_original_nonword_identity
            complete=complete[complete.daily_one_word.eq(0)].copy()
            assert not complete.verified_current_price_shape.eq(1).any()
            complete['base_call']=np.where(complete.verified_current_price_shape.eq(0),np.where(complete.raw_score.gt(.5),'PROMOTE','FAIL'),'UNKNOWN')
            complete['base_score_logit']=logits(complete.raw_score)
            complete['ranking_score']=complete.raw_score;complete['call']=complete.base_call;complete['inner_block']=block
            complete['was_in_original_cached_validation']=pd.MultiIndex.from_frame(complete[KEYS]).isin(pd.MultiIndex.from_frame(cache[KEYS]))
            oof_blocks.append(complete)
            cache_records.append(dict(month=month,block=block,first_validation=first,max_first_stage_target=past.target_date.max(),
                full_validation_rows=len(raw),original_cache_rows=len(cache),complete_original_B_rows=len(complete),
                new_completion_B_rows=int((~complete.was_in_original_cached_validation).sum()),max_cache_score_error=err,
                original_model_sha256=sha(model_path),first_stage_new_fits=0))
            for p in [model_path,cache_path]:bindings[str(p.resolve())]=sha(p)
        expected_oof=pd.concat(oof_blocks,ignore_index=True)
        oof_path=ROOT/f'execution/{month}_complete_B_OOF_BEFORE_SELECTION.csv';oof=csv(oof_path)
        pd.testing.assert_frame_equal(oof.drop(columns=['raw_score','ranking_score','base_score_logit']),
            expected_oof.drop(columns=['raw_score','ranking_score','base_score_logit']),check_exact=True,check_dtype=False)
        for c in ['raw_score','ranking_score','base_score_logit']:
            np.testing.assert_allclose(oof[c],expected_oof[c],atol=1e-14,rtol=1e-14)
        assert not oof.duplicated(KEYS).any()
        third_first=min(oof.loc[oof.inner_block.eq(3),'date'])
        prefix=oof[(oof.inner_block.lt(3))&eligible(oof)&known(oof)&oof.date.lt(third_first)&oof.target_date.lt(third_first)].copy()
        final=oof[eligible(oof)&known(oof)&oof.date.lt(cutoff)&oof.target_date.lt(cutoff)].copy()
        validation=oof[oof.inner_block.eq(3)].copy()
        prefix_path=ROOT/f'execution/{month}_meta_prefix_training.csv';final_path=ROOT/f'execution/{month}_meta_final_training.csv'
        pd.testing.assert_frame_equal(csv(prefix_path),prefix.reset_index(drop=True),check_exact=True,check_dtype=False)
        pd.testing.assert_frame_equal(csv(final_path),final.reset_index(drop=True),check_exact=True,check_dtype=False)
        assert prefix.target_date.max()<third_first and final.target_date.max()<cutoff
        assert prefix.target.nunique()==2 and final.target.nunique()==2
        selection_path=ROOT/f'execution/{month}_SELECTION_BEFORE_FINAL_FIT_AND_OUTER.json';selection=read(selection_path)
        assert selection['max_prefix_target']==prefix.target_date.max() and selection['max_final_target']==final.target_date.max()
        assert selection['first_third_signal']==third_first and not selection['outer_results_used_for_selection']
        for arm in ARMS:
            columns=protocol['features'][arm]
            prefix_model_path=ROOT/f'execution/{month}_{arm}_PREFIX_model.joblib'
            final_model_path=ROOT/f'execution/{month}_{arm}_FINAL_model.joblib'
            prefix_model,prefix_info=verify_model(prefix_model_path,prefix,columns,bindings)
            final_model,final_info=verify_model(final_model_path,final,columns,bindings)
            model_count+=2
            for phase,info in [('PREFIX',prefix_info),('FINAL',final_info)]:model_records.append(dict(month=month,arm=arm,phase=phase,**info))
            third_scores=predict(prefix_model,validation,columns)
            table={t:stats(validation,third_scores,t) for t in THRESHOLDS}
            threshold=choose(table)
            assert threshold==selection['choices'][arm]['selected_threshold']
            for t in THRESHOLDS:
                saved_stat=selection['choices'][arm]['statistics'][str(t)]
                for key,value in table[t].items():assert value==saved_stat[key]
            final_started=read(ROOT/f'execution/{month}_{arm}_FINAL_FIT_STARTED.json')
            prefix_completed=read(ROOT/f'execution/{month}_{arm}_PREFIX_FIT_COMPLETED.json')
            assert datetime.fromisoformat(prefix_completed['completed_at'])<=datetime.fromisoformat(selection['formed_at'])<datetime.fromisoformat(final_started['formed_at'])
            inference=base[base.date.str[:7].eq(month)].copy();inference['base_call']=inference.evaluation_call
            lookup=full.set_index(KEYS);index=pd.MultiIndex.from_frame(inference[KEYS])
            for field in INDIVIDUAL:
                values=lookup.loc[index,field].to_numpy()
                if field in inference:np.testing.assert_allclose(inference[field],values,equal_nan=True,atol=0,rtol=0)
                else:inference[field]=values
            inference['base_score_logit']=logits(inference.b_score)
            secondary=predict(final_model,inference,columns)
            prediction_path=ROOT/f'execution/{month}_{arm}_predictions_BEFORE_EVALUATION.csv';prediction=csv(prediction_path)
            assert list(prediction.code)==list(inference.code) and list(prediction.date)==list(inference.date)
            mask=eligible(inference).to_numpy()
            error=float(np.max(np.abs(secondary[mask]-prediction.filter_score.to_numpy()[mask]))) if mask.any() else 0.
            max_meta_error=max(max_meta_error,error);assert error<=1e-14
            assert prediction.filter_score[~mask].isna().all()
            reject=mask&(secondary<=threshold) if threshold else np.zeros(len(inference),bool)
            calls=np.where(reject,'FAIL',inference.base_call)
            assert np.array_equal(calls,prediction.call) and np.array_equal(calls,prediction.research_call)
            assert np.array_equal(reject,prediction.filter_applied) and prediction.filter_threshold.eq(threshold).all()
            for field in ['ranking_score','score','b_score','target','label_valid','daily_one_word','verified_current_price_shape']:
                pd.testing.assert_series_equal(prediction[field].reset_index(drop=True),inference[field].reset_index(drop=True),check_exact=True,check_dtype=False)
            assert prediction_path.stat().st_mtime>=selection_path.stat().st_mtime
            qualification_records.append(dict(month=month,arm=arm,meta_prefix_rows=len(prefix),meta_final_rows=len(final),
                max_prefix_target=prefix.target_date.max(),third_first_signal=third_first,max_final_target=final.target_date.max(),outer_cutoff=cutoff,
                third_complete_original_B_rows=len(validation),third_unknown_outcomes=int((~known(validation)).sum()),
                chosen_threshold=threshold,independent_threshold_choice_exact=True,selection_before_final_and_outer=True,
                replay_rows=len(prediction),eligible_positive_replayed=int(mask.sum()),max_saved_model_replay_error=error))
            bindings[str(prediction_path.resolve())]=sha(prediction_path)
        for p in [oof_path,prefix_path,final_path,selection_path]:bindings[str(p.resolve())]=sha(p)
        assert sha(plan['final_model_path'])==plan['final_model_sha256']
        bindings[str(Path(plan['final_model_path']).resolve())]=sha(plan['final_model_path'])
    assert model_count==64 and completed['new_second_stage_fits']==64 and completed['first_stage_new_fits']==0
    assert len(list((ROOT/'execution').glob('*_model.joblib')))==64
    assert len(list((ROOT/'execution').glob('*_FIT_STARTED.json')))==64
    assert len(list((ROOT/'execution').glob('*_FIT_COMPLETED.json')))==64
    assert not (ROOT/'FAILURE_STOP.json').exists()
    output=OUT/'executed_training_qualification';output.mkdir(exist_ok=False)
    pd.DataFrame(model_records).to_csv(output/'all64_meta_models_preprocessing_and_convergence.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(cache_records).to_csv(output/'all48_first_stage_cache_completion_replays.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(qualification_records).to_csv(output/'all32_meta_training_threshold_and_saved_prediction_replays.csv',index=False,encoding='utf-8-sig')
    receipt=dict(independent_training_qualification_pass=True,independent_new_fits=0,official_target16_new_outcomes_read=False,
        full9315_and92_fields_and_original_identity_unchanged=True,all16_original252_windows_exact=True,
        first_stage_new_fits=0,original48_cache_models_and16_final_model_hashes_unchanged=True,
        first_stage_models_replayed=48,original_cached_scores_max_difference=max_cache_error,
        complete_original_B_OOF_unknowns_retained=True,prefix_targets_strictly_before_third_block=True,
        final_targets_strictly_before_outer_cutoff=True,all64_saved_meta_models_verified=True,
        all_meta_n_iter_below2000=True,min_meta_n_iter=min(r['n_iter'] for r in model_records),max_meta_n_iter=max(r['n_iter'] for r in model_records),
        row_equal_fit_without_sample_or_class_weight=True,preprocessing_fit_on_declared_training_only=True,
        all32_threshold_choices_independently_recomputed=True,full_B_third_block_recall_denominator=True,
        selection_precedes_final_refit_and_outer_predictions=True,outer_meta_saved_model_replay_max_difference=max_meta_error,
        raw_scores_rankings_and_noneligible_routes_unchanged=True,second_stage_research_fits_recorded=64,
        historical_XML_recovery_explicit=recovery_status,outer_predictive_metrics_repeated=False,
        final_refit_threshold_transfer_risk_still_retained=True,source_bindings=bindings)
    (output/'independent_training_and_saved_model_replay_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_bindings'},ensure_ascii=False,indent=2))


if __name__=='__main__':run()
