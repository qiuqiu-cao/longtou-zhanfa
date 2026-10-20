"""Independent monthly252/PREFIX qualification and saved-score replay. No fit."""
from pathlib import Path
from datetime import datetime
import hashlib
import importlib.util
import json
import sys

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
REPO=ROOT.parents[2]
sys.path.insert(0,str(REPO))
import model_upgrade_history as history
import expectation_345_research as original_weights
import relay_local_recovery as recovery

PRIOR=REPO/'research/model_upgrade_validation/positive_filter_20261016T091007Z'
RECOVERY=REPO/'research/local_repair_20261016T051940Z/recovery_manifest.json'
HELPER=PRIOR/'training_audit/verify_saved_positive_heads_NO_FITS.py'
spec=importlib.util.spec_from_file_location('readonly_prior_meta_audit',HELPER)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
KEYS=['date','code']
ARMS=['MONTHLY252_OOF_FROZEN_SCORE_FILTER','MONTHLY252_OOF_FROZEN_INDIVIDUAL_FILTER']
KIND=dict(zip(ARMS,['POSITIVE_FILTER_SCORE_ONLY','POSITIVE_FILTER_INDIVIDUAL_CONTEXT']))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def csv(p):return pd.read_csv(p,dtype={'code':str},float_precision='round_trip',low_memory=False)


def training(full,cutoff):
    mask=helper.known(full)&full.date.lt(cutoff)&full.target_date.lt(cutoff)
    known=full[mask].copy();days=sorted(known.date.unique())[-252:]
    result=known[known.date.isin(days)].copy()
    assert len(days)==252 and result.target_date.max()<cutoff
    return result


def run():
    done_path=ROOT/'execution/run_COMPLETED_BEFORE_EVALUATION.json';done=read(done_path)
    recovery_status=recovery.verify_sources(done['source_bindings'],RECOVERY)
    protocol_path=ROOT/'protocol_BEFORE_ANY_FIT.json';protocol=read(protocol_path)
    data_path=ROOT/'fixed_input_BEFORE_ANY_FIT.joblib';data=joblib.load(data_path)
    prior_input=PRIOR/'fixed_input_BEFORE_ANY_FIT.joblib';previous=joblib.load(prior_input)
    pd.testing.assert_frame_equal(data['full'],previous['full'],check_exact=True)
    assert data['fields']==previous['fields'] and len(data['fields'])==92 and len(data['full'])==9315
    pd.testing.assert_frame_equal(data['shapes'],previous['shapes'],check_exact=True)
    full,fields,shapes=data['full'],data['fields'],data['shapes']
    reference_path=PRIOR/'execution/all49101_predictions_BEFORE_EVALUATION.csv'
    reference=csv(reference_path);base=reference[reference.arm.eq('SOURCE_PATCHED_FEATURE_CLASSIFIER')].copy()
    assert len(base)==3777
    bindings={str(p.resolve()):sha(p) for p in [done_path,protocol_path,data_path,prior_input,reference_path,HELPER,RECOVERY,Path(__file__)]}
    first_records=[];meta_records=[];choice_records=[];score_parts=[]
    max_primary_error=0.;max_meta_error=0.;new_HGB=0;reuse_HGB=0;meta_count=0
    for plan in data['monthly_source_plans']:
        month,cutoff=plan['month'],plan['cutoff']
        train=training(full,cutoff);digest=history._frame_digest(train,fields)
        assert plan['training_days']==252 and plan['training_digest']==digest and plan['max_target']==train.target_date.max()
        completed_path=ROOT/f'execution/{month}_SOURCE_SCORES_COMPLETED.json';completed=read(completed_path)
        model_path=Path(completed['model_path']);saved=joblib.load(model_path);model=saved['model']
        assert saved['features']==fields and saved['training_digest']==digest and saved['cutoff']==cutoff
        assert completed['model_sha256']==sha(model_path)
        estimator=model.named_steps['hgb']
        assert estimator.n_iter_==120 and estimator.early_stopping is False
        for key,value in protocol['first_stage_estimator'].items():assert estimator.get_params()[key]==value
        w=original_weights.weights(train,1.)
        np.testing.assert_allclose([w.sum(),w[train.target.eq(0)].sum(),w[train.target.eq(1)].sum()],[252,126,126],atol=1e-10,rtol=0)
        if plan['reuse']:
            reuse_HGB+=1
            assert sha(model_path)==plan['model_sha256']
            assert not (ROOT/f'execution/{month}_SOURCE_BACKFILL_FIT_STARTED.json').exists()
        else:
            new_HGB+=1
            started_path=ROOT/f'execution/{month}_SOURCE_BACKFILL_FIT_STARTED.json'
            fit_completed_path=ROOT/f'execution/{month}_SOURCE_BACKFILL_FIT_COMPLETED.json'
            started,fit_done=read(started_path),read(fit_completed_path)
            assert started['plan']==fit_done['plan']==plan and started['estimator']==protocol['first_stage_estimator']
            assert fit_done['model_sha256']==sha(model_path) and fit_done['n_iter']==120
            assert datetime.fromisoformat(started['formed_at'])<datetime.fromisoformat(fit_done['completed_at'])<=datetime.fromisoformat(completed['completed_at'])
            for p in [started_path,fit_completed_path]:bindings[str(p.resolve())]=sha(p)
        current=full[full.date.str[:7].eq(month)].copy()
        with threadpool_limits(limits=2):raw=model.predict_proba(current[fields])[:,list(model.classes_).index(1)]
        score_path=ROOT/f'execution/{month}_monthly252_OOF_raw_BEFORE_META.csv';scores=csv(score_path)
        pd.testing.assert_frame_equal(scores[KEYS],current[KEYS].reset_index(drop=True),check_exact=True,check_dtype=False)
        error=float(np.max(np.abs(scores.raw_score.to_numpy()-raw)))
        max_primary_error=max(max_primary_error,error);assert error<=1e-14
        assert scores.first_stage_model_cutoff.eq(cutoff).all()
        if plan['reuse']:
            frozen=base[base.date.str[:7].eq(month)]
            scored=scores.set_index(KEYS).loc[pd.MultiIndex.from_frame(frozen[KEYS])].raw_score.to_numpy()
            np.testing.assert_allclose(scored,frozen.b_score,atol=1e-14,rtol=0)
        score_parts.append(scores)
        first_records.append(dict(month=month,cutoff=cutoff,training_days=252,training_rows=len(train),
            max_training_target=train.target_date.max(),strict_maturity=True,training_digest=digest,
            new_fit=not plan['reuse'],reused_exact_deployment_head=plan['reuse'],n_iter=120,early_stopping=False,
            weight_total=float(w.sum()),weight_negative_mass=float(w[train.target.eq(0)].sum()),weight_positive_mass=float(w[train.target.eq(1)].sum()),
            replay_rows=len(scores),replay_error=error,model_path=str(model_path.resolve()),model_sha256=sha(model_path)))
        for p in [completed_path,model_path,score_path]:bindings[str(p.resolve())]=sha(p)
    all_scores=pd.concat(score_parts,ignore_index=True)
    assert not all_scores.duplicated(KEYS).any()
    score_lookup=all_scores.set_index(KEYS)
    for original_plan in data['plans']:
        assert sha(original_plan['final_model_path'])==original_plan['final_model_sha256']
        bindings[str(Path(original_plan['final_model_path']).resolve())]=sha(original_plan['final_model_path'])
    for plan in data['alignment_plans']:
        month=plan['month'];selection_path=ROOT/f'execution/{month}_SELECTION_BEFORE_OUTER_SCORE.json';selection=read(selection_path)
        assert selection['month']==month and not selection['outer_results_used_for_selection'] and not selection['final_meta_refit']
        source_qualification=[]
        for cutoff in plan['source_month_cutoffs']:
            mature=full[helper.known(full)&full.date.lt(cutoff)&full.target_date.lt(cutoff)]
            available=mature.date.nunique()
            assert plan['source_qualified_days'][cutoff]==min(available,252)
            source_qualification.append(available>=252)
        assert plan['eligible']==all(source_qualification)==selection['eligible']
        if plan['eligible']:
            parts=[]
            for number,days in enumerate(plan['validation_dates'],1):
                rows=full[full.date.isin(days)].copy();index=pd.MultiIndex.from_frame(rows[KEYS]);got=score_lookup.loc[index]
                rows['raw_score']=got.raw_score.to_numpy();rows['first_stage_model_cutoff']=got.first_stage_model_cutoff.to_numpy()
                assert rows.first_stage_model_cutoff.eq(rows.date.str[:7]+'-01').all()
                rows=rows.merge(shapes[KEYS+['verified_current_price_shape']],on=KEYS,validate='one_to_one')
                rows['daily_one_word']=rows.frozen_original_nonword_identity;rows=rows[rows.daily_one_word.eq(0)].copy()
                assert not rows.verified_current_price_shape.eq(1).any()
                rows['base_call']=np.where(rows.verified_current_price_shape.eq(0),np.where(rows.raw_score.gt(.5),'PROMOTE','FAIL'),'UNKNOWN')
                rows['call']=rows.base_call;rows['ranking_score']=rows.raw_score;rows['base_score_logit']=helper.logits(rows.raw_score);rows['inner_block']=number
                parts.append(rows)
            reconstructed=pd.concat(parts,ignore_index=True)
            oof_path=ROOT/f'execution/{month}_complete_monthly252_OOF_BEFORE_META.csv';oof=csv(oof_path)
            pd.testing.assert_frame_equal(oof,reconstructed,check_exact=True,check_dtype=False)
            assert min(oof.loc[oof.inner_block.eq(3),'date'])==plan['first_third_signal']
            before=plan['first_third_signal']
            prefix=oof[(oof.inner_block.lt(3))&helper.eligible(oof)&helper.known(oof)&oof.date.lt(before)&oof.target_date.lt(before)].copy()
            prefix_path=ROOT/f'execution/{month}_frozen_meta_training.csv'
            pd.testing.assert_frame_equal(csv(prefix_path),prefix.reset_index(drop=True),check_exact=True,check_dtype=False)
            assert prefix.target_date.max()<before<plan['cutoff']
            validation=oof[oof.inner_block.eq(3)].copy()
            for p in [oof_path,prefix_path]:bindings[str(p.resolve())]=sha(p)
        for arm in ARMS:
            kind=KIND[arm];choice=selection['choices'][arm];columns=protocol['features'][kind]
            if plan['eligible']:
                model_path=ROOT/f'execution/{month}_{kind}_ALIGNED_FROZEN_PREFIX_model.joblib'
                model,info=helper.verify_model(model_path,prefix,columns,bindings)
                meta_count+=1;meta_records.append(dict(month=month,arm=arm,**info))
                validation_scores=helper.predict(model,validation,columns)
                tables={t:helper.stats(validation,validation_scores,t) for t in helper.THRESHOLDS}
                threshold=helper.choose(tables)
                assert threshold==choice['threshold']
                for t in helper.THRESHOLDS:
                    for key,value in tables[t].items():assert value==choice['statistics'][str(t)][key]
                assert choice['max_training_target']==prefix.target_date.max() and choice['first_third_signal']==before
                fit_completed=read(ROOT/f'execution/{month}_{kind}_ALIGNED_FROZEN_PREFIX_FIT_COMPLETED.json')
                assert datetime.fromisoformat(fit_completed['completed_at'])<datetime.fromisoformat(selection['formed_at'])
                model_sha_before=sha(model_path)
            else:
                model=None;threshold=0.
                assert choice['threshold']==0. and choice['reason']=='INSUFFICIENT_MONTHLY252_OOF_WARMUP'
                assert not list((ROOT/'execution').glob(f'{month}_*ALIGNED_FROZEN_PREFIX*'))
            inference=base[base.date.str[:7].eq(month)].copy();inference['base_call']=inference.evaluation_call
            got=full.set_index(KEYS).loc[pd.MultiIndex.from_frame(inference[KEYS])]
            for c in helper.INDIVIDUAL:inference[c]=got[c].to_numpy()
            inference['base_score_logit']=helper.logits(inference.b_score)
            score=helper.predict(model,inference,columns)
            prediction_path=ROOT/f'execution/{month}_{arm}_predictions_BEFORE_EVALUATION.csv';prediction=csv(prediction_path)
            assert list(prediction.date)==list(inference.date) and list(prediction.code)==list(inference.code)
            mask=helper.eligible(inference).to_numpy();error=float(np.max(np.abs(score[mask]-prediction.filter_score.to_numpy()[mask]))) if mask.any() else 0.
            max_meta_error=max(max_meta_error,error);assert error<=1e-14
            reject=mask&(score<=threshold) if threshold else np.zeros(len(inference),bool)
            calls=np.where(reject,'FAIL',inference.base_call)
            assert np.array_equal(prediction.call,calls) and np.array_equal(prediction.filter_applied,reject)
            for c in ['ranking_score','score','b_score','target','label_valid','daily_one_word']:
                pd.testing.assert_series_equal(prediction[c].reset_index(drop=True),inference[c].reset_index(drop=True),check_exact=True,check_dtype=False)
            assert prediction_path.stat().st_mtime>=selection_path.stat().st_mtime
            if plan['eligible']:assert sha(model_path)==model_sha_before
            choice_records.append(dict(month=month,arm=arm,qualified_month=plan['eligible'],threshold=threshold,
                same_prefix_model_after_threshold_selection=True,final_meta_refit=False,exact_independent_threshold_choice=True,
                complete_replay_rows=len(prediction),saved_replay_error=error,
                max_prefix_target=prefix.target_date.max() if plan['eligible'] else None,
                third_first=plan['first_third_signal'],outer_cutoff=plan['cutoff'],selection_formed_at=selection['formed_at']))
            bindings[str(prediction_path.resolve())]=sha(prediction_path)
        bindings[str(selection_path.resolve())]=sha(selection_path)
    assert new_HGB==6 and reuse_HGB==15 and meta_count==24
    assert done['first_stage_backfill_fits']==6 and done['reused_source_models']==15 and done['meta_fits']==24 and done['first_stage_deployment_refits']==0
    assert len(list((ROOT/'execution').glob('*_SOURCE_BACKFILL_model.joblib')))==6
    assert len(list((ROOT/'execution').glob('*_ALIGNED_FROZEN_PREFIX_model.joblib')))==24
    assert len(list((ROOT/'execution').glob('*_model.joblib')))==30
    assert not list((ROOT/'execution').glob('*_FINAL*')) and not (ROOT/'FAILURE_STOP.json').exists()
    output=OUT/'verified_monthly252_and_PREFIX';output.mkdir(exist_ok=False)
    pd.DataFrame(first_records).to_csv(output/'all21_primary_monthly252_training_and_replay.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(meta_records).to_csv(output/'all24_PREFIX_models_preprocessing_and_convergence.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(choice_records).to_csv(output/'all32_frozen_model_or_warmup_routing_replays.csv',index=False,encoding='utf-8-sig')
    receipt=dict(independent_training_and_replay_pass=True,independent_new_fits=0,official_target16_new_outcomes_read=False,
        full9315_92_features_and_original_identity_unchanged=True,all_monthly_models_strict252_mature_signals_before_natural_month=True,
        exact_first_stage_parameters_and_class_balanced_daily_weights=True,new_early_month_HGB_checked=6,
        exact_old_OOF_HGB_reuses_checked=15,all16_old_deployment_heads_SHA_unchanged=True,
        all21_monthly_source_models_saved_score_replay_max_error=max_primary_error,
        prefix_meta_models_checked=24,all_PREFIX_targets_strict_before_third_first_signal=True,
        no_final_meta_refit_model_artifacts=True,selected_saved_PREFIX_reused_for_outer=True,
        all24_independent_threshold_selections_exact=True,all32_outer_policy_replay_max_error=max_meta_error,
        four_whole_outer_warmup_months_bypassed_per_arm=True,all_meta_n_iter_below2000=True,
        min_meta_n_iter=min(x['n_iter'] for x in meta_records),max_meta_n_iter=max(x['n_iter'] for x in meta_records),
        exact30_new_research_fit_artifacts_no_extra_fits=True,no_outer_metrics_recomputed=True,
        historical_XML_recovery_explicit=recovery_status,
        source_bindings=bindings,
        output_bindings={str(p.resolve()):sha(p) for p in output.iterdir() if p.suffix=='.csv'})
    (OUT/'monthly252_training_and_saved_model_replay_BINDINGS.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['source_bindings','output_bindings']},ensure_ascii=False,indent=2))


if __name__=='__main__':run()
