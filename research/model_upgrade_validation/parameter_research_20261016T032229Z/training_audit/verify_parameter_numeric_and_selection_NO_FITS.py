"""Independent parameter, temporal membership and OOF selection audit. No fit."""
from pathlib import Path
from fractions import Fraction
from datetime import datetime,timezone
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
REPO=ROOT.parents[2]
sys.path.insert(0,str(REPO))
import model_upgrade_history as history
import expectation_345_research as original


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def rational_stats(rows,threshold):
    assert rows.target.isin([0,1]).all() and rows.label_valid.all()
    positive=int(rows.target.eq(1).sum())
    p=rows.verified_current_price_shape.eq(0)&rows.raw_score.gt(threshold)
    tp=int((p&rows.target.eq(1)).sum());fp=int((p&rows.target.eq(0)).sum());fn=positive-tp
    return dict(tp=tp,fp=fp,positive=positive,F05=Fraction(5*tp,5*tp+fn+4*fp),Recall=Fraction(tp,positive),F1=Fraction(2*tp,2*tp+fn+fp))


def guarded(candidate,base):
    return candidate['Recall']>=base['Recall']-Fraction(1,40) and candidate['F1']>=base['F1']-Fraction(1,100)


def run():
    protocol_path=ROOT/'protocol_BEFORE_ANY_FIT.json';protocol=read(protocol_path)
    data_path=ROOT/'fixed_input_BEFORE_ANY_FIT.joblib';data=joblib.load(data_path)
    previous_path=REPO/'research/model_upgrade_validation/false_positive_20261016T030307Z/fixed_input_BEFORE_ANY_FIT.joblib'
    previous=joblib.load(previous_path)
    pd.testing.assert_frame_equal(data['full'],previous['full'],check_exact=True)
    assert data['fields']==previous['fields'] and len(data['fields'])==92
    baseline=protocol['baseline'];configs=protocol['capacity_candidates']
    capacity_keys={'max_depth','max_leaf_nodes','min_samples_leaf','l2_regularization'}
    for params in configs.values():
        assert set(k for k in params if params[k]!=baseline[k])==capacity_keys
    assert not (ROOT/'FAILURE_STOP.json').exists()
    records=[];selections=[];bindings={str(p.resolve()):sha(p) for p in [protocol_path,data_path,previous_path,Path(__file__)]}
    final_count=0;inner_count=0;old_final_count=0;fit_numbers=[];final_fit_numbers=[]
    for plan in data['plans']:
        month=plan['month'];full=data['full'];fields=data['fields'];cutoff=plan['cutoff']
        valid=full.label_valid.fillna(False).astype(bool)&full.target.notna()&full.date.lt(cutoff)&full.target_date.lt(cutoff)
        mature=full[valid];days=sorted(mature.date.unique())[-252:]
        window=mature[mature.date.isin(days)].copy()
        assert len(days)==252 and history._frame_digest(window,fields)==plan['training_digest']
        selection_path=ROOT/f'execution/{month}_SELECTION_BEFORE_FINAL_FIT_AND_OUTER_SCORE.json'
        selection=read(selection_path);completed_path=ROOT/f'execution/{month}_COMPLETED.json';completed=read(completed_path)
        assert not selection['outer_scores_or_labels_read_for_selection']
        assert selection['max_OOF_target']<cutoff
        assert sha(plan['final_model_path'])==plan['final_model_sha256']
        bindings[str(Path(plan['final_model_path']).resolve())]=sha(plan['final_model_path'])
        for block_number in (1,2,3):
            validation_days=days[63*block_number:63*(block_number+1)];first=validation_days[0]
            past=window[window.date.lt(first)&window.target_date.lt(first)].copy()
            assert past.target_date.max()<first
            expected=history._frame_digest(past,fields)
            weights=original.weights(past,1.)
            old_model_path=previous_path.parent/f'execution/{month}_inner{block_number}_model.joblib'
            assert sha(old_model_path)==protocol['source_bindings'][str(old_model_path.resolve())]
            old_model=joblib.load(old_model_path)
            assert old_model['training_digest']==expected and old_model['features']==fields
            old_estimator=old_model['model'].named_steps['hgb']
            assert old_estimator.n_iter_==120 and old_estimator.early_stopping is False
            for k,v in baseline.items():assert old_estimator.get_params()[k]==v
            bindings[str(old_model_path.resolve())]=sha(old_model_path)
            for name,params in configs.items():
                prefix=ROOT/f'execution/{month}_inner{block_number}_{name}'
                model_path=Path(str(prefix)+'_model.joblib');started_path=Path(str(prefix)+'_STARTED.json');done_path=Path(str(prefix)+'_COMPLETED.json')
                saved=joblib.load(model_path);started=read(started_path);done=read(done_path)
                estimator=saved['model'].named_steps['hgb']
                assert saved['params']==started['params']==params
                assert saved['training_digest']==started['training_digest']==expected
                assert saved['features']==fields and saved['cutoff']==started['first_validation']==first
                assert started['max_train_target']==past.target_date.max()<first
                assert estimator.n_iter_==120 and estimator.early_stopping is False
                for k,v in params.items():assert estimator.get_params()[k]==v
                assert done['model_sha256']==sha(model_path)
                assert datetime.fromisoformat(started['formed_at'])<datetime.fromisoformat(selection['formed_at'])
                fit_numbers.append(started['fit_number']);inner_count+=1
                records.append(dict(month=month,block=block_number,config=name,training_days=past.date.nunique(),training_rows=len(past),
                    max_train_target=past.target_date.max(),validation_first=first,strict_maturity=True,params_exact=True,
                    n_iter=estimator.n_iter_,early_stopping=estimator.early_stopping,original_weights_sha256=history.digest(weights.tolist()),
                    training_digest=expected,model_sha256=sha(model_path)))
                for p in [model_path,started_path,done_path]:bindings[str(p.resolve())]=sha(p)
        oof={}
        for name in ['BASE','COMPACT','EXPANDED']:
            p=ROOT/f'execution/{month}_{name}_complete_OOF_BEFORE_SELECTION.csv'
            rows=pd.read_csv(p,dtype={'code':str},float_precision='round_trip')
            assert not rows.duplicated(['date','code']).any() and rows.target_date.max()<cutoff
            assert rows.frozen_original_nonword_identity.eq(0).all()
            oof[name]=rows
            if name!='BASE':pd.testing.assert_frame_equal(rows.drop(columns='raw_score'),oof['BASE'].drop(columns='raw_score'),check_exact=True)
            bindings[str(p.resolve())]=sha(p)
        stats={name:rational_stats(rows,.5) for name,rows in oof.items()}
        eligible=[name for name in ['COMPACT','EXPANDED'] if guarded(stats[name],stats['BASE']) and stats[name]['F05']>stats['BASE']['F05']]
        capacity=max(eligible,key=lambda name:stats[name]['F05']) if eligible else 'BASE'
        thresholds=protocol['threshold_candidates'];th={t:rational_stats(oof[capacity],t) for t in thresholds}
        eligible_t=[t for t in thresholds if t!=.5 and guarded(th[t],stats['BASE']) and th[t]['F05']>th[.5]['F05']]
        threshold=min(eligible_t,key=lambda t:(-th[t]['F05'],abs(Fraction(str(t))-Fraction(1,2)),-Fraction(str(t)))) if eligible_t else .5
        assert selection['selected_capacity']==capacity and selection['selected_threshold']==threshold
        for name,stat in stats.items():
            for key in ['tp','fp','positive']:assert selection['capacity_statistics'][name][key]==stat[key]
        before=datetime.fromisoformat(selection['formed_at']).timestamp()
        for arm in protocol['policy_arms']:
            prediction=ROOT/f'execution/{month}_{arm}_predictions_BEFORE_EVALUATION.csv'
            assert selection_path.stat().st_mtime<=prediction.stat().st_mtime and before<=prediction.stat().st_mtime
            bindings[str(prediction.resolve())]=sha(prediction)
        if capacity=='BASE':
            assert not completed['selected_final_new_fit'] and completed['selected_final_model_sha256']==plan['final_model_sha256']
            assert not (ROOT/f'execution/{month}_selected_FINAL_STARTED.json').exists()
            assert not (ROOT/f'execution/{month}_selected_FINAL_model.joblib').exists()
            old_final_count+=1
        else:
            final_count+=1
            started_path=ROOT/f'execution/{month}_selected_FINAL_STARTED.json';started=read(started_path)
            assert datetime.fromisoformat(selection['formed_at'])<datetime.fromisoformat(started['formed_at'])
            assert selection_path.stat().st_mtime<=started_path.stat().st_mtime
            model_path=Path(completed['selected_final_model_path']);saved=joblib.load(model_path);estimator=saved['model'].named_steps['hgb']
            assert saved['capacity']==started['capacity']==capacity
            assert saved['params']==started['params']==configs[capacity]
            assert saved['training_digest']==started['training_digest']==plan['training_digest']
            assert saved['features']==fields and saved['cutoff']==cutoff
            assert estimator.n_iter_==120 and estimator.early_stopping is False
            for k,v in configs[capacity].items():assert estimator.get_params()[k]==v
            assert completed['selected_final_model_sha256']==sha(model_path)
            final_fit_numbers.append(started['fit_number'])
            for p in [started_path,model_path]:bindings[str(p.resolve())]=sha(p)
        assert completed['inner_fits_this_month']==6
        selections.append(dict(month=month,selected_capacity=capacity,selected_threshold=threshold,
            exact_OOF_selection=True,guards_against_original_BASE=True,strict_improvement_and_tie_rules=True,
            receipt_precedes_final_fit_and_outer_score=True,final_new_fit=capacity!='BASE',original252_members_exact=True))
        for p in [selection_path,completed_path]:bindings[str(p.resolve())]=sha(p)
    assert inner_count==96 and final_count==11 and old_final_count==5
    assert sorted(fit_numbers)==list(range(1,97)) and sorted(final_fit_numbers)==list(range(1,12))
    assert len(list((ROOT/'execution').glob('*_inner*_model.joblib')))==96
    assert len(list((ROOT/'execution').glob('*_selected_FINAL_model.joblib')))==11
    result=read(ROOT/'execution/parameter_fixed_result.json')
    assert result['inner_new_fits']==96 and result['selected_final_new_fits']==11 and result['total_new_fits']==107
    out=ROOT/'training_audit/executed_numeric_qualification';out.mkdir(exist_ok=False)
    pd.DataFrame(records).to_csv(out/'all96_inner_numerical_and_training_qualification.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(selections).to_csv(out/'all16_exact_OOF_selection_qualification.csv',index=False,encoding='utf-8-sig')
    receipt=dict(independent_numeric_qualification_pass=True,independent_new_fits=0,outer_prediction_metrics_repeated=False,
        official_target16_results_read=False,complete9315_members_and_original_identity_unchanged=True,fields92_unchanged=True,
        all16_original252_training_digests_exact=True,all96_inner_maturity_boundaries_strict=True,
        only_four_capacity_parameters_differ=True,learning_rate_iterations_seed_and_weight_code_fixed=True,
        all107_saved_HGB_n_iter120_early_stopping_false=True,all16_exact_fraction_OOF_selections_verified=True,
        threshold_guards_use_original_BASE_no_stacked_loss=True,selection_receipts_precede_final_fit_and_outer_scores=True,
        new_inner_fits=96,new_selected_final_fits=11,baseline_final_reused=5,extra_fit_artifacts_found=False,
        old16_final_classifier_hashes_unchanged=True,no_failure_stop_receipt=True,
        old48_inner_classifier_hashes_params_n_iter_unchanged=True,
        parameter_research_predictive_screen_not_asserted=True,source_bindings=bindings)
    (out/'independent_parameter_numeric_qualification_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_bindings'},ensure_ascii=False,indent=2))


if __name__=='__main__':run()
