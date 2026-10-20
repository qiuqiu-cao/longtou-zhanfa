"""Same-input arithmetic audit of saved November meta models; no fitting.

Only the already-filtered five outer rows and November's third OOF block are
read as scoring inputs. No PREFIX outer-month aggregate performance is run.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import itertools
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
OLD=REPO/'research/model_upgrade_validation/positive_filter_20261016T091007Z'
EXEC=OLD/'execution'
ARM='POSITIVE_FILTER_INDIVIDUAL_CONTEXT'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def csv(p):return pd.read_csv(p,dtype={'code':str},float_precision='round_trip',low_memory=False)

def main():
    HERE.mkdir(parents=True,exist_ok=True)
    receipt_path=HERE/'meta_mechanism_same_input_receipt.json'
    if receipt_path.exists():raise FileExistsError('immutable mechanism audit already exists')
    prefix_path=EXEC/f'2025-11_{ARM}_PREFIX_model.joblib'
    final_path=EXEC/f'2025-11_{ARM}_FINAL_model.joblib'
    completed_path=EXEC/f'2025-11_{ARM}_COMPLETED.json'
    completed=read(completed_path)
    assert sha(prefix_path)==completed['prefix_fit']['model_sha256']
    assert sha(final_path)==completed['final_fit']['model_sha256']
    prefix=joblib.load(prefix_path);final=joblib.load(final_path)
    assert prefix['features']==final['features'] and prefix['estimator']==final['estimator']
    fields=prefix['features'];pm=prefix['model'];fm=final['model']
    assert len(fields)==10
    pcoef=pm['logistic'].coef_[0];fcoef=fm['logistic'].coef_[0]
    pi=float(pm['logistic'].intercept_[0]);fi=float(fm['logistic'].intercept_[0])
    assert not len(pm['imputer'].indicator_.features_) and not len(fm['imputer'].indicator_.features_)
    coeff=[]
    for i,field in enumerate(fields):
        coeff.append(dict(feature=field,prefix_imputer_median=pm['imputer'].statistics_[i],final_imputer_median=fm['imputer'].statistics_[i],
            prefix_scaler_mean=pm['scale'].mean_[i],final_scaler_mean=fm['scale'].mean_[i],
            prefix_scaler_scale=pm['scale'].scale_[i],final_scaler_scale=fm['scale'].scale_[i],
            prefix_standardized_coefficient=pcoef[i],final_standardized_coefficient=fcoef[i],
            standardized_coefficient_delta=fcoef[i]-pcoef[i],prefix_raw_unit_slope=pcoef[i]/pm['scale'].scale_[i],
            final_raw_unit_slope=fcoef[i]/fm['scale'].scale_[i]))
    five_path=OLD/'evaluation/all_filtered_candidates_GAINS_LOSSES_UNKNOWN.csv'
    five=csv(five_path)
    assert len(five)==5 and five.arm.eq(ARM).all() and five.date.str.startswith('2025-11').all()
    oof_path=EXEC/'2025-11_complete_B_OOF_BEFORE_SELECTION.csv'
    oof=csv(oof_path)
    third=oof[oof.inner_block.eq(3)]
    third_eligible=third[third.verified_current_price_shape.eq(0)&third.base_call.eq('PROMOTE')&third.daily_one_word.eq(0)].copy()
    assert len(third)==562 and len(third_eligible)==86
    prefix_train_path=EXEC/'2025-11_meta_prefix_training.csv'
    final_train_path=EXEC/'2025-11_meta_final_training.csv'
    prefix_train=csv(prefix_train_path);final_train=csv(final_train_path)
    assert len(prefix_train)==341 and len(final_train)==427
    assert prefix_train.target_date.max()<third.date.min() and final_train.target_date.max()<'2025-11-01'
    old_ids=set(map(tuple,prefix_train[['date','code']].to_numpy()))
    final_ids=set(map(tuple,final_train[['date','code']].to_numpy()))
    third_ids=set(map(tuple,third_eligible[['date','code']].to_numpy()))
    assert old_ids.issubset(final_ids) and third_ids.issubset(final_ids) and old_ids.isdisjoint(third_ids)
    added=final_train[~pd.MultiIndex.from_frame(final_train[['date','code']]).isin(pd.MultiIndex.from_frame(prefix_train[['date','code']]))]
    assert len(added)==86 and set(map(tuple,added[['date','code']].to_numpy()))==third_ids
    states=dict(month='2025-11',arm=ARM,prefix_rows=341,prefix_positives=int(prefix_train.target.eq(1).sum()),prefix_negatives=int(prefix_train.target.eq(0).sum()),
        final_rows=427,final_positives=int(final_train.target.eq(1).sum()),final_negatives=int(final_train.target.eq(0).sum()),
        added_third_block_rows=86,added_positives=int(added.target.eq(1).sum()),added_negatives=int(added.target.eq(0).sum()),
        prefix_intercept=pi,final_intercept=fi,standardized_intercept_delta=fi-pi,
        raw_input_effective_intercept_prefix=pi-float(np.sum(pcoef*pm['scale'].mean_/pm['scale'].scale_)),
        raw_input_effective_intercept_final=fi-float(np.sum(fcoef*fm['scale'].mean_/fm['scale'].scale_)),
        missing_indicator_dimensions_prefix=0,missing_indicator_dimensions_final=0,
        first_third_signal=third.date.min(),last_third_signal=third.date.max(),max_prefix_training_target=prefix_train.target_date.max(),max_final_training_target=final_train.target_date.max(),
        third_PREFIX_role='THRESHOLD_SELECTION_DATA_NOT_UNBIASED_VALIDATION',third_FINAL_role='INCLUDES_THIS_DATA_IN_FINAL_TRAINING_NOT_GENERALIZATION_COMPARISON')
    rows=[];feature_rows=[];component_rows=[];coalitions=[]
    groups=[('ALREADY_FILTERED_FIVE_ONLY',five),('THIRD_BLOCK_ELIGIBLE_SELECTION_TRAINING_DIAGNOSTIC_ONLY',third_eligible)]
    for scope,frame in groups:
        assert frame[fields].notna().all().all(),'new missing-value path requires a separately explicit attribution definition'
        with threadpool_limits(limits=2):
            pp=pm.predict_proba(frame[fields])[:,list(pm['logistic'].classes_).index(1)]
            fp=fm.predict_proba(frame[fields])[:,list(fm['logistic'].classes_).index(1)]
        x=frame[fields].to_numpy(float)
        px=pm['imputer'].transform(frame[fields]);fx=fm['imputer'].transform(frame[fields])
        np.testing.assert_array_equal(px,x);np.testing.assert_array_equal(fx,x)
        pz=pm['scale'].transform(px);fz=fm['scale'].transform(fx)
        pcon=pz*pcoef;fcon=fz*fcoef
        # Exact symmetric two-factor decomposition. It splits coefficient x
        # standardized-input interaction equally, independent of update order.
        scaler=(pcoef+fcoef)[None,:]*(fz-pz)/2
        coefficients=(pz+fz)*(fcoef-pcoef)[None,:]/2
        np.testing.assert_allclose(scaler+coefficients,fcon-pcon,atol=1e-14,rtol=0)
        plogit=pi+pcon.sum(axis=1);flogit=fi+fcon.sum(axis=1)
        np.testing.assert_allclose(expit(plogit),pp,rtol=0,atol=1e-14)
        np.testing.assert_allclose(expit(flogit),fp,rtol=0,atol=1e-14)
        if scope=='ALREADY_FILTERED_FIVE_ONLY':np.testing.assert_allclose(fp,frame.filter_score,rtol=0,atol=1e-14)
        for idx,record in enumerate(frame.itertuples(index=False)):
            common=dict(scope=scope,date=record.date,code=record.code,name=getattr(record,'name',''),
                target_for_already_known_diagnostic=record.target)
            row=common|dict(prefix_same_input_score=float(pp[idx]),final_same_input_score=float(fp[idx]),same_input_score_delta=float(fp[idx]-pp[idx]),
                prefix_logit=float(plogit[idx]),final_logit=float(flogit[idx]),logit_delta=float(flogit[idx]-plogit[idx]),
                imputer_component_logit=0.,intercept_component_logit=fi-pi,
                scaler_symmetric_component_logit=float(scaler[idx].sum()),coefficient_symmetric_component_logit=float(coefficients[idx].sum()),
                fixed_existing_threshold=.3,prefix_at_or_below_existing_threshold=bool(pp[idx]<=.3),final_at_or_below_existing_threshold=bool(fp[idx]<=.3),
                score_delta_is_arithmetic_not_generalization_gain=True)
            np.testing.assert_allclose(row['logit_delta'],row['intercept_component_logit']+row['scaler_symmetric_component_logit']+row['coefficient_symmetric_component_logit'],atol=1e-14,rtol=0)
            rows.append(row)
            for j,field in enumerate(fields):
                feature_rows.append(common|dict(feature=field,raw_input=x[idx,j],prefix_standardized_input=pz[idx,j],final_standardized_input=fz[idx,j],
                    prefix_coefficient=pcoef[j],final_coefficient=fcoef[j],prefix_feature_logit_contribution=pcon[idx,j],final_feature_logit_contribution=fcon[idx,j],
                    feature_logit_delta=fcon[idx,j]-pcon[idx,j],scaler_symmetric_component_logit=scaler[idx,j],
                    coefficient_symmetric_component_logit=coefficients[idx,j],imputer_component_logit=0.))
            # Probability-scale attribution also uses a symmetric Shapley split
            # of the three changed component groups, not a claimed causal path.
            values={}
            for bits in itertools.product([0,1],repeat=3):
                z=fz[idx] if bits[0] else pz[idx]
                beta=fcoef if bits[1] else pcoef
                intercept=fi if bits[2] else pi
                values[bits]=float(expit(intercept+np.dot(beta,z)))
                coalitions.append(common|dict(final_scaler=bits[0],final_coefficients=bits[1],final_intercept=bits[2],
                    hybrid_same_input_score=values[bits],hybrid_is_algebraic_counterfactual_not_fitted_model=True))
            contributions=np.zeros(3)
            for order in itertools.permutations(range(3)):
                bits=[0,0,0]
                for component in order:
                    old_value=values[tuple(bits)];bits[component]=1
                    contributions[component]+=(values[tuple(bits)]-old_value)/6
            np.testing.assert_allclose(contributions.sum(),fp[idx]-pp[idx],atol=1e-14,rtol=0)
            component_rows.append(common|dict(imputer_probability_component=0.,
                scaler_probability_ShAP=contributions[0],coefficient_probability_ShAP=contributions[1],intercept_probability_ShAP=contributions[2],
                total_score_delta=fp[idx]-pp[idx],decomposition='All6 component update orders equally averaged; arithmetic only'))
    scores=pd.DataFrame(rows);features=pd.DataFrame(feature_rows);components=pd.DataFrame(component_rows)
    output={'saved_meta_parameters_and_preprocessing.csv':pd.DataFrame(coeff),'same_input_scores_and_exact_logit_components.csv':scores,
        'all_feature_logit_contributions_and_symmetric_changes.csv':features,'same_input_probability_component_ShAP_arithmetic.csv':components,
        'all8_component_hybrids_per_input_NOT_FITTED_MODELS.csv':pd.DataFrame(coalitions)}
    for name,frame in output.items():frame.to_csv(HERE/name,index=False,encoding='utf-8-sig')
    (HERE/'training_membership_and_intercept_state.json').write_text(json.dumps(states,ensure_ascii=False,indent=2),encoding='utf-8')
    sources=[Path(__file__),prefix_path,final_path,completed_path,five_path,oof_path,prefix_train_path,final_train_path,OLD/'protocol_BEFORE_ANY_FIT.json']
    receipt=dict(formed_at_UTC=datetime.now(timezone.utc).isoformat(),verification_fits=0,network_calls=0,
        scope='November saved Individual meta heads; only existing5 filtered rows and eligible86 third-block rows',
        no_PREFIX_outer_month_aggregate_effect_computed=True,no_threshold_selected_or_recommended=True,no_new_oct16_market_results_read=True,
        saved_model_hashes_match_original_receipts=True,imputer_changes_effect_zero_for_all91_complete_inputs=True,
        third_PREFIX_is_selection_data=True,third_FINAL_is_partly_in_sample=True,
        observed_changes_are_model_arithmetic_not_capital_flow_causation=True,
        components_identifiable='Exact fitted score/logit change, exact feature contribution, symmetric split accounting',
        not_identifiable='Unique causal blame among correlated features; isolated training-data vs effective regularization vs distribution shift; generalization effect of NO_META_REFIT',
        source_bindings={str(p.resolve()):sha(p) for p in sources},
        output_bindings={str((HERE/name).resolve()):sha(HERE/name) for name in output}|{str((HERE/'training_membership_and_intercept_state.json').resolve()):sha(HERE/'training_membership_and_intercept_state.json')})
    receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(scores[scores.scope.eq('ALREADY_FILTERED_FIVE_ONLY')][['name','prefix_same_input_score','final_same_input_score','imputer_component_logit','intercept_component_logit','scaler_symmetric_component_logit','coefficient_symmetric_component_logit']].to_string(index=False))

if __name__=='__main__':main()
