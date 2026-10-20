"""Independent numerical and membership audit; no fit/prediction/result lookup."""
from pathlib import Path
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))
import model_upgrade_history as history


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def run():
    protocol_path = ROOT/'protocol_BEFORE_ANY_FIT.json'
    protocol = read(protocol_path)
    data_path = ROOT/'fixed_input_BEFORE_ANY_FIT.joblib'
    data = joblib.load(data_path)
    original_path = REPO/'research/model_upgrade_validation/ranking_optimization_20261015T173045Z/source_feature_training/prepared.joblib'
    original = joblib.load(original_path)
    full, fields = data['full'], data['fields']
    assert fields == original['fields'] and len(fields) == 92
    pd.testing.assert_frame_equal(full.drop(columns='frozen_original_nonword_identity'), original['full'], check_exact=True)
    identity_path = REPO/'research/model_upgrade_validation/short_term_thinking_20261015T071000Z/dataset.joblib'
    identity = joblib.load(identity_path)['features'][['date','code','daily_one_word']].set_index(['date','code'])
    refs_path = REPO/'research/model_upgrade_validation/ranking_optimization_20261015T173045Z/source_feature_training/same_source_route_alignment/complete26439_predictions_WITH_SAME_SOURCE_ROUTES.csv'
    refs = pd.read_csv(refs_path, dtype={'code':str}, low_memory=False)
    frozen = refs[refs.arm.eq('BASE')].set_index(['date','code'])
    assert len(frozen) == 3777 and not frozen.index.duplicated().any()
    for row in full.itertuples():
        key = (row.date,row.code)
        value = frozen.loc[key,'daily_one_word'] if key in frozen.index else identity.loc[key,'daily_one_word']
        assert pd.isna(value) and pd.isna(row.frozen_original_nonword_identity) or value == row.frozen_original_nonword_identity
    maps, blocks, sources = [], [], [protocol_path,data_path,original_path,identity_path,refs_path,Path(__file__)]
    for plan in data['plans']:
        month, cutoff = plan['month'], plan['cutoff']
        known = full[full.label_valid.fillna(False).astype(bool) & full.target.notna() & full.date.lt(cutoff) & full.target_date.lt(cutoff)].copy()
        days = sorted(known.date.unique())[-252:]
        window = known[known.date.isin(days)].copy()
        assert len(days) == 252 and history._frame_digest(window,fields) == plan['training_digest']
        expected_parts = []
        for block_number in (1,2,3):
            validation_days = days[block_number*63:(block_number+1)*63]
            first = validation_days[0]
            past = window[window.date.lt(first) & window.target_date.lt(first)].copy()
            valid = window[window.date.isin(validation_days)]
            assert not set(past.date).intersection(valid.date)
            assert past.target_date.max() < first and past.target.nunique() == 2
            key = ROOT/f'execution/{month}_inner{block_number}'
            started_path, completed_path = Path(str(key)+'_STARTED.json'),Path(str(key)+'_COMPLETED.json')
            model_path, validation_path = Path(str(key)+'_model.joblib'),Path(str(key)+'_all_validation_scores_BEFORE_EFFECTS.csv')
            started,completed = read(started_path),read(completed_path)
            saved = joblib.load(model_path)
            digest = history._frame_digest(past,fields)
            assert started['training_digest'] == saved['training_digest'] == digest
            assert saved['features'] == fields and saved['cutoff'] == started['validation_first'] == first
            assert started['max_train_target'] == past.target_date.max() < first
            assert completed['model_sha256'] == sha(model_path)
            scored = pd.read_csv(validation_path,dtype={'code':str},float_precision='round_trip')
            columns = ['date','code','target_date','target','label_valid','frozen_original_nonword_identity']
            pd.testing.assert_frame_equal(scored[columns].reset_index(drop=True),valid[columns].reset_index(drop=True),check_exact=True,check_dtype=False)
            assert scored.inner_block.eq(block_number).all()
            expected_parts.append(scored[scored.frozen_original_nonword_identity.eq(0)])
            blocks.append(dict(month=month,block=block_number,training_days=past.date.nunique(),training_rows=len(past),
                max_training_target=past.target_date.max(),validation_first=first,validation_days=len(validation_days),
                complete_validation_rows=len(valid),original_B_rows=len(expected_parts[-1]),training_digest=digest,
                strict_maturity=True,disjoint_signals=True,saved_training_digest_exact=True,validation_members_labels_identity_exact=True))
            sources.extend([started_path,completed_path,model_path,validation_path])
        expected_oof = pd.concat(expected_parts,ignore_index=True)
        oof_path = ROOT/f'execution/{month}_OOF_calibration_input_BEFORE_FIT.csv'
        oof = pd.read_csv(oof_path,dtype={'code':str},float_precision='round_trip')
        pd.testing.assert_frame_equal(oof.drop(columns='calibration_weight'),expected_oof,check_exact=True,check_dtype=False)
        assert not oof.duplicated(['date','code']).any() and oof.target_date.max() < cutoff
        assert oof.frozen_original_nonword_identity.eq(0).all() and oof.label_valid.all() and oof.target.notna().all()
        expected_weights = 1./oof.groupby('date').date.transform('size')
        np.testing.assert_allclose(oof.calibration_weight,expected_weights,atol=1e-15,rtol=0)
        per_day = oof.groupby('date').calibration_weight.sum()
        np.testing.assert_allclose(per_day,np.ones(len(per_day)),atol=1e-14,rtol=0)
        map_path = ROOT/f'execution/{month}_calibrator.joblib'
        map_started_path,month_completed_path = ROOT/f'execution/{month}_calibrator_STARTED.json',ROOT/f'execution/{month}_COMPLETED.json'
        map_record,month_completed = read(map_started_path),read(month_completed_path)
        saved_map = joblib.load(map_path)
        model = saved_map['calibrator']
        slope,intercept = float(model.coef_[0,0]),float(model.intercept_[0])
        assert model.coef_.shape == (1,1) and np.isfinite([slope,intercept]).all() and slope > 0
        assert int(model.n_iter_.max()) < 2000 and model.max_iter == 2000
        assert saved_map['features'] == fields and saved_map['plan'] == plan
        assert model.C == 1 and model.solver == 'lbfgs' and model.class_weight is None
        assert map_record['input_sha256'] == sha(oof_path) and month_completed['calibrator_sha256'] == sha(map_path)
        assert month_completed['final_classifier_refits'] == 0 and sha(plan['final_model_path']) == plan['final_model_sha256']
        maps.append(dict(month=month,coefficient=slope,intercept=intercept,n_iter=int(model.n_iter_.max()),max_iter=model.max_iter,
            positive_slope=True,converged_below_max_iter=True,OOF_rows=len(oof),OOF_days=len(per_day),
            OOF_day_mass_min=float(per_day.min()),OOF_day_mass_max=float(per_day.max()),OOF_weight_mass=float(oof.calibration_weight.sum()),
            exact_expected_OOF_members=True,original_B_only=True,no_class_balancing=True,no_row_renormalization=True,
            final_classifier_refit=False,final_saved_classifier_sha_unchanged=True))
        sources.extend([oof_path,map_path,map_started_path,month_completed_path,Path(plan['final_model_path'])])
    out = ROOT/'training_audit/executed_numeric_qualification'
    out.mkdir(exist_ok=False)
    pd.DataFrame(maps).to_csv(out/'all16_map_numerical_qualifications.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(blocks).to_csv(out/'all48_inner_training_members_maturity_verified.csv',index=False,encoding='utf-8-sig')
    receipt = dict(numerical_qualification_pass=True,independent_new_fits=0,prediction_metrics_recomputed=False,
        future_target16_results_read=False,complete_input_rows_unchanged=9315,fields_unchanged=92,
        original_historical_identity_and_BASE3777_override_exact=True,complete252_day_members_each_month_exact=True,
        inner_models_checked=48,strict_train_target_before_validation=True,validation_members_and_labels_exact=True,
        calibration_models_checked=16,all_positive_slopes=True,all_n_iter_below_2000=True,
        min_n_iter=min(x['n_iter'] for x in maps),max_n_iter=max(x['n_iter'] for x in maps),
        min_slope=min(x['coefficient'] for x in maps),max_slope=max(x['coefficient'] for x in maps),
        calibration_weight_each_day_mass1=True,no_calibration_class_balancing_or_row_renormalization=True,
        final_classifiers_reused_without_new_fit=16,final_saved_classifier_hashes_unchanged=True,
        fixed_predictive_screen_pass_not_asserted=True,
        source_bindings={str(p.resolve()):sha(p) for p in sources})
    (out/'independent_numeric_qualification_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_bindings'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    run()
