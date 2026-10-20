"""One fixed, isolated source-feature challenge; unchanged windows reuse models.

Only causal_price_shape is filled from verified same/adjacent raw sources.
Late-source retrospective evidence never becomes original-time or role truth.
"""
from pathlib import Path
import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

import short_term_thinking_dataset_builder as b
import short_term_thinking_research as old_training
import model_upgrade_history as history
import expectation_345_research as original
import expectation_345_evaluation as metrics
import relay_source_qualification_repair as qualification
from short_term_thinking_preprocessing import EmptyTrainingColumnGuard

ROOT=b.ROOT/'research/model_upgrade_validation/ranking_optimization_20261015T173045Z/source_feature_training'
OLD=original.ROOT
REFERENCE=ROOT.parent/'source_repair/verified_22662_final'
AUDIT=ROOT.parent/'source_audit'
ARM='SOURCE_PATCHED_FEATURE_CLASSIFIER'
FIELD='causal_price_shape'
REF_ARMS=['BASE','ST_DUAL','ST_DUAL_LAGGED_PRICE_RELATION','CORRECTED_CAUSAL']
PERIODS=history.FOLDS+[dict(id='2026Oct',start='2026-10-01',end='2026-10-31')]


def patched_feature_copy(full,fields,verified,expected_changes=47):
    if len(set(fields))!=len(fields) or FIELD not in fields:
        raise ValueError('fixed feature schema required')
    if full.duplicated(['date','code']).any() or verified.duplicated(['date','code']).any():
        raise ValueError('duplicate dated source identity')
    result=full.copy();lookup=result.set_index(['date','code'])
    changes=[]
    for row in verified[verified.verified_current_price_shape.notna()].itertuples():
        if row.verified_current_price_shape not in [0,1] or (row.date,row.code) not in lookup.index:
            raise ValueError('verified shape/key outside fixed input')
        old=lookup.loc[(row.date,row.code),FIELD]
        if pd.notna(old):raise ValueError('refuse nonmissing feature overwrite')
        mask=result.date.eq(row.date)&result.code.eq(row.code)
        result.loc[mask,FIELD]=float(row.verified_current_price_shape)
        changes.append(dict(date=row.date,code=row.code,feature=FIELD,old_value=None,new_value=float(row.verified_current_price_shape),late_source_research_only=True,original_PIT_eligible=False))
    if len(changes)!=expected_changes:raise ValueError('unexpected feature change count')
    pd.testing.assert_frame_equal(full.drop(columns=FIELD),result.drop(columns=FIELD),check_exact=True)
    if sum((~(full[FIELD].eq(result[FIELD])|(full[FIELD].isna()&result[FIELD].isna()))))!=expected_changes:
        raise ValueError('unexpected feature mutation')
    return result,pd.DataFrame(changes)


def choose_reuse(old_digest,new_digest):
    return old_digest==new_digest


def verified_shapes(full,verified):
    output=full[['date','code','current_price_shape']].rename(columns={'current_price_shape':'verified_current_price_shape'}).copy()
    lookup=verified.set_index(['date','code']).verified_current_price_shape
    for i,row in output.iterrows():
        if (row.date,row.code) in lookup.index and pd.notna(lookup.loc[(row.date,row.code)]):
            if pd.notna(row.verified_current_price_shape):raise ValueError('nonmissing route overwrite')
            output.loc[i,'verified_current_price_shape']=lookup.loc[(row.date,row.code)]
    return output


def route_predictions(test,raw_score,shape,base):
    result=test[['date','code','name','boards','target_date','target','label_valid','daily_one_word','current_price_shape','review_cycle_state','current_price_path_status']].copy()
    result['ranking_score']=raw_score;result['b_score']=raw_score;result['score']=raw_score
    result['call']=np.where(np.asarray(raw_score)>.5,'PROMOTE','FAIL')
    result=result.merge(shape,on=['date','code'],validate='one_to_one')
    for i,row in result.iterrows():
        value=row.verified_current_price_shape
        if pd.isna(value):result.loc[i,['score','call']]=[np.nan,'UNKNOWN']
        elif value==1:
            value=base.loc[(row.date,row.code)]
            result.loc[i,['score','call']]=[value,'PROMOTE' if value>.5 else 'FAIL']
    result['arm']=ARM;result['evaluation_call']=result.call
    result['qualification_late_research_only']=True;result['original_PIT_eligible']=False
    return result


def full_slot_ranking(frame,k=3):
    group=frame[frame.daily_one_word.eq(0)].sort_values(['date','ranking_score','code'],ascending=[True,False,True],na_position='last')
    chosen=group.groupby('date',sort=True).head(k)
    known=chosen.label_valid.fillna(False).astype(bool)&chosen.target.notna()
    hits=int((known&chosen.target.eq(1)).sum());slots=k*frame.date.nunique()
    return dict(k=k,signal_days=frame.date.nunique(),full_slots=slots,selected=len(chosen),empty_slots=slots-len(chosen),
        known_selected=int(known.sum()),unknown_selected=int((~known).sum()),hits=hits,
        full_slot_hit_rate=hits/slots if slots else None,unknown_direction_selected=int(chosen.evaluation_call.eq('UNKNOWN').sum())),chosen


def prepare():
    if (ROOT/'prepared.joblib').exists():raise FileExistsError('new source-feature input already prepared')
    ROOT.mkdir(parents=True,exist_ok=True)
    qualified,proof=qualification.load_qualified_rows(AUDIT)
    source=OLD/'trainable_corrected_dataset_QUALIFIED.joblib';data=joblib.load(source)
    full=data['full'];fields=data['fields']
    if len(full)!=9315 or len(fields)!=92:raise ValueError('complete9315/92 fixed source expected')
    patched,changes=patched_feature_copy(full,fields,qualified)
    shape=verified_shapes(full,qualified)
    reference_path=REFERENCE/'all_predictions_with_isolated_qualification.csv'
    ref_receipt=b.read(REFERENCE/'source_qualification_repair_receipt.json');b.verify(ref_receipt['output_bindings'])
    refs=pd.read_csv(reference_path,dtype={'code':str},low_memory=False)
    refs=refs[refs.arm.isin(REF_ARMS)].copy()
    for arm,frame in refs.groupby('arm'):
        if len(frame)!=3777:raise ValueError('complete3777 reference required')
    # Preserve all original reference columns; event metrics use the source-
    # qualified research attachment, never overwrite original frozen call.
    refs['evaluation_call']=refs.recovered_B_call
    refs.to_csv(ROOT/'frozen_source_qualified_references.csv',index=False,encoding='utf-8-sig')
    original_protocol=b.read(OLD/'protocol_BEFORE_ANY_FIT.json')
    checks=[];models={}
    for check in original_protocol['training_checks']:
        train=history.training_rows(full,check['cutoff'],'relay')
        fixed=history.training_rows(patched,check['cutoff'],'relay')
        pd.testing.assert_frame_equal(train.drop(columns=FIELD),fixed.drop(columns=FIELD),check_exact=True)
        old_digest=history._frame_digest(train,fields);new_digest=history._frame_digest(fixed,fields)
        if old_digest!=check['training_digest'] or train.date.nunique()!=252 or train.target_date.max()>=check['cutoff']:
            raise ValueError('old training window or maturity drift')
        model_path=OLD/'fixed_comparison'/f"{check['month']}_CORRECTED_CAUSAL_model.joblib"
        saved=joblib.load(model_path)
        if saved['training_digest']!=old_digest or saved['features']!=fields or saved['estimator']!=old_training.PARAMS or saved['negative_cost']!=1:
            raise ValueError('old saved model/schema differs')
        models[str(model_path.resolve())]=b.sha(model_path)
        diff=~(train[FIELD].eq(fixed[FIELD])|(train[FIELD].isna()&fixed[FIELD].isna()))
        checks.append(dict(month=check['month'],cutoff=check['cutoff'],training_rows=len(train),training_days=252,
            max_mature_target=train.target_date.max(),old_digest=old_digest,new_digest=new_digest,
            changed_training_cells=int(diff.sum()),reuse_original_model=choose_reuse(old_digest,new_digest),
            original_model_path=str(model_path.resolve()),original_model_sha256=b.sha(model_path)))
    changes.to_csv(ROOT/'all47_single_field_changes_BEFORE_ANY_FIT.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(checks).to_csv(ROOT/'all16_training_window_plans_BEFORE_ANY_FIT.csv',index=False,encoding='utf-8-sig')
    joblib.dump(dict(full=patched,fields=fields,verified_shapes=shape,sessions=data['sessions'],training_checks=checks),ROOT/'prepared.joblib',compress=3)
    bindings=proof['source_bindings']|models|{str(path.resolve()):b.sha(path) for path in [source,reference_path,REFERENCE/'source_qualification_repair_receipt.json',ROOT/'prepared.joblib',ROOT/'frozen_source_qualified_references.csv',ROOT/'all47_single_field_changes_BEFORE_ANY_FIT.csv',ROOT/'all16_training_window_plans_BEFORE_ANY_FIT.csv',Path(__file__),Path(qualification.__file__),Path(original.__file__),Path(metrics.__file__),Path(history.__file__),Path(old_training.__file__),b.ROOT/'short_term_thinking_preprocessing.py']}
    b.write_new(ROOT/'preparation_BEFORE_ANY_FIT.json',dict(formed_at=b.now(),changed_feature=FIELD,changed_cells=47,other91_features_unchanged=True,
        all_metadata_members_labels_unchanged=True,planned_new_fits=sum(not x['reuse_original_model'] for x in checks),planned_reuses=sum(x['reuse_original_model'] for x in checks),
        source_bindings=bindings,new_fits=0,original_models_unchanged=True))
    return dict(changes=47,planned_new_fits=sum(not x['reuse_original_model'] for x in checks),planned_reuses=sum(x['reuse_original_model'] for x in checks))


def freeze(tests_xml):
    if (ROOT/'protocol_BEFORE_ANY_FIT.json').exists():raise FileExistsError('one fixed challenge already frozen')
    prep=b.read(ROOT/'preparation_BEFORE_ANY_FIT.json');b.verify(prep['source_bindings'])
    import xml.etree.ElementTree as ET
    tree=ET.parse(tests_xml);suites=tree.getroot().findall('.//testsuite')
    if not suites or any(int(x.get('errors',0))+int(x.get('failures',0)) for x in suites):raise ValueError('passing preflight tests required')
    data=joblib.load(ROOT/'prepared.joblib')
    bindings=prep['source_bindings']|{str(Path(tests_xml).resolve()):b.sha(tests_xml),str((b.ROOT/'tests/test_relay_source_feature_research.py').resolve()):b.sha(b.ROOT/'tests/test_relay_source_feature_research.py')}
    b.write_new(ROOT/'protocol_BEFORE_ANY_FIT.json',dict(formed_at=b.now(),arm=ARM,
        hypothesis='Previously verified same/adjacent raw sources resolve47 missing causal_price_shape cells; changing only this feature may improve an otherwise identical classifier',
        features=data['fields'],only_changed_field=FIELD,changed_cells=47,training_checks=data['training_checks'],estimator=old_training.PARAMS,
        training='Original full252 mature signal-day windows, same members/labels/inverse-day and class-balanced cost1 weights',threshold=.5,strict_greater_than=True,
        reuse='Identical complete training digest: load exact original CORRECTED_CAUSAL saved model, no fit. Changed digest: one new fit per month',
        fit_max=16,fit_budget_seconds=1800,search=False,repeat_failed_old_trials=False,
        comparator='Same verified source qualification, old92 features and old weights: source_repair/verified_22662_final CORRECTED_CAUSAL; BASE/ST_DUAL/LAGGED retained',
        qualification='Verified current price shape only; original continuous-path/parent/role UNKNOWN and metadata never expanded',
        acceptance=dict(pooled_historical_nonword_F1_gain_min=.01,each_three_fold_Recall_loss_max=.025,each_three_fold_Precision_loss_max=.025,pooled_Top3_full_slot_rate_not_lower=True),
        top3='Original frozen nonword membership, descending raw ranking_score/code ascending ties; 3 slots for EVERY complete signal date including unknown/empty slots, unknown outcomes never hits',
        hist_periods=history.FOLDS,October='Supplemental, current15 six target16 remain unknown, not part of historical acceptance',
        late_source_research_only=True,strict_original_PIT=False,previously_seen_historical_data=True,probability_calibrated=False,production_authorized=False,
        source_bindings=bindings))
    return dict(frozen=True,new_fits=0)


def evaluate(pred):
    records=[];rankings=[];selected=[]
    for period in PERIODS+[dict(id='HISTORICAL_POOLED',start='2025-07-01',end='2026-09-30')]:
        p=pred[pred.date.between(period['start'],period['end'])]
        for arm,frame in p.groupby('arm',sort=True):
            for scope,part in [('FULL_POOL',frame),('ORIGINAL_NONWORD',frame[frame.daily_one_word.eq(0)])]:
                copy=part.copy();copy['call']=copy.evaluation_call
                records.append(dict(period=period['id'],arm=arm,scope=scope,**metrics.classification(copy)))
            for k in [1,2,3]:
                stat,picks=full_slot_ranking(frame,k)
                rankings.append(dict(period=period['id'],arm=arm,**stat))
                picks=picks.copy();picks['period']=period['id'];picks['k']=k;selected.append(picks)
    records=pd.DataFrame(records);rankings=pd.DataFrame(rankings)
    table=records[records.scope.eq('ORIGINAL_NONWORD')].set_index(['period','arm'])
    new=table.loc[('HISTORICAL_POOLED',ARM)];ref=table.loc[('HISTORICAL_POOLED','CORRECTED_CAUSAL')]
    guards={'pooled_F1_gain_at_least1pp':bool(new.F1>=ref.F1+.01)}
    for fold in history.FOLDS:
        a=table.loc[(fold['id'],ARM)];z=table.loc[(fold['id'],'CORRECTED_CAUSAL')]
        for key in ['Recall','Precision']:guards[fold['id']+'_'+key+'_loss_within2p5pp']=bool(a[key]>=z[key]-.025) if pd.notna(a[key]) and pd.notna(z[key]) else False
    ranks=rankings[rankings.period.eq('HISTORICAL_POOLED')&rankings.k.eq(3)].set_index('arm')
    guards['pooled_Top3_full_slot_not_lower']=bool(ranks.loc[ARM,'full_slot_hit_rate']>=ranks.loc['CORRECTED_CAUSAL','full_slot_hit_rate'])
    return dict(formed_at=b.now(),guards=guards,screen_pass=all(guards.values()),production_deployed=False,not_blind=True,
        qualifier='Compared with equally source-qualified old model, so qualification gain is not counted twice'),records,rankings,pd.concat(selected,ignore_index=True)


def run():
    protocol=b.read(ROOT/'protocol_BEFORE_ANY_FIT.json');b.verify(protocol['source_bindings'])
    if protocol['estimator']!=old_training.PARAMS:raise ValueError('fixed HGB parameters drift')
    b.write_new(ROOT/'experiment_STARTED.json',dict(formed_at=b.now(),protocol_sha256=b.sha(ROOT/'protocol_BEFORE_ANY_FIT.json')))
    output=ROOT/'execution';output.mkdir(exist_ok=False)
    data=joblib.load(ROOT/'prepared.joblib');full=data['full'];fields=data['fields']
    refs=pd.read_csv(ROOT/'frozen_source_qualified_references.csv',dtype={'code':str},low_memory=False)
    base=refs[refs.arm.eq('BASE')].set_index(['date','code']).score
    source_identity=refs[refs.arm.eq('BASE')].set_index(['date','code']).daily_one_word
    shape=data['verified_shapes'];parts=[];fits=0;reuses=0;start=time.monotonic()
    for check in protocol['training_checks']:
        if time.monotonic()-start>protocol['fit_budget_seconds']:raise TimeoutError('fixed1800sec budget')
        train=history.training_rows(full,check['cutoff'],'relay')
        if history._frame_digest(train,fields)!=check['new_digest']:raise ValueError('new training digest drift')
        prefix=output/check['month'];test=full[full.date.str[:7].eq(check['month'])].copy()
        if choose_reuse(check['old_digest'],check['new_digest'])!=check['reuse_original_model']:raise ValueError('reuse decision drift')
        b.write_new(str(prefix)+'_STARTED.json',dict(formed_at=b.now(),plan=check,attempt=1,new_fit=not check['reuse_original_model']))
        try:
            if check['reuse_original_model']:
                if b.sha(check['original_model_path'])!=check['original_model_sha256']:raise ValueError('original model changed')
                saved=joblib.load(check['original_model_path']);model=saved['model'];model_path=Path(check['original_model_path']);reuses+=1
            else:
                model=Pipeline([('empty_training_columns',EmptyTrainingColumnGuard()),('hgb',HistGradientBoostingClassifier(**protocol['estimator']))])
                with threadpool_limits(limits=2):model.fit(train[fields],train.target.astype(int),hgb__sample_weight=original.weights(train,1.))
                fits+=1;model_path=Path(str(prefix)+'_model.joblib')
                joblib.dump(dict(model=model,features=fields,cutoff=check['cutoff'],training_digest=check['new_digest'],negative_cost=1.,estimator=protocol['estimator'],arm=ARM,trained_at=b.now()),model_path)
            with threadpool_limits(limits=2):score=model.predict_proba(test[fields])[:,list(model.classes_).index(1)]
            result=route_predictions(test,score,shape,base)
            result['engine_metadata_daily_one_word']=result.daily_one_word
            result['daily_one_word']=[source_identity.loc[(r.date,r.code)] for r in result.itertuples()]
            result['frozen_original_nonword_identity']=result.daily_one_word;result['model_cutoff']=check['cutoff']
            # Save every complete prediction before computing any effect.
            result.to_csv(str(prefix)+'_predictions_BEFORE_EVALUATION.csv',index=False,encoding='utf-8-sig');parts.append(result)
            b.write_new(str(prefix)+'_COMPLETED.json',dict(formed_at=b.now(),month=check['month'],new_fit=not check['reuse_original_model'],model_path=str(model_path.resolve()),model_sha256=b.sha(model_path),
                training_digest=check['new_digest'],prediction_sha256=b.sha(str(prefix)+'_predictions_BEFORE_EVALUATION.csv'),rows=len(result)))
            print(check['month'],'REUSE' if check['reuse_original_model'] else 'NEW_FIT',len(result),flush=True)
        except Exception as exc:
            b.write_new(ROOT/'FAILURE_STOP.json',dict(formed_at=b.now(),error=str(exc),type=type(exc).__name__,new_fits=fits,reuses=reuses,retry_authorized=False));raise
    challenger=pd.concat(parts,ignore_index=True)
    if len(challenger)!=3777 or challenger.duplicated(['date','code']).any():raise ValueError('complete3777 challenger required')
    pred=pd.concat([refs,challenger],ignore_index=True)
    pred.to_csv(output/'complete18885_predictions_BEFORE_EVALUATION.csv',index=False,encoding='utf-8-sig')
    result,stats,ranks,picks=evaluate(pred)
    stats.to_csv(output/'same_pool_classification_metrics.csv',index=False,encoding='utf-8-sig')
    ranks.to_csv(output/'complete_full_slot_Top1_Top2_Top3.csv',index=False,encoding='utf-8-sig')
    picks.to_csv(output/'all_selected_rows_with_unknown.csv',index=False,encoding='utf-8-sig')
    current=challenger[challenger.date.eq('2026-10-15')]
    if len(current)!=6 or current.target.notna().any() or current.label_valid.any():raise ValueError('future6 maturity drift')
    current.to_csv(output/'current15_six_target16_UNMATURED.csv',index=False,encoding='utf-8-sig')
    result.update(new_fits=fits,windows_reused=reuses,fit_total=fits+reuses,elapsed_seconds=time.monotonic()-start,
        original_model_refits=0,source_bindings=protocol['source_bindings']|{str(p.resolve()):b.sha(p) for p in output.iterdir() if p.is_file()})
    b.verify(result['source_bindings']);b.write_new(output/'fixed_source_feature_result.json',result)
    return dict(new_fits=fits,windows_reused=reuses,screen_pass=result['screen_pass'])


def verify():
    p=b.read(ROOT/'protocol_BEFORE_ANY_FIT.json');b.verify(p['source_bindings'])
    output=ROOT/'execution';data=joblib.load(ROOT/'prepared.joblib');full=data['full'];fields=data['fields']
    records=[]
    for check in p['training_checks']:
        receipt=b.read(output/(check['month']+'_COMPLETED.json'))
        if b.sha(receipt['model_path'])!=receipt['model_sha256']:raise ValueError('saved model hash changed')
        model=joblib.load(receipt['model_path'])['model'];test=full[full.date.str[:7].eq(check['month'])]
        with threadpool_limits(limits=2):raw=model.predict_proba(test[fields])[:,list(model.classes_).index(1)]
        saved=pd.read_csv(output/(check['month']+'_predictions_BEFORE_EVALUATION.csv'),dtype={'code':str})
        if not np.allclose(raw,saved.b_score,atol=1e-15,rtol=0):raise ValueError('prediction replay differs')
        if b.sha(check['original_model_path'])!=check['original_model_sha256']:raise ValueError('old model mutated')
        records.append(dict(month=check['month'],saved_model_replay=True,max_absolute_difference=float(np.max(abs(raw-saved.b_score))),original_model_hash_unchanged=True,reused=check['reuse_original_model']))
    b.write_new(output/'all16_saved_model_replay_verification.json',dict(formed_at=b.now(),models=records,new_fits=0,all16_original_models_unchanged=True,
        source_bindings={str(path.resolve()):b.sha(path) for path in [Path(__file__),ROOT/'protocol_BEFORE_ANY_FIT.json',output/'fixed_source_feature_result.json']}))
    return dict(replayed_models=16,new_fits=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','freeze','run','verify']);parser.add_argument('--tests-xml')
    args=parser.parse_args();answer=freeze(args.tests_xml) if args.command=='freeze' else globals()[args.command]();print(json.dumps(answer,ensure_ascii=True))
