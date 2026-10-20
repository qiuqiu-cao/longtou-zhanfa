"""One read-only pass over saved capacity models; no fit or outer outcomes."""
from pathlib import Path
import sys,json,hashlib
from datetime import datetime,timezone
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from diagnostic_primitives import eligible_B_metrics,model_tree_structure,feature_profiles
import model_upgrade_history as history
import relay_temporal_score_calibration as temporal

OUT=Path(__file__).resolve().parent
STUDY=OUT.parent
EXEC=STUDY/'execution'
PREVIOUS=ROOT/'research/model_upgrade_validation/false_positive_20261016T030307Z/execution'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csv(name,frame):frame.to_csv(OUT/name,index=False,encoding='utf-8-sig')
def main():
    lock=OUT/'saved_model_diagnostics_STARTED.json'
    if lock.exists():raise FileExistsError('one read-only model pass already started')
    lock.write_text(json.dumps(dict(formed_at=datetime.now(timezone.utc).isoformat(),new_fits=0)),encoding='utf-8')
    source_paths=[STUDY/'protocol_BEFORE_ANY_FIT.json',STUDY/'fixed_input_BEFORE_ANY_FIT.joblib',EXEC/'all16_training_only_selections.csv',ROOT/'relay_temporal_score_calibration.py',ROOT/'model_upgrade_history.py',OUT/'diagnostic_primitives.py',Path(__file__)]
    bindings={str(p.resolve()):sha(p) for p in source_paths}
    data=joblib.load(STUDY/'fixed_input_BEFORE_ANY_FIT.joblib')
    # Every window is strictly before its month. Explicitly remove October rows
    # before accessing training outcomes; the Oct model still trains through Sep.
    full=data['full'][data['full'].date.le('2026-09-30')].copy();fields=data['fields']
    shape=data['shape'].set_index(['date','code']).verified_current_price_shape
    selections=pd.read_csv(EXEC/'all16_training_only_selections.csv')
    metrics=[];structures=[];trees=[];feature_uses=[];profiles=[];replays=[]
    loads=0
    for plan in data['plans']:
        month=plan['month'];window=history.training_rows(full,plan['cutoff'],'relay')
        assert window.date.nunique()==252 and window.target_date.max()<plan['cutoff']
        complete_oof={}
        for config in ['BASE','COMPACT','EXPANDED']:
            path=EXEC/f'{month}_{config}_complete_OOF_BEFORE_SELECTION.csv'
            complete_oof[config]=pd.read_csv(path,dtype={'code':str},float_precision='round_trip')
            bindings[str(path.resolve())]=sha(path)
        for block,(past,valid) in enumerate(temporal.temporal_splits(window),1):
            assert past.target_date.max()<valid.date.min()
            profile=feature_profiles(past,valid,fields)
            profile.insert(0,'block',block);profile.insert(0,'month',month);profiles.append(profile)
            for config in ['BASE','COMPACT','EXPANDED']:
                path=(PREVIOUS/f'{month}_inner{block}_model.joblib' if config=='BASE' else EXEC/f'{month}_inner{block}_{config}_model.joblib')
                bindings[str(path.resolve())]=sha(path);saved=joblib.load(path);loads+=1
                assert saved['features']==fields and saved['training_digest']==history._frame_digest(past,fields)
                model=saved['model']
                with threadpool_limits(limits=2):
                    score_train=model.predict_proba(past[fields])[:,list(model.classes_).index(1)]
                    score_valid=model.predict_proba(valid[fields])[:,list(model.classes_).index(1)]
                frozen=complete_oof[config]
                frozen=frozen[frozen.date.isin(valid.date.unique())].set_index(['date','code'])
                B=valid.frozen_original_nonword_identity.eq(0)
                expected=frozen.loc[pd.MultiIndex.from_frame(valid.loc[B,['date','code']])].raw_score.to_numpy()
                error=float(np.max(np.abs(score_valid[B.to_numpy()]-expected))) if B.any() else 0.
                assert error<1e-14
                replays.append(dict(month=month,block=block,config=config,original_B_validation_rows=int(B.sum()),max_raw_replay_error=error))
                for phase,frame,score in [('TRAIN',past,score_train),('OOF',valid,score_valid)]:
                    kval=shape.loc[pd.MultiIndex.from_frame(frame[['date','code']])].to_numpy()
                    stat=eligible_B_metrics(frame,score,frame.frozen_original_nonword_identity.to_numpy(),kval)
                    metrics.append(dict(month=month,block=block,config=config,phase=phase,
                      first_signal=frame.date.min(),last_signal=frame.date.max(),max_target=frame.target_date.max(),
                      complete_fit_rows=len(past),complete_fit_days=past.date.nunique(),**stat))
                summary,tree,use=model_tree_structure(saved,fields)
                tag=dict(month=month,block=block,config=config,phase='INNER')
                structures.append(tag|summary)
                for key,value in tag.items():tree[key]=value;use[key]=value
                trees.append(tree);feature_uses.append(use)
        # Only the selected final model is saved/loaded; do not invent the
        # unselected candidate's full-window performance or refit anything.
        selected=selections[selections.month.eq(month)].iloc[0]
        path=Path(selected.selected_final_model_path)
        assert sha(path)==selected.selected_final_model_sha256
        bindings[str(path.resolve())]=sha(path);saved=joblib.load(path);loads+=1
        assert saved['features']==fields and saved['training_digest']==history._frame_digest(window,fields)
        summary,tree,use=model_tree_structure(saved,fields)
        tag=dict(month=month,block=0,config=selected.selected_capacity,phase='SELECTED_FINAL')
        structures.append(tag|summary)
        for key,value in tag.items():tree[key]=value;use[key]=value
        trees.append(tree);feature_uses.append(use)
        model=saved['model']
        with threadpool_limits(limits=2):score=model.predict_proba(window[fields])[:,list(model.classes_).index(1)]
        kval=shape.loc[pd.MultiIndex.from_frame(window[['date','code']])].to_numpy()
        stat=eligible_B_metrics(window,score,window.frozen_original_nonword_identity.to_numpy(),kval)
        metrics.append(dict(month=month,block=0,config=selected.selected_capacity,phase='SELECTED_FINAL_TRAIN_ONLY',
          first_signal=window.date.min(),last_signal=window.date.max(),max_target=window.target_date.max(),
          complete_fit_rows=len(window),complete_fit_days=252,**stat))
    metric=pd.DataFrame(metrics);structure=pd.DataFrame(structures)
    csv('all_inner_train_OOF_and_selected_final_train_metrics.csv',metric)
    csv('all160_saved_model_structure_summaries.csv',structure)
    csv('all_saved_tree_structure_details.csv',pd.concat(trees,ignore_index=True))
    uses=pd.concat(feature_uses,ignore_index=True);csv('all160_models_92_feature_split_usage.csv',uses)
    profiles=pd.concat(profiles,ignore_index=True);csv('all48_inner_train_OOF_92_feature_coverage.csv',profiles)
    csv('all144_inner_saved_score_replays.csv',pd.DataFrame(replays))
    inner=metric[metric.phase.isin(['TRAIN','OOF'])]
    gaps=[]
    for (month,block,config),f in inner.groupby(['month','block','config']):
        pair=f.set_index('phase');a=pair.loc['TRAIN'];z=pair.loc['OOF']
        record=dict(month=month,block=block,config=config,train_B_rows=int(a.eligible_known_B_rows),oof_B_rows=int(z.eligible_known_B_rows))
        for field in ['raw_uncalibrated_logloss','precision','recall','F1','F0_5','positive_call_rate']:
            record['train_'+field]=a[field];record['oof_'+field]=z[field]
            record['OOF_minus_train_'+field]=z[field]-a[field]
        gaps.append(record)
    gap=pd.DataFrame(gaps);csv('all48_blocks_three_capacity_train_OOF_gaps.csv',gap)
    compare=[]
    for (month,block),f in gap.groupby(['month','block']):
        f=f.set_index('config');base=f.loc['BASE']
        for config in ['COMPACT','EXPANDED']:
            item=dict(month=month,block=block,config=config)
            for field in ['raw_uncalibrated_logloss','precision','recall','F1','F0_5']:
                for phase in ['train','oof']:
                    item[phase+'_'+field+'_minus_BASE']=f.loc[config,phase+'_'+field]-base[phase+'_'+field]
            compare.append(item)
    comparison=pd.DataFrame(compare);csv('all96_candidate_differences_from_matched_BASE.csv',comparison)
    # Means summarize 48 overlapping chronological blocks, not 48 independent
    # experiments, and are not used to choose any parameter.
    means=gap.groupby('config').mean(numeric_only=True).drop(columns='block').reset_index()
    csv('descriptive_overlapping48_block_means_NOT_PARAMETER_SELECTION.csv',means)
    matched=uses.merge(profiles,on=['month','block','field'],how='left',validate='many_to_one')
    csv('all_inner_feature_usage_with_train_OOF_coverage.csv',matched[matched.phase.eq('INNER')])
    assert loads==160 and len(replays)==144 and len(structure)==160 and len(metric)==304
    for p,h in bindings.items():assert sha(p)==h,p
    receipt=dict(formed_at=datetime.now(timezone.utc).isoformat(),new_fits=0,fetches=0,model_loads=loads,
      saved_inner_models=144,selected_final_models=16,all_inner_score_replays_pass=True,
      maximum_replay_error=max(x['max_raw_replay_error'] for x in replays),Brier_calculated=False,
      outer_results_read=False,October16_results_read=False,new_selection_or_threshold=False,
      metrics_scope='Original frozen B AND verified0 AND known target. Unknown/non-B reported separately, never negative',
      caveat='Train/OOF difference includes time change, not a causal overfit measure. Joint capacity comparison cannot isolate one parameter. Scores not validated probabilities.',
      source_bindings=bindings)
    (OUT/'saved_training_diagnostic_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(means[['config','train_raw_uncalibrated_logloss','oof_raw_uncalibrated_logloss','train_F0_5','oof_F0_5']].to_string(index=False))
    print(json.dumps(dict(model_loads=loads,new_fits=0,replayed=144),ensure_ascii=False))

if __name__=='__main__':main()
