"""Independent dated-price, artifact and frozen-prediction checks (no fitting)."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import joblib,numpy as np,pandas as pd
import short_term_thinking_dataset_builder as b
import relay_multiday_scenarios as ev
OUT=Path(__file__).resolve().parent;MODEL=OUT/'model_optimization'
def main():
 own=b.read(OUT/'independent_full39_BEFORE_MODEL_AND_GEMINI.json');receipt=b.read(OUT/'independent_archive_receipt_BEFORE_MODEL.json')
 assert b.sha(OUT/'independent_full39_BEFORE_MODEL_AND_GEMINI.json')==receipt['sha256']
 assert len(own['rows'])==39 and len([r for r in own['rows'] if r['boards']>=2])==11
 b.verify(own['sources']);facts=b.read(OUT/'facts/qualified_current_facts_SCORES_MASKED.json');b.verify(facts['source_bindings'])
 dataset=joblib.load(MODEL/'qualified_dataset.joblib');full=dataset['full'];p=dataset['protocol'];result=b.read(MODEL/'fixed_research_result.json')
 checks=dataset['checks'];completed=list(MODEL.glob('*_fit_COMPLETED.json'));started=list(MODEL.glob('*_fit_STARTED.json'))
 assert len(completed)==len(started)==result['actual_new_research_fits']==47 and len(checks)==48
 assert len([c for c in checks if not c['qualified']])==1
 maximum_delta=0;prediction_rows=0
 for c in checks:
  prefix=MODEL/(c['month']+'_h'+str(c['horizon']))
  if not c['qualified']:
   assert not Path(str(prefix)+'_model.joblib').exists();continue
  assert c['max_training_target']<c['cutoff'] and c['full_signal_days']==c['known_label_days']==252
  model=joblib.load(str(prefix)+'_model.joblib');test=full[full.date.str[:7].eq(c['month'])].copy()
  saved=pd.read_csv(str(prefix)+'_predictions_BEFORE_VALIDATION.csv',dtype={'code':str});assert not any(col in saved for col in ['y','target','cumulative_up'])
  assert list(zip(saved.date,saved.code))==list(zip(test.date,test.code));again=model.predict_proba(test[p['fields']])[:,1]
  maximum_delta=max(maximum_delta,float(np.max(np.abs(again-saved.score.to_numpy()))));prediction_rows+=len(saved)
  assert np.array_equal(saved.predicted.to_numpy(),(again>.5).astype(int))
 assert maximum_delta<1e-12
 original=joblib.load(Path(next(iter(p['inputs']))));bars=original['bars'].set_index(['date','code']);sessions=dataset['sessions'];ix={d:i for i,d in enumerate(sessions)}
 actual_lookup=bars[['close','volume']].to_dict('index');hist=pd.read_csv(MODEL/'all_original_rows_WITH_UNKNOWN_and_references.csv',dtype={'code':str});assert len(hist)==11121 and not hist.duplicated(['date','code','horizon']).any()
 known=hist[hist.y.notna()];label_matches=0
 for r in known.itertuples():
  assert r.target_date==sessions[ix[r.date]+int(r.horizon)]
  a=actual_lookup[(r.date,r.code)];z=actual_lookup[(r.target_date,r.code)];assert a['volume']>0 and z['volume']>0
  assert bool(r.y)==(z['close']>a['close']);label_matches+=1
 recomputed=[]
 for f in result['fold_metrics']:
  start,end=p['folds'][f['fold']];x=hist[hist.horizon.eq(f['horizon'])&hist.date.between(start,end)];known=x[x.y.notna()]
  assert len(x)==f['direct']['full'] and len(known)==f['direct']['known']
  for col,key in [('predicted','direct'),('proxy_predicted','proxy_nextlimit_direction'),('always_up','always_up')]:
   hit=int(known[col].eq(known.y).sum());assert abs(hit/len(known)-f[key]['accuracy'])<1e-12
  recomputed.append(dict(horizon=f['horizon'],fold=f['fold'],known=len(known),unknown=int(x.y.isna().sum()),accuracy=f['direct']['accuracy']))
 # Every new event is still future; this cannot add mature samples.
 daily=b.read(OUT/'facts/normalized_daily.json');boundbars=pd.DataFrame([dict(code=c,date=d,**{k:r[k]/100 for k in ['open','high','low','close']},volume=r['volume_native']) for c,it in daily.items() for d,r in it['days'].items()])
 events=ev.evaluate(own,boundbars,sessions,b.now());assert events['new_mature_labels']==0 and len(events['rows'])==39*11
 b.write_new(OUT/'forward429_events_PENDING_at_creation.json',events)
 old=b.read(OUT/'BASE_inference_verification.json');b.verify(old['protected_hashes'])
 q=dict(formed_at=b.now(),checks_passed=True,independent39_unchanged=True,model11_complete=True,fits_verified=47,unqualified_windows_skipped=1,saved_model_prediction_rows_verified=prediction_rows,max_saved_score_delta=maximum_delta,
  historical_full_events=11121,known_labels_independently_recomputed=label_matches,metrics_recomputed=recomputed,new429_events_all_pending=True,
  formal_registry_and_ledger_unchanged=True,new_formal_forecasts=0,new_formal_maturity=0,actual_fills=0,protected_hashes=old['protected_hashes'],new_fits_in_verification=0,
  no_claimed_blind_or_live_gain=True,source_scope='qualifies arithmetic and binding, not exchange-certified completeness, fund causality, tradable fills or new real-forward accuracy')
 b.write_new(OUT/'verification_FINAL.json',q);print(json.dumps({k:v for k,v in q.items() if k not in ['protected_hashes','metrics_recomputed']},ensure_ascii=False))
if __name__=='__main__':main()
