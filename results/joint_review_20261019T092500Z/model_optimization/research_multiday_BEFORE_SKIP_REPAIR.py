"""Fixed direct-horizon research: targets differ from next-close limit up.

No production fit, weight change, deployment, or calibrated probability claim.
All original rows, unknown labels, month cutoffs and failed trials are kept.
"""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,sys,warnings
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import joblib,numpy as np,pandas as pd
from sklearn.exceptions import ConvergenceWarning
from rolling_forecast_challenger_v274 import make
import short_term_thinking_dataset_builder as b
import relay_context_research as common
OUT=Path(__file__).resolve().parent/'model_optimization'
SOURCE=ROOT/'research/model_upgrade_validation/historical_symbolic_context_20261015T081200Z/latest15_prepared_v2/engine_context_WITH_CURRENT15.joblib'
ROLL=ROOT/'research/model_upgrade_validation/recency_walkforward_20261013T134351Z'
HORIZONS=[2,3,5];MONTHS=[str(m) for m in pd.period_range('2025-07','2026-10',freq='M')]
FOLDS={'2025H2':('2025-07-01','2025-12-31'),'2026JanAug':('2026-01-01','2026-08-31'),'2026Sep':('2026-09-01','2026-09-30')}
def now():return datetime.now(timezone.utc).isoformat()
def clean(x):return b.clean(x)
def write(p,x):b.write_new(p,x)
def digest(frame,cols):return hashlib.sha256(frame[cols].sort_values(['date','code']).to_csv(index=False).encode()).hexdigest()
def qualify():
 OUT.mkdir(exist_ok=True)
 protocol=dict(version='DIRECT_CUMULATIVE_TARGETS_ONE_FIXED_TRY_v1',formed_at=now(),user_request='总结经验，试试如何优化模型，提高准确率',
  hypothesis='Separate cumulative cash-close directions at h2/h3/h5 from next-session limit-up classification. Direct targets may improve event-matched accuracy; no change in next-day event accuracy is implied.',
  new_evidence='New official-horizon cumulative-close labels built from existing bound unadjusted daily bars; no new industry truth, no repeated old feature/threshold/model-family challenge.',
  horizons=HORIZONS,events={str(h):'C[T+h]>C[T], exact official target session, unadjusted cash close; not daily-up, limit-up, net return or total return' for h in HORIZONS},
  fields=common.RELAY_FIELDS,months=MONTHS,folds=FOLDS,max_fits=48,one_attempt_per_horizon_month=True,
  estimator='Original make(logistic): median+missing indicators+scaler; C=.25 max_iter2500 random_state271 no class balance; strict score>.5',
  train='Last252 original full signal days whose T+h target strictly precedes month start; preserve full-day members, fit only source-qualified endpoint labels; inverse known rows/date, total weight=252.',
  validation='Frozen complete original monthly relay pools; no deletion for unknown outcomes. Evaluation labels stay outside fitted features. Each month model trained once; save predictions before validation.',
  references=['frozen monthly ROLL252 next-limit decision and ranking used as explicitly mismatched direction proxy','constant UP event baseline','original board-height ranking'],
  acceptance=dict(each_fold_accuracy_gain_over_best_proxy_and_alwaysup_min=.03,each_fold_balanced_accuracy_and_F1_not_below_proxy=True,pooled_Top3_hit_fraction_not_below_proxy=True,
   pooled_accuracy_difference_lower_interval_above_zero_vs_each_reference=True,bootstrap_signal_day_block=5,draws=2000,three_horizon_adjusted_quantiles=[.05/6,1-.05/6]),
  scope='Retrospective previously studied and late-acquired history, current-stock survivor bias retained; not blind or live. No next-day accuracy upgrade; no deployment even if preliminary screen passes.',
  inputs={str(SOURCE.resolve()):b.sha(SOURCE),str((ROOT/'data/v298/exchange_calendar_extension_20261003_v2/calendar.json').resolve()):b.sha(ROOT/'data/v298/exchange_calendar_extension_20261003_v2/calendar.json'),str(Path(__file__).resolve()):b.sha(__file__)},
  old_failures_retried=False,production_fits=0,current20_outcomes_read=False,calibrated_probabilities=False)
 write(OUT/'protocol_BEFORE_TARGET_BUILD_AND_FITS.json',protocol)
 engine=joblib.load(SOURCE);full=engine['prepared']['relay'].copy();full['code']=full.code.astype(str).str.zfill(6)
 if full.duplicated(['date','code']).any():raise ValueError('duplicate original cohort')
 sessions=b.read(ROOT/'data/v298/exchange_calendar_extension_20261003_v2/calendar.json')['sessions'];idx={d:i for i,d in enumerate(sessions)}
 bars=engine['bars'];codes=set(full.code);bars=bars[bars.code.isin(codes)&bars.date.le('2026-10-15')].copy()
 if bars.duplicated(['date','code']).any():raise ValueError('source contains duplicated date/code')
 for f in ['open','high','low','close','volume']:bars[f]=pd.to_numeric(bars[f],errors='coerce')
 valid=np.isfinite(bars[['open','high','low','close','volume']]).all(axis=1)&bars[['open','high','low','close','volume']].gt(0).all(axis=1)&bars.low.le(bars[['open','close']].min(axis=1))&bars.high.ge(bars[['open','close']].max(axis=1))
 bars['bar_valid']=valid
 base=bars[['date','code','close','bar_valid']].rename(columns={'close':'signal_price_observed','bar_valid':'signal_bar_valid'})
 full=full.merge(base,on=['date','code'],how='left',validate='one_to_one');full['signal_price_agreed']=full.signal_price_observed.round(2).eq(pd.to_numeric(full.price,errors='coerce').round(2))&full.signal_bar_valid.eq(True)
 qualifications=[]
 for h in HORIZONS:
  dates={d:sessions[idx[d]+h] if idx[d]+h<len(sessions) else None for d in full.date.unique()}
  full['target_h'+str(h)]=full.date.map(dates)
  end=bars[['date','code','close','bar_valid']].rename(columns={'date':'target_h'+str(h),'close':'close_h'+str(h),'bar_valid':'valid_h'+str(h)})
  full=full.merge(end,on=['target_h'+str(h),'code'],how='left',validate='many_to_one')
  good=full.signal_price_agreed&full['valid_h'+str(h)].eq(True)
  full['y_h'+str(h)]=np.where(good,(full['close_h'+str(h)]>full.signal_price_observed).astype(int),np.nan)
  for name,(start,finish) in FOLDS.items():
   scope=full[full.date.between(start,finish)];qualifications.append(dict(horizon=h,fold=name,full_rows=len(scope),qualified_endpoint_labels=int(scope['y_h'+str(h)].notna().sum()),unknown=int(scope['y_h'+str(h)].isna().sum()),unknown_kept=True))
 # Industry is not manufactured from name diffusion, current pages or returns.
 industry=dict(full_rows=len(full),known_nonempty_industry=int(full.industry.fillna('').astype(str).str.strip().ne('').sum()),known_nonempty_theme=int(full.theme.fillna('').astype(str).str.strip().ne('').sum()),gemini_anchor_sector_hypothesis='NOT_FIT_MISSING_HISTORICAL_ISSUER_TOPIC_AND_SECTOR_MAPPING',old_name_and_price_relation_trials='ALREADY_FAILED_NOT_RERUN',future_parent_amount_prohibited=True)
 write(OUT/'source_qualification_BEFORE_FITS.json',dict(formed_at=now(),rows=len(full),endpoint_coverage=qualifications,industry_hypothesis=industry,source_bundle_hash=b.sha(SOURCE),no_performance_viewed=True))
 checks=[]
 for h in HORIZONS:
  for month in MONTHS:
   cutoff=month+'-01';eligible=sorted(full.loc[full['target_h'+str(h)].lt(cutoff),'date'].unique())[-252:]
   train=full[full.date.isin(eligible)&full['y_h'+str(h)].notna()].copy()
   if len(eligible)!=252 or train.date.nunique()!=252 or train['target_h'+str(h)].ge(cutoff).any() or train['y_h'+str(h)].nunique()!=2:raise ValueError('full mature252 days/class support missing')
   checks.append(dict(horizon=h,month=month,cutoff=cutoff,full_signal_days=len(eligible),training_rows=len(train),max_training_target=train['target_h'+str(h)].max(),training_digest=digest(train,['date','code']+common.RELAY_FIELDS+['target_h'+str(h),'y_h'+str(h)])))
 write(OUT/'training_windows_BEFORE_FITS.json',dict(formed_at=now(),checks=checks,all_maturity_checks_passed=True))
 joblib.dump(dict(full=full,sessions=sessions,checks=checks,protocol=protocol),OUT/'qualified_dataset.joblib',compress=3)
 print(json.dumps(dict(rows=len(full),coverage=qualifications,sector_qualification=industry,windows=len(checks),fits=0),ensure_ascii=False),flush=True)
def metric(x,pred='predicted'):
 k=x[x.y.notna()].copy();truth=k.y.eq(1);pr=k[pred].astype(bool)
 tp=int((truth&pr).sum());tn=int((~truth&~pr).sum());fp=int((~truth&pr).sum());fn=int((truth&~pr).sum())
 return dict(full=len(x),known=len(k),unknown=len(x)-len(k),TP=tp,FP=fp,FN=fn,TN=tn,accuracy=(tp+tn)/len(k) if len(k) else None,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,balanced_accuracy=((tp/(tp+fn))+(tn/(tn+fp)))/2 if tp+fn and tn+fp else None)
def top(x,score):
 s=x.sort_values(['date',score,'code'],ascending=[True,False,True]).groupby('date',sort=False).head(3)
 return dict(slots=len(s),known=int(s.y.notna().sum()),unknown=int(s.y.isna().sum()),hits=int(s.y.eq(1).sum()),hit_fraction_known=float(s.y.mean()))
def interval(x,ref):
 q=x[x.y.notna()].copy();q['diff']=(q.predicted.eq(q.y)-q[ref].eq(q.y).astype(int)).astype(float)
 # Explicit integer comparison avoids bool subtraction in newer numpy/pandas.
 q['diff']=q.predicted.eq(q.y).astype(int)-q[ref].eq(q.y).astype(int)
 day=q.groupby('date').agg(total=('diff','sum'),n=('diff','size')).reset_index();blocks=[day.iloc[i:i+5][['total','n']].sum().to_numpy(float) for i in range(0,len(day),5)];arr=np.asarray(blocks)
 rng=np.random.default_rng(20261019);samples=rng.integers(0,len(arr),size=(2000,len(arr)));s=arr[samples].sum(axis=1);values=s[:,0]/s[:,1]
 return dict(difference=float(q['diff'].mean()),lower=float(np.quantile(values,.05/6)),upper=float(np.quantile(values,1-.05/6)),draws=2000,three_horizon_adjusted=True)
def run():
 p=b.read(OUT/'protocol_BEFORE_TARGET_BUILD_AND_FITS.json');b.verify(p['inputs']);data=joblib.load(OUT/'qualified_dataset.joblib');full=data['full'];checks=data['checks']
 write(OUT/'ONE_ATTEMPT_LOCK.json',dict(started_at=now(),maximum_fits=48,do_not_repeat=True))
 original=[]
 for month in MONTHS[:-1]:
  path=ROLL/(month+'_ROLL252_predictions_BEFORE_VALIDATION.csv');r=pd.read_csv(path,dtype={'code':str});r['code']=r.code.str.zfill(6)
  original.append(r[['date','code','predicted','raw_score']].rename(columns={'predicted':'proxy_predicted','raw_score':'proxy_score'}))
 reference=pd.concat(original,ignore_index=True)
 allpred=[];fit_count=0
 for check in checks:
  h=check['horizon'];month=check['month'];cutoff=check['cutoff'];days=sorted(full.loc[full['target_h'+str(h)].lt(cutoff),'date'].unique())[-252:]
  train=full[full.date.isin(days)&full['y_h'+str(h)].notna()].copy()
  if digest(train,['date','code']+common.RELAY_FIELDS+['target_h'+str(h),'y_h'+str(h)])!=check['training_digest']:raise ValueError('members changed')
  fields=common.RELAY_FIELDS;w=1/train.groupby('date').date.transform('size');assert abs(w.sum()-252)<1e-7
  prefix=OUT/(month+'_h'+str(h));write(str(prefix)+'_fit_STARTED.json',dict(started_at=now(),check=check,weight_sum=float(w.sum()),score_threshold=.5))
  model=make('logistic')
  with warnings.catch_warnings():
   warnings.simplefilter('error',ConvergenceWarning);model.fit(train[fields],train['y_h'+str(h)].astype(int),**{model.steps[-1][0]+'__sample_weight':w})
  fit_count+=1;joblib.dump(model,str(prefix)+'_model.joblib',compress=3)
  test=full[full.date.str[:7].eq(month)].copy()
  # Predictions deliberately contain no evaluation outcome until persisted.
  pred=test[['date','code','name','boards','actual_one_word_daily','target_h'+str(h)]].copy();pred=pred.rename(columns={'target_h'+str(h):'target_date'});pred['horizon']=h
  pred['score']=model.predict_proba(test[fields])[:,1];pred['predicted']=pred.score.gt(.5).astype(int)
  pred.to_csv(str(prefix)+'_predictions_BEFORE_VALIDATION.csv',index=False,encoding='utf-8-sig')
  write(str(prefix)+'_fit_COMPLETED.json',dict(completed_at=now(),model_sha256=b.sha(str(prefix)+'_model.joblib'),predictions_sha256=b.sha(str(prefix)+'_predictions_BEFORE_VALIDATION.csv'),training_check=check,fit_number=fit_count))
  pred['y']=test['y_h'+str(h)].to_numpy();allpred.append(pred)
 # Only after all predictions are immutable, attach labels and compare.
 x=pd.concat(allpred,ignore_index=True);x=x.merge(reference,on=['date','code'],how='left',validate='many_to_one');x['always_up']=1
 if x[x.date.le('2026-09-30')].proxy_predicted.isna().any():raise ValueError('complete frozen comparison unavailable')
 historical=x[x.date.le('2026-09-30')].copy();historical.to_csv(OUT/'all_original_rows_WITH_UNKNOWN_and_references.csv',index=False,encoding='utf-8-sig')
 metrics=[];accepted=[]
 for h in HORIZONS:
  z=historical[historical.horizon.eq(h)];foldpasses=[]
  for name,(start,end) in FOLDS.items():
   sub=z[z.date.between(start,end)];direct=metric(sub);proxy=metric(sub,'proxy_predicted');always=metric(sub,'always_up')
   gate=direct['accuracy']>=max(proxy['accuracy'],always['accuracy'])+.03 and direct['balanced_accuracy']>=proxy['balanced_accuracy'] and direct['F1']>=proxy['F1']
   metrics.append(dict(horizon=h,fold=name,direct=direct,proxy_nextlimit_direction=proxy,always_up=always,Top3_direct=top(sub,'score'),Top3_proxy=top(sub,'proxy_score'),Top3_height=top(sub,'boards'),fold_pass=gate));foldpasses.append(gate)
  ci_proxy=interval(z,'proxy_predicted');ci_up=interval(z,'always_up');t=top(z,'score');tp=top(z,'proxy_score')
  ok=all(foldpasses) and ci_proxy['lower']>0 and ci_up['lower']>0 and t['hit_fraction_known']>=tp['hit_fraction_known']
  accepted.append(dict(horizon=h,passed=ok,accuracy_vs_proxy_interval=ci_proxy,accuracy_vs_always_up_interval=ci_up,Top3_direct=t,Top3_proxy=tp))
 current=b.read(Path(__file__).resolve().parent/'features/current29_BEFORE_MODEL.json');current_rows=[]
 X=pd.DataFrame([r['features'] for r in current['relay']]);assert len(X)==11
 for h in HORIZONS:
  model=joblib.load(str(OUT/('2026-10_h'+str(h)))+'_model.joblib');scores=model.predict_proba(X[common.RELAY_FIELDS])[:,1]
  for r,s in zip(current['relay'],scores):current_rows.append(dict(code=r['code'],name=r['name'],horizon=h,uncalibrated_score=float(s),direction='YES' if s>.5 else 'NO',not_next_limit_probability=True))
 write(OUT/'current11_horizon_scores_RESEARCH_ONLY.json',dict(formed_at=now(),signal_date='2026-10-19',rows=current_rows,probability_validated=False,next_day_accuracy_improved=False,do_not_replace_independent_views=True))
 result=dict(formed_at=now(),actual_new_research_fits=fit_count,production_refits=0,old_failed_trials_repeated=0,next_day_accuracy_improved=False,direct_multiday_acceptance=accepted,fold_metrics=metrics,
  original_historical_row_count=int(historical.shape[0]/3),predictions_with_unknown=len(historical),new_forward_outcomes=0,deployed=False,probability_calibrated=False,
  source_scope='late source, previously seen history, survivor scope; endpoint cash-price event only; no fee-adjusted tradable return',current_inference_after_independent_archive=True)
 write(OUT/'fixed_research_result.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':
 if sys.argv[1]=='qualify':qualify()
 elif sys.argv[1]=='run':run()
 else:raise ValueError('qualify or run')
