from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal,ROUND_HALF_UP
import json,hashlib
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];DAY='2026-10-23';PRIOR='2026-10-22';CAP=9326859896.52
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as w:json.dump(x,w,ensure_ascii=False,indent=2,allow_nan=False)
def limit(c,mult='1.1'):return int((Decimal(c)*Decimal(mult)).quantize(Decimal('1'),rounding=ROUND_HALF_UP))
f=read(OUT/'facts/qualified_full_facts_SCORES_MASKED.json');d=read(OUT/'facts/normalized_daily.json')
raw=read(OUT/'facts/raw/2026-10-23_limit_up.json')['data']['pool'];lut={x['c']:x for x in raw};field_conflicts=[]
for row in f['rows']:
 z=lut.get(row['code'])
 if z:
  past=[v for k,v in sorted(d.get(row['code'],{}).get('days',{}).items()) if k<DAY];cur=d.get(row['code'],{}).get('days',{}).get(DAY)
  mean=sum(v['volume_native'] for v in past[-5:])/5 if len(past)>=5 else None
  ratio=cur['volume_native']/mean if cur and mean else None
  thin=z['ltsz']<=3e9 and z['hs']<5 and (row['one_price'] is True or ratio is not None and ratio<1)
  gap=row['one_price'] is None or ratio is None
  row.update(reported_float_cap_cny=z['ltsz'],reported_total_cap_cny=z['tshare'],primary_amount_cny=z['amount'],volume_ratio_previous5=ratio,original_cap_screen=z['ltsz']<=CAP,thin_proxy_excluded=thin,
   screen_status='EXCLUDED_CAP' if z['ltsz']>CAP else 'EXCLUDED_THIN_PROXY' if thin else 'UNKNOWN_MISSING_DAILY_FOR_RISK_SCREEN' if gap else 'WITHIN_ORIGINAL_RESEARCH_SCREEN',risk_proxy_not_fund_identity=True)
  page=row.get('reported_times'); ft=z.get('fbt'); lt=z.get('lbt')
  fmt=lambda v:f'{int(v)//10000:02d}:{int(v)//100%100:02d}' if v is not None else None
  if page and ' / ' in page and page.split(' / ')!=[fmt(ft),fmt(lt)]:
   conflict=dict(code=row['code'],name=row['name'],field='FIRST_LAST_SEAL_TIME',page=page,primary_first=ft,primary_last=lt,resolution='UNKNOWN_REQUIRE_INTRADAY_CROSSCHECK')
   field_conflicts.append(conflict);row['seal_time_source_conflict']=conflict
  row['seal_fields_qualification']='CONFLICT_REQUIRE_INTRADAY' if row.get('seal_time_source_conflict') else 'PROVIDER_REPORTED_NOT_ORDERFLOW_CERTIFIED'
f['seal_time_source_conflicts']=field_conflicts
f['source_bindings'][str(OUT/'facts/raw/2026-10-23_limit_up.json')]=sha(OUT/'facts/raw/2026-10-23_limit_up.json')
f['market_cap_screen_threshold_cny']=CAP
f['formed_at']=now()
write(OUT/'facts/complete_enriched_facts_SCORES_MASKED.json',f)
oldpath=ROOT/'results/daily_review_20261022T1608Z/facts/complete_enriched_facts_SCORES_MASKED.json';old=read(oldpath);current={x['code']:x for x in f['rows']};feedback=[]
for x in old['rows']:
 ds=d.get(x['code'],{}).get('days',{});a=ds.get(PRIOR);b=ds.get(DAY);known=bool(a and b)
 upper=limit(a['close']) if known else None;lower=limit(a['close'],'0.9') if known else None
 nxt=current.get(x['code'],{})
 feedback.append(dict(code=x['code'],name=x['name'],prior_kind=x['kind'],prior_boards=x.get('boards'),price_known=known,closed_limit_up=int(b['close']==upper) if known else None,promoted=int(b['close']==upper) if known and x['kind']=='limit_up' else None,closed_limit_down=int(b['close']==lower) if known else None,intraday_touched_upper=int(b['high']==upper) if known else None,return_pct=(b['close']/a['close']-1)*100 if known else None,open_gap_pct=(b['open']/a['close']-1)*100 if known else None,close=b['close']/100 if b else None,previous_close=a['close']/100 if a else None,current_pool_kind=nxt.get('kind'),current_boards=nxt.get('boards'),unknown_reason=None if known else 'EXACT_OFFICIAL_DAY_PAIR_MISSING',outcome_source='TENCENT_EXACT_UNADJUSTED_CLOSE_WITH_OFFICIAL_ADJACENT_DATES'))
layers={}
for h in sorted({x['prior_boards'] for x in feedback if x['prior_kind']=='limit_up'}):
 q=[x for x in feedback if x['prior_kind']=='limit_up' and x['prior_boards']==h];k=[x for x in q if x['price_known']]
 layers[str(h)]=dict(total=len(q),known=len(k),unknown=len(q)-len(k),promoted=sum(x['promoted'] for x in k),mean_return_pct=sum(x['return_pct'] for x in k)/len(k) if k else None,loss7_count=sum(x['return_pct']<=-7 for x in k),limit_down_count=sum(x['closed_limit_down'] for x in k),mean_open_gap_pct=sum(x['open_gap_pct'] for x in k)/len(k) if k else None)
kind_layers={}
for kind in ['limit_up','failed_board','limit_down']:
 q=[x for x in feedback if x['prior_kind']==kind];k=[x for x in q if x['price_known']]
 kind_layers[kind]=dict(total=len(q),known=len(k),unknown=len(q)-len(k),closed_limit_up=sum(x['closed_limit_up'] for x in k),closed_limit_down=sum(x['closed_limit_down'] for x in k),mean_return_pct=sum(x['return_pct'] for x in k)/len(k) if k else None,loss7_count=sum(x['return_pct']<=-7 for x in k))
write(OUT/'facts/COMPLETE_PRIOR_COHORT_FEEDBACK.json',dict(formed_at=now(),prior_date=PRIOR,signal_date=DAY,prior_full_pool_source=str(oldpath),prior_full_pool_sha256=sha(oldpath),complete_prior_rows=len(old['rows']),layers=layers,kind_layers=kind_layers,rows=feedback,missing_prices_not_zero=True,all_original_rows_preserved=len(feedback)==len(old['rows'])))
# Blind-to-model settlement of the frozen independent whole pool, no old model reads.
pp=ROOT/'results/daily_review_20261022T1608Z/INDEPENDENT_FULL_POOL_BEFORE_MODEL.json';pred=read(pp);by={x['code']:x for x in feedback};settled=[]
for x in pred['rows']:
 y=by.get(x['code'],{});call=x['next_close_limit_up_call'];actual=y.get('closed_limit_up');settled.append(dict(code=x['code'],name=x['name'],frozen_call=call,actual_closed_limit_up=actual,correct=(int(call=='YES')==actual) if call in ['YES','NO'] and actual is not None else None,prior_kind=y.get('prior_kind'),prior_boards=y.get('prior_boards'),old_forecast_unchanged=True))
known=[x for x in settled if x['actual_closed_limit_up'] is not None];called=[x for x in known if x['frozen_call'] in ['YES','NO']]
top=[dict(by.get(c,{}),original_rank=i+1,old_forecast_unchanged=True) for i,c in enumerate(pred['top3'])]
write(OUT/'facts/PRIOR_INDEPENDENT_FULL_POOL_SETTLEMENT_APPEND_ONLY.json',dict(formed_at=now(),source=str(pp),source_sha256=sha(pp),original_prediction_formed_at=pred['formed_at'],signal_date=PRIOR,target_date=DAY,complete_prediction_slots=len(settled),known_outcomes=len(known),unknown_outcomes=len(settled)-len(known),judged_known=len(called),correct=sum(bool(x['correct']) for x in called),abstained_known=len(known)-len(called),top3=top,top3_correct_known=sum(x['closed_limit_up'] for x in top if x.get('closed_limit_up') is not None),top3_complete_slots=3,rows=settled,multiday_windows_not_all_mature=True))
write(OUT/'facts/PRIOR_TOP3_SETTLEMENT_APPEND_ONLY.json',dict(formed_at=now(),source=str(pp),source_sha256=sha(pp),top3=top,correct_known=sum(x['closed_limit_up'] for x in top if x.get('closed_limit_up') is not None),complete_slots=3,multiday_windows_not_all_mature=True))
manifest=read(OUT/'facts/daily_manifest.json');poolmanifest=read(OUT/'facts/pool_capture_manifest.json');hash_bad=[]
for r in manifest['requests']+poolmanifest['requests']:
 if r.get('raw_path') and sha(r['raw_path'])!=r['raw_sha256']:hash_bad.append(r['raw_path'])
assert len(feedback)==97 and len(settled)==97 and not hash_bad
write(OUT/'facts/CAPTURE_AND_RECONCILIATION_QA.json',dict(formed_at=now(),requests=len(manifest['requests']),normalized_received=len(d),current_full_rows=len(f['rows']),current_exact_known=sum(x['qualification']=='EXACT_DAY_OHLC_OBSERVED' for x in f['rows']),unknown_current=[dict(code=x['code'],name=x['name'],qualification=x['qualification']) for x in f['rows'] if x['qualification']!='EXACT_DAY_OHLC_OBSERVED'],requested_daily_failures=[dict(code=r['code'],status=r['status'],error=r.get('error'),parse_error=r.get('parse_error')) for r in manifest['requests'] if r.get('status')!='RAW_RECEIVED' or r.get('parse_status')=='UNAVAILABLE'],price_board_conflicts=f['conflicts'],seal_time_conflicts=field_conflicts,all_receipt_hashes_valid=not hash_bad,prior_full97_preserved=True,models_loaded=0,fits=0,previous_files_changed=0))
print(json.dumps(dict(counts=f['source_counts'],layers=layers,kind_layers=kind_layers,prior_top3=top,seal_conflicts=field_conflicts,current_qualified=sum(x['qualification']=='EXACT_DAY_OHLC_OBSERVED' for x in f['rows']),current_rows=len(f['rows']),first_screen_pass=[dict(code=x['code'],name=x['name'],close=x['close'],float_cap_yi=x.get('reported_float_cap_cny',0)/1e8,seal_first=x.get('seal_first'),seal_last=x.get('seal_last'),turnover_pct=x.get('turnover_pct'),breaks=x.get('reported_open_count')) for x in f['rows'] if x['kind']=='limit_up' and x['boards']==1 and x.get('screen_status')=='WITHIN_ORIGINAL_RESEARCH_SCREEN']),ensure_ascii=False))
