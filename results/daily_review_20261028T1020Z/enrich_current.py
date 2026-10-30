from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal,ROUND_HALF_UP
import json,hashlib
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];DAY='2026-10-28';PRIOR='2026-10-27';CAP=9326859896.52
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def limit(v,m='1.1'):return int((Decimal(int(v))*Decimal(m)).quantize(Decimal('1'),rounding=ROUND_HALF_UP))
f=read(OUT/'facts/qualified_full_facts_SCORES_MASKED.json');daily=read(OUT/'facts/normalized_daily.json')
raw=read(OUT/'facts/raw/2026-10-28_limit_up.json');lut={r['c']:r for r in raw['data']['pool']}
priorpath=ROOT/'results/daily_review_20261027T0928Z/facts/RESOLVED_FACTS_v2_SCORES_MASKED.json';prior=read(priorpath)
oldlut={r['code']:r for r in prior['rows']};changes=[]
for r in f['rows']:
 c=r['code'];ds=daily.get(c,{}).get('days',{});a=ds.get(PRIOR);b=ds.get(DAY)
 if c=='605488' and a and b and b['close']==limit(a['close']):
  r['original_provider_kind']=r['kind'];r['kind']='limit_up';r['boards']=oldlut.get(c,{}).get('boards',0)+1 if oldlut.get(c,{}).get('kind')=='limit_up' else 1
  r['qualification']='CLOSING_PRICE_CONFIRMED_POOL_CONFLICT';r['seal_first']=None;r['seal_last']=None
  changes.append({'code':c,'name':r['name'],'original_kind':'failed_board','resolved_event':'close_at_limit','boards':r['boards'],'close':r['close'],'queue_and_seal_time_unknown':True})
 p=lut.get(c)
 if p:
  past=[v for k,v in sorted(ds.items()) if k<DAY];mean=sum(v['volume_native'] for v in past[-5:])/5 if len(past)>=5 else None
  vr=b['volume_native']/mean if b and mean else None
  thin=p['ltsz']<=3e9 and p['hs']<5 and (r['one_price'] is True or vr is not None and vr<1)
  r.update(reported_float_cap_cny=p['ltsz'],reported_total_cap_cny=p['tshare'],volume_ratio_previous5=vr,original_cap_screen=p['ltsz']<=CAP,thin_proxy_excluded=thin,
   screen_status='EXCLUDED_CAP' if p['ltsz']>CAP else 'EXCLUDED_THIN_PROXY' if thin else 'UNKNOWN_DAILY' if vr is None or r['one_price'] is None else 'WITHIN_ORIGINAL_RESEARCH_SCREEN')
 else:r.update(screen_status='UNKNOWN_SOURCE_CAP',original_cap_screen=None,thin_proxy_excluded=None)
 r['seal_delta_1430_to_close']=None;r['seal_delta_status']='UNAVAILABLE_NO_COMPARABLE_TIMESTAMPED_QUEUE_PAIR'
f.update(formed_at=now(),relay=[r for r in f['rows'] if r['kind']=='limit_up' and r['boards']>=2],resolved_closing_counts={k:sum(r['kind']==k for r in f['rows']) for k in ['limit_up','failed_board','limit_down']},closing_price_resolutions=changes,market_cap_screen_threshold_cny=CAP)
f['source_bindings'][str(OUT/'facts/qualified_full_facts_SCORES_MASKED.json')]=sha(OUT/'facts/qualified_full_facts_SCORES_MASKED.json')
write(OUT/'facts/RESOLVED_FACTS_v2_SCORES_MASKED.json',f)
feedback=[]
for r in prior['rows']:
 ds=daily.get(r['code'],{}).get('days',{});a=ds.get(PRIOR);b=ds.get(DAY);known=bool(a and b)
 feedback.append(dict(code=r['code'],name=r['name'],prior_kind=r['kind'],prior_boards=r.get('boards'),known=known,
  close=b['close']/100 if b else None,previous_close=a['close']/100 if a else None,return_pct=(b['close']/a['close']-1)*100 if known else None,
  open_gap_pct=(b['open']/a['close']-1)*100 if known else None,closed_limit_up=int(b['close']==limit(a['close'])) if known else None,
  closed_limit_down=int(b['close']==limit(a['close'],'0.9')) if known else None))
layers={}
for h in sorted({r['prior_boards'] for r in feedback if r['prior_kind']=='limit_up'}):
 allr=[r for r in feedback if r['prior_kind']=='limit_up' and r['prior_boards']==h];kr=[r for r in allr if r['known']]
 layers[str(h)]={'total':len(allr),'known':len(kr),'unknown':len(allr)-len(kr),'promoted':sum(r['closed_limit_up'] for r in kr),'mean_return_pct':sum(r['return_pct'] for r in kr)/len(kr) if kr else None,'limit_down':sum(r['closed_limit_down'] for r in kr)}
write(OUT/'facts/COMPLETE_PRIOR_COHORT_FEEDBACK.json',{'formed_at':now(),'signal_date':DAY,'prior_date':PRIOR,'source':str(priorpath),'sha256':sha(priorpath),'layers':layers,'rows':feedback,'unknown_not_zero':True})
print(json.dumps({'counts':f['resolved_closing_counts'],'changes':changes,'layers':layers,'first_screen':[{'code':r['code'],'name':r['name'],'cap_yi':r.get('reported_float_cap_cny',0)/1e8,'close':r['close'],'seal':r.get('seal_first'),'turnover':r['turnover_pct']} for r in f['rows'] if r['kind']=='limit_up' and r['boards']==1 and r['screen_status']=='WITHIN_ORIGINAL_RESEARCH_SCREEN']},ensure_ascii=False))
