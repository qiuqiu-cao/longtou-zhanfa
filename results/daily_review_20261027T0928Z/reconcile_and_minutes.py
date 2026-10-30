from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal,ROUND_HALF_UP
import requests,json,hashlib,copy
O=Path(__file__).resolve().parent
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def get(job):
 c,name=job;s=('sh' if c.startswith('6') else 'sz')+c;rec=dict(symbol=s,name=name,requested_at=now(),attempts=1)
 try:
  r=requests.get('https://web.ifzq.gtimg.cn/appstock/app/minute/query',params={'code':s},timeout=(4,8),headers={'User-Agent':'Mozilla/5.0'});p=O/'minutes'/(s+'_additional.response');p.write_bytes(r.content);r.raise_for_status();d=r.json()['data'][s];assert d['data']['date']=='20261027';q=d['qt'][s]
  rows=[]
  for line in d['data']['data']:
   v=line.split();t=v[0]
   if '0930'<=t<='1130' or '1300'<=t<='1500':rows.append(dict(time=t,price=float(v[1]),cum_volume=float(v[2]),cum_amount=float(v[3]) if len(v)>3 else None))
  rec.update(status='EXACT_DATE_MINUTE_POINTS_SINGLE_SOURCE',source_date='20261027',url=r.url,raw_path=str(p),sha256=sha(p),received_at=now(),previous_close=float(q[4]),last_quote=float(q[3]),quote_clock=q[30],rows=rows)
 except Exception as e:rec.update(status='UNAVAILABLE',error=str(e)[:250])
 return rec
jobs=[('600865','百大集团'),('603758','秦安股份'),('601956','东贝集团'),('600828','茂业商业'),('001400','江顺科技'),('603376','大明电子'),('603370','华新精科'),('601566','九牧王'),('002187','广百股份'),('000058','深赛格'),('603089','正裕工业')]
with ThreadPoolExecutor(max_workers=6) as e:extra=list(e.map(get,jobs))
write(O/'minutes/additional_capture_receipt.json',dict(formed_at=now(),date='20261027',records=extra,secondary_domain_all_failed=True))
records=read(O/'minutes/capture_receipt.json')['records']+extra;mm={r['symbol'][-6:]:r for r in records if r.get('rows')}
f=read(O/'facts/complete_enriched_facts_SCORES_MASKED.json');raw=copy.deepcopy(f);daily=read(O/'facts/normalized_daily.json');old=read(O.parent/'daily_review_20261026T1323Z/facts/complete_enriched_facts_SCORES_MASKED.json');oldlut={r['code']:r for r in old['rows']};decisions=[]
for r in f['rows']:
 if r['code'] not in ['600826','600865']:continue
 c=r['code'];bar=daily[c]['days']['2026-10-27'];before=daily[c]['days']['2026-10-26'];qt=daily[c]['quote'];m=mm.get(c);pool=r['source_rows'].get('primary_failed_board',{})
 cap=int((Decimal(before['close'])*Decimal('1.1')).quantize(Decimal('1'),rounding=ROUND_HALF_UP))
 assert bar['close']==cap and pool['price_cents']==cap and (m is None or m['rows'][-1]['price']==cap/100)
 r['original_provider_kind']=r['kind'];r['original_provider_boards']=r.get('boards');r['kind']='limit_up';r['boards']=oldlut[c]['boards']+1 if c in oldlut and oldlut[c]['kind']=='limit_up' else 1
 r['qualification']='CLOSING_PRICE_CONFIRMED_POOL_MEMBERSHIP_CONFLICT';r['seal_last']=None;r['one_price']=False
 qv=qt['values'];r['reported_float_cap_cny']=float(qv['float_cap_native'])*1e8 if 'float_cap_native' in qv else None
 # Quote field positions are the same established adapter used in enrichment; preserve explicit units.
 if r['reported_float_cap_cny'] is None:
  vals=qv.get('fields',[]) if isinstance(qv,dict) else []
 # Existing parsed quote adapter stores ordered original fields under raw_fields; inspect before using cap.
 r['screen_status']='UNKNOWN_POOL_MEMBERSHIP_AND_CAP_BINDING';r['original_cap_screen']=None;r['thin_proxy_excluded']=None
 r['pool_resolution']='Exact unadjusted daily close, primary failed-pool price and Tencent minute/quote agree at upper limit; original failed classification retained. Closing-price event confirmed; seal queue/time and final board-pool membership not certified.'
 decisions.append(dict(code=c,name=r['name'],original_kind='failed_board',resolved_kind='limit_up',boards=r['boards'],close=cap/100,minute_received=m is not None,primary_price=pool['price_cents']/100,source_qualification=r['qualification']))
f['formed_at']=now();f['raw_source_counts']=f['source_counts'];f['resolved_closing_counts']={k:sum(r['kind']==k for r in f['rows']) for k in ['limit_up','failed_board','limit_down']};f['relay']=[r for r in f['rows'] if r['kind']=='limit_up' and r['boards']>=2];f['closing_price_resolutions']=decisions;f['source_bindings'][str(O/'facts/complete_enriched_facts_SCORES_MASKED.json')]=sha(O/'facts/complete_enriched_facts_SCORES_MASKED.json')
write(O/'facts/RECONCILED_COMPLETE_FACTS_SCORES_MASKED.json',f)
write(O/'facts/CLOSING_PRICE_POOL_CONFLICT_APPENDIX.json',dict(formed_at=now(),decisions=decisions,original_full_rows_unchanged=True,raw_counts=raw['source_counts'],resolved_counts=f['resolved_closing_counts'],all_raw_files_retained=True,model_source_gate=False))
fb=read(O/'facts/COMPLETE_PRIOR_COHORT_FEEDBACK.json');supp=[]
for r in fb['rows']:
 if r['price_known']:continue
 m=mm.get(r['code']);prior=oldlut[r['code']].get('close')
 if m and prior and m['rows'][-1]['time']=='1500':
  cap=float((Decimal(str(prior))*Decimal('1.1')).quantize(Decimal('.01'),rounding=ROUND_HALF_UP));cl=m['rows'][-1]['price'];supp.append(dict(code=r['code'],name=r['name'],close=cl,previous_close=prior,change_pct=(cl/prior-1)*100,closed_limit_up=int(cl==cap),source='Tencent exact-date minute point plus quote; no daily OHLC certification',prior_boards=r.get('prior_boards')))
write(O/'facts/PRIOR_MISSING_CLOSE_MINUTE_SUPPLEMENT.json',dict(formed_at=now(),rows=supp,daily_unknowns_preserved=True))
windows=[]
for node in ['0937','1012','1400','1449','1453']:
 n=int(node[:2])*60+int(node[2:]);times=[f'{(n+k)//60:02d}{(n+k)%60:02d}' for k in [-5,-3,-1,0,1,3,5]]
 windows.append(dict(node=node,times=times,prices={c:[{x['time']:x['price'] for x in mm.get(c,{}).get('rows',[])}.get(t) for t in times] for c in ['000001','399001','399006','600838','603528','000678','603028','001219','002820','001318','002724','002696','002086','600558','600825','002185','603089']}))
write(O/'minutes/RELATIONSHIP_WINDOWS.json',dict(formed_at=now(),date='2026-10-27',offsets_minutes=[-5,-3,-1,0,1,3,5],frequency='minute price points; not interval highs/lows, no tick causality',windows=windows))
print(json.dumps(dict(extra=[(x['name'],x['status']) for x in extra],resolutions=decisions,counts=f['resolved_closing_counts'],minute_settlement=supp,quote_cap_keys=list(daily['600826']['quote']['values'])),ensure_ascii=False))
