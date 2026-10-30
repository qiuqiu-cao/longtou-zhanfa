from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
import requests,json,re,hashlib,sys
O=Path(__file__).resolve().parent;ROOT=O.parents[1];sys.path.insert(0,str(ROOT))
import public_outcome_sources_v293 as p
(O/'raw').mkdir(exist_ok=True)
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
old=read(ROOT/'results/daily_review_20261029T1912Z/facts/RESOLVED_FACTS_v2_SCORES_MASKED.json')
codes={r['code'] for r in old['rows']}
codes|={'000678','600825','600838','002696','002724','603278','002896','002050','002472','002185','002156','600584','601811','003028','603897','002164','000670',
 '601398','601939','600036','002948','601288','601988','601318','601601','601628','600900','600886','600011','600795','601985','600025','600028','601857','601088','601225','600188','601899','600547','600489','000975','600362','600050','601728','600941','600487','600089','601615','002028','601012','600031','600150'}
syms=['sh000001','sz399001','sz399006','sh000688']+[('sh' if c.startswith('6') else 'sz')+c for c in sorted(codes)]
def fetch(job):
 name,url,params=job;rec=dict(name=name,url=url,params=params,requested_at=now())
 try:
  r=requests.get(url,params=params,timeout=(4,9),headers={'User-Agent':'Mozilla/5.0'});raw=O/'raw'/(name+'.response');raw.write_bytes(r.content);r.raise_for_status()
  rec.update(status='RECEIVED',available_at=now(),raw_path=str(raw),sha256=hashlib.sha256(r.content).hexdigest());return rec,r
 except Exception as e:rec.update(status='UNAVAILABLE',error=str(e)[:300],completed_at=now());return rec,None
jobs=[('quotes_'+str(i),'https://qt.gtimg.cn/q='+','.join(syms[i:i+60]),None) for i in range(0,len(syms),60)]
for kind,endpoint in p.ENDPOINTS.items():
 jobs.append((kind,p.EM_BASE+endpoint,dict(ut='7eea3edcaed734bea9cbfc24409ed989',dpt='wz.ztzt',Pageindex='0',pagesize='10000',sort='fbt:asc' if kind in {'limit_up','failed_board'} else 'fund:asc' if kind=='limit_down' else 'zs:desc',date='20261030')))
quotes=[];pools={};receipts=[]
with ThreadPoolExecutor(max_workers=8) as ex:
 for rec,r in ex.map(fetch,jobs):
  receipts.append(rec)
  if r is None:continue
  if rec['name'].startswith('quotes'):
   for sym,body in re.findall(r'v_([a-z]+\d+)="([^"]*)"',r.content.decode('gbk',errors='replace')):
    x=body.split('~')
    try:
     q=dict(symbol=sym,code=x[2],name=x[1],price=float(x[3]),preclose=float(x[4]),open=float(x[5]),volume_lots=float(x[6]),clock=x[30],pct=float(x[32]),high=float(x[33]),low=float(x[34]),amount_wan=float(x[37]),turnover_pct=float(x[38]),float_cap_yi=float(x[44]),total_cap_yi=float(x[45]),upper=float(x[47]),lower=float(x[48]),fields=x)
     assert q['clock'].startswith('20261030');q['vwap']=float(x[57])*10000/(float(x[36])*100) if float(x[36])>0 and sym not in {'sh000001','sz399001','sz399006','sh000688'} else None
     quotes.append(q)
    except Exception as e:rec.setdefault('parse_errors',[]).append(dict(symbol=sym,error=str(e)[:180]))
  else:
   try:pools[rec['name']]=p.parse_pool(r.json(),'2026-10-30',rec['name'])
   except Exception as e:rec['parse_error']=str(e)[:200]
write(O/'capture_manifest.json',dict(formed_at=now(),session='2026-10-30_INTRADAY',requests=receipts,prior_review_pool=102,declared_extra_defensive_and_prior_core=True,exchange_wide_screen=False))
write(O/'live_quotes.json',dict(formed_at=now(),rows=quotes,scope='Explicit prior-review candidates and defensive/core comparators; not full-market screen',vwap_units='Tencent quote amount57 in wan CNY divided by volume36 in lots times100; positional source inference'))
write(O/'live_pools.json',dict(formed_at=now(),pools=pools,closing_data=False))
print(json.dumps(dict(quotes=len(quotes),pools={k:v['all_market_raw_count'] for k,v in pools.items()},failures=[r for r in receipts if r['status']!='RECEIVED'],summary=[{k:q[k] for k in ['symbol','name','price','pct','open','high','low','turnover_pct','vwap','clock']} for q in quotes]),ensure_ascii=True))
