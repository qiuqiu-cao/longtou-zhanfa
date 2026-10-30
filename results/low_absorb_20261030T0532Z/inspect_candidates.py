from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
import requests,json,hashlib,importlib.util,sys
O=Path(__file__).resolve().parent;ROOT=O.parents[1];sys.path.insert(0,str(ROOT))
import public_outcome_sources_v293 as public
def now():return datetime.now(timezone.utc).isoformat()
(O/'detail').mkdir(exist_ok=True)
quotes=json.loads((O/'live_quotes.json').read_text(encoding='utf8'))['rows'];qlut={r['code']:r for r in quotes}
codes=['000678','002708','002565','002696','001317','603382','002185','600487','600362','603897','002531','603279','603067','600997']
def get(job):
 code,kind=job;sym=('sh' if code.startswith('6') else 'sz')+code;rec=dict(code=code,kind=kind,requested_at=now())
 try:
  if kind=='minute':url='https://web.ifzq.gtimg.cn/appstock/app/minute/query';params={'code':sym}
  else:url=public.DAY_URL;params={'param':f'{sym},day,2026-06-01,2026-10-30,180,'}
  r=requests.get(url,params=params,timeout=(4,9),headers={'User-Agent':'Mozilla/5.0'});raw=O/'detail'/(code+'_'+kind+'.response');raw.write_bytes(r.content);r.raise_for_status()
  rec.update(url=r.url,available_at=now(),path=str(raw),sha256=hashlib.sha256(r.content).hexdigest())
  obj=r.json()
  if kind=='minute':
   v=obj['data'][sym]['data'];assert v['date']=='20261030';rec.update(status='OBSERVED_INTRADAY',source_date=v['date'],rows=[dict(time=x.split()[0],price=float(x.split()[1]),cum_volume=float(x.split()[2]),extra=x.split()[3:]) for x in v['data']])
  else:
   d=public.parse_daily(obj,code,'2026-06-01','2026-10-30');bars=[(k,v) for k,v in sorted(d['days'].items()) if k<'2026-10-30'];cl=[v['close']/100 for k,v in bars]
   rec.update(status='OBSERVED_HISTORY',days=d['days'],ma20_live=(sum(cl[-19:])+qlut[code]['price'])/20 if len(cl)>=19 else None,
    ma5_live=(sum(cl[-4:])+qlut[code]['price'])/5 if len(cl)>=4 else None,prior20_high=max(v['high']/100 for k,v in bars[-20:]),prior5_high=max(v['high']/100 for k,v in bars[-5:]),
    prior20_mean_volume_lots=sum(v['volume_native'] for k,v in bars[-20:])/20,amount20_verified=False)
 except Exception as e:rec.update(status='UNAVAILABLE',error=str(e)[:250],completed_at=now())
 return rec
with ThreadPoolExecutor(max_workers=8) as ex:records=list(ex.map(get,[(c,k) for c in codes for k in ['minute','daily']]))
(O/'detail/observations.json').write_text(json.dumps(dict(formed_at=now(),records=records),ensure_ascii=False,indent=2),encoding='utf8')
for c in codes:
 a=next(r for r in records if r['code']==c and r['kind']=='daily');b=next(r for r in records if r['code']==c and r['kind']=='minute');points={r['time']:r['price'] for r in b.get('rows',[])}
 print(json.dumps(dict(code=c,name=qlut[c]['name'],price=qlut[c]['price'],ma20=a.get('ma20_live'),ma5=a.get('ma5_live'),prior5_high=a.get('prior5_high'),minute={t:points.get(t) for t in ['0930','0945','1000','1030','1100','1130','1300','1310','1320','1330']},minute_status=b['status'],daily_status=a['status']),ensure_ascii=True))
