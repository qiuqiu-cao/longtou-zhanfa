from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json,hashlib,requests
OUT=Path(__file__).resolve().parent/'minutes';OUT.mkdir(exist_ok=True)
DAY='20261023'
SPECS=[('sh000001','沪指'),('sz399001','深成指'),('sz399006','创业板'),
('sh603278','大业股份'),('sz002724','海洋王'),('sh600293','三峡新材'),
('sh600838','上海九百'),('sz002343','慈文传媒'),('sh600825','新华传媒'),
('sz002696','百洋股份'),('sz002242','九阳股份'),('sz002185','华天科技'),
('sz002156','通富微电'),('sz002896','中大力德'),('sz002347','泰尔股份'),
('sz002068','黑猫股份'),('sh603028','赛福天'),('sh603280','南方路机'),('sh600261','阳光照明')]
def now():return datetime.now(timezone.utc).isoformat()
def get(spec):
 symbol,name=spec;rec=dict(symbol=symbol,name=name,requested_at=now(),attempts=1)
 try:
  r=requests.get('https://web.ifzq.gtimg.cn/appstock/app/minute/query',params={'code':symbol},timeout=(4,9),headers={'User-Agent':'Mozilla/5.0'})
  p=OUT/(symbol+'.response');p.write_bytes(r.content)
  rec.update(url=r.url,received_at=now(),http_status=r.status_code,path=str(p),sha256=hashlib.sha256(r.content).hexdigest());r.raise_for_status()
  obj=r.json()['data'][symbol];v=obj['data'];assert v['date']==DAY,repr(v.get('date'))
  q=obj['qt'][symbol];rows=[]
  for x in v['data']:
   f=x.split();t=f[0]
   if '0930'<=t<='1130' or '1300'<=t<='1500':rows.append(dict(time=t,price=float(f[1]),cum_volume=float(f[2]),cum_amount=float(f[3]) if len(f)>3 else None))
  rec.update(status='EXACT_DATE_MINUTE_POINTS_SINGLE_SOURCE',source_date=v['date'],previous_close=float(q[4]),last_quote=float(q[3]),quote_clock=q[30],count=len(rows),rows=rows)
 except Exception as e:rec.update(status='UNAVAILABLE',error=str(e)[:300])
 return rec
with ThreadPoolExecutor(max_workers=6) as e:records=list(e.map(get,SPECS))
(OUT/'capture_receipt.json').write_text(json.dumps(dict(formed_at=now(),date=DAY,scope='Declared index, old emotion, robot and capacity comparators; no fund-flow causality',records=records),ensure_ascii=False,indent=2),encoding='utf-8')
for x in records:
 if x['status']=='UNAVAILABLE':print(json.dumps({k:x.get(k) for k in ['name','status','error']},ensure_ascii=False));continue
 rs=x['rows'];lo=min(rs,key=lambda z:z['price']);hi=max(rs,key=lambda z:z['price'])
 print(json.dumps(dict(name=x['name'],previous=x['previous_close'],close=x['last_quote'],low=lo,high=hi,samples=[z for z in rs if z['time'] in ['0930','1000','1030','1100','1130','1300','1330','1400','1430','1500']]),ensure_ascii=False))
