from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json,hashlib,requests
OUT=Path(__file__).resolve().parent/'minutes';OUT.mkdir(exist_ok=True)
DAY='20261023'
SPECS=[('sh603089','正裕工业'),('sz002535','林州重机'),('sh603528','多伦科技'),('sh603933','睿能科技'),('sh603677','奇精机械'),('sz002009','天奇股份'),('sh600215','派斯林'),('sh603081','大丰实业'),('sh603721','中广天择'),('sz000678','襄阳轴承'),('sz003021','兆威机电'),('sh600785','新华百货')]
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

def em_get(spec):
 symbol,name=spec;code=symbol[2:];rec=dict(symbol=symbol,name=name,requested_at=now(),attempts=1,source='EASTMONEY_MINUTE_OHLC')
 try:
  r=requests.get('https://push2his.eastmoney.com/api/qt/stock/trends2/get',params=dict(secid=('1.' if symbol.startswith('sh') else '0.')+code,fields1='f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13',fields2='f51,f52,f53,f54,f55,f56,f57,f58',ndays='1',iscr='0',iscca='0'),timeout=(4,9),headers={'User-Agent':'Mozilla/5.0'})
  p=OUT/(symbol+'_eastmoney.response');p.write_bytes(r.content)
  rec.update(url=r.url,received_at=now(),http_status=r.status_code,path=str(p),sha256=hashlib.sha256(r.content).hexdigest());r.raise_for_status()
  obj=r.json()['data'];assert obj['code']==code
  rows=[]
  for x in obj['trends']:
   f=x.split(',');assert f[0].startswith('2026-10-23 '),f[0]
   rows.append(dict(time=f[0][-5:].replace(':',''),datetime=f[0],open=float(f[1]),price=float(f[2]),high=float(f[3]),low=float(f[4]),volume_hands=float(f[5]),amount_cny=float(f[6]),vwap=float(f[7])))
  rec.update(status='EXACT_DATE_MINUTE_OHLC',source_date=DAY,previous_close=obj.get('preClose'),count=len(rows),rows=rows)
 except Exception as e:rec.update(status='UNAVAILABLE',error=str(e)[:300])
 return rec
with ThreadPoolExecutor(max_workers=6) as e:records=list(e.map(get,SPECS))
with ThreadPoolExecutor(max_workers=2) as e:em_records=list(e.map(em_get,[('sz002724','海洋王'),('sh603721','中广天择')]))
with (OUT/'additional_capture_receipt.json').open('x',encoding='utf-8') as w:json.dump(dict(formed_at=now(),date=DAY,scope='Exactly declared additional 12 Tencent stocks and 2 Eastmoney seal conflict checks, one call each',records=records+em_records),w,ensure_ascii=False,indent=2)
for x in records+em_records:
 print(json.dumps({k:x.get(k) for k in ['symbol','name','source','status','count','previous_close','last_quote','error']},ensure_ascii=False))
