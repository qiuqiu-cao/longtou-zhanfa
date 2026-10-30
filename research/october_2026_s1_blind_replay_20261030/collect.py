"""Bounded provider data retrieval; never retrain and never request 10/30 labels."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime, timezone
import hashlib,json,time,sys
import pandas as pd,requests
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
RAW=OUT/'raw_kline'; RAW.mkdir(exist_ok=True)
def now():return datetime.now(timezone.utc).isoformat()
def fetch(code):
    raw=RAW/f'{code}.json'; symbol=('1.' if code.startswith('6') else '0.')+code
    params={'secid':symbol,'fields1':'f1,f2,f3,f4,f5,f6','fields2':'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61','klt':'101','fqt':'0','beg':'20260901','end':'20261029'}
    info={'code':code,'requested_at_utc':now(),'params':params}
    for attempt in range(2):
        try:
            r=requests.get('https://push2his.eastmoney.com/api/qt/stock/kline/get',params=params,timeout=(4,12)); r.raise_for_status(); content=r.content
            j=json.loads(content); d=j.get('data'); rows=(d or {}).get('klines',[])
            if not rows or d.get('code')!=code: raise ValueError('empty or wrong code')
            if max(x.split(',')[0] for x in rows)>'2026-10-29':raise ValueError('future row returned')
            raw.write_bytes(content);info.update(received_at_utc=now(),http_status=r.status_code,sha256=hashlib.sha256(content).hexdigest(),rows=len(rows),first=rows[0][:10],last=rows[-1][:10],attempts=attempt+1);return info
        except Exception as e:info['error']=str(e)[:300]
    info.update(received_at_utc=now(),attempts=2);return info

path=ROOT/'data/v16/point_in_time_rebuild_cutoff_20260831_v2/point_in_time_current_daily_bars.csv'
bars=pd.read_csv(path,usecols=['date','code'],dtype={'code':str});codes=sorted(bars[ bars.date.eq('2026-08-31')].code.unique())
assert len(codes)>3000
print('frozen-universe',len(codes),flush=True)
items=[]
with ThreadPoolExecutor(max_workers=12) as pool:
    fs=[pool.submit(fetch,c) for c in codes]
    for n,f in enumerate(as_completed(fs),1):
        items.append(f.result())
        if n%200==0:print('done',n,'success',sum('sha256' in x for x in items),flush=True)
manifest={'formed_at_utc':now(),'scope':'original 2026-08-31 archived survivor universe, not complete exchange universe','attempts_max':2,'requests':items}
(OUT/'KLINE_RECEIPTS.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('complete',len(items),'success',sum('sha256' in x for x in items),flush=True)
