"""Bounded independent retry for HTTP 503, preserving both prior receipts and raw responses."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,hashlib,requests
OUT=Path(__file__).resolve().parent;raw=OUT/'raw_sohu_extra';raw.mkdir(exist_ok=False)
def now():return datetime.now(timezone.utc).isoformat()
receipt=json.loads((OUT/'SOHU_REPAIR_RECEIPTS.json').read_text(encoding='utf-8'));codes=[x['code'] for x in receipt['requests'] if not x['valid']]
def get(code):
    params={'code':f'cn_{code}','start':'20260901','end':'20261029','stat':'1','order':'A','period':'d'};attempts=[]
    for attempt in range(3):
        rec={'code':code,'attempt':attempt+1,'requested_at_utc':now(),'url':'https://q.stock.sohu.com/hisHq','params':params}
        try:
            r=requests.get(rec['url'],params=params,headers={'User-Agent':'Mozilla/5.0','Referer':'https://q.stock.sohu.com/'},timeout=(5,10));blob=r.content
            (raw/f'{code}_attempt{attempt+1}.response').write_bytes(blob);rec.update(received_at_utc=now(),http_status=r.status_code,sha256=hashlib.sha256(blob).hexdigest());r.raise_for_status()
            j=json.loads(blob.decode('gb18030'));rows=j[0]['hq']
            if j[0]['status']!=0 or not rows or max(x[0] for x in rows)>'2026-10-29':raise ValueError('empty or wrong date')
            rec.update(valid=True,rows=len(rows));attempts.append(rec);break
        except Exception as e:rec.update(valid=False,error=repr(e)[:200]);attempts.append(rec)
    (raw/f'{code}.receipt.json').write_text(json.dumps(attempts,ensure_ascii=False),encoding='utf-8');return attempts
print('retry candidates',len(codes),flush=True);items=[]
with ThreadPoolExecutor(max_workers=4) as p:
    for n,f in enumerate(as_completed([p.submit(get,c) for c in codes]),1):
        items.append(f.result())
        if n%100==0:print('done',n,'valid',sum(any(r['valid'] for r in q) for q in items),flush=True)
(OUT/'SOHU_EXTRA_RECEIPTS.json').write_text(json.dumps({'formed_at':now(),'requests':items},ensure_ascii=False,indent=2),encoding='utf-8')
print('completed',len(items),'valid',sum(any(r['valid'] for r in q) for q in items),flush=True)
