from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,hashlib,requests
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;raw=OUT/'raw_sohu';raw.mkdir(exist_ok=True)
def now():return datetime.now(timezone.utc).isoformat()
codes=set()
for path in (OUT/'pages').glob('*.json'):
    if not '2026-10-08'<=path.stem<='2026-10-29':continue
    p=json.loads(path.read_text(encoding='utf-8'));codes.update(x['code'] for x in p['rows'] if x['scope'] and x['kind']=='limit_up' and x['boards']==1)
codes=sorted(codes)
def get(code):
    p={'code':f'cn_{code}','start':'20260901','end':'20261029','stat':'1','order':'A','period':'d'};rec={'code':code,'params':p,'requested_at_utc':now(),'url':'https://q.stock.sohu.com/hisHq'}
    for attempt in range(2):
        try:
            r=requests.get(rec['url'],params=p,timeout=(5,15));r.raise_for_status();content=r.content;j=json.loads(content); rows=j[0]['hq']
            if j[0]['status']!=0 or not rows or max(x[0] for x in rows)>'2026-10-29':raise ValueError('bad/empty response')
            (raw/f'{code}.json').write_bytes(content);rec.update(received_at_utc=now(),sha256=hashlib.sha256(content).hexdigest(),rows=len(rows),first=rows[0][0],last=rows[-1][0],attempts=attempt+1);break
        except Exception as e:rec['error']=repr(e)[:250]
    else:rec['received_at_utc']=now();rec['attempts']=2
    (raw/f'{code}.receipt.json').write_text(json.dumps(rec,ensure_ascii=False),encoding='utf-8');return rec
print('candidate unique codes',len(codes),flush=True);items=[]
with ThreadPoolExecutor(max_workers=6) as p:
    for n,f in enumerate(as_completed([p.submit(get,c) for c in codes]),1):
        items.append(f.result())
        if n%100==0:print('done',n,'success',sum('sha256' in x for x in items),flush=True)
(OUT/'SOHU_RECEIPTS.json').write_text(json.dumps({'formed_at':now(),'requests':items},ensure_ascii=False,indent=2),encoding='utf-8')
print('completed',len(items),'success',sum('sha256' in x for x in items),flush=True)
