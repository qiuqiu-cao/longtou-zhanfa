"""New bounded collection after prior run's encoding failure; preserve old receipts."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,hashlib,requests
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;raw=OUT/'raw_sohu_repair';raw.mkdir(exist_ok=False)
def now():return datetime.now(timezone.utc).isoformat()
codes=set()
for path in (OUT/'pages').glob('*.json'):
    if '2026-10-08'<=path.stem<='2026-10-29':
        p=json.loads(path.read_text(encoding='utf-8'));codes.update(x['code'] for x in p['rows'] if x['scope'] and x['kind']=='limit_up' and x['boards']==1)
def get(code):
    p={'code':f'cn_{code}','start':'20260901','end':'20261029','stat':'1','order':'A','period':'d'};rec={'code':code,'params':p,'requested_at_utc':now(),'url':'https://q.stock.sohu.com/hisHq'}
    try:
        r=requests.get(rec['url'],params=p,timeout=(5,12));content=r.content; (raw/f'{code}.response').write_bytes(content)
        rec.update(received_at_utc=now(),http_status=r.status_code,sha256=hashlib.sha256(content).hexdigest());r.raise_for_status()
        j=json.loads(content.decode('gb18030'));rows=j[0]['hq']
        if j[0]['status']!=0 or not rows or max(x[0] for x in rows)>'2026-10-29':raise ValueError('bad or future rows')
        rec.update(valid=True,rows=len(rows),first=rows[0][0],last=rows[-1][0])
    except Exception as e:rec.update(valid=False,error=repr(e)[:250],completed_at_utc=now())
    (raw/f'{code}.receipt.json').write_text(json.dumps(rec,ensure_ascii=False),encoding='utf-8');return rec
print('candidates',len(codes),flush=True);items=[]
with ThreadPoolExecutor(max_workers=4) as p:
    for n,f in enumerate(as_completed([p.submit(get,c) for c in sorted(codes)]),1):
        items.append(f.result())
        if n%100==0:print('done',n,'valid',sum(x.get('valid',False) for x in items),flush=True)
(OUT/'SOHU_REPAIR_RECEIPTS.json').write_text(json.dumps({'formed_at':now(),'requests':items,'repair':'Provider payload is GB18030, not UTF-8; prior failed receipts preserved.'},ensure_ascii=False,indent=2),encoding='utf-8')
print('completed',len(items),'valid',sum(x.get('valid',False) for x in items),flush=True)
