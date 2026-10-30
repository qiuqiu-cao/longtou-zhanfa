from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,hashlib,requests,pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
raw=OUT/'raw_tencent';raw.mkdir(exist_ok=True)
def now():return datetime.now(timezone.utc).isoformat()
bars=pd.read_csv(ROOT/'data/v16/point_in_time_rebuild_cutoff_20260831_v2/point_in_time_current_daily_bars.csv',usecols=['date','code'],dtype={'code':str})
codes=set(bars[bars.date.eq('2026-08-31')].code)
for path in (OUT/'pages').glob('*.json'):
    p=json.loads(path.read_text(encoding='utf-8'))
    codes.update(x['code'] for x in p['rows'] if x['scope'])
codes=sorted(codes)
def get(code):
    symbol=('sh' if code.startswith('6') else 'sz')+code; params={'param':f'{symbol},day,2026-09-01,2026-10-29,100,'}
    rec={'code':code,'params':params,'requested_at_utc':now(),'url':'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get'}
    for attempt in range(2):
        try:
            r=requests.get(rec['url'],params=params,timeout=(4,10));r.raise_for_status();content=r.content;j=json.loads(content);rows=j['data'][symbol]['day']
            if not rows or max(x[0] for x in rows)>'2026-10-29':raise ValueError('empty or future rows')
            (raw/f'{code}.json').write_bytes(content);rec.update(received_at_utc=now(),sha256=hashlib.sha256(content).hexdigest(),rows=len(rows),first=rows[0][0],last=rows[-1][0],attempts=attempt+1);break
        except Exception as e:rec['error']=repr(e)[:250]
    else:rec['received_at_utc']=now();rec['attempts']=2
    (raw/f'{code}.receipt.json').write_text(json.dumps(rec,ensure_ascii=False),encoding='utf-8');return rec
print('requested codes',len(codes),flush=True); items=[]
with ThreadPoolExecutor(max_workers=8) as p:
    for n,f in enumerate(as_completed([p.submit(get,c) for c in codes]),1):
        items.append(f.result())
        if n%200==0:print('done',n,'success',sum('sha256' in x for x in items),flush=True)
(OUT/'TENCENT_RECEIPTS.json').write_text(json.dumps({'formed_at':now(),'requests':items},ensure_ascii=False,indent=2),encoding='utf-8')
print('completed',len(items),'success',sum('sha256' in x for x in items),flush=True)
