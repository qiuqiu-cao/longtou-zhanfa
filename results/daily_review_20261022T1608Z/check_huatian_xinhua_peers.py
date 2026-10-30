"""Capture a predeclared small comparator set around the user-nominated event."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import requests

OUT=Path(__file__).resolve().parent / 'huatian_xinhua_minutes_20261022T170331Z'
SPECS=[('002156','通富微电','0.002156'),('600584','长电科技','1.600584'),
       ('002181','粤传媒','0.002181'),('002696','百洋股份','0.002696'),
       ('002724','海洋王','0.002724'),('603278','大业股份','1.603278'),
       ('sh_index','上证指数','1.000001'),('cyb_index','创业板指','0.399006')]
URL='https://push2his.eastmoney.com/api/qt/stock/trends2/get'

def now():return datetime.now(timezone.utc).isoformat()

def get(spec):
    code,name,secid=spec
    rec={'code':code,'name':name,'requested_at':now()}
    try:
        r=requests.get(URL,params={'secid':secid,'fields1':'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13',
            'fields2':'f51,f52,f53,f54,f55,f56,f57,f58','ndays':'1','iscr':'0','iscca':'0'},
            headers={'User-Agent':'Mozilla/5.0','Referer':'https://quote.eastmoney.com/'},timeout=10)
        p=OUT/('peer_eastmoney_'+code+'.response');p.write_bytes(r.content)
        rec.update(url=r.url,received_at=now(),http_status=r.status_code,raw_path=str(p),sha256=hashlib.sha256(r.content).hexdigest())
        r.raise_for_status();d=r.json().get('data')
        if not d:raise ValueError('Missing data')
        trends=d.get('trends',[])
        if not trends or any(not x.startswith('2026-10-22 ') for x in trends):raise ValueError('Missing or wrong-date trends')
        rec.update(status='EXACT_DATE_PUBLIC_MINUTES_SINGLE_SOURCE',preclose=d.get('preClose'),count=len(trends),source_name=d.get('name'))
        samples=[]
        for line in trends:
            v=line.split(',')
            if v[0][11:] in ['13:50','13:54','13:57','13:58','14:00','14:03','14:05','14:10','14:30','15:00']:
                samples.append({'time':v[0],'close':float(v[2]),'volume':float(v[5]),'amount':float(v[6])})
        rec['samples']=samples
    except Exception as e:rec.update(status='UNAVAILABLE',error=str(e)[:350])
    return rec

with ThreadPoolExecutor(max_workers=6) as pool:rows=list(pool.map(get,SPECS))
report={'target_date':'2026-10-22','formed_at':now(),'scope':'Fixed two semiconductor, four emotion comparators, and two indices; not the full market or proof of fund migration.',
        'sources':rows,'old_predictions_modified':False,'training_runs':0}
(OUT/'peer_capture_receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=True),flush=True)
