from pathlib import Path
from datetime import datetime,timezone
import requests,json,hashlib
OUT=Path(__file__).resolve().parent
u='https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketData.getKLineData';params={'symbol':'sh603838','scale':'240','ma':'no','datalen':'300'}
rec={'requested_at_utc':datetime.now(timezone.utc).isoformat(),'url':u,'params':params,'historical_use_cutoff':'2026-08-31'}
r=requests.get(u,params=params,timeout=(5,15));r.raise_for_status();blob=r.content
with (OUT/'603838_sina_history_raw.response').open('xb') as f:f.write(blob)
rows=json.loads(blob);selected=[x for x in rows if x['day']<='2026-08-31']
rec.update(received_at_utc=datetime.now(timezone.utc).isoformat(),sha256=hashlib.sha256(blob).hexdigest(),rows_before_cutoff=len(selected),first=selected[0]['day'],last=selected[-1]['day'],status_change_source='https://static.cninfo.com.cn/finalpage/2026-10-17/1225608268.PDF',listing_source='https://static.cninfo.com.cn/finalpage/2015-06-30/1201214267.PDF')
with (OUT/'603838_sina_history_receipt.json').open('x',encoding='utf-8') as f:json.dump(rec,f,ensure_ascii=False,indent=2)
print(json.dumps(rec,ensure_ascii=False,indent=2))
