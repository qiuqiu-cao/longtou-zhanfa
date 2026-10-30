from pathlib import Path
from datetime import datetime,timezone
import requests,json,hashlib
OUT=Path(__file__).resolve().parent
u='https://q.stock.sohu.com/hisHq';params={'code':'cn_603337','start':'20260901','end':'20261029','stat':'1','order':'A','period':'d'};attempts=[]
for i in range(3):
    rec={'code':'603337','attempt':i+1,'requested_at_utc':datetime.now(timezone.utc).isoformat(),'url':u,'params':params}
    try:
        r=requests.get(u,params=params,headers={'User-Agent':'Mozilla/5.0','Referer':'https://q.stock.sohu.com/'},timeout=(5,12));blob=r.content
        (OUT/f'603337_sohu_attempt{i+1}.response').write_bytes(blob);rec.update(received_at_utc=datetime.now(timezone.utc).isoformat(),status=r.status_code,sha256=hashlib.sha256(blob).hexdigest());r.raise_for_status()
        d=json.loads(blob.decode('gb18030')); rows=d[0]['hq']; assert d[0]['status']==0 and rows and max(x[0] for x in rows)<='2026-10-29';rec['valid']=True;attempts.append(rec);break
    except Exception as e:rec.update(valid=False,error=repr(e));attempts.append(rec)
(OUT/'603337_sohu_receipt.json').write_text(json.dumps(attempts,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(attempts,ensure_ascii=False))
