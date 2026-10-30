from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import importlib.util,json,hashlib,requests
from bs4 import BeautifulSoup
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
spec=importlib.util.spec_from_file_location('notice_readonly_helper',ROOT/'results/joint_review_20261019T092500Z/facts/qualify_and_notices.py')
n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n);n.OUT=OUT/'context';n.DAY='2026-10-28'
def now():return datetime.now(timezone.utc).isoformat()
def capture(job):
 name,url=job;rec={'name':name,'url':url,'requested_at':now(),'scope':'BODY_ONLY_NO_COMMENTS_OR_IMAGES'}
 try:
  r=requests.get(url,timeout=(4,10),headers={'User-Agent':'Mozilla/5.0'});p=n.OUT/(name+'.html');p.write_bytes(r.content);r.raise_for_status()
  b=BeautifulSoup(r.content,'html.parser');t=b.get_text('\n',strip=True);(n.OUT/(name+'.txt')).write_text(t,encoding='utf-8')
  rec.update(status='RECEIVED',received_at=now(),path=str(p),sha256=hashlib.sha256(r.content).hexdigest())
 except Exception as e:rec.update(status='UNAVAILABLE',error=str(e))
 return rec
if __name__=='__main__':
 indexes=json.loads((n.OUT/'notices/index_manifest.json').read_text(encoding='utf-8'))['rows']
 focus={'000678','001219','002708','601956','603037','600715','600828','001400','003028','002164','603897','000670','002526','603633','002862','002879'}
 jobs=[]
 for rec in indexes:
  if rec['code'] not in focus:continue
  its=rec.get('items',[])
  selected=[v for v in its if any(w in v['title'] for w in ['异常波动','风险提示','减持','立案','停牌','三季度报告','权益变动'])][:2]
  if not selected:selected=its[:1]
  jobs.extend((rec['code'],v) for v in selected)
 with ThreadPoolExecutor(max_workers=6) as ex:docs=list(ex.map(n.pdf_one,jobs))
 n.write(n.OUT/'notices/focus_fulltext_manifest.json',{'formed_at':now(),'rows':docs,'late_acquisition_not_historical_signal':True})
 with ThreadPoolExecutor(max_workers=3) as ex:ctx=list(ex.map(capture,[('yanbian_1028','https://m.tgb.cn/a/2wgkSCDy1bR'),('yanbian_1027','https://m.tgb.cn/a/2weGjaPomt8'),('market_close','https://finance.sina.com.cn/roll/2026-10-28/doc-iniwukpx9499564.shtml')]))
 n.write(n.OUT/'context_capture_manifest.json',{'formed_at':now(),'rows':ctx})
 print(json.dumps({'docs':[(r['code'],r['title'],r['status']) for r in docs],'context':ctx},ensure_ascii=False))
