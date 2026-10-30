from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
import json,requests,hashlib,importlib.util
from bs4 import BeautifulSoup
O=Path(__file__).resolve().parent;R=O.parents[1]
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
protected=json.loads((R/'results/daily_review_20261026T1323Z/PROTECTED_BEFORE.json').read_text(encoding='utf-8'))
paths=list(protected)+['results/daily_review_20261026T1323Z/'+p for p in ['INDEPENDENT_FULL_POOL_BEFORE_MODEL.json','BASE_scores_AFTER_INDEPENDENT.json','FINAL_SIX_EXECUTION_CARDS_20261027.json','20261026复盘与1027预期_完整双榜.md']]
write(O/'PROTECTED_BEFORE.json',{p:sha(R/p) for p in paths})
def minute(code):
 rec=dict(code=code,requested_at=now(),max_attempts=1)
 try:
  params=dict(secid=('1.' if code.startswith('6') else '0.')+code,fields1='f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13',fields2='f51,f52,f53,f54,f55,f56,f57,f58',ndays=1,iscr=0,iscca=0)
  r=requests.get('https://push2his.eastmoney.com/api/qt/stock/trends2/get',params=params,timeout=(4,8),headers={'User-Agent':'Mozilla/5.0'})
  p=O/'minutes'/(code+'_eastmoney.response');p.write_bytes(r.content);r.raise_for_status();d=r.json()['data'];assert d and all(t.startswith('2026-10-27') for t in d['trends'])
  rec.update(url=r.url,received_at=now(),sha256=sha(p),raw_path=str(p),status='EXACT_DATE_MINUTE_OHLC',data=d)
 except Exception as e:rec.update(status='UNAVAILABLE',error=str(e)[:250])
 return rec
codes=['600826','600865','600558','600303','002156','603758','601956','600828','001400','603376','603370','601566','002187','000058','603089']
with ThreadPoolExecutor(max_workers=6) as e:minutes=list(e.map(minute,codes))
write(O/'minutes/secondary_capture.json',minutes)
spec=importlib.util.spec_from_file_location('n',R/'results/joint_review_20261019T092500Z/facts/qualify_and_notices.py');n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n);n.OUT=O/'context'
idx=json.loads((O/'context/notices/index_manifest.json').read_text(encoding='utf-8'));jobs=[]
selection={'600838':'风险提示','603528':'风险提示','603028':'风险提示','000678':'异常波动','001219':'三季度报告','001318':'异常波动','002708':'异常波动','000058':'担保责任','603758':'三季度报告','601956':'三季度报告','001400':'三季度报告','600828':'三季度报告'}
for entry in idx['rows']:
 c=entry['code'];word=selection.get(c)
 if word:
  match=[x for x in entry.get('items',[]) if word in x['title']]
  if match:jobs.append((c,match[0]))
with ThreadPoolExecutor(max_workers=6) as e:pdf=list(e.map(n.pdf_one,jobs))
write(O/'context/notices/focus_fulltext_manifest.json',dict(formed_at=now(),records=pdf))
urls=[('yanbian_1026','https://m.tgb.cn/a/2wd3CQePUzU'),('second_1026','https://m.tgb.cn/a/2wdbgSIcLdH'),('context_gelonghui','https://m.gelonghui.com/p/7159024')]
def html(job):
 key,url=job;rec=dict(key=key,url=url,requested_at=now(),max_attempts=1)
 try:
  r=requests.get(url,timeout=(4,8),headers={'User-Agent':'Mozilla/5.0'});r.raise_for_status();r.encoding=r.apparent_encoding
  p=O/'context'/(key+'.html');p.write_bytes(r.content);s=BeautifulSoup(r.text,'html.parser')
  for t in s(['script','style']):t.decompose()
  txt=s.get_text('\n',strip=True);(O/'context'/(key+'.txt')).write_text(txt,encoding='utf-8')
  rec.update(received_at=now(),sha256=sha(p),chars=len(txt),status='RECEIVED_REQUIRES_BODY_REVIEW')
 except Exception as e:rec.update(status='UNAVAILABLE',error=str(e)[:250])
 return rec
with ThreadPoolExecutor(max_workers=3) as e:htmls=list(e.map(html,urls))
write(O/'context/limited_context_capture.json',htmls)
print(json.dumps(dict(minutes=[(r['code'],r['status']) for r in minutes],pdfs=[(r['code'],r['status']) for r in pdf],html=htmls),ensure_ascii=False))
