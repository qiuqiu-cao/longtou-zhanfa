from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import requests,json,hashlib,importlib.util
from bs4 import BeautifulSoup
O=Path(__file__).resolve().parent;ROOT=O.parents[1];C=O/'context';C.mkdir(exist_ok=True)
def now():return datetime.now(timezone.utc).isoformat()
def write(p,v):
 with p.open('x',encoding='utf8') as s:json.dump(v,s,ensure_ascii=False,indent=2)
def capture(job):
 name,url=job;rec=dict(name=name,url=url,requested_at=now(),scope='MAIN_BODY_CAPTURE_COMMENTS_REMAIN_SEPARATE')
 try:
  r=requests.get(url,timeout=(4,12),headers={'User-Agent':'Mozilla/5.0'});p=C/(name+'.html');p.write_bytes(r.content);r.raise_for_status()
  text=BeautifulSoup(r.content,'html.parser').get_text('\n',strip=True);(C/(name+'.txt')).write_text(text,encoding='utf8')
  rec.update(status='RECEIVED',received_at=now(),path=str(p),sha256=hashlib.sha256(r.content).hexdigest(),characters=len(text))
 except Exception as e:rec.update(status='UNAVAILABLE',error=str(e)[:300],received_at=now())
 return rec
spec=importlib.util.spec_from_file_location('notice_readonly',ROOT/'results/joint_review_20261019T092500Z/facts/qualify_and_notices.py');n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n);n.OUT=C;n.DAY='2026-10-29'
(C/'notices').mkdir(exist_ok=True)
facts=json.loads((O/'facts/RESOLVED_FACTS_v2_SCORES_MASKED.json').read_text(encoding='utf8'))
codes=sorted({r['code'] for r in facts['rows'] if r['kind']=='limit_up'})
with ThreadPoolExecutor(max_workers=8) as ex:
 contexts=list(ex.map(capture,[('forum_chunfeng_1029','https://www.tgb.cn/a/2whZtXktFJ7-1'),('forum_greedy_1029','https://www.tgb.cn/a/2wi0UQ6FipC'),('market_fund_1029','https://finance.sina.com.cn/jjxw/2026-10-29/doc-iniwwtuv8779993.shtml')]))
 indexes=list(ex.map(n.notices,codes))
write(C/'capture_manifest.json',dict(formed_at=now(),rows=contexts,scope='Forum正文仅启发假设；评论不自动作为作者观点或行情真值'))
write(C/'notices/index_manifest.json',dict(formed_at=now(),rows=indexes,post_close_acquisition=True))
focus={'002708','001219','601956','600715','603131','002565','003032','603703','600501','600592','000530','002172','002862'}
jobs=[]
for r in indexes:
 if r['code'] not in focus:continue
 selected=[i for i in r.get('items',[]) if any(w in i['title'] for w in ['异常波动','风险提示','减持','立案','停牌','三季度报告','业绩'])][:3]
 if not selected:selected=r.get('items',[])[:1]
 jobs.extend((r['code'],i) for i in selected)
with ThreadPoolExecutor(max_workers=8) as ex:docs=list(ex.map(n.pdf_one,jobs))
write(C/'notices/focus_fulltext_manifest.json',dict(formed_at=now(),rows=docs))
print(json.dumps(dict(context=contexts,indexes=len(indexes),docs=[{k:r.get(k) for k in ['code','title','status','text_path']} for r in docs]),ensure_ascii=True))
